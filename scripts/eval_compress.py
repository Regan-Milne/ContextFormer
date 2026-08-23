"""Phase 1b: analytic compressors -> behavioral Pareto curve.

Each compressor maps the past tokens' per-token state to a smaller stored
representation and back, operating on pre-RoPE K and V (RoPE re-applied at
reconstruction; position is free since every token is retained). The
reconstructed cache replaces the true cache for the past `--past` tokens, and
the model's next-token behavior over the remaining window is compared to the
full-KV baseline (teacher-forced KL, top-1/top-5 agreement, delta-NLL).

Bases/scales are fit on the past tokens only (per-document adaptive basis,
amortized storage) -- an upper bound on analytic compressibility; held-out
bases are Phase 2 business.

Usage: python scripts/eval_compress.py --capture data/capture_PREPRINT_2048.pt
"""

import argparse
import os
import time

import torch

from common import (load_model, load_capture, pre_rope_kv, rope_k,
                    eval_window, behavior_metrics)

FP16_BYTES_PER_TOKEN = None   # filled from config


def qsim(x, bits, dim):
    """Symmetric fake-quant along `dim` (scale computed over that dim)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def pca_fit(X, r):
    """X: (N, D) fit rows. Returns (mean, components (D, r))."""
    mu = X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X - mu, full_matrices=False)
    return mu, Vh[:r].T


def pca_codec(X, mu, W, coeff_bits):
    Z = (X - mu) @ W
    if coeff_bits < 16:
        Z = qsim(Z, coeff_bits, dim=0)   # per-component scale, amortized
    return Z @ W.T + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="data/capture_PREPRINT_2048.pt")
    ap.add_argument("--past", type=int, default=1792)
    ap.add_argument("--report", default="reports/phase1_pareto.md")
    args = ap.parse_args()

    model = load_model()
    cap = load_capture(args.capture)
    ids = cap["token_ids"]
    hidden = cap["hidden"]
    keys_true, values_true = cap["keys"], cap["values"]     # post-RoPE ground truth
    L1, T, D = hidden.shape
    L = L1 - 1
    P = args.past
    k_pre, v_pre = pre_rope_kv(model, hidden)               # (L, kvh, T, hd)
    kvh, hd = k_pre.shape[1], k_pre.shape[3]
    kv_dims = kvh * hd
    pos = torch.arange(T).unsqueeze(0)
    fp16_bytes = L * 2 * kv_dims * 2
    targets = ids.squeeze(0)[P + 1:]

    # sanity: two-pass with exact cache must reproduce a fresh full forward
    print("sanity: two-pass vs full forward...")
    base_logits = eval_window(model, ids, keys_true, values_true, P)
    with torch.no_grad():
        full = model(ids).logits.squeeze(0)[P:].float()
    diff = (base_logits - full).abs().max().item()
    agree = (base_logits.argmax(-1) == full.argmax(-1)).float().mean().item()
    print(f"  max |two-pass - full| logit diff: {diff:.2e} "
          f"(fp16 cache storage noise), top-1 agreement: {agree:.4f}")
    assert diff < 0.5 and agree > 0.99, "two-pass harness does not match full forward"

    def flatK(x):
        return x.permute(2, 0, 1, 3).reshape(T, L * kv_dims)

    def unflatK(x):
        return x.reshape(T, L, kvh, hd).permute(1, 2, 0, 3).contiguous()

    results = []

    def run(name, k_hat_pre, v_hat, bytes_tok):
        t0 = time.time()
        k_hat = rope_k(model, k_hat_pre, pos)
        # only the past is compressed; window tokens' KV are computed live
        logits = eval_window(model, ids, k_hat, v_hat, P)
        m = behavior_metrics(base_logits, logits, targets)
        m.update(name=name, bytes=bytes_tok, ratio=fp16_bytes / bytes_tok)
        results.append(m)
        print(f"  {name:34s} {bytes_tok:6.0f} B/tok ({m['ratio']:5.1f}x)  "
              f"KL {m['kl_mean']:.4f}  top1 {m['top1_agree']:.3f}  "
              f"dNLL {m['delta_nll']:+.4f}  [{time.time() - t0:.0f}s]")

    print("running compressors (bases fit on past tokens only)...")

    # --- quantization ---
    for kb, vb in [(8, 8), (4, 8), (8, 4), (4, 4), (2, 4), (4, 2), (2, 2)]:
        kq = qsim(k_pre, kb, dim=2)                     # per-channel over tokens
        vq = qsim(v_pre, vb, dim=3)                     # per-token over dims
        b = L * kv_dims * (kb + vb) / 8 + L * kvh * 2   # + per-token V scales
        run(f"quant K{kb} V{vb}", kq, vq, b)

    # --- per-layer low-rank (fit per layer, K and V separately) ---
    for r, cb in [(64, 16), (32, 16), (16, 16), (8, 16), (4, 16), (16, 8), (8, 8)]:
        k_hat = torch.empty_like(k_pre)
        v_hat = torch.empty_like(v_pre)
        for l in range(L):
            for src, dst in ((k_pre, k_hat), (v_pre, v_hat)):
                X = src[l].permute(1, 0, 2).reshape(T, kv_dims)
                mu, W = pca_fit(X[:P], r)
                Xh = pca_codec(X, mu, W, cb)
                dst[l] = Xh.reshape(T, kvh, hd).permute(1, 0, 2)
        b = 2 * L * r * cb / 8
        run(f"per-layer rank{r} c{cb}", k_hat, v_hat, b)

    # --- joint cross-layer stack ---
    stack = torch.cat([flatK(k_pre), flatK(v_pre)], dim=1)      # (T, 2*L*kv_dims)
    std = stack[:P].std(0, keepdim=True).clamp_min(1e-6)
    for R, cb in [(768, 16), (384, 16), (192, 16), (96, 16), (48, 16),
                  (768, 8), (768, 4), (384, 8), (192, 8)]:
        Xs = stack / std
        mu, W = pca_fit(Xs[:P], R)
        Xh = pca_codec(Xs, mu, W, cb) * std
        k_hat = unflatK(Xh[:, : L * kv_dims])
        v_hat = unflatK(Xh[:, L * kv_dims:])
        run(f"joint-stack rank{R} c{cb}", k_hat, v_hat, R * cb / 8)

    # --- trajectory-space joint low-rank (decode = PCA expand + frozen proj) ---
    traj = hidden.permute(1, 0, 2).reshape(T, L1 * D)
    tstd = traj[:P].std(0, keepdim=True).clamp_min(1e-6)
    for R, cb in [(384, 16), (384, 8), (384, 4), (192, 16), (192, 8), (96, 16), (96, 8)]:
        Xs = traj / tstd
        mu, W = pca_fit(Xs[:P], R)
        Xh = (pca_codec(Xs, mu, W, cb) * tstd).reshape(T, L1, D).permute(1, 0, 2)
        k_hat, v_hat = pre_rope_kv(model, Xh.contiguous())
        run(f"trajectory rank{R} c{cb}", k_hat, v_hat, R * cb / 8)

    # --- report ---
    results.sort(key=lambda m: -m["bytes"])
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# Phase 1b: bytes/token vs behavior ({cap['model']}, doc {cap['doc']})\n\n")
        f.write(f"Past = {P} tokens compressed, window = {T - P} tokens evaluated "
                f"teacher-forced against the full-KV baseline. fp16 full KV = "
                f"{fp16_bytes} B/token. Bases fit on past tokens (per-doc adaptive, "
                f"amortized). Recompute-from-tokens baseline: 0 B/token marginal, "
                f"full prefill compute -- every method below must justify its bytes "
                f"by decode compute saved.\n\n")
        f.write("| method | B/token | ratio | KL mean | KL p95 | top-1 | top-5 | dNLL |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        f.write(f"| full KV fp16 | {fp16_bytes} | 1.0x | 0 | 0 | 1.000 | 1.000 | +0.0000 |\n")
        for m in results:
            f.write(f"| {m['name']} | {m['bytes']:.0f} | {m['ratio']:.1f}x "
                    f"| {m['kl_mean']:.4f} | {m['kl_p95']:.4f} | {m['top1_agree']:.3f} "
                    f"| {m['top5_overlap']:.3f} | {m['delta_nll']:+.4f} |\n")
        f.write(f"\nBaseline NLL over window: {results[0]['base_nll']:.4f} nats/token.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
