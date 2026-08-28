"""C3: conditional-coding pilot — the founding hypothesis's first direct test.

Question (PREPRINT §6.1): at fixed decoder capacity and matched bytes, does
access to causal context reduce the state needed per token?

Linear pilot: in the behavioral basis, each token has a code Z_i (rank 768).
We fit ridge predictors of Z_i from
    (a) nothing (mean)                       -- unconditional
    (b) the token's own embedding            -- identity only
    (c) identity + previous tokens' codes    -- identity + causal context
and measure, on held-out later tokens:
    - fraction of Z variance explained (test)
    - Gaussian bits-saved proxy: 0.5*log2(var_ratio) summed over dims
    - behavioral: quantize the RESIDUAL at int4 vs quantizing Z at int4
      (same marginal bytes; predictor weights amortize like the basis),
      teacher-forced KL/top-1 over the evaluation window.

If (c) materially beats (b), context conditioning moves the rate-distortion
frontier even at linear capacity -- first positive evidence for
t_i + C_i -> Z_i beyond document granularity. If (c) ~= (b), the hypothesis
remains open pending nonlinear (4090) predictors. Prediction uses CLEAN
previous codes (optimistic bound; a deployed coder sees quantized ones).

Usage: python scripts/c3_conditional_pilot.py
"""

import argparse
import os

import torch

from common import (load_model, load_capture, pre_rope_kv, rope_k, eval_window,
                    behavior_metrics, behavioral_weights, metric_scale,
                    fit_joint, stack_flat, qsim)


def ridge(Xtr, Ytr, lam=10.0):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(Xc.shape[1]), Xc.T @ Yc)
    return mx, my, W


def predict(feats, mx, my, W):
    return (feats - mx) @ W + my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="data/capture_PREPRINT_2048.pt")
    ap.add_argument("--past", type=int, default=1792)
    ap.add_argument("--report", default="reports/c3_conditional_pilot.md")
    args = ap.parse_args()

    model = load_model()
    cap = load_capture(args.capture)
    ids = cap["token_ids"]
    hidden = cap["hidden"]
    T = hidden.shape[1]
    P = args.past
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)
    targets = ids.squeeze(0)[P + 1:]
    base_logits = eval_window(model, ids, cap["keys"], cap["values"], P)

    wK, wV = behavioral_weights(model, hidden)
    codec = fit_joint(k_pre, v_pre, P, 768, scale=metric_scale(wK, wV))
    Xs = stack_flat(k_pre, v_pre) / codec["std"]
    Z = (Xs - codec["mu"]) @ codec["W"]                      # (T, 768)
    emb = model.model.embed_tokens(ids.squeeze(0).to(
        next(model.parameters()).device)).detach().float().cpu()

    Zprev = torch.roll(Z, 1, dims=0)
    Zctx = torch.stack([Z[max(0, i - 8):max(1, i)].mean(0) for i in range(T)])
    feats = {
        "identity only": emb,
        "identity + prev code": torch.cat([emb, Zprev], dim=1),
        "identity + prev code + ctx mean": torch.cat([emb, Zprev, Zctx], dim=1),
    }

    n_tr = int(P * 0.75)
    sl_tr, sl_te = slice(16, n_tr), slice(n_tr, P)
    var_te = Z[sl_te].var(0).clamp_min(1e-9)
    rows = []
    residuals = {"unconditional (mean)": Z - Z[sl_tr].mean(0)}
    base_var = residuals["unconditional (mean)"][sl_te].var(0).clamp_min(1e-9)
    for name, F in feats.items():
        mx, my, W = ridge(F[sl_tr], Z[sl_tr])
        resid = Z - predict(F, mx, my, W)
        residuals[name] = resid
        rv = resid[sl_te].var(0).clamp_min(1e-9)
        r2 = 1 - (rv.sum() / base_var.sum()).item()
        bits = (0.5 * torch.log2(base_var / rv)).clamp_min(0).sum().item()
        rows.append((name, r2, bits))
        print(f"  {name:34s} test R^2 {r2:+.3f}   bits-saved proxy "
              f"{bits:.0f}/token (of {768 * 4} at int4)")

    # behavioral at matched bytes: int4 on Z vs int4 on residual + prediction
    print("behavioral comparison at 384 B/token (rank 768, int4):")
    beh = []

    def run(name, Zhat):
        Xh = (Zhat @ codec["W"].T + codec["mu"]) * codec["std"]
        L, kvh, _, hd = k_pre.shape
        half = Xh.shape[1] // 2
        g = lambda x: x.reshape(T, L, kvh, hd).permute(1, 2, 0, 3).contiguous()
        kh, vh = g(Xh[:, :half]), g(Xh[:, half:])
        m = behavior_metrics(base_logits,
                             eval_window(model, ids, rope_k(model, kh, pos), vh, P),
                             targets)
        beh.append((name, m))
        print(f"  {name:34s} KL {m['kl_mean']:.4f}  top1 {m['top1_agree']:.3f}  "
              f"dNLL {m['delta_nll']:+.4f}")

    run("unconditional: Z int4", qsim(Z, 4, dim=0))
    for name in feats:
        resid = residuals[name]
        run(f"conditional ({name}) resid int4", (Z - resid) + qsim(resid, 4, dim=0))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# C3: conditional-coding pilot (linear, rank-768 behavioral "
                "basis)\n\n## Predictability of per-token codes (held-out "
                "later tokens)\n\n| predictor | test R^2 | bits-saved proxy "
                "/token |\n|---|---|---|\n")
        for name, r2, bits in rows:
            f.write(f"| {name} | {r2:+.3f} | {bits:.0f} |\n")
        f.write("\n## Behavioral at matched 384 B/token (int4)\n\n"
                "| coding | KL | top-1 | dNLL |\n|---|---|---|---|\n")
        for name, m in beh:
            f.write(f"| {name} | {m['kl_mean']:.4f} | {m['top1_agree']:.3f} "
                    f"| {m['delta_nll']:+.4f} |\n")
        f.write("\nReading: the founding hypothesis predicts the "
                "context-conditioned rows beat identity-only at matched "
                "bytes. Linear predictors and clean-code conditioning make "
                "this an optimistic-capacity, first-evidence test only.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
