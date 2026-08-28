"""A8: behavior-weighted subspace selection (the laptop gate).

Replaces the variance metric in the joint codec with a behavioral one:
  - K channels weighted by rms of the post-RoPE *queries* that will probe them
    (pooled over the query heads sharing each KV head, pair-symmetrized so the
    diagonal metric commutes with RoPE and can be applied pre-rotation).
    Rationale: attention-logit error is q . dk, so expected damage is
    dk' E[qq'] dk -- we use the diagonal of that metric.
  - V channels weighted by the column norms of o_proj (how strongly each value
    channel reaches the residual stream).
PCA then allocates components by expected behavioral damage, not by variance.

Gate questions:
  1. Does behavior-vs-rank become monotone (A15/A16)?
  2. Does needle recall recover at 16x+ (A7 frontier)?

Usage: python scripts/a8_behavior_basis.py
"""

import argparse
import os

import torch
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from common import (load_model, load_capture, pre_rope_kv, rope_k, eval_window,
                    behavior_metrics, fit_joint, apply_joint, pca_fit, qsim)
from a7_needle import NEEDLES, build_doc, probe_needle


# behavioral_weights / metric_scale now live in common.py (version- and
# architecture-robust: Qwen3 q_norm/head_dim, device/dtype aware); re-exported
# here so older scripts keep importing from this module.
from common import behavioral_weights, metric_scale  # noqa: F401,E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="data/capture_PREPRINT_2048.pt")
    ap.add_argument("--filler", default="../PREPRINT.md")
    ap.add_argument("--past", type=int, default=1792)
    ap.add_argument("--report", default="reports/a8_behavior_basis.md")
    args = ap.parse_args()

    model = load_model()
    tok = AutoTokenizer.from_pretrained(model.config.name_or_path)
    P = args.past

    # ---------- gate 1: monotonicity on the PREPRINT eval ----------
    cap = load_capture(args.capture)
    ids = cap["token_ids"]
    hidden = cap["hidden"]
    T = hidden.shape[1]
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)
    targets = ids.squeeze(0)[P + 1:]
    base_logits = eval_window(model, ids, cap["keys"], cap["values"], P)
    wK, wV = behavioral_weights(model, hidden)
    scale = metric_scale(wK, wV)

    sweep = []
    print("gate 1: behavioral-metric rank sweep (variance-metric phase-1 numbers "
          "for comparison: r96 KL 4.34, r192 KL 6.54, r384 KL 0.53, r768 KL 0.03)")
    for R in [96, 192, 384, 768]:
        codec = fit_joint(k_pre, v_pre, P, R, scale=scale)
        kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=16)
        logits = eval_window(model, ids, rope_k(model, kh, pos), vh, P)
        m = behavior_metrics(base_logits, logits, targets)
        m["R"] = R
        sweep.append(m)
        print(f"  behavioral rank {R:4d}  KL {m['kl_mean']:.4f}  "
              f"top1 {m['top1_agree']:.3f}  dNLL {m['delta_nll']:+.4f}")

    # ---------- gate 2: needle recall ----------
    print("gate 2: needle recall with behavioral metric")
    n_ids, depths = build_doc(tok, open(args.filler, encoding="utf-8").read())
    Tn = n_ids.shape[1]
    from common import prefill_doc
    hid_n, keys_n, values_n, _ = prefill_doc(model, n_ids)
    kn_pre, vn_pre = pre_rope_kv(model, hid_n)
    pos_n = torch.arange(Tn).unsqueeze(0)
    wKn, wVn = behavioral_weights(model, hid_n)
    scale_n = metric_scale(wKn, wVn)

    recall_rows = []

    def recall_run(name, keys, values):
        hits, nlls, per = 0, [], []
        for (nm, num), d in zip(NEEDLES, depths):
            ok, nll = probe_needle(model, tok, keys, values, Tn, nm, num)
            hits += ok
            nlls.append(nll)
            per.append("Y" if ok else ".")
        recall_rows.append((name, hits, sum(nlls) / len(nlls), "".join(per)))
        print(f"  {name:40s} recall {hits}/8  gold-NLL {recall_rows[-1][2]:.3f}  "
              f"[{recall_rows[-1][3]}]")

    recall_run("full KV", keys_n, values_n)

    for R, cb, label in [(768, 8, "16x"), (768, 4, "32x"), (384, 8, "32x")]:
        codec = fit_joint(kn_pre, vn_pre, Tn, R, scale=scale_n)
        kh, vh = apply_joint(codec, kn_pre, vn_pre, coeff_bits=cb)
        recall_run(f"behavioral joint{R} c{cb} ({label})",
                   rope_k(model, kh, pos_n), vh)

    # asymmetric split: K gets a private budget under its own metric
    def fit_single(X, fit_len, R, sc):
        mu, W = pca_fit(X[:fit_len] / sc, R)
        return mu, W, sc

    def apply_single(X, mu, W, sc, cb):
        Z = (X / sc - mu) @ W
        if cb < 16:
            Z = qsim(Z, cb, dim=0)
        return (Z @ W.T + mu) * sc

    L, kvh, _, hd = kn_pre.shape
    flatK = kn_pre.permute(2, 0, 1, 3).reshape(Tn, L * kvh * hd)
    flatV = vn_pre.permute(2, 0, 1, 3).reshape(Tn, L * kvh * hd)
    scK = (1.0 / (wKn / wKn.mean()).clamp_min(1e-3)).reshape(1, -1)
    scV = (1.0 / (wVn / wVn.mean()).clamp_min(1e-3)).reshape(1, -1)
    for rk, rv, cb, label in [(512, 256, 8, "16x"), (576, 192, 8, "16x")]:
        muK, WK, _ = fit_single(flatK, Tn, rk, scK)
        muV, WV, _ = fit_single(flatV, Tn, rv, scV)
        kh = apply_single(flatK, muK, WK, scK, cb).reshape(Tn, L, kvh, hd).permute(1, 2, 0, 3).contiguous()
        vh = apply_single(flatV, muV, WV, scV, cb).reshape(Tn, L, kvh, hd).permute(1, 2, 0, 3).contiguous()
        recall_run(f"behavioral split K{rk}/V{rv} c{cb} ({label})",
                   rope_k(model, kh, pos_n), vh)

    # ---------- report ----------
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# A8: behavior-weighted subspace selection\n\n"
                "K channels weighted by pooled post-RoPE query rms (diagonal of the "
                "attention-logit damage metric, pair-symmetrized for RoPE); V channels "
                "by o_proj column norms. PCA allocates by behavioral energy.\n\n"
                "## Gate 1: rank sweep, PREPRINT eval (variance metric in parens)\n\n"
                "| rank | KL behavioral | KL variance | top-1 behavioral |\n|---|---|---|---|\n")
        var_ref = {96: 4.3446, 192: 6.5384, 384: 0.5267, 768: 0.0327}
        for m in sweep:
            f.write(f"| {m['R']} | {m['kl_mean']:.4f} | ({var_ref[m['R']]:.4f}) "
                    f"| {m['top1_agree']:.3f} |\n")
        f.write("\n## Gate 2: needle recall (variance-metric A7 results: 16x -> 4/8, "
                "32x -> 2/8, trajectory -> 0/8)\n\n"
                "| method | recall | gold-NLL | per-needle |\n|---|---|---|---|\n")
        for nm, h, nll, per in recall_rows:
            f.write(f"| {nm} | {h}/8 | {nll:.3f} | `{per}` |\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
