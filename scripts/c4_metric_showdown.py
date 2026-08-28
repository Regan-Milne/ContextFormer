"""C4: metric showdown at stress ranks — settle what actually causes the
metric effects before rewording the preprint's A8 claim.

C1 showed the 16x/2k/8-needle test is saturated (even raw PCA passes), and
that the earlier catastrophic "variance metric" was specifically STANDARDIZED
PCA. This run discriminates at the ranks where metrics separate:

  raw PCA (no scaling)      768 / 384 / 192
  standardized PCA          384 / 192          (the published villain)
  behavioral diagonal       192                (known: 768->8/8, 384->7/8)
  non-diagonal              192                (known: 384->8/8; does 64x hold?)

Each config: 8-needle recall + teacher-forced KL over the last 256 tokens.

Usage: python scripts/c4_metric_showdown.py
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from common import (load_model, prefill_doc, pre_rope_kv, rope_k, eval_window,
                    behavior_metrics, behavioral_weights, metric_scale,
                    fit_joint, apply_joint, stack_flat)
from a7_needle import NEEDLES, build_doc, probe_needle
from c1_basis_science import block_grams, nondiag_codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--filler", default="../PREPRINT.md")
    ap.add_argument("--report", default="reports/c4_metric_showdown.md")
    args = ap.parse_args()

    model = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids, depths = build_doc(tok, open(args.filler, encoding="utf-8").read())
    T = ids.shape[1]
    W = T - 256
    hidden, keys_t, values_t, _ = prefill_doc(model, ids)
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)
    targets = ids.squeeze(0)[W + 1:]
    base_logits = eval_window(model, ids, keys_t, values_t, W)

    wK, wV = behavioral_weights(model, hidden)
    scale_beh = metric_scale(wK, wV)
    ones = torch.ones(1, stack_flat(k_pre, v_pre).shape[1])
    CK, CV = block_grams(model, hidden)

    rows = []

    def run(name, kh, vh):
        keys = rope_k(model, kh, pos)
        hits, per = 0, []
        for (nm, num), d in zip(NEEDLES, depths):
            ok, _ = probe_needle(model, tok, keys, vh, T, nm, num)
            hits += ok
            per.append("Y" if ok else ".")
        m = behavior_metrics(base_logits, eval_window(model, ids, keys, vh, W),
                             targets)
        rows.append((name, hits, "".join(per), m))
        print(f"  {name:28s} recall {hits}/8 [{rows[-1][2]}]  "
              f"KL {m['kl_mean']:.4f}  top1 {m['top1_agree']:.3f}")

    def joint(name, scale, R):
        codec = fit_joint(k_pre, v_pre, T, R, scale=scale)
        kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
        run(name, kh, vh)

    for R in (768, 384, 192):
        joint(f"raw rank {R}", ones, R)
    for R in (384, 192):
        joint(f"standardized rank {R}", None, R)
    joint("behavioral rank 192", scale_beh, 192)
    kh, vh = nondiag_codec(k_pre, v_pre, CK, CV, T, 192, 8)
    run("non-diag rank 192", kh, vh)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# C4: metric showdown at stress ranks (2k doc, c8 coeffs; "
                "rank = B/token)\n\n| metric/rank | recall | per-needle | KL | "
                "top-1 | dNLL |\n|---|---|---|---|---|---|\n")
        for name, hits, per, m in rows:
            f.write(f"| {name} | {hits}/8 | `{per}` | {m['kl_mean']:.4f} "
                    f"| {m['top1_agree']:.3f} | {m['delta_nll']:+.4f} |\n")
        f.write("\nContext: published behavioral numbers at these ranks -- "
                "768: 8/8 rec; 384: 7/8 rec, KL 0.11 (c16); 192: KL 2.55 (c16). "
                "Standardized (published as 'variance'): 384 KL 0.53, 192 KL "
                "6.54 (c16), 16x recall 4/8. This table decides whether the "
                "A8 claim should be 'standardization is the villain; explicit "
                "behavioral weighting is the principled form of what raw "
                "magnitude weighting does by accident' -- and which metric "
                "actually owns the stressed-rank frontier.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
