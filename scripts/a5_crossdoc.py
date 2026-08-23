"""A5: does the compression basis generalize across documents?

Fits the joint-stack and trajectory bases on document A (technical preprint)
and evaluates them on document B (Moby-Dick, out of domain), against bases fit
on B itself, at matched bytes. This doubles as the cheapest real context test
(document granularity): if the per-doc basis clearly beats the foreign basis,
document-level context is buying compression (supports A4); if they tie, the
exploited structure is global (weakens the context reading, but strengthens
deployability -- a fixed basis ships with the model). Either outcome informs.

Usage: python scripts/a5_crossdoc.py
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from common import (load_model, load_capture, pre_rope_kv, rope_k, prefill_doc,
                    eval_window, behavior_metrics, qsim,
                    fit_joint, apply_joint, fit_traj, apply_traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-a", default="data/capture_PREPRINT_2048.pt")
    ap.add_argument("--doc-b", default="data/mobydick.txt")
    ap.add_argument("--b-skip-chars", type=int, default=30000,  # skip Gutenberg header
                    help="characters to skip before taking doc B text")
    ap.add_argument("--past", type=int, default=1792)
    ap.add_argument("--report", default="reports/a5_crossdoc.md")
    args = ap.parse_args()

    model = load_model()
    tok = AutoTokenizer.from_pretrained(model.config.name_or_path)
    P = args.past

    # document A state (basis donor)
    capA = load_capture(args.capture_a)
    kA, vA = pre_rope_kv(model, capA["hidden"])
    hA = capA["hidden"]

    # document B state (evaluation target)
    textB = open(args.doc_b, encoding="utf-8", errors="ignore").read()
    idsB = tok(textB[args.b_skip_chars:], return_tensors="pt").input_ids[:, :2048]
    T = idsB.shape[1]
    print(f"doc B: {T} tokens of {args.doc_b}")
    hB, keysB, valuesB, _ = prefill_doc(model, idsB)
    kB, vB = pre_rope_kv(model, hB)
    pos = torch.arange(T).unsqueeze(0)
    targetsB = idsB.squeeze(0)[P + 1:]
    base_logits = eval_window(model, idsB, keysB, valuesB, P)

    results = []

    def run(name, k_hat_pre, v_hat):
        logits = eval_window(model, idsB, rope_k(model, k_hat_pre, pos), v_hat, P)
        m = behavior_metrics(base_logits, logits, targetsB)
        m["name"] = name
        results.append(m)
        print(f"  {name:42s} KL {m['kl_mean']:.4f}  top1 {m['top1_agree']:.3f}  "
              f"dNLL {m['delta_nll']:+.4f}")

    # basis-free reference: is doc B pathological?
    run("quant K8 V4 (basis-free, 2.6x)", qsim(kB, 8, dim=2), qsim(vB, 4, dim=3))

    # joint-stack, 16x
    for src, kf, vf in [("per-doc (fit on B)", kB, vB), ("foreign (fit on A)", kA, vA)]:
        codec = fit_joint(kf, vf, P, 768)
        kh, vh = apply_joint(codec, kB, vB, coeff_bits=8)
        run(f"joint768 c8 16x, basis {src}", kh, vh)

    # trajectory, 32x
    for src, hf in [("per-doc (fit on B)", hB), ("foreign (fit on A)", hA)]:
        codec = fit_traj(hf, P, 384)
        hh = apply_traj(codec, hB, coeff_bits=8)
        kh, vh = pre_rope_kv(model, hh)
        run(f"traj384 c8 32x, basis {src}", kh, vh)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# A5: cross-document basis transfer (eval doc: Moby-Dick, {T} tokens; "
                f"foreign basis donor: kvspace preprint)\n\n"
                f"| method / basis | KL mean | KL p95 | top-1 | top-5 | dNLL |\n"
                f"|---|---|---|---|---|---|\n")
        for m in results:
            f.write(f"| {m['name']} | {m['kl_mean']:.4f} | {m['kl_p95']:.4f} "
                    f"| {m['top1_agree']:.3f} | {m['top5_overlap']:.3f} "
                    f"| {m['delta_nll']:+.4f} |\n")
        f.write("\nReading: per-doc >> foreign  => document context buys compression "
                "(supports A4 at doc granularity, per-doc adaptive bases worth their "
                "storage). per-doc ~= foreign => structure is global; a fixed basis "
                "ships with the model (deployability up, context evidence down).\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
