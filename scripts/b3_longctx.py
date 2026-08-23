"""B3: does the 16x recall result survive at ~8k context?

Builds a ~7.4k-token document (Moby-Dick filler) with 16 needles at depths
spanning the full context, compresses everything, probes recall. Also prints
the net-ratio accounting at this length (basis amortization).

Usage: python scripts/b3_longctx.py
"""

import argparse
import os
import random

import torch
from transformers import AutoTokenizer

from common import (load_model, pre_rope_kv, rope_k, prefill_doc,
                    fit_joint, apply_joint)
from a7_needle import TEMPLATE, probe_needle
from a8_behavior_basis import behavioral_weights, metric_scale
from b1_needle_robust import NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-needles", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=420)
    ap.add_argument("--report", default="reports/b3_longctx.md")
    args = ap.parse_args()

    model = load_model()
    tok = AutoTokenizer.from_pretrained(model.config.name_or_path)
    rng = random.Random(7)
    text = open("data/mobydick.txt", encoding="utf-8", errors="ignore").read()[30000:]
    filler = tok(text, return_tensors="pt").input_ids.squeeze(0)

    names = rng.sample(NAMES, args.n_needles)
    needles = [(nm, str(rng.randint(10000, 99999))) for nm in names]
    pieces, depths, off = [], [], 0
    for nm, num in needles:
        pieces.append(filler[off:off + args.chunk])
        off += args.chunk
        depths.append(sum(p.shape[0] for p in pieces))
        pieces.append(tok(TEMPLATE.format(nm, num),
                          return_tensors="pt").input_ids.squeeze(0))
    pieces.append(filler[off:off + args.chunk])
    ids = torch.cat(pieces).unsqueeze(0)
    T = ids.shape[1]
    print(f"long doc: {T} tokens, {args.n_needles} needles, "
          f"depths {depths[0]}..{depths[-1]}")

    import time
    t0 = time.time()
    hidden, keys_t, values_t, _ = prefill_doc(model, ids)
    print(f"prefill: {time.time() - t0:.0f}s")
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)
    wK, wV = behavioral_weights(model, hidden)
    codec = fit_joint(k_pre, v_pre, T, 768, scale=metric_scale(wK, wV))

    rows = []

    def recall_run(name, keys, values):
        hits, nlls, per = 0, [], []
        for (nm, num), d in zip(needles, depths):
            ok, nll = probe_needle(model, tok, keys, values, T, nm, num)
            hits += ok
            nlls.append(nll)
            per.append("Y" if ok else ".")
        rows.append((name, hits, sum(nlls) / len(nlls), "".join(per)))
        print(f"  {name:24s} recall {hits}/{args.n_needles}  "
              f"gold-NLL {rows[-1][2]:.3f}  [{rows[-1][3]}]")

    recall_run("full KV", keys_t, values_t)
    for name, cb in [("behavioral 16x (c8)", 8), ("behavioral 32x (c4)", 4)]:
        kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=cb)
        recall_run(name, rope_k(model, kh, pos), vh)

    basis_mb = codec["W"].numel() * 2 / 2**20
    net16 = 768 + basis_mb * 2**20 / T
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# B3: recall at {T}-token context (16 needles, full-depth spread)\n\n"
                "| method | recall | gold-NLL | per-needle (shallow->deep) |\n"
                "|---|---|---|---|\n")
        for nm, h, nll, per in rows:
            f.write(f"| {nm} | {h}/{args.n_needles} | {nll:.3f} | `{per}` |\n")
        f.write(f"\nAccounting at this length: coefficients 768 B/token + fp16 basis "
                f"{basis_mb:.1f} MB / {T} tokens = {net16:.0f} B/token net "
                f"({12288 / net16:.1f}x vs full fp16 KV).\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
