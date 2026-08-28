"""C2: what rank does 7.5k context require? First points on the
rank-vs-context-length curve (the scale gate's central question).

At ~1.9k tokens, rank 768 (16x marginal) held 24/24. At ~7.5k it held 14/16.
This sweep asks what rank restores full recall at 7.5k, i.e. how the required
latent size grows with document length under a per-doc behavioral basis.

Usage: python scripts/c2_rank_length.py
"""

import argparse
import os
import random
import time

import torch
from transformers import AutoTokenizer

from common import (load_model, prefill_doc, pre_rope_kv, rope_k, kv_geometry,
                    behavioral_codec, apply_joint)
from a7_needle import TEMPLATE, probe_needle
from b1_needle_robust import NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-needles", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=420)
    ap.add_argument("--ranks", default="768,1152,1536")
    ap.add_argument("--report", default="reports/c2_rank_length.md")
    args = ap.parse_args()

    model = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    _, _, _, _, fp16_b = kv_geometry(model.config)
    rng = random.Random(7)   # same construction as b3 for comparability
    text = open("data/mobydick.txt", encoding="utf-8", errors="ignore").read()[30000:]
    filler = tok(text, return_tensors="pt").input_ids.squeeze(0)
    names = rng.sample(NAMES, args.n_needles)
    needles = [(nm, str(rng.randint(10000, 99999))) for nm in names]
    pieces, depths, off = [], [], 0
    for nm, num in needles:
        pieces.append(filler[off:off + args.chunk]); off += args.chunk
        depths.append(sum(p.shape[0] for p in pieces))
        pieces.append(tok(TEMPLATE.format(nm, num),
                          return_tensors="pt").input_ids.squeeze(0))
    pieces.append(filler[off:off + args.chunk])
    ids = torch.cat(pieces).unsqueeze(0)
    T = ids.shape[1]
    print(f"doc {T} tokens, {args.n_needles} needles")

    t0 = time.time()
    hidden, keys_t, values_t, _ = prefill_doc(model, ids)
    print(f"prefill {time.time() - t0:.0f}s")
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)

    rows = []

    def recall_run(name, keys, values):
        hits, nlls, per = 0, [], []
        for (nm, num), d in zip(needles, depths):
            ok, nll = probe_needle(model, tok, keys, values, T, nm, num)
            hits += ok
            nlls.append(nll)
            per.append("Y" if ok else ".")
        rows.append((name, hits, sum(nlls) / len(nlls), "".join(per)))
        print(f"  {name:36s} {hits}/{args.n_needles}  NLL {rows[-1][2]:.3f}  "
              f"[{rows[-1][3]}]")

    recall_run("full KV", keys_t, values_t)
    for R in [int(r) for r in args.ranks.split(",")]:
        codec = behavioral_codec(model, hidden, k_pre, v_pre, T, R)
        kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
        recall_run(f"behavioral rank {R} c8 ({fp16_b / R:.1f}x)",
                   rope_k(model, kh, pos), vh)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# C2: rank vs context length ({T} tokens, "
                f"{args.n_needles} needles)\n\n"
                "| method | recall | gold-NLL | per-needle |\n|---|---|---|---|\n")
        for nm, h, nll, per in rows:
            f.write(f"| {nm} | {h}/{args.n_needles} | {nll:.3f} | `{per}` |\n")
        f.write("\nReference points: rank 768 held 24/24 at ~1.9k tokens "
                "(B1) and 14/16 at ~7.5k (B3). This sweep locates the rank "
                "that restores full recall at ~7.5k, the first data on how "
                "latent size must scale with context length.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
