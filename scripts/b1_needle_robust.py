"""B1: error bars for the recall claim.

Repetitions with randomized needle entities, numbers, positions (jittered
chunk sizes), and different filler documents. Methods: full KV, behavioral
16x, behavioral 16x with int8-quantized basis (accounting upgrade), and
behavioral 32x. Also aggregates recall by depth quartile (A17).

Usage: python scripts/b1_needle_robust.py
"""

import argparse
import os
import random

import torch
from transformers import AutoTokenizer

from common import (load_model, pre_rope_kv, rope_k, prefill_doc, qsim,
                    fit_joint, apply_joint)
from a7_needle import TEMPLATE, probe_needle
from a8_behavior_basis import behavioral_weights, metric_scale

NAMES = ["Aldery", "Branwick", "Corvane", "Dunmore", "Ellery", "Fenwick",
         "Garrow", "Hollis", "Ilverton", "Jessamy", "Kestrel", "Lorimer",
         "Marbury", "Norwood", "Ostrey", "Pellham", "Quimby", "Rathbone",
         "Selwyn", "Thackery", "Umbers", "Verrell", "Winslow", "Yardley"]

FILLERS = [("../PREPRINT.md", 0), ("data/mobydick.txt", 30000),
           ("data/federalist.txt", 20000)]


def build_random_doc(tok, filler_text, rng, n_needles=8):
    filler_ids = tok(filler_text, return_tensors="pt").input_ids.squeeze(0)
    names = rng.sample(NAMES, n_needles)
    needles = [(nm, str(rng.randint(10000, 99999))) for nm in names]
    pieces, depths, off = [], [], 0
    for nm, num in needles:
        c = rng.randint(150, 230)
        pieces.append(filler_ids[off:off + c])
        off += c
        depths.append(sum(p.shape[0] for p in pieces))
        pieces.append(tok(TEMPLATE.format(nm, num),
                          return_tensors="pt").input_ids.squeeze(0))
    pieces.append(filler_ids[off:off + 200])
    return torch.cat(pieces).unsqueeze(0), needles, depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--report", default="reports/b1_needle_robust.md")
    args = ap.parse_args()

    model = load_model()
    tok = AutoTokenizer.from_pretrained(model.config.name_or_path)
    methods = ["full KV", "behavioral 16x (c8)", "behavioral 16x (c8, int8 basis)",
               "behavioral 32x (c4)"]
    tally = {m: {"hits": 0, "n": 0, "nll": [], "by_q": [0, 0, 0, 0],
                 "nq": [0, 0, 0, 0]} for m in methods}

    for rep in range(args.reps):
        rng = random.Random(1000 + rep)
        path, skip = FILLERS[rep % len(FILLERS)]
        text = open(path, encoding="utf-8", errors="ignore").read()[skip:]
        ids, needles, depths = build_random_doc(tok, text, rng)
        T = ids.shape[1]
        print(f"rep {rep}: filler {path}, {T} tokens, depths {depths}")
        hidden, keys_t, values_t, _ = prefill_doc(model, ids)
        k_pre, v_pre = pre_rope_kv(model, hidden)
        pos = torch.arange(T).unsqueeze(0)
        wK, wV = behavioral_weights(model, hidden)
        codec = fit_joint(k_pre, v_pre, T, 768, scale=metric_scale(wK, wV))
        codec_q = dict(codec, W=qsim(codec["W"], 8, dim=0))

        variants = {"full KV": (keys_t, values_t)}
        for name, cdc, cb in [("behavioral 16x (c8)", codec, 8),
                              ("behavioral 16x (c8, int8 basis)", codec_q, 8),
                              ("behavioral 32x (c4)", codec, 4)]:
            kh, vh = apply_joint(cdc, k_pre, v_pre, coeff_bits=cb)
            variants[name] = (rope_k(model, kh, pos), vh)

        for m in methods:
            keys, values = variants[m]
            per = []
            for i, ((nm, num), d) in enumerate(zip(needles, depths)):
                ok, nll = probe_needle(model, tok, keys, values, T, nm, num)
                t = tally[m]
                t["hits"] += ok
                t["n"] += 1
                t["nll"].append(nll)
                q = min(3, i // 2)
                t["by_q"][q] += ok
                t["nq"][q] += 1
                per.append("Y" if ok else ".")
            print(f"  {m:34s} [{''.join(per)}]")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# B1: needle recall robustness ({args.reps} reps, randomized "
                f"needles/positions, filler rotation)\n\n"
                "| method | recall | gold-NLL mean | by depth quartile (shallow->deep) |\n"
                "|---|---|---|---|\n")
        for m in methods:
            t = tally[m]
            quart = " / ".join(f"{t['by_q'][q]}/{t['nq'][q]}" for q in range(4))
            f.write(f"| {m} | {t['hits']}/{t['n']} "
                    f"| {sum(t['nll']) / len(t['nll']):.3f} | {quart} |\n")
        f.write("\nInt8 basis halves per-doc basis storage (9.4 -> 4.7 MB): net "
                "ratio at 32k improves from 11.6x to ~13.2x if recall holds.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
