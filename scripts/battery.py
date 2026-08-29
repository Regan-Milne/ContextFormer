"""Expanded recall battery (PREPRINT §8): typed exact-detail needles.

Needle types, each stressing a different addressability mode:
  number      -- 5-digit secrets (the original test)
  uuid        -- 8-hex tokens (arbitrary alphanumeric exactness)
  identifier  -- snake_case function names (code-shaped exactness)
  person      -- first+last names (natural-language exactness)
  confusable  -- PAIRS of needles with near-identical entities AND values
                 sharing digits (retrieval-collision stress, the predicted
                 failure mode of compressed keys)

Model- and scale-agnostic: ranks derive from the model's own stack geometry
(16x = stack_fp16_bytes/16 with int8 coeffs, etc.). Works CPU or CUDA.

Not yet implemented here (4090 day-2, needs an instruct model): the
multi-instruction IFEval-style compliance sweep and the system-prompt-leakage
test from "Pitfalls of KV Cache Compression" — scaffold deliberately separate.

Usage:
  python scripts/battery.py                                   # laptop check
  python scripts/battery.py --model Qwen/Qwen3-4B --target-tokens 8000
"""

import argparse
import os
import random

import torch
from transformers import AutoTokenizer

from common import (load_model, prefill_doc, pre_rope_kv, rope_k, kv_geometry,
                    behavioral_codec, apply_joint, log)
from a7_needle import probe_needle

FIRST = ["Marlowe", "Kestrel", "Ottoline", "Bram", "Sable", "Quill", "Vesper",
         "Corin", "Isolde", "Fenn"]
LAST = ["Ashford", "Blackwood", "Carrow", "Dunmore", "Ellery", "Farrow",
        "Grimsby", "Holloway", "Ives", "Jessop"]
ENTITIES = ["Aldery", "Branwick", "Corvane", "Dunmore", "Ellery", "Fenwick",
            "Garrow", "Hollis", "Ilverton", "Jessamy", "Lorimer", "Marbury",
            "Norwood", "Ostrey", "Pellham", "Quimby", "Rathbone", "Selwyn"]


def gen_needles(rng, per_type):
    """List of dicts: {type, insert, query, gold}."""
    ents = rng.sample(ENTITIES, per_type * 4 + 2)
    out = []
    i = 0
    for _ in range(per_type):
        e = ents[i]; i += 1
        v = str(rng.randint(10000, 99999))
        out.append(dict(type="number",
                        insert=f"\nFor the record, the secret number for Project {e} is {v}.\n",
                        query=f"\nFor the record, the secret number for Project {e} is",
                        gold=v))
    for _ in range(per_type):
        e = ents[i]; i += 1
        v = "".join(rng.choice("0123456789abcdef") for _ in range(8))
        out.append(dict(type="uuid",
                        insert=f"\nThe access token for the {e} gateway is {v}.\n",
                        query=f"\nThe access token for the {e} gateway is",
                        gold=v))
    for _ in range(per_type):
        e = ents[i]; i += 1
        v = f"{e.lower()}_merge_v{rng.randint(2, 9)}"
        out.append(dict(type="identifier",
                        insert=f"\nThe function that finalizes the {e} pipeline is called {v}().\n",
                        query=f"\nThe function that finalizes the {e} pipeline is called",
                        gold=v))
    for _ in range(per_type):
        e = ents[i]; i += 1
        v = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        out.append(dict(type="person",
                        insert=f"\nThe auditor assigned to the {e} account is {v}.\n",
                        query=f"\nThe auditor assigned to the {e} account is",
                        gold=v))
    # confusable pair: same entity stem, values sharing 3 leading digits
    e = ents[i]
    stem = rng.randint(100, 999)
    va, vb = f"{stem}{rng.randint(10, 99)}", f"{stem}{rng.randint(10, 99)}"
    while vb == va:
        vb = f"{stem}{rng.randint(10, 99)}"
    for suffix, v in (("Alpha", va), ("Beta", vb)):
        out.append(dict(type="confusable",
                        insert=f"\nFor the record, the secret number for Project {e}-{suffix} is {v}.\n",
                        query=f"\nFor the record, the secret number for Project {e}-{suffix} is",
                        gold=v))
    rng.shuffle(out)
    return out


def build_doc(tok, filler_text, needles, target_tokens):
    filler = tok(filler_text, return_tensors="pt").input_ids.squeeze(0)
    n = len(needles)
    needle_ids = [tok(nd["insert"], return_tensors="pt",
                      add_special_tokens=False).input_ids.squeeze(0)
                  for nd in needles]
    chunk = max(40, (target_tokens - sum(x.shape[0] for x in needle_ids)) // (n + 1))
    pieces, depths, off = [], [], 0
    for nid in needle_ids:
        pieces.append(filler[off:off + chunk]); off += chunk
        depths.append(sum(p.shape[0] for p in pieces))
        pieces.append(nid)
    pieces.append(filler[off:off + chunk])
    return torch.cat(pieces).unsqueeze(0), depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default=None, choices=[None, "fp32", "bf16", "fp16"])
    ap.add_argument("--filler", default="data/mobydick.txt")
    ap.add_argument("--filler-skip", type=int, default=30000)
    ap.add_argument("--target-tokens", type=int, default=1800)
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--ratios", default="16,32")
    ap.add_argument("--metric", default="behavioral",
                    choices=["behavioral", "variance", "raw", "nondiag"],
                    help="codec metric for the main methods")
    ap.add_argument("--variance-control", action="store_true",
                    help="also run the variance-metric codec at each ratio")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="prefill in segments of N tokens (0 = one-shot); "
                         "identical output, bounded VRAM peak")
    ap.add_argument("--vram-frac", type=float, default=1.0,
                    help="hard per-process VRAM cap as a fraction of the "
                         "card; allocations beyond it raise OOM on our side "
                         "instead of evicting other apps (GPU sharing)")
    args = ap.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
             "fp16": torch.float16, None: None}[args.dtype]

    if args.vram_frac < 1.0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.vram_frac)
        log(f"VRAM capped at {args.vram_frac:.0%} of the card")
    model = load_model(args.model, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    L, kvh, hd, stack_dim, fp16_b = kv_geometry(model.config)
    rng = random.Random(args.seed)
    needles = gen_needles(rng, args.per_type)
    text = open(args.filler, encoding="utf-8", errors="ignore").read()[args.filler_skip:]
    ids, depths = build_doc(tok, text, needles, args.target_tokens)
    T = ids.shape[1]
    types = sorted(set(nd["type"] for nd in needles))
    print(f"{args.model}: doc {T} tokens, {len(needles)} needles "
          f"({', '.join(types)}), fp16 KV {fp16_b} B/tok", flush=True)

    log(f"prefill ({T} tokens, chunk={args.prefill_chunk or 'one-shot'})...")
    hidden, keys_t, values_t, _ = prefill_doc(model, ids,
                                              chunk=args.prefill_chunk,
                                              want_logits=False)
    log("pre-RoPE KV recompute...")
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)
    dev, dt = next(model.parameters()).device, model.dtype
    from common import behavioral_weights, metric_scale
    bscale = metric_scale(*behavioral_weights(model, hidden))
    if args.metric != "nondiag":
        del hidden  # (L+1,T,D) fp32, ~12 GB at 4B/32k; only the behavioral
        hidden = None  # weights and nondiag grams ever read it

    methods = [("full KV", None, None)]
    for ratio in [int(r) for r in args.ratios.split(",")]:
        R = min(fp16_b // ratio, T - 8, stack_dim - 8)   # int8 coeffs: R bytes/tok
        methods.append((f"{args.metric} {ratio}x (rank {R} c8)", R, args.metric))
        if args.variance_control and args.metric != "variance":
            methods.append((f"variance {ratio}x (rank {R} c8)", R, "variance"))

    grams = None
    rows = {}
    for name, R, metric in methods:
        if R is None:
            keys, values = keys_t, values_t
        else:
            log(f"fit codec: {name}...")
            from common import fit_joint, stack_flat
            if metric == "behavioral":
                codec = fit_joint(k_pre, v_pre, T, R, scale=bscale)
                kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
            elif metric == "variance":
                codec = fit_joint(k_pre, v_pre, T, R)
                kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
            elif metric == "raw":
                ones = torch.ones(1, stack_flat(k_pre, v_pre).shape[1])
                codec = fit_joint(k_pre, v_pre, T, R, scale=ones)
                kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
            else:  # nondiag
                from c1_basis_science import block_grams, nondiag_codec
                if grams is None:
                    grams = block_grams(model, hidden)
                kh, vh = nondiag_codec(k_pre, v_pre, *grams, T, R, 8)
            keys, values = rope_k(model, kh, pos), vh
        # mount once per method: build_cache's .to() is then a device-local
        # no-op instead of a full fp32 stack upload per probe (2 per needle)
        keys = keys.to(device=dev, dtype=dt)
        values = values.to(device=dev, dtype=dt)
        log(f"probe: {name} ({len(needles)} needles)...")
        per_type = {t: [0, 0] for t in types}
        nlls = []
        for i, (nd, d) in enumerate(zip(needles, depths)):
            ok, nll = probe_needle(model, tok, keys, values, T, "",
                                   nd["gold"], query_template=nd["query"])
            per_type[nd["type"]][0] += ok
            per_type[nd["type"]][1] += 1
            nlls.append(nll)
        del keys, values
        if R is None:
            keys_t = values_t = None  # free the fp32 baseline stacks

        total = sum(v[0] for v in per_type.values())
        rows[name] = (per_type, total, len(needles), sum(nlls) / len(nlls))
        detail = "  ".join(f"{t}:{v[0]}/{v[1]}" for t, v in sorted(per_type.items()))
        print(f"  {name:34s} {total:2d}/{len(needles)}  NLL {rows[name][3]:.3f}  {detail}",
              flush=True)

    slug = args.model.split("/")[-1].replace(".", "")
    seed_tag = "" if args.seed == 11 else f"_s{args.seed}"
    rpt = os.path.join(args.report_dir, f"battery_{slug}_{T}{seed_tag}.md")
    os.makedirs(args.report_dir, exist_ok=True)
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"# Typed-needle battery -- {args.model}, {T} tokens, "
                f"seed {args.seed}\n\n| method | total | gold-NLL | "
                + " | ".join(types) + " |\n|" + "---|" * (3 + len(types)) + "\n")
        for name, (pt, tot, n, nll) in rows.items():
            f.write(f"| {name} | {tot}/{n} | {nll:.3f} | "
                    + " | ".join(f"{pt[t][0]}/{pt[t][1]}" for t in types) + " |\n")
    print(f"wrote {rpt}")


if __name__ == "__main__":
    main()
