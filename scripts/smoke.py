"""Port/environment smoke test — run this FIRST on any new machine, model,
or transformers version, before spending GPU hours.

Validates in ~2 minutes:
  1. model loads on the detected device/dtype
  2. the harness understands the architecture: pre-RoPE recompute + rotation
     reproduces the model's real cache (THE invariant everything rests on)
  3. cache surgery + generation path works (2-needle recall on full KV)

Usage:
  python scripts/smoke.py                          # laptop default
  python scripts/smoke.py --model Qwen/Qwen3-4B    # 4090 day 1
"""

import argparse
import sys
import time

import torch
from transformers import AutoTokenizer

from common import (load_model, prefill_doc, pre_rope_kv, rope_k, kv_geometry)
from a7_needle import TEMPLATE, QUERY, probe_needle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default=None, choices=[None, "fp32", "bf16", "fp16"])
    args = ap.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
             "fp16": torch.float16, None: None}[args.dtype]

    t0 = time.time()
    model = load_model(args.model, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = next(model.parameters()).device
    L, kvh, hd, stack_dim, fp16_b = kv_geometry(model.config)
    print(f"[1/3] {args.model} on {dev} ({model.dtype}) "
          f"L={L} kvh={kvh} hd={hd} stack={stack_dim} "
          f"fp16 {fp16_b} B/tok  [{time.time() - t0:.0f}s]")

    filler = open("assets/filler.txt", encoding="utf-8").read()
    fids = tok(filler, return_tensors="pt").input_ids[:, :300]
    needles = [("Aurora", "40917"), ("Calder", "61893")]
    parts = [fids[:, :150]]
    for nm, num in needles:
        parts.append(tok(TEMPLATE.format(nm, num), return_tensors="pt").input_ids)
        parts.append(fids[:, 150:225] if nm == "Aurora" else fids[:, 225:300])
    ids = torch.cat(parts, dim=1)
    T = ids.shape[1]

    t0 = time.time()
    hidden, keys, values, _ = prefill_doc(model, ids)
    k_pre, v_pre = pre_rope_kv(model, hidden)
    k_rt = rope_k(model, k_pre, torch.arange(T).unsqueeze(0))
    k_err = (k_rt - keys).abs().max().item()
    v_err = (v_pre - values).abs().max().item()
    k_rel = k_err / keys.abs().max().clamp_min(1e-6).item()
    v_rel = v_err / values.abs().max().clamp_min(1e-6).item()
    rt_ok = k_rel < 0.02 and v_rel < 0.02
    print(f"[2/3] roundtrip: K abs {k_err:.3e} (rel {k_rel:.1e}), "
          f"V abs {v_err:.3e} (rel {v_rel:.1e}) -> "
          f"{'OK' if rt_ok else 'FAIL (harness does not match this arch)'} "
          f"[{time.time() - t0:.0f}s]")

    t0 = time.time()
    hits = sum(probe_needle(model, tok, keys, values, T, nm, num)[0]
               for nm, num in needles)
    gen_ok = hits >= 1
    print(f"[3/3] full-KV recall {hits}/2 "
          f"({'OK' if gen_ok else 'FAIL — check probe path / model strength'}) "
          f"[{time.time() - t0:.0f}s]")

    print("SMOKE: PASS" if rt_ok and gen_ok else "SMOKE: FAIL")
    sys.exit(0 if rt_ok and gen_ok else 1)


if __name__ == "__main__":
    main()
