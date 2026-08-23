"""One-command reproduction of the headline result.

    pip install -r requirements.txt
    python reproduce_headline.py

Claim under test (pre-registered): on frozen Qwen2.5-0.5B, replacing every
past token's full layerwise KV stack (12,288 B/token fp16) with a 768-byte
per-token code (behavior-weighted whole-stack linear codec, int8 coefficients,
per-document basis) preserves behavior:

    PASS criteria: needle recall >= 7/8 (reference run: 8/8),
                   teacher-forced mean KL <= 0.05 (reference: ~0.01),
                   top-1 agreement >= 0.90 (reference: ~0.94).

The script downloads the open model, builds a deterministic evaluation
document (fixed filler text shipped in assets/, fixed needle set), constructs
the full KV cache, fits the codec on the document itself (the per-document
basis paradigm; basis storage is reported separately and included in the net
accounting), runs full vs compressed inference, and prints every number.

Expected wall-clock: ~10-20 min on a modern CPU; ~1 GB model download on
first run. No GPU required.
"""

import sys
import time

import torch

sys.path.insert(0, "scripts")

from transformers import AutoTokenizer
from common import (load_model, pre_rope_kv, rope_k, prefill_doc, eval_window,
                    behavior_metrics, fit_joint, apply_joint)
from a7_needle import NEEDLES, build_doc, probe_needle
from a8_behavior_basis import behavioral_weights, metric_scale

MODEL = "Qwen/Qwen2.5-0.5B"
PASS_RECALL, PASS_KL, PASS_TOP1 = 7, 0.05, 0.90

torch.manual_seed(0)

print(f"[1/6] loading {MODEL} (fp32, cpu; downloads ~1 GB on first run)...")
model = load_model(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
fp16_bytes = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * \
    (cfg.hidden_size // cfg.num_attention_heads) * 2

print("[2/6] building deterministic needle document...")
filler = open("assets/filler.txt", encoding="utf-8").read()
ids, depths = build_doc(tok, filler)
T = ids.shape[1]
print(f"      {T} tokens, 8 planted facts at depths {depths}")

print("[3/6] full prefill (ground-truth KV)...")
t0 = time.time()
hidden, keys_true, values_true, _ = prefill_doc(model, ids)
print(f"      {time.time() - t0:.0f}s")
k_pre, v_pre = pre_rope_kv(model, hidden)
pos = torch.arange(T).unsqueeze(0)

print("[4/6] fitting behavior-weighted whole-stack codec (rank 768, int8)...")
wK, wV = behavioral_weights(model, hidden)
codec = fit_joint(k_pre, v_pre, T, 768, scale=metric_scale(wK, wV))
kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
keys_c, values_c = rope_k(model, kh, pos), vh

coeff_bytes = 768
basis_mb = codec["W"].numel() * 2 / 2**20
print(f"      accounting: full KV {fp16_bytes} B/token fp16 | coefficients "
      f"{coeff_bytes} B/token ({fp16_bytes / coeff_bytes:.1f}x marginal) | "
      f"basis {basis_mb:.1f} MB fp16 (per-document)")
for n in (T, 8192, 32768, 131072):
    net = coeff_bytes + basis_mb * 2**20 / n
    print(f"      net at {n:>6,} tokens: {net:6.0f} B/token "
          f"({fp16_bytes / net:4.1f}x)")

print("[5/6] behavior: teacher-forced window (compressed past) vs full KV...")
P = T - 256
base = eval_window(model, ids, keys_true, values_true, P)
comp = eval_window(model, ids, keys_c, values_c, P)
m = behavior_metrics(base, comp, ids.squeeze(0)[P + 1:])
print(f"      KL {m['kl_mean']:.4f} (p95 {m['kl_p95']:.4f}) | top-1 "
      f"{m['top1_agree']:.3f} | top-5 {m['top5_overlap']:.3f} | "
      f"dNLL {m['delta_nll']:+.4f}")

print("[6/6] exact recall: 8 planted facts, whole document compressed...")
hits = 0
for (name, num), d in zip(NEEDLES, depths):
    ok, nll = probe_needle(model, tok, keys_c, values_c, T, name, num)
    hits += ok
    print(f"      depth {d:4d}  Project {name:8s} -> "
          f"{'RECALLED' if ok else 'MISSED'} (gold-NLL {nll:.3f})")

ok = hits >= PASS_RECALL and m["kl_mean"] <= PASS_KL and \
    m["top1_agree"] >= PASS_TOP1
print(f"\nrecall {hits}/8 (need >= {PASS_RECALL})  "
      f"KL {m['kl_mean']:.4f} (need <= {PASS_KL})  "
      f"top-1 {m['top1_agree']:.3f} (need >= {PASS_TOP1})")
print("RESULT: PASS" if ok else "RESULT: FAIL")
sys.exit(0 if ok else 1)
