"""Phase 0: instrument KV storage of a frozen model and capture per-token state.

Captures, for every token of an input document:
  - the hidden-state trajectory h_i^(l), l = 0..L  (input to each layer)
  - the exact KV cache entries K_i^(l), V_i^(l) as used by attention (post-RoPE)

and verifies that the entire KV stack is exactly reproducible from the
trajectory via the frozen weights (input_layernorm -> k_proj/v_proj -> RoPE).
That verification is the foundation of the project: it means the object to
compress is the trajectory (or something smaller), never the KV image itself.

Also writes the bytes/token accounting report.

Usage:
    python scripts/capture.py --input ..\\PREPRINT.md --max-tokens 2048
"""

import argparse
import hashlib
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1}


def kv_accounting(cfg, context_lengths=(1024, 8192, 32768, 131072)):
    L = cfg.num_hidden_layers
    kvh = cfg.num_key_value_heads
    qh = cfg.num_attention_heads
    d_model = cfg.hidden_size
    head_dim = d_model // qh

    kv_dims_per_layer = 2 * kvh * head_dim          # K + V
    kv_dims_total = L * kv_dims_per_layer
    traj_dims_total = (L + 1) * d_model             # h^(0)..h^(L)

    rows = {
        "model": cfg.name_or_path,
        "layers": L,
        "d_model": d_model,
        "q_heads": qh,
        "kv_heads": kvh,
        "head_dim": head_dim,
        "kv_dims_per_layer (K+V)": kv_dims_per_layer,
        "kv_dims_per_token (all layers)": kv_dims_total,
        "trajectory_dims_per_token (L+1 hiddens)": traj_dims_total,
        "kv_bytes_per_token_fp16": kv_dims_total * 2,
        "kv_bytes_per_token_fp32": kv_dims_total * 4,
        "trajectory_bytes_per_token_fp16": traj_dims_total * 2,
        "mha_equivalent_kv_bytes_per_token_fp16": L * 2 * qh * head_dim * 2,
        "gqa_builtin_reduction_vs_mha": qh / kvh,
    }
    totals = []
    for T in context_lengths:
        totals.append({
            "context": T,
            "kv_total_fp16_MB": T * kv_dims_total * 2 / 2**20,
            "trajectory_total_fp16_MB": T * traj_dims_total * 2 / 2**20,
        })
    return rows, totals


def recompute_kv_from_trajectory(model, hidden_states, position_ids):
    """Regenerate every layer's post-RoPE K and pre-attention V from the
    per-token hidden trajectory using only frozen weights."""
    cfg = model.config
    kvh = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    keys, values = [], []
    for l, layer in enumerate(model.model.layers):
        h = hidden_states[l]                                   # (1, T, D)
        T = h.shape[1]
        x = layer.input_layernorm(h)
        k = layer.self_attn.k_proj(x).view(1, T, kvh, head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(x).view(1, T, kvh, head_dim).transpose(1, 2)
        cos, sin = model.model.rotary_emb(h, position_ids)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        keys.append(k)
        values.append(v)
    return keys, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    text = open(args.input, encoding="utf-8").read()
    doc_name = os.path.splitext(os.path.basename(args.input))[0]

    print(f"loading {args.model} (fp32, cpu)...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows, totals = kv_accounting(model.config)
    print(json.dumps(rows, indent=2, default=str))

    ids = tok(text, return_tensors="pt").input_ids[:, : args.max_tokens]
    T = ids.shape[1]
    position_ids = torch.arange(T).unsqueeze(0)
    print(f"document: {args.input}  tokens captured: {T}")

    t0 = time.time()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=True)
    prefill_s = time.time() - t0
    print(f"prefill: {prefill_s:.1f}s ({T / prefill_s:.1f} tok/s, cpu fp32)")

    pkv = out.past_key_values
    legacy = pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv
    cache_k = [kv[0] for kv in legacy]   # each (1, kvh, T, head_dim)
    cache_v = [kv[1] for kv in legacy]
    hidden = list(out.hidden_states)     # L+1 tensors (1, T, d_model)

    # --- verification: KV stack == f(trajectory, frozen weights, position) ---
    print("verifying KV stack is reproducible from hidden trajectory...")
    with torch.no_grad():
        rk, rv = recompute_kv_from_trajectory(model, hidden, position_ids)
    k_err = max((rk[l] - cache_k[l]).abs().max().item() for l in range(len(rk)))
    v_err = max((rv[l] - cache_v[l]).abs().max().item() for l in range(len(rv)))
    k_scale = max(cache_k[l].abs().max().item() for l in range(len(cache_k)))
    v_scale = max(cache_v[l].abs().max().item() for l in range(len(cache_v)))
    print(f"  max |K_recomputed - K_cache| = {k_err:.3e}  (K scale {k_scale:.1f})")
    print(f"  max |V_recomputed - V_cache| = {v_err:.3e}  (V scale {v_scale:.1f})")
    verified = k_err < 1e-3 * max(k_scale, 1.0) and v_err < 1e-3 * max(v_scale, 1.0)
    print(f"  verified: {verified}")

    # --- save capture (fp16 on disk; fp32 is recoverable enough for analysis) ---
    os.makedirs(args.out_dir, exist_ok=True)
    capture_path = os.path.join(args.out_dir, f"capture_{doc_name}_{T}.pt")
    torch.save({
        "model": args.model,
        "doc": args.input,
        "doc_sha1": hashlib.sha1(text.encode()).hexdigest(),
        "token_ids": ids,
        "hidden": torch.stack([h.squeeze(0).half() for h in hidden]),   # (L+1, T, D)
        "keys": torch.stack([k.squeeze(0).half() for k in cache_k]),    # (L, kvh, T, hd)
        "values": torch.stack([v.squeeze(0).half() for v in cache_v]),
        "verified_kv_from_trajectory": verified,
        "k_recompute_max_abs_err": k_err,
        "v_recompute_max_abs_err": v_err,
    }, capture_path)
    print(f"saved {capture_path} ({os.path.getsize(capture_path) / 2**20:.1f} MB)")

    # --- report ---
    os.makedirs(args.report_dir, exist_ok=True)
    rpt = os.path.join(args.report_dir, "phase0_instrumentation.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"# Phase 0: KV instrumentation ({rows['model']})\n\n")
        f.write("## Per-token storage accounting\n\n| quantity | value |\n|---|---|\n")
        for k, v in rows.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Totals by context length (fp16)\n\n")
        f.write("| context | full KV | hidden trajectory |\n|---|---|---|\n")
        for t in totals:
            f.write(f"| {t['context']:,} | {t['kv_total_fp16_MB']:.0f} MB "
                    f"| {t['trajectory_total_fp16_MB']:.0f} MB |\n")
        f.write("\n## Trajectory -> KV verification\n\n")
        f.write(f"- document: `{args.input}`, {T} tokens, prefill {prefill_s:.1f}s (cpu fp32)\n")
        f.write(f"- max abs error K: {k_err:.3e} (scale {k_scale:.1f}), "
                f"V: {v_err:.3e} (scale {v_scale:.1f})\n")
        f.write(f"- **verified: {verified}** -- the full post-RoPE KV stack is an exact\n"
                f"  deterministic image of the per-token hidden trajectory under the\n"
                f"  frozen weights. The compression target is therefore the trajectory\n"
                f"  (or anything smaller that regenerates it), never raw KV.\n")
    print(f"wrote {rpt}")


if __name__ == "__main__":
    main()
