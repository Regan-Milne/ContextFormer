# Phase 0: KV instrumentation (Qwen/Qwen2.5-0.5B)

## Per-token storage accounting

| quantity | value |
|---|---|
| model | Qwen/Qwen2.5-0.5B |
| layers | 24 |
| d_model | 896 |
| q_heads | 14 |
| kv_heads | 2 |
| head_dim | 64 |
| kv_dims_per_layer (K+V) | 256 |
| kv_dims_per_token (all layers) | 6144 |
| trajectory_dims_per_token (L+1 hiddens) | 22400 |
| kv_bytes_per_token_fp16 | 12288 |
| kv_bytes_per_token_fp32 | 24576 |
| trajectory_bytes_per_token_fp16 | 44800 |
| mha_equivalent_kv_bytes_per_token_fp16 | 86016 |
| gqa_builtin_reduction_vs_mha | 7.0 |

## Totals by context length (fp16)

| context | full KV | hidden trajectory |
|---|---|---|
| 1,024 | 12 MB | 44 MB |
| 8,192 | 96 MB | 350 MB |
| 32,768 | 384 MB | 1400 MB |
| 131,072 | 1536 MB | 5600 MB |

## Trajectory -> KV verification

- document: `../PREPRINT.md`, 2048 tokens, prefill 34.1s (cpu fp32)
- max abs error K: 0.000e+00 (scale 221.4), V: 0.000e+00 (scale 19.3)
- **verified: True** -- the full post-RoPE KV stack is an exact
  deterministic image of the per-token hidden trajectory under the
  frozen weights. The compression target is therefore the trajectory
  (or anything smaller that regenerates it), never raw KV.
