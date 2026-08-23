# ContextFormer

Research question: once context has resolved a token's meaning, how little
per-token state must remain resident in place of the conventional
layer-by-layer KV stack?

Every token is retained and addressable. The target is the stored vector state
per token: replace `t_i -> {K_i^(l), V_i^(l)}_{l=1..L}` with `t_i + C_i -> Z_i`
where `Z_i` is radically smaller, and reconstruct (or functionally approximate)
the KV stack from `(t_i, C_i, Z_i)` with a cheap decoder on a frozen model.

This is not token dropping, summarization, retrieval, or plain quantization,
and it is distinct from the KV-routing work in the parent `kvspace` project
(routing selects which entries to *read*; this asks why each entry is so large
to begin with).

The problem is only well-posed as a three-way Pareto frontier:
**(bytes/token, reconstruction compute, behavioral fidelity)** -- since all
tokens are stored, full recomputation is the degenerate zero-byte solution and
is a mandatory baseline.

## Phases

- **Phase 0** (CPU laptop): instrumentation + capture. Frozen small model,
  exact bytes/token accounting, per-token hidden-state trajectories and KV
  stacks captured to disk, with a verification that the KV stack is exactly
  reproducible from the trajectory via the frozen weights (pre-RoPE K path).
- **Phase 1** (CPU laptop): analytic compressors on captured data
  (quantization, per-layer SVD, joint cross-layer low-rank of the trajectory),
  plugged back into the frozen model via `past_key_values`, scored by
  teacher-forced KL / top-k agreement vs the full-KV baseline. First
  bytes-vs-fidelity Pareto curve; per-layer / per-head / K-vs-V breakdown.
- **Phase 2** (RTX 4090): learned context-conditioned encoders/decoders with
  distillation losses, the context-ablation at fixed latent size, adaptive
  per-token budgets, evolutionary search over allocation policies.

## Layout

- `scripts/capture.py` -- Phase 0: load frozen model, print/write KV
  accounting, capture trajectories + KV + tokens, verify KV = f(trajectory).
- `reports/` -- generated instrumentation and results (committed).
- `data/` -- captured tensors (gitignored).

## Environment

Laptop phase: Python 3.11, torch 2.0.1 (CPU), transformers 4.45.1,
model `Qwen/Qwen2.5-0.5B` (24 layers, GQA 2 KV heads x 64 dim, RoPE).
