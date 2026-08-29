# 4090 scale-gate runbook

Ordered so failures surface cheap and early. Do not skip step 2 or 3 — every
downstream number is meaningless if the roundtrip check fails.

**Status 2026-08-28:** steps 1-3 DONE (smokes PASS both models; 0.5B parity
reproduces on GPU/bf16). Step 4 in progress: 8k complete, 16k in flight
(PREPRINT §8 tranche table). Two lessons for anyone rerunning:
(1) rank <= T, so 16x at 4B is only informative at 16k+ (ledger A23);
(2) long batteries: use `python -u`, `--prefill-chunk 1024` (bounds VRAM
peak), `--vram-frac` when sharing the GPU with anything you care about,
and never run two batteries concurrently (the fp32 stacks at 16k plus a
second run exceed 64 GB host RAM).

## 0. Environment (once)

```
git clone https://github.com/Regan-Milne/ContextFormer
cd ContextFormer
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.51" safetensors "numpy<2" accelerate
```

(The laptop pin `transformers<4.47` in requirements.txt is for the Qwen2.5
CPU reproduction; the gate needs >=4.51 for Qwen3. The harness supports both
via compat shims in scripts/common.py.)

## 1. Validate the environment (2 min)

```
python scripts/smoke.py
```

Runs Qwen2.5-0.5B on the GPU in bf16. Expect: roundtrip rel err ~0 in fp32 /
up to ~1e-2 in bf16 (bf16 rounding on K values of magnitude ~200 is real);
recall 2/2; `SMOKE: PASS`.

## 2. Validate the Qwen3 architecture path (5 min) — THE critical unknown

```
python scripts/smoke.py --model Qwen/Qwen3-4B
```

Qwen3 differs from Qwen2.5 in exactly the ways the harness had to be ported
for: `config.head_dim` (128) is decoupled from hidden/heads, and per-head
`q_norm`/`k_norm` (RMSNorm) apply BEFORE RoPE. If the roundtrip check fails
here, the bug is in `_proj_kv` / `behavioral_weights` in scripts/common.py
(norm placement or head_dim) — fix there before anything else.

## 3. Parity with the laptop result (10 min)

```
python scripts/battery.py --target-tokens 1800 --variance-control
python scripts/battery.py --model Qwen/Qwen3-4B --target-tokens 2000 --variance-control
```

First command must reproduce laptop-class numbers (behavioral 16x ~= full KV)
on GPU/bf16. Second is the first 4B data point.

## 4. The scaling sweep (the gate proper)

```
python scripts/battery.py --model Qwen/Qwen3-4B --target-tokens 8000  --variance-control
python scripts/battery.py --model Qwen/Qwen3-4B --target-tokens 16000 --variance-control
python scripts/battery.py --model Qwen/Qwen3-4B --target-tokens 32000 --variance-control
```

Repeat with `--seed 12 --seed 13` at the most informative length. Then a
second family (e.g. `--model meta-llama/Llama-3.2-3B` if licensed, or
another non-Qwen model available).

Primary open question this answers: does a fixed-rank basis stretch with
context length, or must rank grow with T (laptop data: 24/24 at ~1.9k,
14/16 at ~7.5k for 16x marginal)?

## 5. Pre-registered criteria (PREPRINT.md §8)

- PASS: recall across the typed battery intact at >=16x marginal where the
  0.5B result held, dNLL within noise of the 0.5B result, behavioral >
  variance control at matched bytes.
- FAIL: material recall degradation at >=16x, or matched-byte baselines
  reach equivalent fidelity. Either outcome is reported verbatim in
  PREPRINT.md §8. Anomalous runs are retained.

## 6. Still to build (day 2+)

- Multi-instruction IFEval-style compliance + system-prompt-leakage tests
  (needs an instruct model; protocols in RELATED_WORK.md "Evaluation
  practice").
- KIVI/TurboQuant-class quantization baseline and xKV-style layer-group SVD
  analogue at matched bytes.
- Conditional rate-distortion context experiment (PREPRINT §6.1).
- Teacher-forced KL windows at each length (eval_window path — currently the
  battery reports recall + gold-NLL only).

## Notes

- All battery/smoke runs write reports/ files; commit them as they land.
- bf16 vs fp32: codec math is always fp32 (common.py casts); caches are cast
  to model dtype on build. If bf16 recall looks anomalously low at parity
  (step 3), rerun with `--dtype fp32` to isolate precision from mechanism.
- 32k prefill at 4B in bf16 is ~10-20 GB KV + activations; if OOM, drop to
  16k or use `--dtype bf16` (default on CUDA) and close other GPU users.
