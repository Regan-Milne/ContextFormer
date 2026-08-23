# A8: behavior-weighted subspace selection

K channels weighted by pooled post-RoPE query rms (diagonal of the attention-logit damage metric, pair-symmetrized for RoPE); V channels by o_proj column norms. PCA allocates by behavioral energy.

## Gate 1: rank sweep, PREPRINT eval (variance metric in parens)

| rank | KL behavioral | KL variance | top-1 behavioral |
|---|---|---|---|
| 96 | 7.4023 | (4.3446) | 0.066 |
| 192 | 2.5485 | (6.5384) | 0.336 |
| 384 | 0.1103 | (0.5267) | 0.816 |
| 768 | 0.0102 | (0.0327) | 0.938 |

## Gate 2: needle recall (variance-metric A7 results: 16x -> 4/8, 32x -> 2/8, trajectory -> 0/8)

| method | recall | gold-NLL | per-needle |
|---|---|---|---|
| full KV | 8/8 | 0.123 | `YYYYYYYY` |
| behavioral joint768 c8 (16x) | 8/8 | 0.130 | `YYYYYYYY` |
| behavioral joint768 c4 (32x) | 7/8 | 0.286 | `YYYYYYY.` |
| behavioral joint384 c8 (32x) | 7/8 | 0.263 | `YYYYYYY.` |
| behavioral split K512/V256 c8 (16x) | 0/8 | 2.754 | `........` |
| behavioral split K576/V192 c8 (16x) | 0/8 | 2.841 | `........` |
