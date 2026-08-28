# C1: held-out-tokens basis (A20), metric ablation, non-diagonal metric

1870-token doc, needle depths [190, 399, 610, 820, 1030, 1241, 1451, 1661], fit boundary 1122 (3 needles beyond it).

| experiment | recall | gold-NLL | per-needle (shallow->deep) |
|---|---|---|---|
| full KV | 8/8 | 0.123 | `YYYYYYYY` |
| behavioral 16x, fit on ALL tokens (ref) | 8/8 | 0.130 | `YYYYYYYY` |
| A20: behavioral 16x, fit on first 60% | 7/8 | 0.229 | `YYYYY.YY` |
| ABL: K-weights only | 8/8 | 0.135 | `YYYYYYYY` |
| ABL: V-weights only | 8/8 | 0.121 | `YYYYYYYY` |
| ABL: no weighting (raw PCA) | 8/8 | 0.130 | `YYYYYYYY` |
| ND: non-diag metric joint768 c8 (16x) | 8/8 | 0.130 | `YYYYYYYY` |
| ND: non-diag metric joint384 c8 (32x) | 8/8 | 0.105 | `YYYYYYYY` |
