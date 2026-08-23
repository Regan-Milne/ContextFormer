# A7: exact-recall (needle) test under compression

1870-token document, 8 planted 5-digit facts (depths [190, 399, 610, 820, 1030, 1241, 1451, 1661]), whole document compressed including the needles, greedy retrieval.

| method | recall | gold-digit NLL | per-needle (shallow->deep) |
|---|---|---|---|
| full KV | 8/8 | 0.123 | `YYYYYYYY` |
| quant K8 V4 (2.6x) | 8/8 | 0.131 | `YYYYYYYY` |
| joint768 c8 (16x) | 4/8 | 0.709 | `YYY.Y...` |
| joint768 c4 (32x) | 2/8 | 1.230 | `YY......` |
| traj384 c8 (32x) | 0/8 | 2.019 | `........` |
| traj384 c4 (64x) | 0/8 | 1.879 | `........` |

Assumption A7 under test: 'mean KL captures the damage that matters.' If recall collapses at ratios whose mean KL looked acceptable, A7 is falsified and every Pareto point must carry a recall score.
