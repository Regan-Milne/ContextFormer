# B1: needle recall robustness (3 reps, randomized needles/positions, filler rotation)

| method | recall | gold-NLL mean | by depth quartile (shallow->deep) |
|---|---|---|---|
| full KV | 24/24 | 0.134 | 6/6 / 6/6 / 6/6 / 6/6 |
| behavioral 16x (c8) | 24/24 | 0.130 | 6/6 / 6/6 / 6/6 / 6/6 |
| behavioral 16x (c8, int8 basis) | 2/24 | 2.012 | 0/6 / 2/6 / 0/6 / 0/6 |
| behavioral 32x (c4) | 21/24 | 0.304 | 6/6 / 4/6 / 5/6 / 6/6 |

Int8 basis halves per-doc basis storage (9.4 -> 4.7 MB): net ratio at 32k improves from 11.6x to ~13.2x if recall holds.
