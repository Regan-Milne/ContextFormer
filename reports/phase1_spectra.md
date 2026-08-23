# Phase 1a: spectral structure (Qwen/Qwen2.5-0.5B, 2048 tokens of ../PREPRINT.md)

## Per-layer rank (centered PCA over tokens, ranks for 90/95/99% energy, dim = 128)

| layer | K90 | K95 | K99 | V90 | V95 | V99 |
|---|---|---|---|---|---|---|
| 0 | 4 | 7 | 15 | 46 | 65 | 96 |
| 1 | 40 | 57 | 89 | 73 | 91 | 115 |
| 2 | 49 | 65 | 93 | 71 | 89 | 115 |
| 3 | 38 | 55 | 82 | 74 | 92 | 115 |
| 4 | 39 | 56 | 88 | 85 | 102 | 121 |
| 5 | 39 | 55 | 88 | 84 | 101 | 121 |
| 6 | 44 | 62 | 93 | 62 | 81 | 111 |
| 7 | 41 | 56 | 86 | 74 | 93 | 117 |
| 8 | 39 | 56 | 89 | 73 | 93 | 118 |
| 9 | 28 | 42 | 72 | 67 | 86 | 113 |
| 10 | 28 | 43 | 79 | 82 | 100 | 121 |
| 11 | 29 | 40 | 63 | 57 | 77 | 107 |
| 12 | 36 | 51 | 83 | 80 | 98 | 120 |
| 13 | 33 | 46 | 77 | 72 | 91 | 117 |
| 14 | 28 | 42 | 78 | 86 | 103 | 122 |
| 15 | 36 | 51 | 83 | 68 | 88 | 115 |
| 16 | 22 | 29 | 54 | 79 | 95 | 117 |
| 17 | 36 | 51 | 84 | 77 | 94 | 117 |
| 18 | 37 | 55 | 90 | 84 | 101 | 121 |
| 19 | 35 | 51 | 84 | 75 | 94 | 118 |
| 20 | 24 | 35 | 68 | 72 | 88 | 113 |
| 21 | 22 | 29 | 53 | 63 | 75 | 96 |
| 22 | 22 | 31 | 55 | 50 | 57 | 66 |
| 23 | 32 | 44 | 68 | 51 | 59 | 72 |
| **mean** | 33 | 46 | 76 | 71 | 88 | 111 |

## Joint cross-layer stack (standardized, dim 6144)

- joint rank for 90/95/99% energy: 377 / 623 / 1211
- sum of per-layer ranks (same energy): 2485 / 3221 / 4478
- cross-layer advantage at 95%: 5.2x fewer coefficients for the whole stack treated as one object
- hidden-trajectory joint rank (dim 22400): 703 / 1011 / 1593

## Trajectory smoothness (mean cosine of adjacent-layer hidden states)

| layers | 0 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| cos | 0.033 | 0.867 | 0.914 | 0.938 | 0.913 | 0.936 |

mean over all adjacent pairs: 0.857

## Context test: predicting h^(L/2) per token (ridge, train first 75% of tokens, test last 25%)

| features | test R^2 on h^(mid) |
|---|---|
| token embedding only | 0.165 |
| + prev 2 tokens | 0.200 |
| + prev 8 tokens | 0.207 |
| + prev 8 + mean of prev 64 | 0.207 |

(Caveat: 2048 tokens of one document; indicative, not conclusive. The question is the *gap* between rows, i.e. whether context access buys predictability beyond token identity.)
