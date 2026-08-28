# C4: metric showdown at stress ranks (2k doc, c8 coeffs; rank = B/token)

| metric/rank | recall | per-needle | KL | top-1 | dNLL |
|---|---|---|---|---|---|
| raw rank 768 | 8/8 | `YYYYYYYY` | 0.0113 | 0.918 | -0.0185 |
| raw rank 384 | 8/8 | `YYYYYYYY` | 0.1936 | 0.809 | +0.1248 |
| raw rank 192 | 0/8 | `........` | 5.2749 | 0.172 | +5.3145 |
| standardized rank 384 | 1/8 | `Y.......` | 0.4723 | 0.719 | +0.4717 |
| standardized rank 192 | 0/8 | `........` | 7.1925 | 0.102 | +7.1372 |
| behavioral rank 192 | 2/8 | `Y.Y.....` | 1.2526 | 0.570 | +1.2636 |
| non-diag rank 192 | 0/8 | `........` | 5.6922 | 0.051 | +5.6513 |

Context: published behavioral numbers at these ranks -- 768: 8/8 rec; 384: 7/8 rec, KL 0.11 (c16); 192: KL 2.55 (c16). Standardized (published as 'variance'): 384 KL 0.53, 192 KL 6.54 (c16), 16x recall 4/8. This table decides whether the A8 claim should be 'standardization is the villain; explicit behavioral weighting is the principled form of what raw magnitude weighting does by accident' -- and which metric actually owns the stressed-rank frontier.
