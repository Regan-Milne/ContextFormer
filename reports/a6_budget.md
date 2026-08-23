# A6: per-token budget distribution and adaptive allocation

## Residual fraction at joint rank 384 (past tokens)

| p10 | p25 | median | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 0.064 | 0.077 | 0.092 | 0.112 | 0.127 | 0.153 | 0.179 |

Correlations: resid vs surprisal +0.264, vs log in-doc freq -0.265, vs position -0.101.

Hardest 12 tokens: ' was', ' by', '"', ' yet', ' from', ' in', ' then', '-range', ' itself', ' when', ' them', 'see'

## Adaptive vs uniform at matched mean rank (coeff fp16, 768 B/token)

| allocation | mean rank | KL | top-1 | top-5 | dNLL |
|---|---|---|---|---|---|
| uniform rank 384 | 384 | 0.5267 | 0.637 | 0.680 | +0.6415 |
| adaptive 768/192 (top 33% hard get 768) | 383 | 6.1761 | 0.141 | 0.162 | +6.4270 |
| adaptive 768/96 (top 43% hard get 768) | 383 | 3.7540 | 0.246 | 0.331 | +3.8904 |

A6 under test: 'every token deserves the same budget.' Adaptive winning at matched mean bytes falsifies it; the residual tail and its correlates say which tokens are the expensive ones.
