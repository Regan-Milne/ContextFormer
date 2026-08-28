# C2: rank vs context length (7465 tokens, 16 needles)

| method | recall | gold-NLL | per-needle |
|---|---|---|---|
| full KV | 16/16 | 0.110 | `YYYYYYYYYYYYYYYY` |
| behavioral rank 768 c8 (16.0x) | 14/16 | 0.197 | `YY.YYYY.YYYYYYYY` |
| behavioral rank 1152 c8 (10.7x) | 15/16 | 0.096 | `YYYYYYY.YYYYYYYY` |
| behavioral rank 1536 c8 (8.0x) | 15/16 | 0.108 | `YYYYYYY.YYYYYYYY` |

Reference points: rank 768 held 24/24 at ~1.9k tokens (B1) and 14/16 at ~7.5k (B3). This sweep locates the rank that restores full recall at ~7.5k, the first data on how latent size must scale with context length.
