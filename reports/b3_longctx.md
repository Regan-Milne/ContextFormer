# B3: recall at 7465-token context (16 needles, full-depth spread)

| method | recall | gold-NLL | per-needle (shallow->deep) |
|---|---|---|---|
| full KV | 16/16 | 0.110 | `YYYYYYYYYYYYYYYY` |
| behavioral 16x (c8) | 14/16 | 0.197 | `YY.YYYY.YYYYYYYY` |
| behavioral 32x (c4) | 9/16 | 0.588 | `Y.Y.Y..Y.YY..YYY` |

Accounting at this length: coefficients 768 B/token + fp16 basis 9.0 MB / 7465 tokens = 2032 B/token net (6.0x vs full fp16 KV).
