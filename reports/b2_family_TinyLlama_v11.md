# B2: family transfer -- TinyLlama/TinyLlama_v1.1

L=22, kvh=4, head_dim=64, fp16 KV 22528 B/token. Roundtrip max err K 0.00e+00 / V 0.00e+00.

| method | recall | gold-NLL | per-needle |
|---|---|---|---|
| full KV | 7/8 | 5.225 | `.YYYYYYY` |
| variance 16x (rank 1408) | 5/8 | 5.308 | `.YYYY.Y.` |
| behavioral 16x (rank 1408) | 8/8 | 5.254 | `YYYYYYYY` |
| behavioral 32x (rank 704) | 8/8 | 5.391 | `YYYYYYYY` |

Moby-Dick teacher-forced window, behavioral 16x: KL 0.0034, top-1 0.961, dNLL -0.0038.
