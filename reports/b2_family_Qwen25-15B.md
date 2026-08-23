# B2: family transfer -- Qwen/Qwen2.5-1.5B

L=28, kvh=2, head_dim=128, fp16 KV 28672 B/token. Roundtrip max err K 0.00e+00 / V 0.00e+00.

| method | recall | gold-NLL | per-needle |
|---|---|---|---|
| full KV | 8/8 | 0.043 | `YYYYYYYY` |
| variance 16x (rank 1792) | 8/8 | 0.042 | `YYYYYYYY` |
| behavioral 16x (rank 1792) | 8/8 | 0.043 | `YYYYYYYY` |
| behavioral 32x (rank 896) | 8/8 | 0.039 | `YYYYYYYY` |

Moby-Dick teacher-forced window, behavioral 16x: KL 0.0001, top-1 0.996, dNLL -0.0011.
