# A5b: corpus-fit basis on held-out documents

Basis fit on 5 docs x 2048 tokens (fiction x2, science, political, code), behavioral weights pooled over corpus. Eval docs never seen by the basis.

## Needle recall (held-out ML-preprint needle doc)

| method | recall | gold-NLL | per-needle |
|---|---|---|---|
| full KV | 8/8 | 0.123 | `YYYYYYYY` |
| CORPUS basis joint768 c8 (16x) | 3/8 | 0.378 | `Y.Y.Y...` |
| CORPUS basis joint768 c4 (32x) | 1/8 | 0.539 | `Y.......` |
| CORPUS basis joint384 c8 (32x) | 2/8 | 0.567 | `Y.Y.....` |
| per-doc basis joint768 c8 (16x, reference) | 8/8 | 0.130 | `YYYYYYYY` |

## Teacher-forced eval (held-out Grimm doc)

| method | KL | top-1 | top-5 | dNLL |
|---|---|---|---|---|
| CORPUS basis joint768 c8 (16x) | 0.1577 | 0.789 | 0.805 | +0.1870 |
| CORPUS basis joint384 c8 (32x) | 1.5838 | 0.355 | 0.407 | +1.7075 |

If the corpus basis holds recall at 16x, basis storage is a fixed model asset (zero marginal bytes/doc) and the 16x accounting is clean at every context length.
