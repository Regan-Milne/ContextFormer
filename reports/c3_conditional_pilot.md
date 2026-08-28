# C3: conditional-coding pilot (linear, rank-768 behavioral basis)

## Predictability of per-token codes (held-out later tokens)

| predictor | test R^2 | bits-saved proxy /token |
|---|---|---|
| identity only | +0.177 | 10 |
| identity + prev code | +0.077 | 9 |
| identity + prev code + ctx mean | -1.682 | 0 |

## Behavioral at matched 384 B/token (int4)

| coding | KL | top-1 | dNLL |
|---|---|---|---|
| unconditional: Z int4 | 0.1365 | 0.770 | +0.1287 |
| conditional (identity only) resid int4 | 0.1459 | 0.789 | +0.1707 |
| conditional (identity + prev code) resid int4 | 0.2007 | 0.777 | +0.2808 |
| conditional (identity + prev code + ctx mean) resid int4 | 0.4223 | 0.648 | +0.4837 |

Reading: the founding hypothesis predicts the context-conditioned rows beat identity-only at matched bytes. Linear predictors and clean-code conditioning make this an optimistic-capacity, first-evidence test only.
