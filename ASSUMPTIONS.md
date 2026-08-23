# Assumptions ledger

Every default we've inherited or introduced, tagged with where it came from,
what evidence exists, and the cheapest decisive test. Rule of engagement: spend
questioning budget strictly by test cost, cheapest first. A default is not a
conclusion; promotion requires an experiment that could have failed.

Status codes: **UNTESTED** / **WOBBLED** (evidence against, not decisive) /
**SUPPORTED** (evidence for, not decisive) / **FALSIFIED** / **CONFIRMED**
(decisive within current scope: one small model, short contexts).

| id | assumption | origin | status | evidence so far | cheapest decisive test | cost |
|----|-----------|--------|--------|-----------------|------------------------|------|
| A1 | KV should be compressed per layer | field default (cache memory layout promoted to method) | **FALSIFIED** | joint cross-layer PCA beats per-layer ~100x in KL at matched bytes (phase 1) | done | done |
| A2 | K and V deserve symmetric treatment | field default | **FALSIFIED** | K: low-rank but 2-bit catastrophic; V: near-full-rank but 4-bit nearly free (phase 1) | done | done |
| A3 | the KV image is the right object to store | field default (store what attention reads) | **WOBBLED** | trajectory-space (generator) compression wins at >=32x and survives int4 coeffs (phase 1) | sweep trajectory vs stack across models/docs | laptop-days |
| A4 | context conditioning enables compression (founding prior) | our hypothesis | **WOBBLED (weakly tested)** | linear ridge from raw embeddings: R^2 0.165 token-only vs 0.207 +context. Probe is weak: linear, raw embeddings as features, full-vector target, 2k tokens | causal predictive coder: predict token i's trajectory from *previous tokens' compressed states*, store only the residual; bits-of-residual vs unconditional coding at matched fidelity | 4090-days |
| A5 | per-document basis generalizes / is legitimate | our phase-1 shortcut | **UNTESTED** | bases fit and evaluated on same doc | fit basis on corpus, eval held-out doc; ALSO the reverse comparison (per-doc vs global basis at matched bytes) is itself a *document-level context test* for A4 | laptop-hours |
| A6 | every token deserves the same budget (uniform Z size) | our phase-1 simplification | **UNTESTED** | heavy-tail prediction from theory discussion | per-token reconstruction-error and behavioral-impact distribution; adaptive allocation vs uniform at matched mean bytes | laptop-hours |
| A7 | mean KL captures the damage that matters | eval convention | **UNTESTED** | none; predicted failure mode is exact-recall collapse invisible in mean KL | needle test: plant exact details in past, compress, query; compare at matched KL | laptop-hours |
| A8 | variance (PCA energy) is the right objective for subspace choice | tool default | **WOBBLED** | per-layer K rank-64 keeps 95%+ energy yet ruins behavior; energy != behavior | behavior-weighted PCA: weight directions by observed query subspace / attention gradients, compare at matched rank | laptop-days |
| A9 | all layers matter equally (uniform rank across layers) | our phase-1 simplification | **UNTESTED** | layer-0 K is rank ~7; late-layer V ranks collapse (spectra) | leave-one-layer-out and per-layer rank allocation sweep | laptop-hours |
| A10 | recent tokens need full precision (exactness window) | field intuition | **UNTESTED** | none; phase 1 window tokens were uncompressed by construction (confound noted) | shrink/remove the exact window, measure | laptop-hours |
| A11 | compressed state must be expanded back to full KV before attention | our harness convention | **UNTESTED** | none; with a linear codec, attention can run *in latent space* (absorb basis into q/o projections, MLA-style), changing the compute Pareto entirely | derive absorbed form for the PCA codec, verify logits match expansion path | laptop-days |
| A12 | small-model findings transfer to larger models | necessity (hardware) | **UNTESTED** | none | rerun phases 0-1 on 4B-9B class on the 4090 | 4090-hours |
| A13 | fp16-GQA cache is the honest denominator for ratios | our accounting choice | adopted | GQA already bakes in 7x vs MHA; we quote on top of it | n/a (accounting policy, revisit if models change) | -- |
| A14 | pre-RoPE storage with rotation re-applied at read | our design choice | **SUPPORTED** | exact round-trip verified (phase 0); position free since tokens retained | none needed at this scope | done |
| A15 | joint-stack rank-192 anomaly is a bug/outlier interaction, not a real cliff | our guess | **UNTESTED** | rank-192 KL 6.5 >> rank-96 KL 4.3 (non-monotone) | isolate offending components; check K outlier channels x standardization | laptop-hours |

## Priority queue (by cost, then information value)

1. **A7 needle test** -- our predicted failure mode for everything above 16x; nothing else is trustworthy until this exists.
2. **A5 cross-doc basis** -- doubles as the cheapest real context test (document granularity).
3. **A6 adaptive budgets** + **A9 per-layer allocation** -- likely free ratio gains.
4. **A15 anomaly** -- cheap, and non-monotonic failures usually teach something.
5. **A8 behavior-weighted subspaces** -- the biggest conceptual upgrade available on the laptop.
6. **A11 latent-space attention** -- turns storage wins into compute wins; changes the Pareto's third axis.
7. **A4 predictive coder** (4090) -- the founding prior, tested with infrastructure that could actually confirm it, not just a linear probe.
8. **A12 scale transfer** (4090) -- everything above, bigger.
