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
| A4 | context conditioning enables compression (founding prior) | our hypothesis | **SUPPORTED (doc granularity)** | A5 test: per-doc basis beats foreign basis 24x in KL at matched bytes (0.011 vs 0.269 at 16x). Document-level context demonstrably buys compression. Token-level conditioning still untested (linear probe was weak) | causal predictive coder: predict token i's trajectory from *previous tokens' compressed states*, store only the residual; bits-of-residual vs unconditional coding at matched fidelity | 4090-days |
| A5 | per-document basis generalizes / is legitimate | our phase-1 shortcut | **RESOLVED: doc-specific** | foreign basis degrades KL 24x; corpus basis (5 diverse docs, held-out eval) recalls only 3/8 at 16x vs per-doc 8/8 at identical bytes, held-out KL 0.158 vs ~0.01. The enabling structure is constructed per document -- strongest evidence yet FOR A4 at doc granularity, and it fixes the accounting: 16x is marginal-only; net with 9.4MB fp16 basis = 6.4x @8k, 11.6x @32k, 14.6x @128k. Short contexts need a generated (learned) adaptation | done (analytic); learned encoder is the follow-on | done |
| A6 | every token deserves the same budget (uniform Z size) | our phase-1 simplification | **SUPPORTED (surprising)** | residual tail is THIN: median 0.092, p99 0.153, max 0.179 -- no expensive-token minority at the linear-codec level. Weak correlates (surprisal +0.26, log-freq -0.27). Adaptive two-tier LOST badly (KL 6.2 vs 0.53 uniform at matched bytes) -- confounded by A15 anomaly, but the thin tail independently says there's little to reallocate | retry adaptive with anomaly-free tiers after A15 is understood; re-test under learned codecs (tail may be codec-induced) | laptop-hours |
| A7 | mean KL captures the damage that matters | eval convention | **FALSIFIED** | at 16x, mean KL 0.033 "looks intact" but needle recall is 4/8; trajectory codecs recall 0/8 at KL ~0.6. Mean KL and exact recall dissociate completely. Every Pareto point now requires a recall column | done; recall harness now standing (a7_needle.py) | done |
| A8 | variance (PCA energy) is the right objective for subspace choice | tool default | **REFINED (C4)** | The catastrophic "variance metric" was specifically STANDARDIZED PCA (per-dim std division): std-384 recalls 1/8 (KL 0.47) where RAW PCA at 384 recalls 8/8 (KL 0.19) -- lower KL, worse recall (dissociation again). Raw magnitude-weighting ~= behavioral at 16-32x because magnitude correlates with query-probed K channels; behavioral weighting is the principled form and wins at stress (192: behavioral 2/8 > all others 0/8); non-diagonal wins the 32x typed battery (13/14 vs diag 11/14) but collapses at 64x (0/14). 64x is a hard wall for every linear codec AT 0.5B ONLY: at 4B/8k behavioral 64x recalls 14/14 (gold-NLL 0.048), the wall does not transfer (2026-08-28). Separation vs the variance control at 4B is directional but mild (13-12/14 vs 14/14; ~1.6x gold-NLL), not the 0.5B collapse; T/rank stress vs scale robustness unresolved, 16k 64x rows discriminate | learned metric on 4090; understand nondiag's 64x collapse; 16k 64x separation test (in flight) | done (linear, 0.5B) |
| A9 | all layers matter equally (uniform rank across layers) | our phase-1 simplification | **UNTESTED** | layer-0 K is rank ~7; late-layer V ranks collapse (spectra) | leave-one-layer-out and per-layer rank allocation sweep | laptop-hours |
| A10 | recent tokens need full precision (exactness window) | field intuition | **UNTESTED** | none; phase 1 window tokens were uncompressed by construction (confound noted) | shrink/remove the exact window, measure | laptop-hours |
| A11 | compressed state must be expanded back to full KV before attention | our harness convention | **UNTESTED** | none; with a linear codec, attention can run *in latent space* (absorb basis into q/o projections, MLA-style), changing the compute Pareto entirely | derive absorbed form for the PCA codec, verify logits match expansion path | laptop-days |
| A12 | small-model findings transfer to larger models | necessity (hardware) | **SUPPORTED (first tranche)** | 4090 gate 2026-08-28: Qwen3-4B typed battery, behavioral joint codec: 8k 32x 14/14 (gold-NLL 0.019 vs full 0.013), 8k 64x 14/14 (0.048); 16k 16x 14/14 (0.022 vs full 0.018). Transfer is not merely intact, the 0.5B 64x wall VANISHES at 4B (see A8 scope note). Single seed so far; replicates + second family pending | second family + seed replicates (in flight); 32k | 4090-hours |
| A13 | fp16-GQA cache is the honest denominator for ratios | our accounting choice | adopted | GQA already bakes in 7x vs MHA; we quote on top of it | n/a (accounting policy, revisit if models change) | -- |
| A14 | pre-RoPE storage with rotation re-applied at read | our design choice | **SUPPORTED** | exact round-trip verified (phase 0); position free since tokens retained | none needed at this scope | done |
| A15 | joint-stack rank-192 anomaly is a bug/outlier interaction, not a real cliff | our guess | **RESOLVED** | the cliff was a variance-metric artifact: under the behavioral metric the sweep is strictly monotone (7.40 / 2.55 / 0.11 / 0.010 for ranks 96/192/384/768). Variance ordering half-reconstructs behaviorally-critical directions and misdirects attention | done | done |
| A16 | behavior degrades monotonically as rank drops (more components = better) | implicit in all rank sweeps | **FALSIFIED (variance ordering only)** | monotonicity restored under behavioral ordering; non-monotonicity was diagnostic of the wrong metric, not of the model | done | done |
| A18 | K-stack and V-stack can be compressed with separate bases (K deserves a private budget) | our A8 side-experiment | **FALSIFIED** | split codecs (K512/V256, K576/V192, behavioral metric, 16x) recall 0/8 with gold-NLL ~2.8 -- catastrophically worse than the joint codec at identical bytes. K and V of a token are strongly coupled; cutting the stack along the K/V seam is as destructive as cutting it along layers. The per-token whole-stack really is the natural object | understand the K-V coupling (shared components' K/V loadings) | laptop-hours |
| A17 | needle recall is depth-uniform | implicit | **RESOLVED (no systematic bias)** | the shallow-cluster pattern did not reproduce: B1 (24 randomized needles, 3 domains) recalls 24/24 at 16x; failures at 32x and at 7.5k context are scattered across depths. Original pattern was small-sample noise | done | done |
| A19 | the per-doc basis can be quantized (int8) to halve its storage | our accounting hope | **FALSIFIED** | int8 basis: 2/24 recall at 16x vs 24/24 with fp16 basis, identical coefficients. The basis is precision-critical like the keys it reconstructs. Net ratios must use fp16 basis (or a smarter basis-compression scheme than naive int8) | structured basis compression (low-rank of W, mixed precision by component) if net ratios at short contexts ever matter | laptop-hours |
| A20 | the basis works only for tokens it was fit on (deployment risk: decode-time tokens arrive after prefill fit) | our harness shape | **SUPPORTED w/ penalty (C1)** | basis fit on first 60% of doc, needles beyond the boundary: 7/8 recall (one miss = first needle past boundary), gold-NLL 0.229 vs 0.130. Forward generalization within a document works; decode-time encoding with a prefill basis is viable | quantify penalty vs distance past boundary at scale | 4090 |
| A21 | required rank grows linearly with context length | scaling fear | **WOBBLED toward sublinear (C2, 4090 tranche 1)** | 768 covers 1.9k (24/24); 1152 (+50%) recovers 7.5k (15/16) where 768 held 14/16; 1536 adds nothing -- the last miss is rank-INSENSITIVE (fails at 8x too, full KV passes). Rank demand grows much slower than T on this evidence; the residual failure mode is not capacity. 4B (2026-08-28): rank 4608 perfect at 8k (14/14) and rank 2304 perfect at 8k (14/14); fixed-rank-across-lengths rows at 16k in flight | 16k/32k fixed-rank stretch on 4090 (in flight); diagnose the rank-insensitive miss | 4090 |
| A22 | token-level causal context conditioning reduces per-token bits (founding hypothesis at token granularity) | our hypothesis | **NULL at linear capacity (C3)** | ridge prediction of Z_i from previous codes UNDERPERFORMS token identity alone (R^2 0.077 vs 0.177); at matched 384 B/tok, all conditional variants lose behaviorally to unconditional int4. Pattern with A5: context's benefit is document-GLOBAL (which subspace), not local-sequential (predict next code). Underpowered linear test -- not a verdict | nonlinear predictor + doc-summary-conditioned codec on 4090 | 4090 |
| A23 | a compression ratio quoted from a battery row is informative | implicit in all rank sweeps | **SCOPED (accounting rule)** | PCA rank <= T (sample count). Any row whose byte budget implies rank >= T is vacuous: the codec is near-exact by construction. Bit us live: the 4B 2k battery (rank capped at 1985 = T-8) returned every row identical to full KV; retained in reports/ as the cautionary example. Rule: informative rows need T comfortably above rank = stack_fp16_bytes/ratio (at 4B c8: T >> 9216 for 16x, 4608 for 32x, 2304 for 64x); quote T/rank alongside any cross-scale comparison, since matched RATIO does not match per-sample stress across stack sizes | none needed; standing accounting rule | done |

## Priority queue (updated 2026-08-23 after A5/A6/A7 results)

1. **A15/A16 anomaly** -- promoted to top: non-monotone rank-vs-behavior has now
   poisoned two experiments and blocks trustworthy allocation work. Understanding
   it likely resolves A8 too.
2. **A8 behavior-weighted subspaces** -- A16's falsification says energy ordering
   is actively wrong, not just suboptimal; recall (A7) may be recoverable at 16x+
   by selecting components for addressability instead of variance.
3. **A17 depth-dependent recall** -- cheap, and shapes how recall scores are read.
4. **A9 per-layer allocation** -- still likely free gains, now with recall column.
5. **A5 follow-up** -- corpus-fit global basis to separate domain from document.
6. **A11 latent-space attention** -- unchanged.
7. **A4 predictive coder** (4090) -- now with positive doc-level evidence; also the
   escape from the basis-storage amortization problem (generate the adaptation
   instead of storing it).
8. **A12 scale transfer** (4090).

## Standing rule (from A7's falsification)

No compression result is quotable from mean KL alone. Every Pareto point
carries: mean KL, top-1 agreement, **needle recall**, and gold-digit NLL.

**Honest frontier as of 2026-08-23 (behavioral metric, laptop gate passed):**
- 16x: behavioral joint-768 c8 -- recall 8/8, gold-NLL 0.130 (full KV: 0.123),
  KL 0.010, top-1 0.938. Effectively intact.
- 32x: behavioral joint-768 c4 or joint-384 c8 -- recall 7/8 (deepest needle
  lost), gold-NLL ~0.27.
- Superseded: variance-metric points (16x held only 4/8); trajectory-space
  points (0/8); split K/V bases (0/8).

Next gate: 4090 -- replicate at 4B-9B scale, 32k context, held-out documents.

**4090 tranche 1 (2026-08-28, Qwen3-4B, seed 11, behavioral joint codec):**
- 8k: full KV 14/14 (gold-NLL 0.013); 32x 14/14 (0.019); 64x 14/14 (0.048).
  Variance control: 32x 13/14 (0.029), 64x 12/14 (0.076).
- 16k: full KV 14/14 (0.018); 16x (rank 9216, first informative 16x) 14/14
  (0.022). Remaining 16k rows + 8k seed replicates in flight.
- Still open at 4B: variance separation is mild (see A8); second family,
  32k, and the full §8 battery components remain.
