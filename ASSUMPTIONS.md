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
| A8 | variance (PCA energy) is the right objective for subspace choice | tool default | **FALSIFIED; behavioral metric CONFIRMED better** | diagonal query/o_proj-weighted metric: KL 0.010 vs 0.033 at rank 768, 0.11 vs 0.53 at rank 384, AND needle recall 8/8 at 16x (variance metric: 4/8), 7/8 at 32x (variance: 2/8). The laptop gate passed | non-diagonal metric (full E[qq']), learned metric on 4090 | done (diag) |
| A9 | all layers matter equally (uniform rank across layers) | our phase-1 simplification | **UNTESTED** | layer-0 K is rank ~7; late-layer V ranks collapse (spectra) | leave-one-layer-out and per-layer rank allocation sweep | laptop-hours |
| A10 | recent tokens need full precision (exactness window) | field intuition | **UNTESTED** | none; phase 1 window tokens were uncompressed by construction (confound noted) | shrink/remove the exact window, measure | laptop-hours |
| A11 | compressed state must be expanded back to full KV before attention | our harness convention | **UNTESTED** | none; with a linear codec, attention can run *in latent space* (absorb basis into q/o projections, MLA-style), changing the compute Pareto entirely | derive absorbed form for the PCA codec, verify logits match expansion path | laptop-days |
| A12 | small-model findings transfer to larger models | necessity (hardware) | **UNTESTED** | none | rerun phases 0-1 on 4B-9B class on the 4090 | 4090-hours |
| A13 | fp16-GQA cache is the honest denominator for ratios | our accounting choice | adopted | GQA already bakes in 7x vs MHA; we quote on top of it | n/a (accounting policy, revisit if models change) | -- |
| A14 | pre-RoPE storage with rotation re-applied at read | our design choice | **SUPPORTED** | exact round-trip verified (phase 0); position free since tokens retained | none needed at this scope | done |
| A15 | joint-stack rank-192 anomaly is a bug/outlier interaction, not a real cliff | our guess | **RESOLVED** | the cliff was a variance-metric artifact: under the behavioral metric the sweep is strictly monotone (7.40 / 2.55 / 0.11 / 0.010 for ranks 96/192/384/768). Variance ordering half-reconstructs behaviorally-critical directions and misdirects attention | done | done |
| A16 | behavior degrades monotonically as rank drops (more components = better) | implicit in all rank sweeps | **FALSIFIED (variance ordering only)** | monotonicity restored under behavioral ordering; non-monotonicity was diagnostic of the wrong metric, not of the model | done | done |
| A18 | K-stack and V-stack can be compressed with separate bases (K deserves a private budget) | our A8 side-experiment | **FALSIFIED** | split codecs (K512/V256, K576/V192, behavioral metric, 16x) recall 0/8 with gold-NLL ~2.8 -- catastrophically worse than the joint codec at identical bytes. K and V of a token are strongly coupled; cutting the stack along the K/V seam is as destructive as cutting it along layers. The per-token whole-stack really is the natural object | understand the K-V coupling (shared components' K/V loadings) | laptop-hours |
| A17 | needle recall is depth-uniform | implicit | **WOBBLED** | at 16x, recalled needles cluster shallow (`YYY.Y...`) even though deeper needles sit closer to the query | vary depth systematically, more needles, multiple seeds | laptop-hours |

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
