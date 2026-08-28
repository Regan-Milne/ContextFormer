# One Object per Token: Behavior-Weighted Compression of the Cross-Layer KV Stack on Frozen LLMs

**Author:** Regan Milne (github.com/Regan-Milne)
**Date:** 2026-08-23
**Status:** Preprint / technical report; quiet timestamped disclosure of the
mechanism and small-scale results. Of two pre-registered gates, the
corpus-basis test (§6.2) is complete (failed; reported verbatim — the 16×
figure is long-context only, pending a learned encoder) and the
scale-transfer gate (§8) is pending. This document is updated in place;
gate outcomes will be reported here regardless of result, and this
disclosure will not be withdrawn. This document will serve as a timestamped disclosure of the
mechanism and results upon release.
**License (this document & the forthcoming code):** Apache-2.0.

---

## Abstract

Every token an LLM reads leaves behind a layer-by-layer stack of key/value
vectors — the KV cache — whose size, not the token's, dominates long-context
memory. Most post-hoc compression operates on per-layer, per-K/V pieces of
this cache; a growing line of work (MiniCache's adjacent-layer merging,
xKV's cross-layer SVD over layer groups, and architecture-level latent
caches such as DeepSeek's MLA and cross-layer latent attention) shows that
substantial redundancy runs in the depth dimension. Our results are
consistent with that literature and sharpen it with matched-bytes,
behavior-scored evidence for a specific unit: the **complete per-token
stack** — all layers, K and V together, as one coupled object — on a frozen
model, with every token retained and exact-detail recall as a mandatory
metric.

On a frozen Qwen2.5-0.5B (no training, no finetuning) with a plain linear
codec (PCA), three matched-bytes results:

1. **The whole-token object.** Compressing each token's complete 24-layer
   stack as one object beats independent per-layer compression by ~**100×
   in next-token KL at identical bytes/token** (0.033 vs 3.66 at 768
   B/token) in this model. Per-layer factorization cannot represent
   cross-layer structure by construction; our matched-bytes comparison
   quantifies how much behavioral fidelity that costs here, and extends the
   cross-layer observation from adjacent pairs and layer groups to the full
   stack of a single token.
2. **Behavioral metric.** *Standardized* PCA (per-dimension variance
   normalization, the common default) is behavior-misordered: quality is
   non-monotone in rank, and at matched bytes it destroys exact recall that
   raw (magnitude-weighted) PCA preserves — while scoring *lower* KL, a
   second metric/recall dissociation. Raw PCA works largely by accident
   (magnitude correlates with the query-probed key channels); an explicit
   behavior-weighted metric — K channels by the rms of the queries that
   will probe them, V channels by o_proj column norms, or their
   non-diagonal gram generalizations — is the principled form, and wins at
   stressed ranks where raw PCA and standardization both fail.
3. **K/V coupling.** Giving K and V separate bases — even with K favored,
   even under the behavioral metric — collapses exact recall to 0/8 at
   matched bytes. Cutting the stack along the K/V seam is as destructive as
   cutting it along layers.

With the whole-token object and the behavioral metric, the frozen model runs
at **16× fewer KV bytes/token (768 B vs 12,288 B fp16; marginal rate — net
accounting including the per-document basis is given in §6.2/§7) with
exact-detail recall fully intact**: 8/8 planted 5-digit facts recovered by greedy decoding
(gold-digit NLL 0.130 vs 0.123 full-KV; mean next-token KL 0.010; top-1
agreement 0.938). At 32×, recall is 7/8. We further show that mean-KL-style
metrics alone are untrustworthy for this problem: codecs exist that score
*better* mean KL while recalling **0/8** exact details — average-behavior
metrics and addressable-detail retention dissociate completely, so every
result above carries a recall score.

All results are small-scale (0.5B, 2k contexts, single-document bases except
§6.2) and analytic (linear codecs only); the pre-registered scale gate is
§8. Code and the full assumptions ledger (four falsified defaults, each with
the experiment that killed it) accompany this document.

---

## 1. Hypothesis and framing

The project asks: once context has resolved most of a token's meaning, how
little per-token state must actually remain resident? Conventionally a token
`t_i` stores `{K_i^(l), V_i^(l)}` for every layer `l` — on our test model
12,288 bytes at fp16 *after* the 7× reduction its grouped-query attention
already provides. The target is `t_i + C_i -> Z_i`: token identity retained,
context retained, and a radically smaller per-token state `Z_i` from which
the stack's *function* is recovered.

**What is and is not claimed as novel.** Cross-layer/depth redundancy in KV
caches is established prior art (MiniCache's adjacent-layer merging, NeurIPS
2024; xKV's cross-layer SVD over layer groups with per-token coefficients,
arXiv:2503.18893; tensor-factorization approaches such as JoLT; and trained
architectures — DeepSeek's MLA, cross-layer latent attention — that build a
per-token latent in from scratch). See RELATED_WORK.md for a field-by-field
comparison. Against that background, the specific claims of this report are:

1. **Whole-token object across all tested layers** — one coupled object per
   token spanning the entire depth, rather than adjacent-pair merging or
   fixed layer groups — with matched-bytes behavioral evidence for the gap.
2. **Joint K+V representation**, with matched-bytes evidence that separating
   K and V (as prior factorization methods do) destroys exact recall in our
   setup (§5).
3. **Behavior-weighted component selection**, where importance derives from
   how perturbations affect attention and output — including the
   observation that *standardized*-PCA ordering is behavior-misordered
   (non-monotone) in this model, that raw magnitude-weighting captures much
   of the behavioral effect by accident, and that explicit behavioral
   weighting is the principled form that wins at stressed ranks (§4, §7).
4. **Exact-detail recall as a mandatory metric**, with a demonstration that
   low KL can coexist with catastrophic retrieval failure (§4a).
5. **Document-adaptive structure**, quantified same-document vs foreign vs
   corpus-basis at matched bytes (§6).
6. Context-conditioned per-token latent state (`t_i + C_i → Z_i`) is the
   founding *hypothesis*, and is **not claimed as established**; §6.1
   specifies the conditional rate-distortion experiment that would establish
   it.

Two framing commitments distinguish this from adjacent work:

- **Not token dropping, not retrieval, not summarization.** Every token
  remains addressable and recoverable. We compress the state attached to a
  token, never the token set.
- **The three-way Pareto.** Because all tokens are stored, full
  recomputation from text is a degenerate zero-byte, perfect-fidelity
  solution. The problem is only well-posed over **(bytes/token,
  reconstruction compute, behavioral fidelity)**, with recompute as a
  mandatory baseline. Every stored byte purchases avoided FLOPs, nothing
  else: given the text and the frozen weights, the KV cache carries zero
  information — it is a compute cache, not a memory. [Also see §7 on how
  this reframes what `Z_i` should contain.]

## 2. Setup and instrumentation

Model: frozen Qwen2.5-0.5B (24 layers, d_model 896, 14 query heads, 2 KV
heads × 64 dims, RoPE). All experiments CPU, fp32 compute, fp16 storage
reference. Documents: 2,048-token slices (a technical ML preprint,
Moby-Dick, and for §6.2 a five-document corpus). Evaluation: the first 1,792
tokens' KV is replaced by its compressed reconstruction; the model then
processes the remaining 256 tokens teacher-forced, and its logits are
compared to the full-KV run (mean/95p KL, top-1/top-5 agreement, ΔNLL).
Exact recall: 8 five-digit facts planted at depths 190–1,661 in filler text,
the **entire document including the needles** compressed, each fact then
queried by prompt and scored on greedy exact-match and gold-digit NLL.

Two structural facts established first:

- **The stack is an exact linear image of the token's hidden trajectory.**
  Every `K_i^(l)`, `V_i^(l)` is reproduced to zero error from the per-token
  hidden states `{h_i^(l)}` via the frozen weights (layernorm → projection →
  rotation). The object to compress is therefore the trajectory or anything
  smaller — never raw KV, which is a redundant materialization of it.
- **Keys are stored pre-RoPE** and re-rotated at read time; position indices
  are free because every token is retained. This preserves the low-rank
  structure that position rotation otherwise destroys. (Pre-RoPE key
  factorization with prefill-time SVD has direct precedent in ShadowKV,
  arXiv:2410.21465, whose key side is low-rank per-token coefficients; see
  RELATED_WORK.md.)

## 3. Finding 1: the object, not the algorithm

At 768 B/token — a 16× reduction — with the identical algorithm (PCA, rank
matched to bytes) and identical data, only the object boundary changed:

| carving | KL vs full-KV | top-1 agreement |
|---|---|---|
| per layer (24 objects/token, field default) | 3.66 | 0.297 |
| whole-token stack (1 object/token) | 0.033 | 0.887 |

Spectra agree: 95% of the joint stack's energy lies in 623 of 6,144
dimensions — 5.2× fewer coefficients than the per-layer treatments need for
the same energy, because adjacent layers' contributions are highly
correlated (mean adjacent-layer hidden cosine 0.86–0.94 after layer 2).
Depth redundancy itself is consistent with prior observations (MiniCache;
xKV; see RELATED_WORK.md). What this experiment adds is the matched-bytes,
behavior-scored comparison at the granularity of one token's *complete*
stack: in this model, the common per-layer storage boundary is not aligned
with the cache's most behavior-preserving compressible structure, and the
cost of that misalignment is large. Whether the same gap holds at scale is
the pre-registered gate of §8.

## 4. Finding 2: behavior, not energy — and recall, not KL

Two defaults of the compression literature fail here, measurably:

**(a) Mean KL is not a safe fidelity metric.** A trajectory-space codec
(compress hidden trajectory, regenerate KV through frozen weights) achieves
*better* mean KL than the KV-space codec at 32–64× — while recalling **0/8**
planted facts at every ratio. The variance-metric whole-stack codec at 16×
scores mean KL 0.033 ("nearly intact") yet recalls only 4/8. Average
behavior and addressable detail dissociate completely; results quoted on
KL/perplexity alone can be arbitrarily misleading about whether the model
can still *find things*.

**(b) Standardized-PCA components are behavior-misordered.** Under
per-dimension variance normalization (the common default before PCA),
quality is non-monotone in rank: rank 192 (KL 6.54) is far worse than rank
96 (KL 4.34), and at rank 384 the standardized codec recalls 1/8 while
*raw* (unnormalized) PCA at identical bytes recalls 8/8 — despite the
standardized codec's lower KL, a second dissociation. Partially
reconstructing a direction attention is sensitive to misdirects retrieval
worse than omitting it, and standardization inflates exactly the wrong
directions while suppressing the high-magnitude query-probed key channels
that raw PCA keeps by accident. We replace the metric with an explicit
diagonal behavioral one:

- **K channels** weighted by the rms of the post-RoPE *queries* that will
  probe them (pooled over the query heads sharing each KV head,
  pair-symmetrized so the metric commutes with RoPE and applies to pre-RoPE
  storage). Rationale: attention-logit damage is `q·Δk`, so the expected
  damage metric is `E[qq^T]`; we use its diagonal.
- **V channels** weighted by o_proj column norms (how strongly each value
  channel reaches the residual stream).

PCA in this metric allocates components by expected behavioral damage.
Effects, at identical bytes:

| rank | KL (variance metric) | KL (behavioral metric) |
|---|---|---|
| 96 | 4.34 | 7.40 |
| 192 | **6.54** (non-monotone) | 2.55 (monotone) |
| 384 | 0.53 | 0.11 |
| 768 | 0.033 | **0.010** |

| method @ bytes | needle recall | gold-digit NLL |
|---|---|---|
| full KV | 8/8 | 0.123 |
| variance metric, 16× | 4/8 | 0.709 |
| **behavioral metric, 16×** | **8/8** | **0.130** |
| behavioral metric, 32× | 7/8 | 0.263–0.286 |

The non-monotonicity was never a property of the model — it was a diagnostic
of the wrong objective.

## 5. Finding 3: K and V are one object too

Given finding 2's evidence that keys are precision-critical, we tried
granting K a private basis and budget (K512/V256 and K576/V192 splits,
behavioral metric, 768 B total — identical bytes to the joint 16× point).
Result: **0/8 recall, gold-digit NLL ~2.8** — catastrophically worse than
the joint codec. The shared principal components couple each token's K and V
across all layers; severing that coupling destroys addressability even when
K's own budget *increases*. Three independent experiments now point the same
way: cut the per-token stack anywhere — between layers, or along the K/V
seam — and you pay heavily at matched bytes. The whole stack is the object.

## 6. Is context doing the work?

The founding hypothesis holds that *context* is what renders per-token state
compressible. Current evidence, honestly split:

**6.1 Sequence-specific structure: yes. Semantic causation: not yet
established.** A basis fit on the evaluation document compresses it to KL
0.011 at 16×; the same-rank basis fit on a *different* document (technical
preprint applied to Moby-Dick): KL 0.269 — **24× worse at identical bytes**.
The exploited structure is therefore substantially **document-specific**. We
deliberately describe this as sequence/document-specific compressible
structure rather than proof that *semantic context* causes the effect —
domain, register, or token-distribution shift could each contribute, and a
weak token-level linear probe was inconclusive. The founding hypothesis
`t_i + C_i → Z_i` becomes established only by a controlled **conditional
rate-distortion comparison**: at identical latent budget and comparable
decoder capacity, compare `Z_i → X̂_i` against `(t_i, C_i, Z_i) → X̂_i`
(a causal predictive coder conditioning on previous tokens' *compressed*
states), and plot bytes/token vs behavioral fidelity for both. If context
conditioning moves that frontier strongly left, the hypothesis holds; this
is pre-registered as part of §8.

**6.2 Corpus basis — the accounting question. [COMPLETED — gate failed;
results verbatim as pre-registered.]** A per-document basis is ~9 MB and
only amortizes on long documents. If a single basis fit across a diverse
corpus (two novels, scientific prose, political prose, source code)
preserved recall on held-out documents, the basis would ship with the model
at zero marginal cost. It does not:

| method (768 B/token, 16× marginal) | recall (held-out needle doc) | gold-NLL |
|---|---|---|
| full KV | 8/8 | 0.123 |
| corpus basis (5 diverse docs, held-out eval) | **3/8** | 0.378 |
| per-doc basis (reference) | **8/8** | 0.130 |

Held-out teacher-forced KL (Grimm, never in corpus): corpus basis 0.158 at
16× vs ~0.01-scale for per-doc bases. A five-document corpus basis behaves
closer to the single-foreign-document basis of §6.1 than to the per-doc
basis at identical bytes.

Two consequences. **Scientifically**, this strengthens the evidence that
the low-dimensional structure enabling 16× is substantially constructed per
sequence — generic model-level structure is not sufficient. It is
*consistent with* the founding context hypothesis but does not establish
semantic causation (§6.1). **For accounting**, the 16× marginal figure requires the per-doc
basis to be amortized: net bytes/token including a 9.4 MB fp16 basis are
1,920 (6.4×) at 8k, 1,056 (11.6×) at 32k, 840 (14.6×) at 128k. The claim
therefore stands for long contexts — the regime where KV compression matters
— and short-context deployment awaits a learned encoder that *generates* the
document adaptation instead of storing it (§8).

## 7. Honest frontier and accounting

All ratios are quoted against the model's real fp16 GQA cache (12,288
B/token), which already embeds a 7× reduction versus equivalent MHA. Under
the standing rule that no point is quotable without a recall score:

| bytes/token | ratio | method | recall | KL | top-1 |
|---|---|---|---|---|---|
| 12,288 | 1× | full KV fp16 | 8/8 | 0 | 1.000 |
| 4,704 | 2.6× | quant K-8bit V-4bit | 8/8 | 0.003 | 0.957 |
| **768** | **16×** | **behavioral joint-768, int8 coeffs** | **8/8** | **0.010** | **0.938** |
| 384 | 32× | behavioral joint-768, int4 coeffs | 7/8 | — | — |
| 0 (marginal) | ∞ | recompute from text | 8/8 | 0 | 1.000 |

The last row is the degenerate baseline every method must justify itself
against: reconstruction here is a rank-768 matrix multiply per read versus a
full prefill — the codec buys its bytes with ~4 orders of magnitude less
reconstruction compute than recomputation. A further compute option not yet
exploited: with a linear codec the expansion can be absorbed into the
query/output projections (MLA-style), letting attention run directly in the
768-dim latent without ever materializing the stack.

**Small-scale robustness and transfer (laptop suite, 2026-08-23).** Four
follow-ups sharpen the frontier above:

- **Randomized-needle robustness:** across 3 repetitions with fresh random
  entities, values, jittered positions, and three filler domains (technical,
  fiction, political prose), behavioral 16× recalls **24/24** (full KV:
  24/24); behavioral 32× (int4 coeffs): 21/24. The original 8/8 was not a
  lucky draw, and no systematic depth bias appears at 16×.
- **Family transfer:** on frozen TinyLlama-1.1B (Llama architecture), with
  ranks scaled to its own stack (16× = rank 1408 of 11,264), behavioral 16×
  recalls 8/8 where the variance-metric control gets 5/8 and full KV itself
  gets 7/8 (this weaker model is flaky on one needle); teacher-forced KL
  0.0034. The behavioral-vs-variance gap is not Qwen-specific. On
  Qwen2.5-1.5B every method including the variance control passes 8/8 (KL
  0.0001) — at 2k tokens and rank 1792 the per-document basis nearly
  interpolates the document, so this scale lacks discriminative power;
  longer contexts (§8) are the real test there.
- **Longer context:** at 7,465 tokens with 16 needles spanning full depth,
  behavioral 16× (marginal) recalls 14/16 (gold-NLL 0.197 vs 0.110 full-KV);
  32× drops to 9/16. Net ratio at this length including the fp16 basis:
  ~5.9×. The rank-vs-context-length scaling law is a primary question for
  the scale gate.
- **A negative result:** quantizing the *basis* to int8 (which would halve
  basis storage) destroys recall — 2/24 at 16×. The basis, like the keys it
  reconstructs, is precision-critical; all net-ratio figures in this
  document therefore assume an fp16 basis.

**Overnight follow-ups (C-suite, same date).** Five further results, the
sharp ones first:

- **Decode-time viability (held-out tokens):** a basis fit on only the first
  60% of the document, applied to everything including needles planted
  *beyond* the fit boundary: 7/8 recall (gold-NLL 0.229 vs 0.130). Bases
  generalize forward to tokens that did not exist at fit time — the
  deployment shape — with a measurable but modest penalty.
- **Rank vs context length is sublinear so far:** rank 768 covers ~1.9k
  tokens (24/24); at ~7.5k, +50% rank (1152, 10.7× marginal) recovers 15/16
  where 768 held 14/16 — and doubling to 1536 adds nothing: the last miss is
  rank-*insensitive* (fails at 8× too; full KV passes), so the residual
  failure mode at length is not capacity.
- **Metric resolution (the C4 showdown):** the villain behind the
  non-monotonicity is *standardization* specifically; raw PCA matches the
  behavioral metric at 16–32× on simple needles; the metrics separate on the
  typed battery at 32× (non-diagonal 13/14 > diagonal behavioral 11/14 >
  full KV reference 14/14) and at rank 192, where behavioral (2/8) beats
  raw, standardized, and non-diagonal (all 0/8). **64× is a hard wall for
  every linear codec tested.**
- **Typed-needle battery:** at 16×, both behavioral and non-diagonal codecs
  recall 14/14 across numbers, UUIDs, code identifiers, person names, and
  confusable near-collision pairs; at 32× the failures concentrate exactly
  where key-precision theory predicts (identifiers, confusables).
- **Token-level conditional coding (linear pilot): null.** Predicting a
  token's code from previous tokens' codes explains less held-out variance
  than token identity alone, and at matched bytes all conditional variants
  lose behaviorally to unconditional quantization. Combined with §6, the
  context that enables compression appears to be *document-global* (which
  subspace) rather than *local-sequential* (predict the next code) — a
  design steer for the learned codecs of §8, not a closed question: the
  pilot is linear and underpowered by construction.

Additional measured structure: per-token residuals are thin-tailed (99th
percentile 0.153 vs median 0.092 at rank 384) — under linear codecs there is
no expensive-token minority, and uniform per-token budgets are justified;
keys occupy a *lower*-dimensional subspace than values (rank ~46 vs ~88 of
128 at 95% energy) but are precision-critical (2-bit K catastrophic, 4-bit V
nearly free); recalled needles at stressed ratios cluster shallow even
though deeper needles sit closer to the query (unexplained; logged).

## 8. Pre-registered scale gate and limitations

Everything above is one 0.5B model, 2k-token contexts, 8 needles, one seed,
linear codecs, CPU. The claims are staked on the following gate, to be run
on RTX 4090-class hardware and reported regardless of outcome:

- **models:** Qwen3-4B-class dense model plus a substantially different
  family, frozen, no retuning;
- **contexts:** 8k, 16k, 32k+; multiple data domains; held-out documents;
- **recall battery** (protocols adapted from "The Pitfalls of KV Cache
  Compression", arXiv:2510.00231, ACL 2026): many more than 8 needles at
  systematic depths and multiple seeds; repeated/confusable exact values;
  code identifiers; UUID-like strings; names; arithmetic-relevant numbers;
  IFEval-style multi-instruction prompts with **per-instruction-class
  degradation curves over a dense compression-ratio sweep** (their leakage
  results peak at *mid-range* ratios — endpoint-only evaluation is
  insufficient); a system-prompt-leakage test (defense + directive scored
  by ROUGE-L, both instruction orders); per-span reconstruction-error
  instrumentation as the non-eviction analogue of their keep-rate metric;
  **free-running generation**, not only teacher forcing; and RULER's harder
  variants (multi-key NIAH with distractors, multi-value/multi-query NIAH,
  variable tracking — not only single-needle, which models ace while
  failing the rest). Evaluation documents will not be used for any tuning
  before reporting;
- **baselines at matched bytes** (matched bytes, not matched nominal
  rank): full fp16/bf16 KV; straightforward KV quantization and a strong
  modern quantization baseline (KIVI/TurboQuant-class) where practical;
  per-layer low-rank; a faithful cross-layer-merging analogue
  (MiniCache-style) and xKV-style layer-group SVD at matched effective
  storage; and full recomputation from text as the zero-cache compute
  baseline. Reconstruction FLOPs and wall-clock reported separately from
  representation size;
- **context test:** the conditional rate-distortion comparison of §6.1
  (context-conditioned vs unconditioned codec at identical latent budget) —
  the founding hypothesis is claimed only if this moves the frontier;
- **pre-registered failure criteria:** the mechanism fails the gate if
  recall (across the battery, not only simple needles) degrades materially
  at ≥16× marginal where the 0.5B result held, or if matched-byte baselines
  reach equivalent fidelity — in which case the contribution reduces to the
  evaluation methodology and the matched-bytes comparisons, and will be
  reported as such. Anomalous runs are retained; observation is
  distinguished from explanation.

> [PENDING: scale-gate results table]

Beyond the gate: a learned, context-conditioned encoder/decoder trained by
distillation against the frozen teacher (the token-level test of the
founding hypothesis), non-diagonal behavioral metrics, and composition with
write-time KV routing (mapped KV), where routing reads only a small fraction
of entries per step and thus amortizes even expensive reconstruction.

## 9. Method note: falsification-first

Every experimental default in this project is tracked in a public
assumptions ledger (ASSUMPTIONS.md) with origin, status, and the cheapest
decisive test; results above correspond to ledger entries A1–A18. Four
inherited defaults were falsified by matched-bytes experiments (per-layer
carving; K/V symmetric treatment; variance as objective; mean-KL as
sufficient metric); one founding intuition was demoted and partially
re-earned (context, §6); one anomaly was resolved as metric artifact rather
than model property (§4b). The ledger, not the narrative, is the record.

## 10. Reproduction

Scripts (Python, torch CPU-sufficient at 0.5B scale) in `scripts/`:
`capture.py` (instrumentation + trajectory verification), `analyze_spectra.py`
(§3 spectra), `eval_compress.py` (§3–4 Pareto), `a7_needle.py` (recall
harness), `a5_crossdoc.py` / `a5b_corpus_basis.py` (§6), `a6_budget.py`
(budget distribution), `a8_behavior_basis.py` (§4 behavioral metric).
Model: `Qwen/Qwen2.5-0.5B` via HuggingFace; documents: Project Gutenberg +
this repository's parent preprint.
