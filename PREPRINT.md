# One Object per Token: Behavior-Weighted Compression of the Cross-Layer KV Stack on Frozen LLMs

**Author:** Regan Milne (github.com/Regan-Milne)
**Date:** 2026-08-23 (draft; not yet released)
**Status:** DRAFT. Of two pre-registered gates, the corpus-basis test (§6.2)
is complete (failed; reported verbatim — the 16× figure is long-context
only, pending a learned encoder) and the scale-transfer gate (§8) is
pending and will be reported regardless of outcome. This document will serve as a timestamped disclosure of the
mechanism and results upon release.
**License (this document & the forthcoming code):** Apache-2.0.

---

## Abstract

Every token an LLM reads leaves behind a layer-by-layer stack of key/value
vectors — the KV cache — whose size, not the token's, dominates long-context
memory. Prior compression work carves this cache along its storage layout:
per layer, and within a layer per K or per V, then quantizes or factorizes
each piece. We present evidence that **this carving is wrong in every
direction it cuts**, and that the natural unit of compression is the
**whole per-token stack**: all layers, K and V together, treated as one
object.

On a frozen Qwen2.5-0.5B (no training, no finetuning) with a plain linear
codec (PCA), three matched-bytes results:

1. **Cross-layer carving.** Compressing each token's whole 24-layer stack as
   one object beats per-layer compression by ~**100× in next-token KL at
   identical bytes/token** (0.033 vs 3.66 at 768 B/token). The dominant
   redundancy in KV runs *between* layers — invisible, by construction, to
   any per-layer method.
2. **Behavioral metric.** Variance (PCA energy) is the wrong objective:
   rank-vs-quality is *non-monotone* under it (rank 192 is far worse than
   rank 96), an artifact that vanishes under a simple behavior-weighted
   metric — K channels weighted by the rms of the queries that will probe
   them, V channels by o_proj column norms. The behavioral metric also
   converts a failing exact-recall score into a passing one (below).
3. **K/V coupling.** Giving K and V separate bases — even with K favored,
   even under the behavioral metric — collapses exact recall to 0/8 at
   matched bytes. Cutting the stack along the K/V seam is as destructive as
   cutting it along layers.

With the whole-token object and the behavioral metric, the frozen model runs
at **16× fewer KV bytes/token (768 B vs 12,288 B fp16) with exact-detail
recall fully intact**: 8/8 planted 5-digit facts recovered by greedy decoding
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
  structure that position rotation otherwise destroys.

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
correlated (mean adjacent-layer hidden cosine 0.86–0.94 after layer 2). The
transformer re-derives nearly the same representation layer after layer;
per-layer compression discards exactly that structure. The convention it
replaces was never a finding — it is the cache's memory layout promoted,
unexamined, into a method.

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

**(b) Variance-ordered components are behavior-misordered.** Under the
variance metric, quality is non-monotone in rank: rank 192 (KL 6.54) is far
worse than rank 96 (KL 4.34). Partially reconstructing a direction that
attention logits are sensitive to misdirects retrieval worse than omitting
it. We replace the metric with a diagonal behavioral one:

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

**6.1 Document-level context: yes.** A basis fit on the evaluation document
compresses it to KL 0.011 at 16×; the same-rank basis fit on a *different*
document (technical preprint applied to Moby-Dick): KL 0.269 — **24× worse
at identical bytes**. The exploited structure is substantially specific to
the document. (Token-level context conditioning — a causal predictive coder
storing only the residual after previous tokens' compressed states predict
the current token's stack — is future work, §8; a weak linear probe at token
level was inconclusive.)

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

Two consequences. **Scientifically**, this is the strongest evidence yet
*for* the founding hypothesis: the low-dimensional structure that enables
16× is substantially constructed per document — context, at document
granularity, is doing the work, and generic model-level structure is not
sufficient. **For accounting**, the 16× marginal figure requires the per-doc
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

- models: Qwen3-4B (dense) and a hybrid-family model, frozen, no retuning;
- contexts: 8k–32k, held-out documents, needles at systematic depths;
- protocol: identical harness (compressed past, teacher-forced window,
  greedy needle retrieval), behavioral-metric whole-stack codec at 8×/16×/32×,
  with quantization and per-layer low-rank as controls;
- success: recall intact at ≥16× with ΔNLL within noise of the 0.5B result;
- failure modes to be characterized, not hidden.

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
