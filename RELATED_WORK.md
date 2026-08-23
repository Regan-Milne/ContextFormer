# Related work: field-by-field comparison

Facts below were extracted from the papers' method/experiment sections (not
titles or abstracts); "NOT FOUND" marks fields we could not verify. Corrections
welcome — this table exists so that our claims can be checked against prior art
precisely.

| method | frozen model? | all tokens retained? | cross-layer? | K/V joint? | per-token latent? | quantization? | token eviction? | reconstruction at read? | reported compression | exact/needle recall | end-to-end speed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MiniCache** (2405.14366, NeurIPS'24) | yes | yes (~5% kept unmerged per pair) | adjacent-layer pairs, deep half only | separate | no (merged direction + per-layer norms) | no (composable w/ KIVI) | no | rescale only | ~41% alone; 5.02× w/ KIVI-4bit | none reported | ~5× throughput |
| **xKV** (2503.18893, ICML'26) | yes (SVD at prefill) | yes (decode tokens uncompressed) | layer groups (W=4), concat + SVD | separate (K:V rank 1:1.5) | yes — r-dim coeffs per group | composable | no | low-rank matmul (selective variant) | ~8× incl. factor storage | RULER 88.5 vs 91.9 baseline @8× | up to 4.23× e2e |
| **JoLT** (2607.12550, 2026) | yes | yes | layer *groups for budget allocation* only; per-layer Tucker | separate (separate budgets) | r_T-dim token coeffs | 4-bit JL-rotated residual | no | Tucker reconstruction each step (no kernel) | 2–3× near-lossless | RULER NIAH ~100% @2–3× | compression-time only |
| **CLLA** (2410.15252) | **no — trained from scratch** | yes | latent shared across layer pairs | **joint** (MLA-style latent) | yes, 512-dim | 4-bit trained-in | no | up-projection per read | ~48× vs **MHA** baseline | none reported | none reported |
| **MLA** (DeepSeek-V2/V3) | **no — architecture** | yes | no (per-layer latent) | **joint** + decoupled RoPE dims | yes (e.g. 512-dim + rope dims) | composable | no | up-projection (absorbable) | ~an order vs MHA | (production models) | production |
| **KIVI** (2402.02750, ICML'24) | yes | yes (+128-token fp16 window) | no | asymmetric: per-channel K / per-token V | no | 2-bit, group 32 | no | dequant fused in matmul | ~2.6× peak mem | NIAH maintained (heatmaps) | 2.35–3.47× |
| **TurboQuant** (2504.19874) | yes (data-oblivious) | yes | no | same scheme both; outlier-channel bits | codes + 1-bit QJL residual | 2.5–3.5 bit/channel | no | lookup + unrotate (unbiased) | >5× | NIAH 0.997 = baseline @>4× | NOT FOUND in paper |
| **ShadowKV** (2410.21465) | yes (SVD at prefill) | yes (sparse *access*) | no | strongly asymmetric: low-rank pre-RoPE K / offloaded full V | yes — rank-160 K coeffs | no (bf16) | no | K rebuilt for selected chunks + RoPE | ~6× GPU mem | NIAH ok 16K–1M; RULER-128K ~86.9 | 3.04× throughput |
| **SnapKV** (2404.14469, NeurIPS'24) | yes | **no** | no | joint indices | no | no | **yes — permanent** | none | up to 380× (@380K ctx) | 3.6× speed, 8.2× mem |
| **PyramidKV** (2406.02069) | yes | **no** | layerwise *budgets* only | joint indices | no | no | **yes — permanent** | 6–25% of cache | NIAH 100% (70B, 8K, 128 entries) | latency ≈ SnapKV |
| **Quest** (2406.10774, ICML'24) | yes | yes (storage lossless) | no (dense first 2 layers) | joint pages | no (page min/max metadata) | no | no (sparse access) | none | bandwidth ~8×, storage 1× | passkey 96–100% | 7.03× attention |
| **ContextFormer (this work)** | yes | yes | **all layers, one object per token** | **joint K+V (separating them fails, §5)** | yes — 768-dim behavioral-PCA coeffs | int8/int4 coeffs | no | linear expand + RoPE (absorbable in principle) | 16× marginal; net 6.4–14.6× incl. per-doc basis | **8/8 @16×; 7/8 @32×** (gated metric) | not yet measured |

## Nearest neighbors, and exactly what differs

**xKV** is the closest storage-side method: cross-layer SVD with per-token
coefficients, computed per prompt at prefill (the same per-sequence-basis
paradigm as ours), factor storage included in its accounting. Differences:
xKV groups W=4 contiguous layers and factorizes **K and V separately**
(rank ratio 1:1.5); we treat the **entire depth and both K and V as one
coupled object**, and our matched-bytes experiments show both choices
matter in our model — per-layer/grouped decompositions lose ~100× KL, and
separate K/V bases lose exact recall entirely at matched bytes. xKV also
selects components by singular value (energy); we show energy ordering is
behavior-misordered in our model and derive a behavioral metric instead.

**ShadowKV** anticipates part of our key-side mechanism: low-rank **pre-RoPE
keys** via prefill-time SVD with per-token coefficients. Its value side is
uncompressed (CPU-offloaded), its rank selection is energy-based, and its
contribution is a serving system (sparse access + tiering) rather than a
study of the per-token representation. We credit it as precedent for
pre-RoPE key factorization.

**MiniCache** established training-free cross-layer redundancy (adjacent-pair
merging, deep layers). Our result extends the depth axis from pairs to the
full stack and adds the K/V-coupling and behavioral-metric findings.

**MLA / CLLA** are the architectural existence proofs that a small joint
per-token latent can carry K/V state — trained in from scratch. Our question
is the post-hoc frozen-model version: how much of that latent structure
already exists in a pretrained model's cache, and how to extract it without
touching weights. Ratios are not directly comparable (CLLA quotes ~48×
against an MHA baseline; our 16× is against an already-GQA-compressed
cache).

**KIVI / TurboQuant** are strong storage baselines orthogonal to structure:
TurboQuant's NIAH-neutral >4–5× is the quantization bar any structural
method must beat *in combination*, and both are composable with ours
(our coefficients are already int8/int4).

**SnapKV / PyramidKV / Quest** answer a different question (which tokens to
keep or read), not how much state each token needs. SnapKV and PyramidKV
permanently evict; Quest keeps storage lossless and sparsifies access. Our
framing retains every token by construction; Quest-style sparse access
composes naturally with per-token compression (reconstruction cost is then
paid only on read).

## Evaluation practice

"The Pitfalls of KV Cache Compression" (ACL 2026) warns that aggregate
benchmarks hide failures in realistic multi-instruction behavior (details
in §8 of PREPRINT.md, whose expanded gate adopts its recommendations). Our
KL-vs-recall dissociation result (§4a of the preprint) is an independent,
mechanism-level instance of the same warning.
