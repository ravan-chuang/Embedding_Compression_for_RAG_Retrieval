# RARS-v10 PCA-Anchored Harm-Constrained Sidecar

## Status and evidence boundary

RARS-v10 is a **single-configuration, post-confirmation development
experiment**. It is frozen before its first run on the historically opened
2,307-query `oracle_design` role. It is not an independent confirmation, an
external evaluation, or a continuation of the V9 confirmation set.

V9 established two facts that define the new question: a compact frozen-index
sidecar improves over Base, but RARS-v8 does not establish an advantage over a
storage-matched PCA sidecar. V10 therefore asks a narrower question: can a
learned rank-16 basis exceed PCA while explicitly controlling per-query tail
harm and remaining close to the PCA subspace?

The V10 executable is forbidden from reading a V9 packet, V9 per-query arrays,
`future_method_holdout`, `oracle_audit`, or any external test set. V9 is used
only as disclosed motivation. No V9 query can authorize, reject, tune, or
select V10.

## Deployment contract

V10 preserves the same frozen-index retrofit boundary as V8:

- 1,000,000 documents, 384 dimensions, inner product;
- IVF `nlist=512`, `nprobe=16`, PQ `M=32`, 8 bits/subquantizer;
- immutable query/document embeddings, coarse quantizer, inverted lists,
  codebooks, and PQ codes;
- one rank-16 int8 residual sidecar, `16 B/document` representation payload;
- `alpha=0.75`, candidate pool 100, correction depth 40, final cutoff 10;
- no query adapter and no stacking of PCA and learned document codes.

The score is

```text
s_v10(q,d) = s_pq(q,d)
             + 0.75 <q B, dequant_int8(code_B(r_d))>.
```

Thus a positive outcome cannot be attributed to additional document bytes,
deeper reranking, changed queries, or a rebuilt index.

## Method

The initial basis is deterministic uncentred residual PCA. V10 then optimizes
one registered objective on four folds and scores the fifth fold out of sample.
The five validation folds are concatenated into the only method-quality result.

The objective combines four terms:

1. query-balanced smooth-L1 distillation of exact pairwise residual margins;
2. a smooth barrier for positive documents near the Top-10 cutoff;
3. query-equal CVaR tail-harm relative to the PCA-corrected pair margin;
4. a projector-overlap anchor that keeps the learned subspace near PCA.

Cutoff pairs inherit the frozen V8 mining rule. A challenger is merely absent
from positive qrels and is never described as an explicitly non-relevant
document.

Optimization uses deterministic full-batch Riemannian gradient descent on the
Stiefel manifold. Every proposal is QR-retracted, must satisfy Armijo decrease,
and must stay within 20 degrees of PCA. The final accepted step is used; there
is no metric-based checkpoint selection, seed sweep, loss-weight sweep, or
fallback configuration.

Each fold and the final fit must pass a central finite-difference directional
gradient audit on the first 512 deterministic training pairs. All accepted
objective values must be monotone. A failed numerical audit is a failed method,
not a repair opportunity after metrics are visible.

## ScaNN-inspired scalar-quantization headroom diagnostic

The same run also records one **diagnostic-only** comparison motivated by
ScaNN's anisotropic vector-quantization objective. It scores PCA rank-16
coefficients once in FP32 and once with the existing per-dimension int8 scalar
quantizer, while holding the basis, Base candidates, `alpha`, Top-B, and query
role fixed. Their Recall@10 difference isolates coefficient-quantization loss;
it excludes rank truncation, IVF routing, and candidate-pool loss.

This is not an implementation or benchmark of ScaNN. No codebook is trained in
V10. `GO_TO_SEPARATE_AVQ_CODEBOOK_PROTOCOL` requires at least `+0.003`
Recall@10, a positive bootstrap lower bound, one-sided randomization `p<=0.05`,
at least 20 improved queries, and net support at least 10. Otherwise the result
is `STOP_AVQ_CODEBOOK_NO_SCALAR_HEADROOM`.

If the diagnostic passes, a later protocol may test 16 learned 256-entry scalar
codebooks: one byte per projected coefficient, with the global codebook
overhead reported separately. The V10 run itself never fits that codebook and
does not combine it with the learned V10 basis.

## Frozen inference and decision

The primary endpoint is paired known-positive Recall@10 for V10 minus the
storage-matched PCA sidecar. Inference uses 50,000 paired-bootstrap replicates
with seed `20260730` and 100,000 paired-randomization replicates with seed
`20260731`; no seed offsets are permitted.

`GO_TO_FRESH_EXTERNAL_V10_PROTOCOL` requires every registered condition:

- Recall@10 gain over PCA at least `0.005` and bootstrap lower bound above 0;
- one-sided paired-randomization `p <= 0.025`;
- at least 30 improved queries and net improvement at least 15 over PCA;
- non-negative V10-minus-PCA gain in every fold;
- Recall@10 gain over Base at least `0.01`;
- at least 15% recovery of the same-candidate gap;
- MRR@10 and nDCG@10 losses versus PCA no worse than `-0.002`;
- all gradient audits pass and all accepted losses are monotone.

If any condition fails, the decision is
`STOP_V10_NO_STABLE_PCA_ADVANTAGE`. Even a GO does not authorize opening V9 or
an old holdout: it only permits writing a new, preregistered protocol for a
genuinely fresh dataset/model.

## Execution

The source-hash-pinned Colab entry point is
[MSMARCO_RARS_v10_Stable_Sidecar_Development.ipynb](../notebooks/MSMARCO_RARS_v10_Stable_Sidecar_Development.ipynb).
The machine-readable source of truth is
[rars_v10_pca_anchored_harm_constrained_v1.json](../protocols/rars_v10_pca_anchored_harm_constrained_v1.json).
Run the notebook once in a fresh T4 runtime, return both the downloaded ZIP and
executed notebook, and do not edit or rerun after metrics appear.
