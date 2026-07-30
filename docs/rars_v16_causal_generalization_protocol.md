# RARS-v16 Same-Encoder Mechanism Decomposition

## Purpose

V16 does not add another learned sidecar. It determines what should be built
next by separating five explanations that earlier experiments confounded:

1. insufficient headroom inside the frozen Top-100 candidate set;
2. insufficient rank capacity;
3. loss from int8 coefficient coding;
4. weak value of the cutoff-aware training objective over geometric PCA;
5. interaction between the corpus used to fit the basis and the corpus used
   for evaluation.

The machine-readable freeze is
[`protocols/rars_v16_causal_generalization_diagnostic_v1.json`](../protocols/rars_v16_causal_generalization_diagnostic_v1.json).

## Evidence level

This experiment was designed after V1--V15 and external benchmark outcomes
were inspected. It therefore uses **outcome-informed development evidence**.
FiQA and SciFact are not treated as unopened confirmation roles, and V16
cannot support universal cross-domain or cross-encoder claims.

The primary reason to reuse these opened roles is diagnostic efficiency: the
experiment can test the factor matrix without spending a new holdout before
the method family is stable.

## Controlled data design

Both domains must use the same pinned encoder ID, revision, embedding
dimension, pooling, normalization, candidate width, and IVF-PQ recipe. Each
domain has disjoint `fit` and `evaluation` query roles. The evaluator accepts
only two hash-registered domain IDs:

- `fiqa_bge_same_encoder`
- `scifact_bge_same_encoder`

### Pre-metric feasibility amendment

The first preparation attempt exposed an arithmetically impossible data
contract before any retrieval metric was computed or sidecar basis was fit:
SciFact has 300 judged queries, while the original disjoint-role minima
required 150 fit plus 200 evaluation queries. Revision 2 therefore uses an
exact hash-ranked 60/40 split and lowers only the per-domain evaluation
minimum to 100 queries. The fit minimum remains 150. This correction does not
change the method, encoder, index recipe, factor matrix, hyperparameters,
statistical thresholds, or decision order. SciFact intervals must be
interpreted with its smaller evaluation sample explicitly visible.

A second preparation-only failure exposed an encoder provenance typo before
model construction: `b8903db...` belongs to
`sentence-transformers/all-MiniLM-L6-v2`, not
`BAAI/bge-small-en-v1.5`. Revision 3 pins the intended BGE model to its
verified repository revision
`88885630388d6249d876a3ab145b78b34665b79a`. Preparation now downloads that
exact snapshot, registers every snapshot-file hash, checks required files and
the 384-dimensional BERT configuration, and runs a finite, normalized
one-vector probe before encoding either corpus. No retrieval metric was
computed and no basis was fit before this correction.

A third pre-metric implementation failure occurred when Faiss 1.12 returned a
generic IVF Python wrapper from `extract_index_ivf()`. The serialized indexes
were valid `IndexIVFPQ` objects, but the generic wrapper did not expose `.pq`,
so the builder rejected them before candidate search. Revision 4 applies
`faiss.downcast_index()` before inspecting PQ fields, preserves explicit
rejection of non-PQ IVF indexes, and streams each builder's combined output.
This compatibility repair does not change any prepared vector or index byte.

It consumes prepared candidate/residual bundles and never opens a full corpus
or rebuilds an index. Closed MS MARCO, V3 audit, V9 future, TREC DL, and BEIR
NQ confirmation paths are rejected.

## Factor matrix

All reported contrasts use Recall@10 at the query level.

| Factor | Treatment | Control | What changes |
|---|---|---|---|
| Candidate headroom | same-candidate exact | Base IVF-PQ | candidate scoring only |
| Rank capacity | local PCA rank-64 FP32 | local PCA rank-16 FP32 | rank only |
| Int8 coding | local PCA rank-16 FP32 | same basis rank-16 int8 | coding only |
| Objective value | local cutoff-aware RARS rank-16 int8 | local PCA rank-16 int8 | fitting objective only |
| Fit-domain interaction | local RARS rank-16 int8 | source-domain RARS rank-16 int8 | fit domain only |
| Pooled repair | pooled RARS rank-16 int8 | source-domain RARS rank-16 int8 | multi-domain fit only |

The cutoff-aware arms use the unchanged V8 promotion/protection pair mining,
query-role balancing, PCA anchor, optimizer, `alpha=0.75`, and Top-40
correction. Target candidate residuals set int8 scales without reading
relevance labels.

## Why this is better than another version sweep

Earlier experiments changed rank, code structure, allocation, objective, or
gating in separate versions but did not always hold the other factors fixed.
V16 puts the relevant controls in one packet:

- if same-candidate exact has little headroom, stop post-hoc reranking;
- if rank-64 FP32 wins but int8 is not the bottleneck, develop a
  storage-matched higher-effective-rank code;
- if FP32 beats int8 materially, improve coefficient coding;
- if local RARS does not beat local PCA, drop the learned-basis claim;
- if local wins over source and pooled fitting recovers the loss, develop a
  separately frozen robust multi-domain objective;
- if none passes, stop expanding RARS and retain the strongest simple
  geometric sidecar.

## Statistical contract

Each domain is reported separately. Paired query bootstrap and sign-flip
randomization provide the point estimate, confidence interval, p-value, and
improved/harmed/unchanged support for every factor contrast. The equal-domain
summary gives each domain one vote; it is not described as a larger pooled
query sample.

The final decision is deterministic and uses the preregistered order:

```text
STOP_FROZEN_CANDIDATE_METHOD
OBJECTIVE_REPAIR_SUPPORTED
DOMAIN_SHIFT_SUPPORTED
CODING_BOTTLENECK_SUPPORTED
CAPACITY_BOTTLENECK_SUPPORTED
STOP_LEARNING_CLAIM_KEEP_UNIFORM_RPQ
STOP_RARS_METHOD_EXPANSION
```

No result permits retuning V16 on its evaluation roles. Any new algorithm
requires a later protocol and a new holdout.
