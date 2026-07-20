# RARS-v5 PQ-Aware 100K Pilot Protocol

## Goal

This development pilot asks one narrow question before any new RARS experiment
is allowed:

> Can a small rank-8 query/document adapter, trained through the exact hard
> residual-PQ operation used at deployment, recover end-to-end Top-100 recall
> on a deterministic 100K corpus without damaging Top-10 recall or FP32
> retrieval quality?

The machine-readable contract is
[`protocols/rars_v5_pq_aware_100k_pilot_v1.json`](../protocols/rars_v5_pq_aware_100k_pilot_v1.json).
Its configuration and decision thresholds are frozen before the first v5
outcome is observed.

This is an outcome-informed development experiment. It is not independent
confirmation, an official MS MARCO result, a deployment benchmark, a novelty
claim for quantization-aware training, or evidence that the project is ready
for SIGIR.

## Method

The BGE query and document encoders remain frozen. For an original normalized
embedding `x`, the pilot learns a rank-8 residual adapter:

```text
x' = normalize(x + U(V x))
```

Both adapters start as exact identity functions because `U` is initialized to
zero. The IVF centroids, every document's IVF list assignment, and all residual
PQ codebooks are fixed. Residual PQ code assignments are recomputed by hard
nearest-codeword assignment in the forward pass:

```text
r = d' - coarse[d]
code[m] = argmin_j ||r[m] - codebook[m, j]||²
d_hat = coarse[d] + concat_m codebook[m, code[m]]
score(q, d) = q'ᵀ d_hat
```

The forward value is the hard reconstruction. An identity straight-through
estimator supplies the document-side gradient; no soft-PQ score may be used for
checkpoint selection or the formal decision.

Training pairs compare an observed relevant passage with an unjudged hard
candidate near the base IVF-PQ rank-100 boundary. A zero label means only
"unjudged mined hard negative" and is never called explicit non-relevance. The
loss combines:

```text
L = L_pairwise_hard_PQ
  + 0.5 L_FP32_margin_distillation
  + 0.01 L_reconstruction
  + 0.01 L_cosine_drift
```

Pairs with a PQ-induced order flip or larger FP32-to-PQ margin damage receive
more weight. This objective is a differentiable surrogate for Top-k behavior;
Recall itself is not differentiated.

## Data Separation

- Training: the already-observed 2,307-query v3 `oracle_design` role.
- Checkpoint selection and the one-seed decision: the already-observed
  851-query v3 `oracle_audit` role.
- Corpus: 100,000 documents containing every known-positive passage observed
  in those two roles plus a deterministic random fill from the frozen 1M
  corpus.
- The 803-query `future_method_holdout` remains identity-only. Its candidate
  arrays, labels, and metrics may not be materialized or opened.
- No external collection or previously closed test set may be opened.

Missing relevant passages are appended only to the training pair-mining
candidate matrix. They are never appended to selection retrieval results.
Every selection epoch performs a fresh, end-to-end 100K search with the frozen
IVF list assignments, `nprobe=16`, newly assigned hard M32 PQ codes, and no
label-dependent candidate construction. The FP32 guardrail uses full-scan
adapted vectors.

The relevance denominator contains only positives already observed in the
source v3 Top-100 roles. All reported metrics must therefore be called
"100K known-positive development Recall," not official MS MARCO Recall.

## Frozen Configuration

| Item | Value |
|---|---:|
| Pilot documents | 100,000 |
| IVF lists / probes | 256 / 16 |
| PQ | M32, 8 bits per subquantizer |
| Training candidates | Base IVF-PQ Top-200 plus missing training positives |
| Query adapter | Rank-8 residual linear adapter |
| Document adapter | Rank-8 residual linear adapter |
| Trainable parameters | 12,288 |
| Epochs / seed | 8 / 42 |
| Batch size | 1,024 pairs |
| Optimizer | AdamW, learning rate `3e-4`, weight decay `1e-4` |
| Primary selection metric | End-to-end hard-IVF-PQ known-positive Recall@100 |

The registered Colab environment uses Python 3.12.13, NumPy 1.26.4, PyTorch
2.11.0+cu128, CUDA 12.8, a T4 GPU, deterministic PyTorch algorithms, cuDNN
benchmarking disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Although the learned matrices are small, this QAT path is **not** a
frozen-index retrofit: applying the document adapter requires re-encoding and
rewriting document PQ codes. RARS retains a different deployment role and is
not trained in this stage.

## Stage-A Decision

All gates must pass:

1. Mean hard-IVF-PQ Recall@100 gain is at least `+0.005`.
2. The 20,000-replicate paired-bootstrap 95% lower bound is above zero.
3. At least 15% of the base-to-exact Recall@100 gap is recovered.
4. At least `max(20 queries, 2% of selection queries)` improve.
5. Hard-IVF-PQ Recall@10 drops by no more than `0.002`.
6. Adapted-FP32 Recall@100 drops by no more than `0.002` from the original
   exact teacher.

Passing yields `GO_TO_THREE_SEED_100K_REPLICATION`. It authorizes only exact
seed-43 and seed-44 replication under a new execution contract. It does not
authorize a 1M rebuild, RARS combination, external evaluation, paper claim, or
opening the future holdout. Any failed gate yields
`STOP_PQ_AWARE_100K_PILOT`.

## Expected Artifacts

The bundle builder writes the 100K IVF-PQ index, frozen lineage hashes, local
document mappings, IVF assignments, codebooks, observed-positive sets, and
label-independent base/exact retrieval outputs. The trainer writes:

- query and document adapter matrices;
- selected hard-IVF-PQ and adapted-FP32 Top-100 rows;
- paired per-query Recall@100 arrays;
- training history, gate outcomes, source commit, environment, and SHA-256
  records.

The Colab notebook is intentionally pinned to the implementation commit. Run
it top to bottom on a T4 runtime; do not edit the protocol, open the future
holdout, or reuse a partially populated output directory.

## Next Stage, Only After a Replicated GO

The later paper-quality matrix must compare plain M32 IVF-PQ, OPQ-M32,
reconstruction-only adaptation, the rank-aware adapter, PCA and RARS alone,
adapter plus PCA/RARS, and an equal-storage higher-rate M48 rebuild. It must
report end-to-end quality, bytes, indexing/re-encoding cost, latency, QPS, and
paired uncertainty. The combination with RARS is a separate preregistered
experiment, not an automatic continuation of this pilot.
