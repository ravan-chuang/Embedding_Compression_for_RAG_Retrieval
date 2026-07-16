# RARS-v2 Boundary-Loss Feasibility Protocol

## Status and evidence boundary

This is a new development protocol. It does not revise RARS-v1 and must not read
the closed BEIR NQ test queries, qrels, Stage-3 evaluation arrays, or post-hoc
per-query outputs. The NQ post-hoc summary motivates the hypothesis only.

Development uses a fresh deterministic split of the BEIR NQ **train** archive.
A different, unopened collection must be selected and frozen before final v2
testing.

## Hypothesis

The frozen Top-40 candidate set contains material relevance headroom, but the
RARS-v1 exact-overlap proxy is weakly aligned with relevance gain. A rank-16
sidecar trained directly on relevant-versus-boundary-negative pairs may recover
more of that headroom at the same 16-byte document payload.

## Input bundle contract

Each train or validation bundle is a directory containing:

| File | Shape | Meaning |
|---|---|---|
| `manifest.json` | JSON | Declares `split_role`, source, hashes, and `test_qrels_accessed: false` |
| `query_vectors.float32.npy` | `[Q, D]` | Development query embeddings |
| `ann_rows.int64.npy` | `[Q, 100]` | Frozen-index internal document rows |
| `ann_scores.float32.npy` | `[Q, 100]` | Frozen IVF-PQ scores |
| `candidate_relevance.uint8.npy` | `[Q, 100]` | Binary relevance within the candidate pool |
| `document_residuals.float32.npy` | `[N, D]` | Original minus frozen-index reconstructed document vectors |

The training script rejects any split role other than `train` or `validation`
and rejects manifests that reference closed-test markers.

## First feasibility run

```bash
python scripts/train_boundary_loss_sidecar.py \
  --bundle-dir /path/to/nq-train-development-bundle \
  --output-dir /path/to/rars-v2-boundary-loss/run-001 \
  --rank 16 \
  --epochs 5 \
  --device cuda
```

The first run learns untied query/document rank-16 projections plus a query
confidence gate. Document projections use fake symmetric int8 quantization
during training. The primary loss is pairwise softplus loss between candidate
relevant documents and non-relevant documents near the frozen Top-10 boundary.

## Required decomposition before a v2 claim

Run the following under the same candidate pool and query split:

1. Base M32;
2. exact Top-40 candidate rescoring;
3. PCA rank-16 FP32 and int8;
4. RARS-v1 rank-16 FP32 and int8;
5. boundary-loss rank-16 FP32 and int8.

This separates candidate headroom, subspace loss, quantization loss, and
objective mismatch.

## Go/no-go rule

Proceed to a larger v2 implementation only if the int8 boundary-loss model:

- improves validation Recall@10 over Base by at least `+0.01`;
- beats the storage-matched PCA sidecar;
- retains at least 70% of its FP32 gain; and
- improves more queries than it harms.

Otherwise stop the rank-16 learned-sidecar line or revisit the byte budget. Do
not use the closed NQ test set to rescue a failed development result.
