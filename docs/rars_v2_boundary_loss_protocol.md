# RARS-v2 Boundary-Loss Feasibility Protocol

## Status and evidence boundary

This is a new development protocol. It does not revise RARS-v1 and must not read
the closed BEIR NQ test queries, qrels, Stage-3 evaluation arrays, or post-hoc
per-query outputs. The NQ post-hoc summary motivates the hypothesis only.

Development uses the existing deterministic MS MARCO 1M clean train/validation
split, whose qrels and document IDs match the frozen corpus. BEIR NQ train qrels
cannot supervise this method: they refer to a different corpus and a coverage
audit found 117,750 positive documents absent from the frozen 2.68M confirmation
corpus. A different, unopened collection must be selected and frozen before
final v2 testing.

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
| `relevant_counts.int32.npy` | `[Q]` | Full-qrels relevant count; required for validation Recall@10 |
| `candidate_residuals.float32.npy` | `[U, D]` | Residuals for the union of development candidates only |
| `candidate_doc_rows.int64.npy` | `[U]` | Frozen-index document row for each candidate residual |
| `ann_residual_rows.int64.npy` | `[Q, 100]` | Maps every ANN position to its local candidate-residual row |

The training script rejects any split role other than `train` or `validation`
and rejects manifests that reference closed-test markers.

Build the compact bundles from the completed MS MARCO clean-split artifacts:

```bash
python scripts/build_msmarco_rars_v2_boundary_bundles.py \
  --embeddings /path/to/msmarco_basis_gate0_cache/embeddings.fp16.memmap \
  --doc-ids /path/to/msmarco_basis_gate0_cache/doc_ids.int64.memmap \
  --query-vectors /path/to/msmarco_basis_gate0_cache/query_vectors.fp32.npy \
  --index /path/to/frozen_ivfpq_m32_nlist512.index \
  --qrels /path/to/msmarco_basis_gate0_cache/qrels_subset.json \
  --output-root /content/rars-v2-boundary-work/bundles \
  ...
```

Only candidate-union residuals are materialized. This avoids another full
multi-gigabyte FP32 residual matrix on Drive. The feasibility trainer therefore
runs with `--skip-full-encoding`; full-corpus codes are exported only after the
validation go/no-go gate passes.

## First feasibility run

```bash
python scripts/train_boundary_loss_sidecar.py \
  --bundle-dir /path/to/inner_train \
  --selection-bundle-dir /path/to/inner_validation \
  --validation-bundle-dir /path/to/outer_validation \
  --output-dir /path/to/rars-v2-boundary-loss/run-001 \
  --rank 16 \
  --epochs 10 \
  --skip-full-encoding \
  --device cuda
```

The first run learns untied query/document rank-16 projections plus a query
confidence gate. Document projections use learned, fixed per-coefficient int8
scales during QAT. The same scales are written to the artifact, used to encode
the complete document residual matrix, and reused unchanged on validation; no
batch-dependent or validation recalibration is allowed. The primary loss is
pairwise softplus loss between candidate relevant documents and non-relevant
documents near the frozen Top-10 boundary.

`scripts/train_boundary_aware_sidecar.py` is retained as a secondary qrels-free
exact-score distillation ablation. It is not the registered primary method and
must be reported separately rather than silently mixed with relevance training.

The trainer writes portable NumPy artifacts for the query projection, document
projection, query gate, fixed int8 scales, and full-corpus int8 document codes.
When a validation bundle is supplied it also writes Base, FP32, and int8
Recall@10 plus improved/harmed/unchanged query counts.

## v2.1 implementation correction after run-0

Run-0 used relevant positives from the complete Top-100 candidate cache even
though inference corrected only Top-40. Its one-epoch model gained `+0.0100`
Recall@10 over Base but remained below PCA; five epochs collapsed to `-0.2922`
because correction magnitude was not bounded. That run is retained as a
negative feasibility result and is not used as paper evidence.

v2.1 makes the implementation match the registered serving budget:

- pairs may reference only documents in the correctable Top-40;
- relevant ranks 11--40 are paired against non-relevant Top-10 documents;
- relevant Top-10 documents are paired against non-relevant ranks 11--40;
- each per-document correction is bounded to `[-0.05, +0.05]`;
- the query gate starts near 0.12 (`bias=-2`), learning rate is `1e-4`, and
  correction L2 is `1e-3`;
- the original 4,980-query train split is deterministically divided into inner
  train/validation; best epoch is chosen only on inner validation;
- the original 1,000-query validation is an outer diagnostic and cannot select
  epochs or optimizer settings.

The frozen seed-42 v2.1 run selected epoch 9. On the 1,000-query outer
validation split it obtained `0.67867` Recall@10, a `+0.0020` gain over Base,
with 4 improved, 2 harmed, and 994 unchanged queries. FP32 and int8 results were
identical, but the model remained `-0.02567` below the storage-matched PCA
sidecar. The registered decision is therefore `NO_GO_OR_REVISE`.

## v2.2 intervention diagnosis

No additional seeds or outer-validation tuning are authorized after the v2.1
no-go result. Before defining another model, run the fixed v2.1 checkpoint on
the **inner validation split only** with:

```bash
python scripts/diagnose_rars_v2_intervention.py \
  --bundle-dir /path/to/inner_validation \
  --model-dir /path/to/v2.1-seed42-max10epochs \
  --output-dir /path/to/v2.1-seed42-max10epochs/inner_validation_diagnostic
```

The diagnostic performs no fitting, search, parameter sweep, or selection. It
reports Top-40 relevance-oracle headroom, gate and correction distributions,
Top-10 membership changes, and a gate-equals-one counterfactual. These values
distinguish insufficient candidate headroom from gate suppression and wrong-way
correction signals. The outer validation split remains frozen and is not read
by this analysis.

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
