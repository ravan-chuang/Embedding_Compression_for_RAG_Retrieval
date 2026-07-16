# Boundary-aware residual sidecar: development protocol

This experiment follows the BEIR NQ post-hoc diagnosis without using NQ to
revise or select the method.  BEIR NQ is explicitly rejected by the trainer.

## Fixed method budget

- Base index: unchanged IVF-PQ index.
- Candidate depth: Top-40.
- Final depth: Top-10.
- Sidecar: rank 16, one signed int8 coefficient per dimension (16 bytes/doc).
- Teacher: full-precision candidate inner products, with no relevance labels.
- Objective: weighted pairwise softplus across the exact Top-10 boundary.
- Quantization: int8 fake quantization with a straight-through estimator.

Hard pairs use exact ranks 7--10 as positives by default.  Negatives come from
exact ranks 11--40; ANN inversions are selected first, then the smallest exact
margin.  This targets the failure observed in the NQ diagnosis while keeping
the development experiment qrels-free.

## First development run

Use an existing clean train split such as MS MARCO.  The candidate and exact
score files can be produced by `train_select_rars_clean_split.py`; stop that
script after its train cache and residual memmap have been built if necessary.

```bash
python scripts/train_boundary_aware_sidecar.py \
  --development-dataset msmarco-clean-train \
  --query-vectors /content/drive/MyDrive/rars-dev/query_vectors.fp32.npy \
  --query-rows /content/drive/MyDrive/rars-dev/train_query_rows.npy \
  --ann-rows /content/drive/MyDrive/rars-dev/candidate_cache/train/ann_rows.npy \
  --ann-scores /content/drive/MyDrive/rars-dev/candidate_cache/train/ann_scores.npy \
  --exact-scores /content/drive/MyDrive/rars-dev/candidate_cache/train/exact_scores.npy \
  --residuals /content/drive/MyDrive/rars-dev/residual_ivfpq_m32.float32.memmap \
  --initial-basis /content/drive/MyDrive/rars-dev/bases/pca_rank16.float32.npy \
  --output-dir /content/drive/MyDrive/rars-boundary-v1/msmarco-seed42 \
  --n-docs 1000000 \
  --dim 384 \
  --rank 16 \
  --top-b 40 \
  --final-k 10 \
  --epochs 10 \
  --pair-batch-size 2048 \
  --seed 42
```

The output is resumable at the run level (completed outputs are retained), but
the first implementation does not checkpoint within an epoch.  A T4 run is
expected to be short once caches and the residual memmap exist; cache and
residual construction remain the expensive stage.

## Decision sequence

1. Smoke-test one seed on MS MARCO train and inspect loss convergence.
2. Encode rank-16 int8 document coefficients using the learned basis.
3. Select only on the clean validation split, comparing against the same PCA
   and original RARS baselines at Top-40.
4. If Recall@10 or exact-Top-10 overlap improves, repeat seeds 42, 43, and 44.
5. Freeze all hyperparameters before evaluating FiQA and SciFact test data.
6. Keep BEIR NQ diagnostic-only; do not use it for another selection decision.

The first go/no-go target is recovery of at least 10% of the exact Top-40
headroom at the same 16-byte budget, with consistent direction across fresh
collections.
