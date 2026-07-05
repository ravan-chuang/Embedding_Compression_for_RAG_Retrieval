# Selective Residual Refinement Protocol

## Objective
Test whether calibration-derived ranking-risk allocation improves
compressed retrieval more efficiently than uniform or non-ranking-aware
allocation under a fixed average storage budget.

## Base retrieval configuration
- Dataset: FiQA-2018 / BEIR
- Embedding model: BAAI/bge-small-en-v1.5
- Exact reference: GPU Float32 FlatIP
- Base index: GPU IVF-PQ ADC
- Base PQ: M=32, 8 bits/subquantizer
- IVF: nlist=256, nprobe=16
- Candidate refinement depths: Top-20 and Top-50
- Final evaluation cutoff: Top-10

## Fixed storage budget
- Base code: 32 bytes/vector
- Target extra budget: 16 average bytes/vector
- Uniform baseline: IVF-PQ M=48, 48 bytes/vector
- Selective sidecar:
  - Store FP16 residual vectors for selected documents
  - Residual payload per selected document: 384 * 2 = 768 bytes
  - Document ID metadata per selected document: 4 bytes
  - Sidecar entry cost: 772 bytes
  - Selected-document count:
    floor(N * 16 / 772)
- Storage accounting must include payload and document ID metadata.

## Allocation policies
All selective policies use the same selected-document count:
1. Random allocation
2. Reconstruction-error allocation
3. Calibration relevant-drop-risk allocation

Drop-risk score:
3 * relevant_drop_count
+ 1 * boundary_exposure_count

Only calibration-query labels may construct drop-risk scores.

## Candidate-side refinement
For a selected candidate document d:

corrected_score(q, d)
= base_pq_score(q, d) + q · residual(d)

Residual refinement is applied only when d is:
- in the compressed candidate Top-L, and
- present in the selected sidecar set.

## Evaluation protocol
- Construct risk labels and select sidecars using calibration queries only.
- Evaluate all policies on held-out FiQA queries.
- Report Recall@10, MRR@10, nDCG@10, Top-10 overlap with exact,
  recoverable-drop recovery, refinement latency, and total bytes/vector.
- Compare Top-20 and Top-50 refinement separately.
- Do not use held-out relevance labels for sidecar selection.

## Interpretation constraint
Candidate-side residual refinement cannot recover a relevant document
that is absent from the compressed Top-L candidate pool. Results must
therefore be interpreted alongside the recoverability audit.

## Residual-PQ storage accounting
Residual-PQ experiments must report both:
- residual code payload bytes;
- amortized shared PQ codebook bytes.

For a residual PQ stage with `M_r` subquantizers and 8-bit codes:
- residual code payload per encoded document: `M_r` bytes;
- codebook storage: `D * 256 * 4` bytes;
- amortized codebook cost: codebook storage / N documents.

Sparse sidecars must additionally include document-ID metadata.
Diagnostic code-size sweeps may vary code size with a fixed selected set,
but they must not be interpreted as fixed-budget comparisons.
Fixed-budget experiments must include code payload, metadata, and
amortized codebook cost.
