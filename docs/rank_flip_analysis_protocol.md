# FiQA PQ / OPQ Rank-Flip Analysis Protocol

## Objective
Identify retrieval errors caused by vector compression and create
calibration-only document-level ranking-risk labels for later residual
allocation experiments.

## Fixed retrieval configurations
- Dataset: FiQA-2018 / BEIR
- Embedding model: BAAI/bge-small-en-v1.5
- Exact reference: GPU Float32 FlatIP
- Compressed backends:
  - IVF-PQ ADC: M=96, nlist=256, nprobe=16, 8 bits/subquantizer
  - Native Faiss OPQMatrix + IVF-PQ ADC: M=96, nlist=256, nprobe=16,
    8 bits/subquantizer
- Candidate depth: Top-100
- Evaluation cutoff: Top-10

## Query partitions
Use the existing deterministic FiQA split:
- calibration: sha256(query_id) mod 5 == 0
- held-out: all remaining queries

Only calibration query relevance labels may be used to construct
document-level ranking-risk labels.

Held-out query labels are reserved for later evaluation and descriptive
reporting only. They must not affect risk scores, allocation tiers, or
hyperparameter choices.

## Required ranking exports
For exact Float32, IVF-PQ, and OPQ-IVF-PQ, export one row per
query-backend-rank with:
- query_id
- split
- backend
- rank
- document index
- document id
- retrieval score
- qrel score
- is_relevant

## Required error events
For each compressed backend, relative to exact Float32 Top-10:
- relevant_drop:
  relevant document in exact Top-10 but absent from compressed Top-10
- nonrelevant_intrusion:
  non-relevant document in compressed Top-10 but absent from exact Top-10
- relevant_recovery:
  relevant document absent from exact Top-10 but present in compressed Top-10
- shared_relevant:
  relevant document appearing in both Top-10 lists

## Boundary instability
For each query and backend, record:
- exact score at rank 10 and rank 11
- exact Top-10 boundary margin
- compressed score at rank 10 and rank 11
- compressed Top-10 boundary margin
- Top-10 overlap with exact retrieval
- count of relevant drops and non-relevant intrusions

## Calibration-only document risk labels
For each document, aggregate calibration-query events:
- relevant_drop_count
- nonrelevant_intrusion_count
- exact_top10_exposure_count
- compressed_top10_exposure_count
- boundary_exposure_count
- mean exact rank when involved in an event
- mean exact Top-10 boundary margin when involved in an event

The first risk score will be a transparent heuristic, not a learned model:

risk =
  3.0 * relevant_drop_count
+ 1.0 * nonrelevant_intrusion_count
+ 1.0 * boundary_exposure_count

This is a starting baseline. Residual allocation must later compare it
against random, uniform, and reconstruction-error-aware allocation under
the same average bytes/vector budget.

## Outputs
- rankings_long.csv
- rank_flip_events.csv
- query_boundary_summary.csv
- document_risk_calibration.csv
- rank_flip_summary.md

## Interpretation constraint
This analysis identifies compression-related ranking instability. It does
not by itself prove that residual allocation improves retrieval. That claim
requires a separate fixed-budget residual-refinement experiment evaluated
on held-out queries and SciFact.
