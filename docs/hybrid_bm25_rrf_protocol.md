# Hybrid BM25 + Compressed Dense Retrieval + RRF

## Goal
Test whether sparse BM25 retrieval can recover relevant documents missed by
compressed dense ANN retrieval, without removing the storage advantages of IVF-PQ.

## Phase 1: FiQA prototype
- Dataset: FiQA / BEIR
- Sparse retriever: BM25Okapi
- Dense retrievers:
  - Float32 FlatIP reference
  - IVF-PQ M=96, nprobe=16
  - OPQ-IVF-PQ M=96, nprobe=16
- Candidate depth per retriever: 100
- Final ranking depth: 10

## Fusion
Use Reciprocal Rank Fusion:

score(d) = w_sparse / (rrf_k + rank_sparse(d))
         + w_dense  / (rrf_k + rank_dense(d))

Initial fixed configuration:
- rrf_k = 60
- w_sparse = 1.0
- w_dense = 1.0

## Evaluation
Report:
- Recall@10
- MRR@10
- nDCG@10
- Sparse retrieval latency
- Dense ANN search latency
- Sequential hybrid latency
- Candidate overlap between sparse and dense retrieval
- Per-query wins, losses, and ties versus dense-only retrieval

## Fairness protocol
- Use a deterministic FiQA calibration split for any future RRF tuning.
- Keep a held-out FiQA subset untouched until the configuration is frozen.
- Apply the frozen configuration to SciFact without retuning.
- Do not claim hybrid improvement unless it holds on held-out FiQA and is checked on SciFact.

## Planned comparisons
1. BM25 only
2. Float32 dense only
3. IVF-PQ only
4. OPQ-IVF-PQ only
5. BM25 + IVF-PQ RRF
6. BM25 + OPQ-IVF-PQ RRF
