# Benchmark Methodology

This document describes the evaluation methodology for the embedding-compression and GPU ANN experiments in this repository.

## 1. Objective

The benchmark answers two related but distinct questions:

1. **Compression quality:** After compressing document embeddings, how much semantic retrieval quality is retained?
2. **Serving efficiency:** When the system searches directly over compressed PQ codes, how do latency and throughput change?

These questions are evaluated separately because reconstructing compressed embeddings before search measures representation distortion, but does not measure the performance of compressed-domain approximate nearest-neighbor (ANN) search.

## 2. Dataset and Retrieval Task

| Item | Configuration |
|---|---|
| Benchmark | FiQA from the BEIR retrieval benchmark suite |
| Corpus size | 57,638 documents |
| Query count | 648 |
| Relevance labels | Official FiQA qrels |
| Retrieval setting | Query-to-document dense retrieval |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |

Every query is embedded with the same sentence-transformer model as the document corpus. Rankings are evaluated against FiQA relevance labels rather than using document-to-document nearest-neighbor agreement.

## 3. Evaluation Metrics

The project reports ranking metrics and systems metrics.

### Ranking quality

- **Recall@5 / Recall@10:** Fraction of relevant documents retrieved in the top 5 or top 10 results.
- **MRR@10:** Mean Reciprocal Rank for the first relevant document within the top 10.
- **nDCG@10:** Normalized Discounted Cumulative Gain; rewards highly ranked relevant documents and supports graded ranking quality.

### Systems behavior

- **Deployment compression ratio:** Float32 baseline storage divided by deployment storage. The accounting includes PQ codes, IVF/index metadata, codebooks, and—for external OPQ experiments—the learned rotation matrix.
- **P50 latency:** Median per-query latency.
- **P95 latency:** Tail latency, intended to expose variability beyond median performance.
- **QPS:** Queries per second.

## 4. Experimental Modes

## 4.1 Reconstructed-vector quality evaluation

This mode evaluates how accurately a compression method preserves retrieval rankings.

Pipeline:

```text
Float32 document embeddings
        ↓
compression
        ↓
compressed representation
        ↓
reconstruction to dense vectors
        ↓
GPU dense inner-product search
        ↓
FiQA ranking metrics
```

Methods evaluated in this mode include:

- Float32 baseline
- INT8 scalar quantization
- INT4 scalar quantization
- Product Quantization (PQ)
- Optimized Product Quantization (OPQ) + PQ

### Interpretation

This mode isolates the **quality cost of compression**. Search occurs over reconstructed dense vectors, so it must not be interpreted as PQ/ADC serving speed.

## 4.2 Compressed-domain GPU ANN evaluation

This mode evaluates an actual ANN serving path.

Pipeline:

```text
Float32 document embeddings
        ↓
IVF training + PQ encoding
        ↓
inverted lists containing PQ codes
        ↓
Faiss GPU IVF-PQ ADC search
        ↓
top-k document rankings
        ↓
FiQA ranking metrics + latency/QPS
```

Methods evaluated in this mode include:

- `GpuIndexFlatIP`: exact Float32 dense inner-product baseline
- `GpuIndexIVFPQ`: GPU IVF-PQ with asymmetric distance computation (ADC)
- External PyTorch-learned OPQ rotation + `GpuIndexIVFPQ`

### Why this is a genuine compressed-domain benchmark

For IVF-PQ ADC retrieval, documents are stored as PQ codes. During search, Faiss compares the query with those codes through lookup-table-based asymmetric distance computation. It does not reconstruct every document vector to Float32 before scoring.

## 5. Index Configuration

Representative GPU ANN configuration:

| Parameter | Value |
|---|---|
| IVF partition count | `nlist = 256` |
| Primary probe setting | `nprobe = 16` |
| Embedding dimension | 384 |
| PQ subquantizers tested | `M = 24, 48, 64, 96` |
| Bits per subquantizer | 8 |
| GPU layout | Interleaved Faiss GPU layout |
| GPU lookup tables | Float16 lookup tables enabled where supported |

The benchmark also explores lower-compute settings such as `M=24, nprobe=4` to show the quality/latency/storage frontier.

## 6. OPQ Treatment

The project includes an externally learned OPQ-style rotation before IVF-PQ encoding.

For OPQ runs:

1. A learned orthogonal rotation is applied to document embeddings.
2. The IVF-PQ index is trained on rotated document embeddings.
3. Queries receive the same rotation before GPU ANN search.
4. Deployment storage accounting includes the rotation matrix because it is required at query time.

This avoids overstating compression by ignoring model-side state that is necessary for production deployment.

## 7. Latency Measurement Protocol

Latency is measured separately from ranking evaluation.

- GPU warm-up queries are executed before timing.
- Timed search uses the GPU Faiss index directly.
- The benchmark records per-query latency and QPS.
- Each configuration is repeated **five times**.
- README serving tables report median statistics across repeated runs.
- Batch sizes tested: **1, 8, and 64**.

Batch size is reported explicitly because GPU behavior changes substantially between online serving and high-throughput micro-batching.

## 8. Result Interpretation Rules

The following rules are used when interpreting results:

1. **Do not compare reconstructed-vector timing against ADC timing as equivalent serving modes.**
2. **Do not label an experiment as accelerated PQ search if documents were reconstructed into dense Float32 vectors first.**
3. **Report storage accounting consistently**, including codebooks, index components, and OPQ rotation where applicable.
4. **Evaluate both ranking quality and systems efficiency.** A high compression ratio alone is not sufficient if ranking quality collapses.
5. **Evaluate batch size separately.** A method that improves batch-1 latency may not dominate at batch-64.

## 9. Main Limitations

- FiQA is a relevance benchmark with 57,638 documents; it is not a million-scale ANN benchmark.
- Results currently use one English sentence-embedding model.
- Hardware is a Colab NVIDIA Tesla T4, so absolute latency should not be generalized to all GPUs.
- IVF-PQ hyperparameters are intentionally small enough to run reproducibly in a notebook environment; production systems should tune `nlist`, `nprobe`, code size, and batching against their own traffic and corpus.

## 10. Reproducibility

1. Open `notebooks/Ai_embedding_compression.ipynb` in Google Colab.
2. Select an NVIDIA GPU runtime.
3. Run all notebook cells in order.
4. Exported benchmark artifacts are written to:

```text
fiqa_rag_results/readme_artifacts/
```

The repository stores selected exported artifacts under:

```text
results/fiqa_gpu_benchmark/
figures/
```
