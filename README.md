# AI Embedding Compression for RAG Retrieval

A GPU-accelerated benchmark for embedding compression and approximate nearest-neighbor retrieval in Retrieval-Augmented Generation (RAG) systems.

This project separates two questions that are often conflated:

1. **Compression quality:** How much retrieval quality remains after compressing document embeddings?
2. **Retrieval efficiency:** How much latency and throughput improvement is obtained when searching directly in compressed Product Quantization (PQ) code space?

## Highlights

- Evaluates Float32, INT8, INT4, PQ, OPQ, IVF-PQ, and OPQ-IVF-PQ.
- Uses the FiQA / BEIR relevance benchmark instead of document-to-document nearest-neighbor proxies.
- Measures Recall@5, Recall@10, MRR@10, nDCG@10, storage cost, latency, and QPS.
- Implements genuine GPU compressed-domain retrieval with Faiss IVF-PQ ADC; document vectors are not reconstructed to Float32 during ANN search.
- Includes repeated-run serving benchmarks for batch sizes 1, 8, and 64.

## Benchmark Setup

| Item | Configuration |
|---|---|
| Dataset | FiQA / BEIR |
| Corpus | 57,638 documents |
| Queries | 648 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Retrieval metrics | Recall@5, Recall@10, MRR@10, nDCG@10 |
| ANN backend | Faiss GPU IVF-PQ ADC |
| GPU | NVIDIA Tesla T4 |
| IVF configuration | `nlist=256`; representative ANN setting: `nprobe=16` |

## Methods

### Reconstructed-vector quality evaluation

These methods compress document vectors, reconstruct dense vectors, then evaluate retrieval quality with dense GPU search:

- Float32 baseline
- INT8 scalar quantization
- INT4 scalar quantization
- Product Quantization (PQ)
- Optimized Product Quantization (OPQ) + PQ

This mode evaluates compression distortion and ranking preservation. It does **not** claim compressed-domain search acceleration.

### Compressed-domain GPU ANN evaluation

These methods use Faiss GPU indexes directly:

- `GpuIndexFlatIP`: exact Float32 dense retrieval baseline
- `GpuIndexIVFPQ`: compressed-domain IVF-PQ ADC retrieval
- PyTorch-learned OPQ rotation + `GpuIndexIVFPQ`

For IVF-PQ, document vectors remain encoded as PQ codes during search. Faiss uses asymmetric distance computation (ADC) rather than reconstructing every document embedding.

## Main GPU ADC Quality Results

| Method | nprobe | Deployment compression | Recall@10 | nDCG@10 |
|:--|--:|--:|--:|--:|
| GPU Float32 FlatIP | – | 1.00× | 0.4413 | 0.3687 |
| OPQ-IVF-PQ M=96 | 16 | 13.59× | 0.4189 | 0.3441 |
| IVF-PQ M=96 | 16 | 14.94× | 0.4085 | 0.3442 |
| IVF-PQ M=24 | 4 | 49.83× | 0.2806 | 0.2254 |

At **13.59× deployment compression**, OPQ-IVF-PQ retained **94.9%** of Float32 Recall@10 and **93.3%** of Float32 nDCG@10.

> Latency and throughput are reported only in the repeated serving benchmark below. This avoids mixing one-off microbenchmark timing with repeated-run serving measurements.

## Repeated Serving Benchmark

Each configuration was repeated five times. The table reports median latency and median QPS.

### Batch size = 1

| Method | P50 latency | P95 latency | Median QPS |
|:--|--:|--:|--:|
| GPU Float32 FlatIP | 0.663 ms | 0.724 ms | 1,490 |
| IVF-PQ M=96, nprobe=16 | 0.240 ms | 0.313 ms | 4,030 |
| OPQ-IVF-PQ M=96, nprobe=16 | 0.236 ms | 0.283 ms | 4,115 |

### Batch size = 8

| Method | P50 latency | P95 latency | Median QPS |
|:--|--:|--:|--:|
| GPU Float32 FlatIP | 0.133 ms | 0.140 ms | 7,471 |
| IVF-PQ M=96, nprobe=16 | 0.046 ms | 0.050 ms | 21,657 |
| OPQ-IVF-PQ M=96, nprobe=16 | 0.046 ms | 0.049 ms | 21,901 |

### Batch size = 64

| Method | P50 latency | P95 latency | Median QPS |
|:--|--:|--:|--:|
| GPU Float32 FlatIP | 0.021 ms | 0.083 ms | 44,672 |
| IVF-PQ M=96, nprobe=16 | 0.021 ms | 0.035 ms | 47,884 |
| OPQ-IVF-PQ M=96, nprobe=16 | 0.020 ms | 0.036 ms | 48,635 |

## Figures

### Storage-quality trade-off

![Storage-quality trade-off](figures/storage_quality_tradeoff.png)

### Repeated-run throughput stability

![GPU Faiss throughput stability](figures/throughput_stability.png)

## Methodology

For experimental modes, storage accounting, latency protocol, and interpretation rules, see [Benchmark Methodology](docs/benchmark_methodology.md).

## Key Findings

- **High-quality compression:** PQ with a large code budget preserved near-Float32 retrieval quality under substantial deployment compression.
- **OPQ helps most at constrained code budgets:** OPQ improved ranking quality most clearly when the PQ code budget was low or medium.
- **ANN speedup requires candidate pruning:** Full-scan PQ was not automatically faster than dense GPU retrieval at this corpus scale; IVF candidate pruning created the main ANN throughput benefit.
- **Batch size matters:** IVF-PQ and OPQ-IVF-PQ show their strongest relative advantage for online and small-batch serving. Dense GPU matrix multiplication becomes more competitive at large batches.
- **Latency reporting is repeated:** Serving numbers are based on five repeated runs and use median statistics to reduce the effect of transient GPU-runtime variation.

## Repository Structure

```text
notebooks/
  Ai_embedding_compression.ipynb
results/
  fiqa_gpu_benchmark/
    main_gpu_adc_results.csv
    stability_batch_benchmark.csv
    main_results_table.md
figures/
  storage_quality_tradeoff.png
  throughput_stability.png
docs/
  benchmark_methodology.md
```

## Reproducibility

1. Open `notebooks/Ai_embedding_compression.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime.
3. Run all cells from top to bottom.
4. The notebook installs the CUDA-compatible Faiss GPU package and exports benchmark artifacts under:

```text
fiqa_rag_results/readme_artifacts/
```

For the GPU benchmark, use Google Colab with an NVIDIA GPU runtime and install `requirements-colab.txt`.

## Limitations and Next Steps

- FiQA has 57,638 documents, so it is well suited to relevance evaluation but does not fully represent million-scale ANN workloads.
- The current benchmark uses one English embedding model.
- Future work includes a 100K–1M vector scale benchmark, Faiss `OPQMatrix` comparison, a Traditional Chinese retrieval benchmark, and a Dockerized retrieval API.
