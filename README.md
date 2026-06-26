# AI Embedding Compression for RAG Retrieval

[![CI](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Faiss](https://img.shields.io/badge/ANN-Faiss-blue)](https://github.com/facebookresearch/faiss)
[![Docker Verified](https://img.shields.io/badge/Docker-verified-2496ED)](docs/docker_api.md)

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
- Ships a verified FastAPI retrieval service, Docker Compose deployment, unit tests, and GitHub Actions CI.

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

## Retrieval API

The repository also includes a local FastAPI retrieval service backed by the exported
FiQA `IndexIVFPQ` artifact. It exposes:

- `GET /health` for service and artifact readiness.
- `POST /search` for single-query top-k retrieval.
- `POST /batch-search` for true micro-batched retrieval: queries are embedded together
  and sent to Faiss in one matrix search call.

The local service was verified with the FiQA artifact containing 57,638 documents.
A representative single-query request for `What is a dividend stock?` returned relevant
top-ranked FiQA passages with an end-to-end local latency of about 25 ms on Apple Silicon.
This application latency includes query embedding, Faiss search, and response assembly, so
it is intentionally reported separately from the GPU-only serving benchmark above.

### Local API setup

On macOS Apple Silicon, install Faiss through conda-forge to avoid mixing native Faiss and
OpenMP runtimes from Conda and pip:

```bash
conda env create -f environment.yml
conda activate rag-api
pip install -r requirements-api.txt
```

If the environment already exists:

```bash
conda activate rag-api
conda install -c conda-forge faiss-cpu
pip install -r requirements-api.txt
```

Generate the local FiQA metadata copy (the 45 MB metadata file is intentionally ignored by Git):

```bash
python scripts/prepare_fiqa_documents.py
```

Run the service:

```bash
uvicorn app.main:app
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

Example requests:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a dividend stock?","top_k":5,"nprobe":16}'
```

```bash
curl -X POST http://127.0.0.1:8000/batch-search \
  -H "Content-Type: application/json" \
  -d '{"queries":["What is a dividend stock?","How does inflation affect bond prices?"],"top_k":3,"nprobe":16}'
```

For the full artifact contract and operational notes, see [Retrieval API](docs/retrieval_api.md).

### Local API benchmark

The repository includes a real HTTP benchmark for the running service. It measures
client-visible end-to-end latency, API-reported retrieval latency, and query throughput.

```bash
python scripts/benchmark_api.py --warmup 5 --runs 30 --batch-sizes 1 8 32
```

Verified local CPU results on Apple Silicon:

| Endpoint | Batch size | Client P50 | Client P95 | Query throughput |
|:--|--:|--:|--:|--:|
| `/search` | 1 | 6.627 ms | 7.304 ms | 149.70 q/s |
| `/batch-search` | 8 | 9.930 ms | 11.953 ms | 797.30 q/s |
| `/batch-search` | 32 | 21.006 ms | 21.837 ms | 1,519.05 q/s |

Batch size 32 reaches about **10.1×** the query throughput of the single-query
endpoint. These are local CPU application measurements, including HTTP, query
embedding, Faiss search, and response assembly; they are not directly comparable
to the GPU-only Faiss serving benchmark above.

## API Demo

The Swagger UI below shows a verified `POST /search` request against the
serialized FiQA `IndexIVFPQ` artifact. The service returns ranked passages with
similarity scores and document IDs.

![FastAPI retrieval demo](figures/api_demo.png)

### Docker deployment

The service is containerized and verified with Docker Compose:

```bash
docker compose up --build
```

The first start generates the reproducible FiQA metadata file and downloads the
embedding model. Once ready, open:

```text
http://127.0.0.1:8000/docs
```

The Docker deployment was verified with `GET /health` and `POST /search` against
the serialized 57,638-document `IndexIVFPQ` artifact. See [Docker API](docs/docker_api.md).

### Testing and CI

The repository includes **7 offline unit tests** for artifact consistency,
retriever behavior, batch search, and endpoint logic.

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions runs the test suite on pushes to `main` and pull requests. See
[Testing and CI](docs/testing_ci.md).

## Repository Structure

```text
.github/
  workflows/
    ci.yml
app/
  main.py
  retriever.py
artifacts/
  fiqa_ivfpq_m96/
    index.faiss
    service_config.json
    doc_ids.json
docker/
  entrypoint.sh
docs/
  api_benchmark.md
  benchmark_methodology.md
  docker_api.md
  retrieval_api.md
  testing_ci.md
figures/
  storage_quality_tradeoff.png
  throughput_stability.png
notebooks/
  Ai_embedding_compression.ipynb
results/
  api_benchmark/
  fiqa_gpu_benchmark/
scripts/
  benchmark_api.py
  export_service_artifacts.py
  prepare_fiqa_documents.py
tests/
  test_api.py
  test_artifact_contract.py
  test_retriever.py
Dockerfile
docker-compose.yml
environment.yml
environment-ci.yml
requirements-api.txt
requirements-dev.txt
requirements-ci.txt
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
- Future work includes a 100K–1M vector scale benchmark, Faiss `OPQMatrix` comparison, a Traditional Chinese retrieval benchmark, reranking, and production observability / deployment hardening.


## Release Readiness

The current repository represents a complete retrieval-engineering workflow:

```text
GPU benchmark → serialized IVF-PQ artifact → FastAPI serving
→ local API benchmark → Docker Compose deployment → automated CI
```

The next milestone is a `v1.0.0` release after adding a short API demo recording
or screenshot to the repository.
