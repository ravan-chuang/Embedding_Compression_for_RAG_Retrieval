# AI Embedding Compression for RAG Retrieval

[![CI](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Faiss](https://img.shields.io/badge/ANN-Faiss-blue)](https://github.com/facebookresearch/faiss)
[![Docker Verified](https://img.shields.io/badge/Docker-verified-2496ED)](docs/docker_api.md)

A GPU benchmark and deployable retrieval system for embedding compression and approximate nearest-neighbor search in Retrieval-Augmented Generation (RAG).

This project separates two questions that are often conflated:

1. **Compression quality:** How much retrieval quality remains after compressing document embeddings?
2. **Retrieval efficiency:** How much latency and throughput improvement is obtained when searching directly in compressed Product Quantization (PQ) code space?

## Highlights

- Evaluates Float32, INT8, INT4, PQ, OPQ, IVF-PQ, and OPQ-IVF-PQ.
- Uses FiQA and SciFact / BEIR relevance benchmarks across MiniLM and BGE-small embedding models, plus a deterministic 1M-passage MS MARCO scale validation, instead of document-to-document nearest-neighbor proxies.
- Measures Recall@5, Recall@10, MRR@10, nDCG@10, storage cost, latency, and QPS.
- Implements genuine GPU compressed-domain retrieval with Faiss IVF-PQ ADC; document vectors are not reconstructed to Float32 during ANN search.
- Exports a deployable MiniLM OPQ-IVF-PQ artifact, including the learned query-side rotation matrix required for serving.
- Ships a verified FastAPI retrieval service, Docker Compose deployment, metadata regeneration flow, unit tests, and GitHub Actions CI.
- Includes an optional BGE cross-encoder reranking path after OPQ-IVF-PQ candidate retrieval, with experimental FiQA evaluation and true multi-query rerank batching.

## Benchmark Setup

| Item | Configuration |
|---|---|
| Primary benchmark | FiQA / BEIR |
| FiQA corpus / queries | 57,638 documents / 648 queries |
| Cross-dataset validation | SciFact / BEIR: 5,183 documents / 300 queries |
| Million-scale validation | Deterministic MS MARCO subset: 1,000,000 passages / 6,980 dev queries |
| Primary deployment model | `sentence-transformers/all-MiniLM-L6-v2` |
| Cross-model validation | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 for both evaluated models |
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
- Native Faiss `OPQMatrix` + `GpuIndexIVFPQ` comparison

For IVF-PQ, document vectors remain encoded as PQ codes during search. Faiss uses asymmetric distance computation (ADC) rather than reconstructing every document embedding.

## Main GPU ADC Quality Results

| Method | nprobe | Analytical compression | Serialized deployment compression | Recall@10 | nDCG@10 |
|:--|--:|--:|--:|--:|--:|
| GPU Float32 FlatIP | – | 1.00× | 1.00× | 0.4413 | 0.3687 |
| PyTorch OPQ-IVF-PQ M=96 | 16 | 13.59× | 12.01× | 0.4081 | 0.3462 |
| IVF-PQ M=96 | 16 | 14.94× | 13.05× | 0.4085 | 0.3442 |
| PyTorch OPQ-IVF-PQ M=96 | 64 | 13.59× | 12.01× | 0.4216 | 0.3548 |

At the deployed `M=96, nprobe=16` configuration, PyTorch OPQ-IVF-PQ retains about **92.5%** of Float32 Recall@10 and **93.9%** of Float32 nDCG@10 while reducing serialized deployment storage by about **12.01×**.

The serialized deployment figure includes the external `384 × 384` FP32 query-side OPQ rotation matrix. The analytical compression figure describes the index coding budget only.

## Cross-Dataset Validation: SciFact / BEIR

To test whether the FiQA trade-off is specific to a single financial-QA corpus, the same embedding model and ANN configuration were evaluated on SciFact / BEIR.

| Method | `nprobe` | Serialized deployment compression | Recall@10 | nDCG@10 |
|:--|--:|--:|--:|--:|
| GPU Float32 FlatIP | – | 1.00× | 0.7833 | 0.6451 |
| IVF-PQ M=96 | 16 | 6.00× | 0.7206 | 0.5975 |
| PyTorch OPQ-IVF-PQ M=96 | 16 | 4.15× | 0.7056 | 0.5906 |
| Native Faiss OPQMatrix-IVF-PQ M=96 | 16 | 4.15× | 0.7156 | 0.5969 |

The SciFact experiment uses 5,183 documents and 300 judged queries. At `M=96, nprobe=16`, the strongest OPQ baseline was the native Faiss `OPQMatrix` configuration, while plain IVF-PQ slightly outperformed the PyTorch-learned OPQ variant on Recall@10 and nDCG@10.

This is evidence of **cross-dataset evaluation**, not a claim that one OPQ implementation universally wins. The relative OPQ benefit is dataset-dependent, and SciFact is too small to support a million-scale ANN speed claim.

### Why SciFact serialized OPQ compression is lower

The lower SciFact serialized deployment compression ratio is expected and is not a PQ regression.

For OPQ serving, total deployment storage includes both the Faiss index and the fixed-size external query transform:

```text
total deployment bytes
= serialized Faiss index bytes
+ 384 × 384 FP32 OPQ rotation matrix bytes
```

The FP32 rotation matrix has the same size for MiniLM and BGE-small because both use 384-dimensional embeddings. Its storage cost is therefore nearly fixed across datasets, while the index size grows with the number of document vectors.

- **FiQA:** 57,638 documents. The fixed rotation overhead is amortized across a much larger index, so the OPQ artifact reaches about **12.01×** serialized deployment compression.
- **SciFact:** 5,183 documents. The index is much smaller, so the same rotation matrix becomes a material share of total artifact storage and lowers serialized deployment compression to about **4.15×**.

The analytical compression ratio still describes the index coding budget. The serialized ratio is intentionally stricter because it represents the actual bytes required to deploy a correct OPQ service.


## Million-Scale Validation: MS MARCO 1M

To test whether the observed compression and ANN trade-offs remain useful beyond small BEIR corpora, the project includes a deterministic **1,000,000-passage MS MARCO** benchmark using `BAAI/bge-small-en-v1.5` embeddings and a Tesla T4 GPU.

### Setup

| Item | Configuration |
|---|---|
| Corpus | 1,000,000 deterministic MS MARCO passages |
| Queries | 6,980 MS MARCO dev queries |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| IVF configuration | `nlist=4096` |
| PQ configuration | `M=96`, `nbits=8`, FP16 lookup tables |
| OPQ configuration | Native Faiss `OPQMatrix`, `niter=50`, `niter_pq=4` |
| GPU | NVIDIA Tesla T4 |

### Main Results

| Method | `nprobe` | Recall@10 | MRR@10 | nDCG@10 | P95 search latency | QPS | Serialized deployment compression |
|:--|--:|--:|--:|--:|--:|--:|--:|
| GPU FlatIP exact | – | 0.8563 | 0.6370 | 0.6870 | 0.428 ms | 2,418 | 1.00× |
| IVF-PQ M=96 | 16 | 0.7195 | 0.5361 | 0.5770 | 0.021 ms | 53,644 | 13.01× |
| IVF-PQ M=96 | 32 | 0.7596 | 0.5619 | 0.6064 | 0.043 ms | 25,557 | 13.01× |
| IVF-PQ M=96 | 64 | 0.7887 | 0.5834 | 0.6297 | 0.078 ms | 13,607 | 13.01× |
| Native Faiss OPQMatrix + IVF-PQ M=96 | 64 | 0.7895 | 0.5842 | 0.6304 | 0.076 ms | 13,815 | 12.94× |

### Deployment-Aware Interpretation

At `M=96, nprobe=64`, plain IVF-PQ retains **92.1%** of exact Recall@10 while reducing serialized deployment storage by **13.01×**.

Native Faiss OPQ produces only a marginal gain at this high-rate configuration:

- Recall@10: `0.7887 → 0.7895` (`+0.0008`)
- nDCG@10: `0.6297 → 0.6304` (`+0.0008`)
- Build time: `16.0 s → 2,094.4 s` (`~130.6×` longer)

This suggests that when 384-dimensional embeddings are split into 96 PQ subvectors of only 4 dimensions each, plain IVF-PQ is already highly expressive and OPQ has limited remaining quantization error to remove.

The practical default at this scale is therefore plain IVF-PQ at `M=96, nprobe=32`, while `nprobe=64` is the higher-recall serving mode. Native OPQ remains an important baseline, especially for lower-rate PQ settings where rotation may provide more value.

> Timing measures GPU Faiss search only. It excludes embedding generation, HTTP transport, artifact loading, and response serialization.

## Low-Rate PQ / OPQ Full Sweep: MS MARCO 1M

A deployment-aware full sweep evaluates plain GPU IVF-PQ against native Faiss
`OPQMatrix` + GPU IVF-PQ on the deterministic 1,000,000-passage MS MARCO subset.

| Setting | Value |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| IVF `nlist` | 4096 |
| PQ `nbits` | 8 |
| `M` values | 24, 32, 48, 64, 96 |
| `nprobe` values | 4, 16, 32, 64 |
| Total benchmark points | 40 |

The merged result set combines the `M=24/48` run and the `M=32/64/96`
continuation. The available recorded metadata was checked for consistency before
merging.

### OPQ incremental value at `nprobe=64`

| M | Plain Recall@10 | OPQ Recall@10 | Δ Recall@10 | Plain build | OPQ build | Build multiplier |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.6494 | 0.6880 | **+0.0386** | 9.9 s | 550.9 s | 55.8× |
| 32 | 0.6957 | 0.7226 | **+0.0269** | 10.3 s | 863.0 s | 83.6× |
| 48 | 0.7489 | 0.7609 | **+0.0120** | 12.6 s | 1162.7 s | 92.6× |
| 64 | 0.7729 | 0.7752 | **+0.0023** | 12.9 s | 1575.3 s | 122.2× |
| 96 | 0.7887 | 0.7895 | **+0.0008** | 17.8 s | 2168.7 s | 122.1× |

The sweep shows a clear rate-dependent transition: native OPQ provides its
largest retrieval-quality recovery at lower code rates (`M=24–32`), the gain
contracts at `M=48`, and it becomes practically negligible by `M=64–96`.
The OPQ transform adds little serialized storage at this corpus size, but its
offline training and index-build cost rises sharply.

This is a configuration-specific result for this corpus, model, IVF setting,
GPU environment, and evaluation workload. It is not a universal recommendation
to use or avoid OPQ.

Detailed outputs:
[full summary CSV](results/msmarco_low_rate_pareto/1m_full_m24_m96/msmarco_1m_low_rate_pq_opq_full_summary.csv),
[artifact accounting](results/msmarco_low_rate_pareto/1m_full_m24_m96/msmarco_1m_low_rate_pq_opq_full_artifacts.csv),
[merged metadata](results/msmarco_low_rate_pareto/1m_full_m24_m96/msmarco_1m_low_rate_pq_opq_full_metadata.json),
and [full report](results/msmarco_low_rate_pareto/1m_full_m24_m96/msmarco_1m_low_rate_pq_opq_full_report.md).

### Low-rate quality-storage Pareto frontier

![MS MARCO 1M full-sweep quality-storage Pareto](results/msmarco_low_rate_pareto/1m_full_m24_m96/quality_storage_pareto.png)

### Low-rate quality-throughput Pareto frontier

![MS MARCO 1M full-sweep quality-throughput Pareto](results/msmarco_low_rate_pareto/1m_full_m24_m96/quality_qps_pareto.png)


## Cross-Model Validation: MiniLM × BGE-small

The same `M=96`, `nlist=256`, `nprobe=16` protocol was also evaluated with
`BAAI/bge-small-en-v1.5`. The table below reports the quality trade-off across
two datasets and two embedding models. Compression ratios are serialized deployment
ratios, including the external FP32 OPQ rotation when required.

| Dataset | Model | Method | Recall@10 | nDCG@10 | Serialized compression |
|:--|:--|:--|--:|--:|--:|
| FiQA | MiniLM | Float32 FlatIP | 0.4413 | 0.3687 | 1.00× |
| FiQA | MiniLM | IVF-PQ M=96 | 0.4085 | 0.3442 | 13.05× |
| FiQA | MiniLM | PyTorch OPQ-IVF-PQ M=96 | 0.4081 | 0.3462 | 12.01× |
| FiQA | MiniLM | Native Faiss OPQMatrix-IVF-PQ M=96 | 0.4157 | 0.3470 | 12.01× |
| SciFact | MiniLM | Float32 FlatIP | 0.7833 | 0.6451 | 1.00× |
| SciFact | MiniLM | IVF-PQ M=96 | 0.7206 | 0.5975 | 6.00× |
| SciFact | MiniLM | PyTorch OPQ-IVF-PQ M=96 | 0.7056 | 0.5906 | 4.15× |
| SciFact | MiniLM | Native Faiss OPQMatrix-IVF-PQ M=96 | 0.7156 | 0.5969 | 4.15× |
| FiQA | BGE-small | Float32 FlatIP | 0.4396 | 0.3848 | 1.00× |
| FiQA | BGE-small | IVF-PQ M=96 | 0.3966 | 0.3453 | 13.05× |
| FiQA | BGE-small | PyTorch OPQ-IVF-PQ M=96 | 0.4048 | 0.3545 | 12.01× |
| FiQA | BGE-small | Native Faiss OPQMatrix-IVF-PQ M=96 | 0.4067 | 0.3555 | 12.01× |
| SciFact | BGE-small | Float32 FlatIP | 0.8452 | 0.7200 | 1.00× |
| SciFact | BGE-small | IVF-PQ M=96 | 0.7991 | 0.6861 | 6.00× |
| SciFact | BGE-small | PyTorch OPQ-IVF-PQ M=96 | 0.8012 | 0.6863 | 4.15× |
| SciFact | BGE-small | Native Faiss OPQMatrix-IVF-PQ M=96 | 0.8108 | 0.6881 | 4.15× |

### Interpretation

- **PyTorch OPQ helps on both BGE-small experiments:** it improves over plain IVF-PQ on FiQA (`+0.0082` Recall@10, `+0.0092` nDCG@10) and SciFact (`+0.0021` Recall@10, `+0.0002` nDCG@10).
- **MiniLM behavior is dataset-dependent:** PyTorch OPQ improves FiQA nDCG@10 slightly but underperforms plain IVF-PQ on SciFact.
- **Native Faiss `OPQMatrix` is the strongest and most stable OPQ baseline** among the evaluated configurations. The custom PyTorch implementation is retained as a reproducible learned-rotation and serving pipeline, not claimed as a universal replacement for Faiss OPQ.
- **BGE-small is stronger on SciFact:** its Float32 baseline reaches `0.8452` Recall@10 and `0.7200` nDCG@10, compared with MiniLM's `0.7833` and `0.6451`.

## Corrected GPU Faiss Search Timing

The following values measure **GPU Faiss search only**. Quality uses all 648 FiQA queries, while latency and QPS use the 640 full-size queries from 10 batches of 64; the final 8-query tail is excluded from latency percentiles to avoid distortion.

| Method | Batch size | P95 per-query search latency | Search throughput |
|:--|--:|--:|--:|
| GPU Float32 FlatIP | 64 | 0.028350 ms | 35,553.88 q/s |
| IVF-PQ M=96, nprobe=16 | 64 | 0.018839 ms | 53,983.10 q/s |
| PyTorch OPQ-IVF-PQ M=96, nprobe=16 | 64 | 0.019044 ms | 54,260.55 q/s |
| PyTorch OPQ-IVF-PQ M=96, nprobe=64 | 64 | 0.065608 ms | 15,401.17 q/s |

These numbers exclude embedding generation, HTTP transport, artifact loading, and response serialization. They must not be compared directly with the Docker API latency below.

## Figures

### Storage-quality trade-off

![Storage-quality trade-off](figures/storage_quality_tradeoff.png)

### Repeated-run throughput stability

![GPU Faiss throughput stability](figures/throughput_stability.png)

## Methodology

For experimental modes, storage accounting, latency protocol, and interpretation rules, see [Benchmark Methodology](docs/benchmark_methodology.md).

## Key Findings

- **Million-scale MS MARCO validation:** on 1M BGE-small passages, plain IVF-PQ at `M=96, nprobe=64` retains 92.1% of exact Recall@10 with 13.01× serialized deployment compression; native OPQ adds only marginal quality at substantially higher offline build cost.
- **Deployment-aware compression accounting matters:** the external OPQ query rotation is required at serving time. Its fixed storage overhead is negligible for larger corpora such as FiQA but material for smaller corpora such as SciFact, so serialized deployment compression must be interpreted separately from index-only compression.
- **The benchmark now covers two datasets × two embedding models:** FiQA and SciFact are evaluated with MiniLM and BGE-small under the same IVF-PQ / OPQ protocol.
- **PyTorch OPQ is cross-model but not universally dominant:** it improves both BGE-small experiments, while its MiniLM behavior is dataset-dependent.
- **Native Faiss `OPQMatrix` remains the strongest OPQ baseline:** it is the most stable quality performer across all evaluated dataset-model pairs.
- **Higher `nprobe` improves quality at a throughput cost:** `nprobe=64` recovers more retrieval quality but increases search latency.
- **ANN speedup requires candidate pruning:** full-scan PQ is not automatically faster than dense GPU retrieval at this corpus scale; IVF candidate pruning creates the main ANN throughput benefit.
- **Small corpora are not a scale benchmark:** SciFact validates ranking behavior in another domain, but its 5,183-document corpus is not evidence of million-scale ANN performance.
- **Batching matters:** benchmark timing is measured over true matrix search calls, not repeated single-query calls.

## Retrieval API

The repository includes a local FastAPI retrieval service backed by the exported FiQA **OPQ-IVF-PQ** artifact.

The deployed artifact contains:

```text
artifacts/fiqa_opq_ivfpq_m96/
├── index.faiss
├── query_opq_rotation.npy
├── doc_ids.json
└── service_config.json
```

`documents.jsonl` is intentionally ignored by Git because it is reproducible metadata derived from FiQA / BEIR. The Docker entrypoint recreates it automatically when missing.

The serving path is:

```text
query
→ sentence-transformers embedding
→ L2 normalization
→ query @ OPQ rotation
→ Faiss IndexIVFPQ search
→ document metadata lookup
```

The API exposes:

- `GET /health` for service and artifact readiness, including optional reranker readiness metadata.
- `POST /search` for single-query top-k retrieval.
- `POST /batch-search` for true micro-batched retrieval: queries are embedded together and sent to Faiss in one matrix search call. When reranking is enabled, all query-candidate pairs are also scored through one cross-encoder prediction call.

The OPQ contract is validated by the service configuration. When `query_transform.enabled` is true, the retriever loads `query_opq_rotation.npy` and applies it before Faiss search.

### Experimental two-stage reranking

The service also supports an **optional** second-stage BGE cross-encoder reranker:

```text
query
→ MiniLM embedding + OPQ rotation
→ Faiss OPQ-IVF-PQ candidate retrieval
→ optional BGE CrossEncoder reranking
→ final top-k documents
```

The request fields are:

- `rerank`: enable second-stage reranking for the request.
- `candidate_k`: number of ANN candidates retrieved before reranking; it must be at least `top_k`.

The bundled FiQA artifact keeps `reranker.enabled: false` by default. This preserves the verified low-latency ANN serving path and avoids loading the cross-encoder model at startup. To run reranking experiments locally, temporarily enable the reranker block in `artifacts/fiqa_opq_ivfpq_m96/service_config.json`, restart the service, and send `"rerank": true`.

Example experimental request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What is a dividend stock?",
    "top_k":5,
    "candidate_k":10,
    "nprobe":16,
    "rerank":true
  }'
```

Reranked results expose both the original ANN score and the cross-encoder score:

```json
{
  "ann_score": 0.7153,
  "rerank_score": 0.9990
}
```

#### FiQA reranker evaluation

A reproducible 100-query FiQA evaluation is recorded under `results/rerank_fiqa_benchmark/`. The tested `BAAI/bge-reranker-base` CPU configuration did **not** improve the deployed MiniLM OPQ-IVF-PQ baseline on that subset, so reranking remains experimental and disabled by default.

| Pipeline | Recall@10 | MRR@10 | nDCG@10 | P95 local service latency |
|:--|--:|--:|--:|--:|
| OPQ-IVF-PQ only | 0.4782 | 0.4888 | 0.4216 | 6.94 ms |
| OPQ-IVF-PQ + rerank, `candidate_k=10` | 0.4782 | 0.4404 | 0.3939 | 651.84 ms |
| OPQ-IVF-PQ + rerank, `candidate_k=20` | 0.4639 | 0.4390 | 0.3840 | 1509.23 ms |

This result is intentionally retained as a negative result rather than presented as a default-quality claim. On the evaluated subset, reranking degraded MRR@10 and nDCG@10; at `candidate_k=20`, it also pushed some relevant candidates below the final top-10 cutoff. Timing includes local query embedding, OPQ rotation, ANN retrieval, CPU reranking, and result formatting; it excludes HTTP transport, container startup, and model-download time.

The repository also supports true multi-query rerank batching through `rerank_many()`. It combines all `(query, candidate document)` pairs from `/batch-search` into one cross-encoder prediction call. This is an architectural batching capability, not a blanket latency-speedup claim: a small four-query local CPU test did not improve throughput because of short batches and document-length padding.

To reproduce the evaluation, temporarily enable the reranker in the artifact configuration and run:

```bash
python scripts/benchmark_reranker.py \
  --max-queries 100 \
  --candidate-ks 10 20 \
  --top-k 10 \
  --nprobe 16 \
  --output-dir results/rerank_fiqa_benchmark_local
```

### Local API setup

On macOS Apple Silicon, install Faiss through conda-forge to avoid mixing native Faiss and OpenMP runtimes from Conda and pip:

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

Generate the local FiQA metadata copy:

```bash
ARTIFACT_DIR=artifacts/fiqa_opq_ivfpq_m96 \
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

For experimental batched reranking, temporarily enable the reranker in `service_config.json`, then add `"rerank":true` and a `candidate_k` value to the request body.

A verified Docker `POST /search` request returned `query_transform_enabled: true` and relevant dividend-related FiQA passages, confirming that the deployed API uses the OPQ query transform.

For the full artifact contract and operational notes, see [Retrieval API](docs/retrieval_api.md).

### Local API benchmark

The repository includes a real HTTP benchmark for the running service. It measures client-visible end-to-end latency, API-reported retrieval latency, and query throughput.

```bash
python scripts/benchmark_api.py --warmup 5 --runs 30 --batch-sizes 1 8 32
```

Run this benchmark after changing the artifact, model, hardware, or serving configuration. Its results are CPU application measurements, including HTTP, query embedding, OPQ rotation, Faiss search, and response assembly; they are not directly comparable to the GPU-only Faiss search benchmark above.

## API Demo

The Swagger UI below shows a verified `POST /search` request against the serialized FiQA `IndexIVFPQ` artifact.

![FastAPI retrieval demo](figures/api_demo.png)

### Docker deployment

The service is containerized and verified with Docker Compose:

```bash
docker compose up --build
```

The first start generates the reproducible FiQA metadata file and downloads the embedding model. Once ready, open:

```text
http://127.0.0.1:8000/docs
```

Verify the deployed artifact:

```bash
curl http://127.0.0.1:8000/health
```

Expected fields include:

```json
{
  "artifact_dir": "artifacts/fiqa_opq_ivfpq_m96",
  "index_type": "IndexIVFPQ",
  "document_count": 57638
}
```

Then verify the OPQ query transform:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a dividend stock?","top_k":3}'
```

The response should contain:

```json
"query_transform_enabled": true
```

See [Docker API](docs/docker_api.md).

### Testing and CI

The repository includes **10 offline unit tests** for artifact consistency, retriever behavior, endpoint logic, single-query reranking, multi-query reranking, and batch rerank integration.

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions runs the test suite on pushes to `main` and pull requests. See [Testing and CI](docs/testing_ci.md).

## Repository Structure

```text
.github/
  workflows/
    ci.yml
app/
  main.py
  reranker.py
  retriever.py
artifacts/
  fiqa_opq_ivfpq_m96/
    index.faiss
    query_opq_rotation.npy
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
  SciFact_OPQ_IVFPQ_Benchmark.ipynb
  FiQA_BGE_Small_OPQ_IVFPQ_Benchmark.ipynb
  SciFact_BGE_Small_OPQ_IVFPQ_Benchmark.ipynb
  MSMARCO_1M_OPQ_Metrics_Sweep_v1_1.ipynb
results/
  api_benchmark/
  fiqa_gpu_benchmark/
  scifact_gpu_benchmark/
  fiqa_bge_small_gpu_benchmark/
  scifact_bge_small_gpu_benchmark/
  msmarco_scale_results/
  rerank_fiqa_benchmark/
scripts/
  benchmark_api.py
  benchmark_reranker.py
  export_service_artifacts.py
  prepare_fiqa_documents.py
tests/
  test_api.py
  test_artifact_contract.py
  test_reranker.py
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

### FiQA serving and primary MiniLM benchmark

1. Open `notebooks/Ai_embedding_compression.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime.
3. Run all cells from top to bottom.
4. The notebook exports FiQA GPU benchmark artifacts and the deployed MiniLM OPQ-IVF-PQ service artifact.

### MiniLM cross-dataset validation

1. Open `notebooks/SciFact_OPQ_IVFPQ_Benchmark.ipynb`.
2. Enable an NVIDIA GPU runtime and run all cells.
3. The notebook writes independent outputs under `scifact_rag_results/`; it does not overwrite the deployed FiQA API artifact.

### BGE-small cross-model validation

1. Open `notebooks/FiQA_BGE_Small_OPQ_IVFPQ_Benchmark.ipynb` and run all cells.
2. Open `notebooks/SciFact_BGE_Small_OPQ_IVFPQ_Benchmark.ipynb` and run all cells.
3. These notebooks write independent outputs under `fiqa_bge_small_rag_results/` and `scifact_bge_small_rag_results/`.
4. They are benchmark-only and do not overwrite the deployed MiniLM FiQA artifact.


### Million-scale MS MARCO validation

1. Open `notebooks/MSMARCO_1M_OPQ_Metrics_Sweep_v1_1.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime.
3. Run all cells from top to bottom.
4. The notebook downloads MS MARCO source files, creates a deterministic 1M-passage subset, writes FP16 embedding memmaps, and exports benchmark summaries under `msmarco_scale_results/`.
5. Large source data and serialized indexes are intentionally excluded from Git history; commit reproducible code, summaries, metadata, and figures only.

For all GPU experiments, use Google Colab with an NVIDIA GPU runtime and install `requirements-colab.txt`.

## Limitations and Next Steps

- FiQA and SciFact provide cross-dataset ranking validation, while the deterministic MS MARCO 1M experiment provides a single-GPU million-scale retrieval benchmark. It does not yet establish multi-node, billion-vector, or online-production behavior.
- The benchmark currently uses two English embedding models; it does not yet validate multilingual or Traditional Chinese retrieval.
- The deployment uses a learned external OPQ transform; any compatible serving implementation must apply the same query rotation before Faiss search.
- The current BGE CPU reranker configuration is experimental: it did not improve the recorded 100-query FiQA subset and adds substantial local latency. Future reranking work should compare domain-appropriate models, title-aware / truncated document formatting, and throughput under realistic batch loads before making a production-default claim.
- Future work includes hybrid sparse-dense retrieval, a Traditional Chinese retrieval benchmark, rank-aware residual allocation, query-aware retrieval routing, model-specific deployment selection, and production observability / deployment hardening.

## Release Readiness

The repository now represents a retrieval-engineering workflow with initial cross-dataset and cross-model validation:

```text
FiQA GPU benchmark → serialized MiniLM OPQ-IVF-PQ artifact + query rotation
→ FastAPI serving → Docker metadata regeneration
→ Docker end-to-end verification → automated CI
→ FiQA + SciFact × MiniLM + BGE-small validation
→ MS MARCO 1M GPU IVF-PQ / native OPQ scale validation
→ optional BGE reranking experiment + reproducible negative-result evaluation
→ true multi-query cross-encoder batching for `/batch-search`
```

Release `v1.4.0` captures the million-scale validation milestone while retaining the verified MiniLM FiQA artifact as the deployed service baseline. The optional reranker is intentionally disabled in that default artifact because the recorded FiQA evaluation did not justify its latency cost. The next technical milestone is hybrid sparse-dense retrieval and an original compression-method comparison.
