# AI Embedding Compression for RAG Retrieval

[![CI](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Faiss](https://img.shields.io/badge/ANN-Faiss-blue.svg)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](docker-compose.yml)

A research and serving repository for **embedding compression**, **Faiss IVF-PQ retrieval**, and **post-hoc residual correction** in Retrieval-Augmented Generation (RAG).

The project studies two related but distinct questions:

1. **Compression quality:** how much retrieval quality remains after document embeddings are compressed?
2. **Retrieval efficiency:** how much latency, throughput, and storage improvement is obtained when searching directly in compressed code space?

Its main research contribution is **Retrieval-Aware Residual Subspace (RARS)**, a compact residual sidecar that improves a **frozen low-rate IVF-PQ index** without retraining the index or rewriting its existing PQ codes.

---

## Highlights

- Evaluates Float32, INT8, INT4, PQ, OPQ, IVF-PQ, and OPQ-IVF-PQ.
- Uses relevance-based evaluation on FiQA, SciFact, and a deterministic MS MARCO 1M benchmark.
- Reports Recall@5, Recall@10, Success@10, MRR@10, nDCG@10, storage, latency, and QPS.
- Implements genuine Faiss compressed-domain IVF-PQ search with asymmetric distance computation.
- Includes fixed-budget residual refinement, oracle candidate-rescoring ceilings, and paired-bootstrap tests.
- Adds a frozen-index residual sidecar that can be deployed independently of the original Faiss index.
- Adds RARS, a score-error-weighted residual basis designed for retrieval-score correction rather than pure reconstruction.
- Provides vectorized live-Faiss benchmarks, a FastAPI serving path, Docker deployment, tests, and CI.
- Generates paper-ready CSV and LaTeX tables directly from committed result artifacts.

---

## Main Result: Frozen-Index RARS Retrofit

### Problem

A deployed ANN index can be expensive or operationally risky to rebuild. RARS asks:

> Can a frozen low-rate IVF-PQ index recover part of its ranking loss by attaching a compact correction sidecar, without modifying the original index?

For a document embedding \(x_i\), frozen IVF-PQ reconstruction \(\hat{x}_i\), and query \(q\):

\[
r_i = x_i - \hat{x}_i
\]

RARS approximates the residual in a shared low-rank subspace:

\[
r_i \approx U z_i
\]

where \(U \in \mathbb{R}^{d \times r}\) is a shared basis and \(z_i\) is a quantized per-document coefficient vector.

The corrected score is:

\[
\tilde{s}_i
=
s_i^{PQ}
+
\alpha (U^\top q)^\top z_i
\]

Only the highest-ranked ANN candidates are corrected.

### MS MARCO 1M held-out result

| Method | Frozen index | Rewrite PQ codes | Extra representation | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|:---:|:---:|---:|---:|---:|---:|---:|
| IVF-PQ M32 | Yes | No | 0 B/doc | 0.66275 | 0.674 | 0.46588 | 0.50991 |
| PCA sidecar Top40 | Yes | No | 16 B/doc | 0.69142 | 0.701 | 0.47919 | 0.52750 |
| **RARS Top20** | **Yes** | **No** | **16.025 B/doc** | **0.69892** | **0.709** | **0.48455** | **0.53236** |
| RARS Top40 | Yes | No | 16.025 B/doc | 0.69992 | 0.710 | 0.48450 | 0.53251 |
| IVF-PQ M48 | No | Yes | +16 B/doc vs M32 | 0.73108 | 0.741 | — | — |
| Exact rescoring oracle within Top100 | Yes | No | — | 0.77917 | 0.787 | — | — |

RARS Top20 captures nearly all of the Top40 retrieval gain while correcting half as many candidates.

### Interpretation

RARS is a **frozen-index retrofit**, not a replacement for a higher-rate index when full re-encoding is acceptable.

At the same total representation budget, IVF-PQ M48 achieves higher absolute quality than M32 plus a 16-byte sidecar. RARS is valuable when the original index must remain unchanged.

---

## Live Faiss Serving Cost

The live benchmark measures:

1. `index.search()`
2. vectorized RARS correction
3. estimated combined search plus correction
4. alternating paired end-to-end timing

It excludes query encoding, HTTP transport, JSON serialization, artifact loading, and document lookup.

### Vectorized correction

| Threads | Method | Search | Correction | Estimated combined | Correction / Faiss | Paired E2E overhead |
|---:|---|---:|---:|---:|---:|---:|
| 1 | RARS Top20 | 270.445 µs/query | 0.613 µs/query | 271.059 µs/query | 0.23% | 0.49% |
| 1 | RARS Top40 | 270.445 µs/query | 1.058 µs/query | 271.503 µs/query | 0.39% | 0.60% |
| 14 | **RARS Top20** | **30.015 µs/query** | **0.816 µs/query** | **30.831 µs/query** | **2.72%** | **5.41%** |
| 14 | RARS Top40 | 30.015 µs/query | 1.325 µs/query | 31.340 µs/query | 4.41% | 5.51% |

On the tested CPU, vectorization reduces correction time by approximately:

- **8.9×** for Top20
- **5.9×** for Top40

Paired end-to-end timing is reported separately because multithreaded Faiss timing is sensitive to scheduling noise.

---

## Sidecar Storage

The deployed rank-16 int8 artifact contains:

```text
artifacts/msmarco_rars_sidecar_m32_rank16/
├── basis.npy
├── scales.npy
├── codes.int8.npy
├── doc_ids.npy
├── sidecar_config.json
└── manifest.json
```

| Component | Total bytes | Bytes/document |
|---|---:|---:|
| Int8 coefficient codes | 16,000,128 | 16.000128 |
| Shared basis and scales included | 16,025,024 | 16.025024 |
| External document IDs | 8,000,128 | 8.000128 |
| Complete deployable artifact | 24,027,749 | 24.027749 |

Use the following wording when reporting storage:

> The rank-16 int8 RARS representation requires 16.03 bytes per document, while the complete deployable artifact including external document-ID metadata requires 24.03 bytes per document.

---

## Cross-Setting Results

RARS is strongest on the target MS MARCO setting and shows mixed transfer behavior.

### FiQA with BGE-small

| Method | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Base M32 | 0.29351 | 0.29636 | 0.23731 |
| PCA Top40 | **0.32816** | 0.31808 | **0.26004** |
| RARS Top20 | 0.31844 | 0.31804 | 0.25594 |
| RARS Top40 | 0.32345 | **0.32202** | 0.25873 |

### FiQA with MiniLM

Fixed transfer with \(\alpha=1\) does not improve the base consistently. Proxy-selected \(\alpha=0.75\) provides a small Recall@10 improvement, but MRR and nDCG remain close to or below the base.

This repository therefore does **not** claim universal superiority across datasets or embedding models.

---

## Statistical Interpretation

On the 1,000-query MS MARCO held-out split, RARS Top40 versus PCA Top40 produced:

| Metric | Mean difference | 95% bootstrap CI |
|---|---:|---:|
| Recall@10 | +0.00850 | [-0.00200, 0.01950] |
| Success@10 | +0.00900 | [-0.00200, 0.02000] |
| MRR@10 | +0.00530 | [-0.00356, 0.01416] |
| nDCG@10 | +0.00501 | [-0.00232, 0.01238] |

The point estimates favor RARS, but the confidence intervals overlap zero. The supported claim is therefore:

> RARS provides the strongest point estimate on the target MS MARCO setting, but statistical superiority over the matched PCA sidecar is not established.

---

## Benchmark Coverage

| Benchmark | Corpus | Queries | Embedding model | Purpose |
|---|---:|---:|---|---|
| FiQA / BEIR | 57,638 | 648 | MiniLM, BGE-small | relevance quality and transfer |
| SciFact / BEIR | 5,183 | 300 | MiniLM, BGE-small | cross-dataset validation |
| MS MARCO 1M | 1,000,000 | 6,980 total; 1,000 frozen RARS eval | BGE-small | million-scale ANN and retrofit study |

Representative configurations:

- FiQA / SciFact: `nlist=256`, representative `nprobe=16`
- MS MARCO high-rate sweep: `nlist=4096`, `M ∈ {24, 32, 48, 64, 96}`
- Frozen-index RARS study: `nlist=512`, `M=32`, `nprobe=16`, candidate Top100

---

## Methods

### Reconstructed-vector quality evaluation

- Float32
- INT8 scalar quantization
- INT4 scalar quantization
- Product Quantization
- OPQ + PQ

These experiments evaluate representation distortion and retrieval quality. They do not imply compressed-domain search acceleration.

### Compressed-domain ANN evaluation

- Faiss FlatIP
- Faiss IVF-PQ
- PyTorch-learned OPQ + IVF-PQ
- Native Faiss `OPQMatrix` + IVF-PQ

### Residual refinement

- Fixed-budget Residual-PQ
- Compact sidecar variants
- Exact Top100 candidate-rescoring oracle
- PCA residual sidecar
- Retrieval-Aware Residual Subspace
- Fixed Top-B and query-adaptive routing diagnostics

---

## FastAPI Serving

The API supports standard retrieval and optional sidecar correction.

### Endpoints

- `GET /health`
- `POST /search`
- `POST /batch-search`

Example request:

```json
{
  "query_embedding": [0.01, -0.02, 0.03],
  "top_k": 10,
  "sidecar": true,
  "sidecar_top_b": 20
}
```

The retriever validates sidecar dimensions and document counts before enabling correction.

### Run locally

```bash
conda env create -f environment.yml
conda activate rag-api

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
docker compose up --build
```

---

## Reproducing the Paper Tables

Generate all paper-ready CSV and LaTeX tables:

```bash
python scripts/build_rars_paper_tables.py
```

Outputs:

```text
results/paper_tables/
├── paper_main_table.csv
├── paper_main_table.tex
├── paper_rars_cross_setting_table.csv
├── paper_rars_cross_setting_table.tex
├── paper_pca_transfer_table.csv
├── paper_pca_transfer_table.tex
├── paper_system_table.csv
├── paper_system_table.tex
├── paper_ablation_table.csv
├── paper_ablation_table.tex
├── paper_significance_table.csv
├── paper_significance_table.tex
├── paper_storage_table.csv
├── paper_storage_table.tex
└── README.md
```

---

## Running the Live Faiss Benchmark

```bash
python scripts/benchmark_rars_live_faiss.py \
  --threads 1 14 \
  --implementations loop vectorized \
  --top-b 0 20 40 \
  --nprobe 16 \
  --candidate-k 100 \
  --warmup-runs 3 \
  --timed-runs 20
```

Generated results:

```text
results/retrieval_aware_residual_basis/live_faiss_benchmark/
├── README.md
├── benchmark_summary.json
└── live_faiss_benchmark.csv
```

---

## Tests

```bash
python -m pytest -q
```

Current test suite:

```text
19 passed
```

The tests cover:

- sidecar artifact loading
- dimension and document-count validation
- score correction
- API request contracts
- optional `/search` and `/batch-search` sidecar behavior
- fake retriever integration paths

---

## Repository Structure

```text
.
├── app/                         # FastAPI retrieval service
├── artifacts/                   # deployable indexes and sidecars
├── docker/                      # deployment configuration
├── docs/                        # methodology and supporting documentation
├── figures/                     # generated figures
├── notebooks/                   # experiment notebooks
├── results/                     # committed benchmark outputs
│   ├── paper_tables/
│   └── retrieval_aware_residual_basis/
├── scripts/                     # benchmarks, artifact builders, table generation
├── tests/                       # unit and API tests
├── Dockerfile
├── docker-compose.yml
├── environment.yml
└── README.md
```

---

## Research Claims and Boundaries

Supported:

- A compact residual sidecar can improve a frozen low-rate IVF-PQ index.
- RARS improves the target MS MARCO point estimate over a matched PCA sidecar.
- Top20 captures almost all Top40 quality gain at lower correction cost.
- Vectorized correction is inexpensive relative to live Faiss search.
- The sidecar can be stored and served independently of the original index.

Not supported:

- RARS universally outperforms PCA across datasets and embedding models.
- RARS outperforms higher-rate PQ when full index rebuilding is allowed.
- The reported microbenchmarks represent complete application latency.
- The current paired-bootstrap results establish statistical superiority over PCA.
- The exact Top100 oracle is equivalent to full-corpus exact retrieval.

---

## Limitations

- The strongest RARS result is concentrated on one large-scale target setting.
- Cross-setting behavior is model- and dataset-dependent.
- RARS versus PCA confidence intervals overlap zero.
- A higher-rate IVF-PQ index remains stronger when re-encoding is acceptable.
- The live benchmark excludes query encoding, HTTP, serialization, and document lookup.
- The current implementation is vectorized but not a fused native Faiss/CUDA kernel.
- The full deployable sidecar requires additional document-ID metadata.
- Additional datasets, independent splits, and hardware environments would strengthen generality.

---

## Citation

This repository is under active research development. A formal paper citation will be added after publication.

```bibtex
@misc{chuang2026rars,
  title        = {Retrofitting Frozen IVF-PQ Indexes with Retrieval-Aware Residual Sidecars},
  author       = {Chuang, Ravan},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Work in progress}
}
```

---

## License

Released under the [MIT License](LICENSE).
