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
- Includes fixed-budget Residual-PQ refinement with oracle ceilings, compact bitmap/rank-prefix sidecars, FP16 residual codebooks, strict storage accounting, and paired-bootstrap significance tests.
- Adds a frozen-index **PQ-residual sidecar** study on 1M MS MARCO passages: a rank-16, per-dimension int8 correction layer that reranks only the top ANN candidates without retraining or rewriting the original IVF-PQ index.
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
| IVF configuration | FiQA / SciFact: `nlist=256`, representative `nprobe=16`; MS MARCO 1M: `nlist=4096` |

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
| IVF-PQ M=96 | 16 | 0.7195 | 0.5361 | 0.5770 | 0.024 ms | 47,147 | 13.01× |
| IVF-PQ M=96 | 32 | 0.7596 | 0.5619 | 0.6064 | 0.039 ms | 27,328 | 13.01× |
| IVF-PQ M=96 | 64 | 0.7887 | 0.5834 | 0.6297 | 0.072 ms | 14,619 | 13.01× |
| Native Faiss OPQMatrix + IVF-PQ M=96 | 64 | 0.7895 | 0.5842 | 0.6304 | 0.072 ms | 14,712 | 12.94× |

### Deployment-Aware Interpretation

At `M=96, nprobe=64`, plain IVF-PQ retains **92.1%** of exact Recall@10 while reducing serialized deployment storage by **13.01×**.

Native Faiss OPQ produces only a marginal gain at this high-rate configuration:

- Recall@10: `0.7887 → 0.7895` (`+0.0008`)
- nDCG@10: `0.6297 → 0.6304` (`+0.0008`)
- Build time: `17.8 s → 2,168.7 s` (`~122.1×` longer)

This suggests that when 384-dimensional embeddings are split into 96 PQ subvectors of only 4 dimensions each, plain IVF-PQ is already highly expressive and OPQ has limited remaining quantization error to remove.

The practical default at this scale is therefore plain IVF-PQ at `M=96, nprobe=32`, while `nprobe=64` is the higher-recall serving mode. Native OPQ remains an important baseline, especially for lower-rate PQ settings where rotation may provide more value.

> The table above is the `M=96` reference operating point from the completed low-rate sweep. Timing measures GPU Faiss search only and excludes embedding generation, HTTP transport, artifact loading, and response serialization.

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



## Frozen IVF-PQ PQ-Residual Sidecar: 1M Retrofit Study

This study evaluates a separate **frozen-index retrofit** mechanism. It is distinct
from the fixed-budget Residual-PQ experiments below: the original IVF-PQ index,
coarse quantizer, and PQ codes are left unchanged. A compact sidecar stores an
approximation of the error between the original embedding and the vector
reconstructed from that fixed IVF-PQ index.

### Motivation

A deployed ANN index may be expensive or operationally risky to rebuild. The
sidecar asks a narrower systems question:

> Can a frozen IVF-PQ index recover part of its ranking loss by attaching a
> small, separately stored correction payload, without retraining the index or
> rewriting its existing PQ codes?

For document embedding \(x\) and the reconstruction \(\hat{x}_{\mathrm{PQ}}\)
returned by the frozen IVF-PQ representation, the sidecar models:

```text
r_PQ(x) = x - x_hat_PQ
```

A shared rank-16 PCA basis is fit to these residuals. Each document stores only
16 per-dimension int8 coefficients. At query time, the system computes a
low-rank correction for only the highest-ranked ANN candidates:

```text
corrected_score(q, x)
= IVF-PQ_score(q, x)
+ alpha · q · r_hat_PQ(x)
```

The full-1M experiment freezes `alpha=1.0` and corrects only the top-40
documents from a Top-100 IVF-PQ candidate pool.

### Full 1M protocol

| Item | Configuration |
|---|---|
| Corpus | 1,000,000 deterministic MS MARCO passages |
| Evaluation queries | 1,000 frozen MS MARCO dev-query split |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Base index | IVF-PQ, `nlist=512`, `M=32`, `nbits=8`, `nprobe=16` |
| Candidate pool | Top-100 |
| Final cutoff | Top-10 |
| Sidecar | Rank-16 PCA residual basis + per-dimension int8 coefficients |
| Sidecar payload | 16 bytes/document |
| Activated candidates | Top-40 ANN candidates |
| Statistical test | Paired bootstrap, 20,000 resamples |
| GPU | NVIDIA Tesla T4 |

The `nlist=512`, `M=32` configuration is a deliberate low-rate retrofit
operating point. It is separate from the `M=96`, `nlist=4096` quality-throughput
reference sweep reported earlier.

### Main full-1M retrieval result

| Method | Recall@10 | Success@10 |
|:--|--:|--:|
| Frozen IVF-PQ `M=32` | 0.6628 | 0.6740 |
| **Frozen IVF-PQ `M=32` + rank-16 int8 PQ-residual sidecar** | **0.6914** | **0.7010** |
| Oracle exact rerank within the same IVF-PQ Top-100 candidate pool | 0.7792 | 0.7870 |

The sidecar improves Recall@10 by **+0.0287**. A paired bootstrap over the
1,000 frozen evaluation queries gives a 95% confidence interval of
**[+0.0147, +0.0430]**; none of 20,000 bootstrap resamples has a non-positive
mean gain.

The oracle row is a **candidate-pool ceiling** only: it rescored the 100
documents already returned by IVF-PQ with exact Float32 inner products. It is
not a full-corpus exact-search result.

Per-query Recall@10 changes were concentrated near ranking cutoffs:

```text
positive-gain queries: 44
unchanged queries:     941
negative-gain queries: 15
```

### Storage and retrofit properties

| Artifact | Size / requirement |
|---|---:|
| Frozen IVF-PQ `M=32` serialized index | 39.276 MB |
| Int8 sidecar codes for 1M documents | 15.259 MB |
| Shared rank-16 FP32 basis | 24.125 KB |
| Per-dimension scales | 192 bytes on disk |
| Added payload | 16 bytes/document |
| IVF-PQ retraining required for upgrade | No |
| Re-encoding existing PQ codes required | No |
| Original embeddings required to build the sidecar | Yes |

The retrofit computes residuals from the original embeddings and the frozen
index reconstruction, then writes a separate sidecar. It does not modify the
existing IVF coarse quantizer, codebooks, inverted lists, or PQ codes.

### Quality-storage comparison

| Method | PQ bytes/doc | Sidecar bytes/doc | Total representation bytes/doc | Recall@10 | Success@10 |
|:--|--:|--:|--:|--:|--:|
| IVF-PQ `M=16` | 16 | 0 | 16 | 0.5423 | 0.5540 |
| IVF-PQ `M=16` + rank-16 sidecar | 16 | 16 | 32 | 0.5799 | 0.5920 |
| IVF-PQ `M=32` | 32 | 0 | 32 | 0.6628 | 0.6740 |
| IVF-PQ `M=32` + rank-16 sidecar | 32 | 16 | 48 | 0.6914 | 0.7010 |
| IVF-PQ `M=48` | 48 | 0 | 48 | 0.7311 | 0.7410 |

The sidecar substantially improves a fixed base index, but it does **not**
outperform simply increasing the PQ code rate under an equal total
representation-byte budget. For example, `M=32 + 16B sidecar` remains below
the 48-byte `M=48` baseline.

The appropriate claim is therefore a **frozen-index enhancement / retrofit**
path, not a replacement for larger PQ codes when a full index rebuild is
acceptable.

### GPU-resident correction latency

The sidecar was moved to GPU memory and evaluated with a batch size of 64 on a
Tesla T4. The GPU implementation matched the CPU prototype exactly
(maximum correction difference `9.31e-09`; Top-10 row agreement `100%`).

| Stage | P50 ms/query | P95 ms/query |
|:--|--:|--:|
| IVF-PQ search | 0.0545 | 0.0885 |
| GPU sidecar correction + Top-40 rerank | 0.0168 | 0.0413 |
| Combined retrieval pipeline | 0.0797 | 0.1174 |

These timings exclude embedding generation, HTTP transport, artifact loading,
and response serialization. The current implementation uses Python
orchestration: Faiss returns candidate arrays to Python before PyTorch performs
GPU-side correction and reranking. It is therefore a batched GPU prototype,
not a fused CUDA serving kernel or a single-query production tail-latency claim.

### Interpretation

The result establishes that a compact low-rank sidecar can recover a
statistically significant portion of the ranking loss of a frozen low-rate
IVF-PQ index:

- **Recall gain:** `0.6628 → 0.6914` on the frozen 1M evaluation split.
- **Candidate-pool oracle-gap recovery:** `24.62%`.
- **Storage overhead:** 16 bytes/document.
- **Correction scope:** only Top-40 retrieved candidates.
- **Upgrade path:** no IVF-PQ retraining and no PQ code rewrite.

It does not establish that the sidecar is globally storage-optimal versus a
newly trained, higher-rate IVF-PQ index. That negative result is retained as
an important deployment-aware limitation.


## Fixed-Budget Residual-PQ Refinement

Beyond standard PQ / OPQ benchmarking, this project evaluates whether a
candidate-side Residual-PQ sidecar can recover low-rate IVF-PQ ranking loss
under a strict storage budget.

### Protocol

| Item | Configuration |
|---|---|
| Dataset | FiQA / BEIR held-out split |
| Held-out queries | 528 |
| Base ANN index | GPU IVF-PQ ADC, `M=32`, `nlist=256`, `nprobe=16` |
| Final cutoff | Top-10 |
| Candidate refinement depth | Top-50 for compact-sidecar comparisons |
| Storage target | At most `48 bytes/vector` |
| Statistical test | Paired bootstrap, 10,000 resamples |

The original fixed-budget study accounts for:

```text
base PQ code
+ residual-PQ code payload for selected documents
+ per-selected-document ID metadata
+ amortized shared residual-PQ codebook storage
```

The compact-sidecar extension removes the per-selected-document ID field. It
uses Faiss / corpus internal row IDs with:

```text
1-bit selection bitmap
+ block-level uint32 rank-prefix index
+ residual codes ordered by internal document ID
+ FP16 residual-PQ codebook
```

The recorded compact-sidecar latency uses an evaluation-only dense
document-index-to-slot accelerator. That accelerator is not included in the
deployable storage budget; bitmap/rank-prefix lookup equivalence is validated
separately.

### Main Results

| Method | Sidecar coverage | Recall@10 | MRR@10 | nDCG@10 | Interpretation |
|:--|--:|--:|--:|--:|:--|
| Base IVF-PQ `M=32` | – | 0.2907 | 0.3032 | 0.2395 | Low-rate baseline |
| Legacy Residual-PQ 16B + reconstruction-error + Top-50 | 45.9% | 0.3195 | 0.3314 | 0.2678 | Original best sparse sidecar |
| **Compact Residual-PQ 8-bit, `M_r=16` + Top-50** | **77.8%** | **0.3355** | 0.3478 | **0.2835** | Best compact Recall / nDCG point estimate |
| **Compact Residual-PQ 4-bit, `M_r=32` + Top-50** | **97.8%** | 0.3327 | **0.3531** | 0.2804 | Near-full coverage; best compact MRR point estimate |
| Uniform IVF-PQ `M=48` | 100% | 0.3455 | 0.3631 | 0.2942 | Higher point estimates on this split |

Both compact layouts significantly improve Recall@10, MRR@10, and nDCG@10
over base `M=32` on the held-out FiQA split. Compact-8bit also significantly
improves Recall@10 and nDCG@10 over the legacy 16B Residual-PQ sidecar.

Uniform `M=48` has higher point estimates, but paired-bootstrap 95% confidence
intervals for Uniform-versus-Compact differences cross zero for Recall@10,
MRR@10, and nDCG@10 on this held-out split. This does **not** establish
equivalence or superiority of the compact layouts; it means the experiment
does not establish a statistically significant directional difference here.

### Coverage-versus-Precision Trade-off

The compact layout makes the original coverage-versus-precision effect more
useful:

- Compact-8bit (`M_r=16`, 8-bit codes) assigns a higher-fidelity residual
  representation to 77.8% of documents.
- Compact-4bit (`M_r=32`, 4-bit codes) uses the same 16-byte residual payload
  per selected document but much smaller shared codebook storage, reaching
  97.8% coverage.
- Compact-8bit has the strongest compact Recall@10 and nDCG@10 point estimates.
- Compact-4bit has the strongest compact MRR@10 point estimate.
- The paired-bootstrap comparison between the two compact variants does not
  establish a significant difference on this split.

This supports a storage-constrained coverage-versus-precision trade-off rather
than a monotonic rule that longer or lower-bit residual codes always win.

### Compact Sidecar Figures

![Compact Residual-PQ fixed-budget quality comparison](figures/compact_residual_pq_quality.png)

![Compact Residual-PQ coverage versus quality](figures/compact_residual_pq_coverage.png)

These figures are generated from committed result artifacts:

    conda run -n rag-api python scripts/plot_compact_residual_pq.py

### Oracle Candidate-Rescoring Ceiling

To distinguish sidecar limitations from candidate-pool limitations, the project
also performs an oracle experiment: all documents already present in the
compressed `M=32` Top-L candidate pool are rescored with exact Float32 inner
products.

| Method | Candidate depth | Recall@10 | MRR@10 | nDCG@10 |
|:--|--:|--:|--:|--:|
| Base IVF-PQ `M=32` | – | 0.2907 | 0.3032 | 0.2395 |
| Oracle exact candidate rescoring | Top-20 | 0.3424 | 0.3927 | 0.3132 |
| Oracle exact candidate rescoring | Top-50 | 0.3810 | 0.4155 | 0.3386 |
| Oracle exact candidate rescoring | Top-100 | 0.3964 | 0.4198 | 0.3465 |
| Compact Residual-PQ 8-bit, `M_r=16` | Top-50 | 0.3355 | 0.3478 | 0.2835 |
| Uniform IVF-PQ `M=48` | – | 0.3455 | 0.3631 | 0.2942 |

The oracle Top-50 and Top-100 results exceed uniform `M=48`, showing that the
compressed `M=32` candidate pool retains substantial recoverable ranking
signal. Compact storage substantially narrows the practical gap by improving
sidecar coverage, but the oracle ceiling still leaves room for better
residual-code efficiency and candidate-side refinement.

### Interpretation

Compact Residual-PQ sidecars significantly recover low-rate IVF-PQ retrieval
loss under a strict 48 bytes/vector budget. The compact bitmap/rank-prefix
layout and FP16 codebook make the sidecar substantially more storage-efficient
than the legacy per-document-ID layout.

On held-out FiQA, both compact variants significantly outperform base `M=32`;
their differences from Uniform `M=48` are not statistically established by the
paired bootstrap used here. This is reported as a single-split, statistically
disciplined result—not a claim of universal equivalence to uniform higher-rate
PQ.

The reproducible compact result package includes [held-out results](results/compact_residual_pq_sidecar/compact_residual_pq_heldout_results.csv),
[strict storage accounting](results/compact_residual_pq_sidecar/compact_residual_pq_storage_config.csv),
and [paired-bootstrap results](results/compact_residual_pq_sidecar/bootstrap_significance/compact_residual_pq_bootstrap.md).

The full protocol is documented in
[`docs/selective_residual_pq_protocol.md`](docs/selective_residual_pq_protocol.md),
and the reproducible experiment is implemented in
[`notebooks/FiQA_BM25_Hybrid_RRF_Benchmark.ipynb`](notebooks/FiQA_BM25_Hybrid_RRF_Benchmark.ipynb).

### Fixed-Budget Residual-PQ Figures

![Fixed-budget Residual-PQ quality trade-off](figures/fixed_budget_residual_pq_quality.png)

![Residual-PQ precision versus coverage](figures/residual_pq_coverage_tradeoff.png)

The legacy figures are regenerated from the committed baseline CSVs:

    conda run -n rag-api python scripts/plot_fixed_budget_residual_pq.py

See the [legacy fixed-budget result package](results/fixed_budget_residual_pq/README.md)
and the [compact sidecar result package](results/compact_residual_pq_sidecar/)
for reproducible result artifacts.


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

- **Frozen IVF-PQ retrofit sidecar (MS MARCO 1M):** a rank-16 int8 PQ-residual sidecar improves the fixed `M=32` IVF-PQ operating point from Recall@10 `0.6628` to `0.6914` (`+0.0287`; paired-bootstrap 95% CI `[+0.0147, +0.0430]`) while adding 16 bytes/document and correcting only Top-40 candidates. It is a frozen-index enhancement, not a same-byte replacement for a higher-rate PQ index.
- **Compact fixed-budget Residual-PQ refinement:** bitmap/rank-prefix metadata and FP16 residual codebooks raise sidecar coverage to 77.8% (Compact-8bit) or 97.8% (Compact-4bit). Both compact layouts significantly improve low-rate `M=32` IVF-PQ; their differences from Uniform `M=48` are not statistically established on the held-out FiQA split.
- **Candidate-side refinement ceiling:** oracle exact rescoring of compressed `M=32` Top-50 candidates reaches Recall@10 `0.3810`, exceeding uniform `M=48` at `0.3455`; practical sidecar coverage and code efficiency are the current bottlenecks.

- **Million-scale MS MARCO full sweep:** across `M=24/32/48/64/96` and `nprobe=4/16/32/64` (40 benchmark points), OPQ Recall@10 gain at `nprobe=64` contracts from `+0.0386` at `M=24` to `+0.0008` at `M=96`, while its build multiplier rises from `55.8×` to about `122×`.
- **High-rate reference point:** on 1M BGE-small passages, plain IVF-PQ at `M=96, nprobe=64` retains 92.1% of exact Recall@10 with 13.01× serialized deployment compression; native OPQ adds only marginal quality at substantially higher offline build cost.
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
  residual_pq_scale_limitations.md
  retrieval_api.md
  selective_residual_pq_protocol.md
  testing_ci.md
figures/
  storage_quality_tradeoff.png
  throughput_stability.png
  fixed_budget_residual_pq_quality.png
  residual_pq_coverage_tradeoff.png
notebooks/
  Ai_embedding_compression.ipynb
  SciFact_OPQ_IVFPQ_Benchmark.ipynb
  FiQA_BGE_Small_OPQ_IVFPQ_Benchmark.ipynb
  SciFact_BGE_Small_OPQ_IVFPQ_Benchmark.ipynb
  MSMARCO_1M_Low_Rate_PQ_OPQ_Pareto.ipynb
  MSMARCO_1M_PQ_Residual_Sidecar_Gate3.ipynb
  FiQA_BM25_Hybrid_RRF_Benchmark.ipynb
  SciFact_BM25_Hybrid_RRF_Transfer.ipynb
results/
  api_benchmark/
  fiqa_gpu_benchmark/
  scifact_gpu_benchmark/
  fiqa_bge_small_gpu_benchmark/
  scifact_bge_small_gpu_benchmark/
  msmarco_low_rate_pareto/
    1m_pilot_m24_m48/
    1m_full_m24_m96/
  msmarco_low_rate_pareto_results_full_m32_m64_m96/
  msmarco_1m_pq_residual_gate3/
  rerank_fiqa_benchmark/
  fixed_budget_residual_pq/
  compact_residual_pq_sidecar/
    bootstrap_significance/
scripts/
  benchmark_api.py
  benchmark_reranker.py
  export_service_artifacts.py
  merge_msmarco_low_rate_results.py
  prepare_fiqa_documents.py
  plot_fixed_budget_residual_pq.py
  compact_sidecar_layout.py
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



### Frozen IVF-PQ PQ-residual sidecar: MS MARCO 1M

1. Open `notebooks/MSMARCO_1M_PQ_Residual_Sidecar_Gate3.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime and mount the Google Drive cache containing the
   1M BGE-small document embedding memmap, document IDs, query vectors, qids,
   and qrels.
3. Build or load the frozen `M=32`, `nlist=512`, `nprobe=16` IVF-PQ index.
4. Reconstruct the frozen PQ representation, train the rank-16 residual basis,
   and emit the 16-byte/document int8 sidecar.
5. Evaluate the frozen Top-40 correction protocol on the saved 1,000-query
   split, then save recall, storage, latency, and paired-bootstrap artifacts.
6. Treat the `M=16/32/48` quality-storage table as a deployment comparison:
   the sidecar improves a fixed base index but does not replace a rebuilt
   higher-rate PQ index at an equal byte budget.


### Fixed-budget Residual-PQ refinement

1. Open `notebooks/FiQA_BM25_Hybrid_RRF_Benchmark.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime and run the notebook from top to bottom.
3. The notebook records rank-flip audits, candidate recoverability, sparse
   FP16 sidecar baselines, oracle exact candidate rescoring, Residual-PQ
   code-size sweeps, fixed-budget Residual-PQ allocation, and paired-bootstrap
   confidence intervals.
4. Cells 16–17 reproduce the legacy fixed-budget experiment and bootstrap analysis.
5. Cell 18 evaluates compact bitmap/rank-prefix Residual-PQ sidecars with FP16
   codebooks under the same strict 48 bytes/vector target.
6. Cell 19 runs paired-bootstrap comparisons among base `M=32`, legacy
   Residual-PQ, Compact-4bit, Compact-8bit, and Uniform `M=48`.
7. The compact layout counts the bitmap, rank-prefix index, serialized FP16
   codebook, residual-code payload, alignment, and metadata headers in its
   deployable storage accounting.

### Million-scale MS MARCO low-rate PQ / OPQ full sweep

1. Open `notebooks/MSMARCO_1M_Low_Rate_PQ_OPQ_Pareto.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime and install `requirements-colab.txt`.
3. Run the notebook from top to bottom. The configured full continuation evaluates `M=32/64/96` over `nprobe=4/16/32/64`; the resumable runner can store completed-`M` checkpoints in Google Drive.
4. Copy the generated summaries, metadata, reports, and figures into `results/msmarco_low_rate_pareto_results_full_m32_m64_m96/`.
5. Keep the committed `M=24/48` pilot artifacts under `results/msmarco_low_rate_pareto/1m_pilot_m24_m48/`, then regenerate the unified full-sweep artifacts locally:

```bash
conda activate rag-api
python scripts/merge_msmarco_low_rate_results.py
```

6. The merged `M=24/32/48/64/96` outputs are written to `results/msmarco_low_rate_pareto/1m_full_m24_m96/`. Large source data, serialized indexes, and resumable checkpoints are intentionally excluded from Git history; commit reproducible code, summaries, metadata, reports, and figures only.

For all GPU experiments, use Google Colab with an NVIDIA GPU runtime and install `requirements-colab.txt`.

## Limitations and Next Steps

- FiQA and SciFact provide cross-dataset ranking validation, while the deterministic MS MARCO 1M experiment provides a single-GPU million-scale retrieval benchmark. It does not yet establish multi-node, billion-vector, or online-production behavior.
- The benchmark currently uses two English embedding models; it does not yet validate multilingual or Traditional Chinese retrieval.
- The deployment uses a learned external OPQ transform; any compatible serving implementation must apply the same query rotation before Faiss search.
- The current BGE CPU reranker configuration is experimental: it did not improve the recorded 100-query FiQA subset and adds substantial local latency. Future reranking work should compare domain-appropriate models, title-aware / truncated document formatting, and throughput under realistic batch loads before making a production-default claim.
- Future work includes hybrid sparse-dense retrieval, a Traditional Chinese retrieval benchmark, query-aware retrieval routing, model-specific deployment selection, and production observability / deployment hardening.
- Fixed-budget Residual-PQ transfer requires a larger corpus than SciFact for an 8-bit residual codebook, because amortized shared-codebook storage becomes prohibitive on small corpora. Future transfer experiments should test compact sidecars on a sufficiently large corpus and preserve identical storage accounting without re-tuning the allocation policy.
- The frozen IVF-PQ PQ-residual sidecar is currently validated on one 1M-passage MS MARCO configuration. Its GPU timing is a batched Python-orchestrated prototype, not a fused serving kernel, and its storage-quality comparison shows that higher-rate PQ remains stronger when a full index rebuild is permitted.

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

The current `main` branch captures the million-scale low-rate PQ / OPQ full-sweep milestone, the compact fixed-budget Residual-PQ extension, and the frozen IVF-PQ PQ-residual sidecar retrofit study, while retaining the verified MiniLM FiQA artifact as the deployed service baseline. The optional reranker is intentionally disabled in that default artifact because the recorded FiQA evaluation did not justify its latency cost. The next research milestone is zero-retuning validation of compact Residual-PQ sidecars on a larger transfer corpus where shared codebook amortization remains storage-feasible, followed by deployable sidecar serving and hybrid sparse-dense retrieval.
