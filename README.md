# AI Embedding Compression for RAG Retrieval

[![CI](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Faiss](https://img.shields.io/badge/ANN-Faiss-blue)](https://github.com/facebookresearch/faiss)
[![Docker Verified](https://img.shields.io/badge/Docker-verified-2496ED)](docs/docker_api.md)

A research-grade benchmark, frozen-index retrofit study, and deployable retrieval system for embedding compression and approximate nearest-neighbor search in Retrieval-Augmented Generation (RAG).

The repository separates three questions that are often conflated:

1. **Compression quality:** How much retrieval quality remains after document embeddings are compressed?
2. **Compressed-domain efficiency:** What latency and throughput are obtained when Faiss searches PQ codes directly?
3. **Frozen-index recovery:** How much ranking quality can be recovered by attaching a compact residual sidecar without rebuilding or rewriting an existing IVF-PQ index?

The current research focus is **Retrieval-Aware Residual Subspace (RARS)**, a rank-16 int8 post-hoc sidecar for frozen IVF-PQ indexes. On the held-out MS MARCO 1M evaluation, RARS Top20 improves Recall@10 from `0.66275` to `0.69892`, while a vectorized live-Faiss implementation adds `0.816 µs/query` of correction work in a 1,000-query batch.

## Research Status

| Area | Status |
|---|---|
| PQ / OPQ quality and GPU ADC benchmarking | Complete |
| MS MARCO 1M low-rate sweep | Complete |
| Frozen IVF-PQ residual sidecar | Complete |
| Retrieval-Aware Residual Subspace (RARS) | Complete |
| Deployable rank-16 int8 sidecar artifact | Complete |
| FastAPI sidecar serving path | Complete |
| Artifact-backed and live-Faiss benchmarks | Complete |
| Paper-ready CSV / LaTeX table pipeline | Complete |
| SIGIR short-paper manuscript | In preparation |

## Highlights

- Evaluates Float32, INT8, INT4, PQ, OPQ, IVF-PQ, and OPQ-IVF-PQ across FiQA, SciFact, and a deterministic 1M-passage MS MARCO benchmark.
- Measures Recall@5, Recall@10, Success@10, MRR@10, nDCG@10, serialized storage, analytical code size, latency, and QPS.
- Implements genuine compressed-domain GPU retrieval with Faiss IVF-PQ ADC; document vectors are not reconstructed to Float32 during ANN search.
- Provides a complete low-rate MS MARCO 1M sweep over `M ∈ {24, 32, 48, 64, 96}` and `nprobe ∈ {4, 16, 32, 64}`.
- Introduces a frozen-index rank-16 int8 residual sidecar that improves IVF-PQ `M=32` Recall@10 from `0.66275` to `0.69142` without retraining the coarse quantizer or rewriting PQ codes.
- Introduces **RARS-Score**, a score-error weighted residual basis that improves the same frozen-index setting to Recall@10 `0.69992`.
- Identifies **RARS Top20** as the preferred deployable operating point: Recall@10 `0.69892`, within `0.0010` of Top40 while halving correction depth.
- Packages the 1M-document RARS sidecar as a deployable artifact with a `16.025 B/document` representation cost and `24.028 B/document` complete artifact cost including external document IDs.
- Adds vectorized live-Faiss correction: on the recorded 14-thread CPU benchmark, Top20 requires `0.816 µs/query`, equal to `2.72%` of independently timed Faiss search cost; alternating paired measurements observed `5.41%` mean end-to-end overhead.
- Verifies live-Faiss and cached candidates with score-close ratio `1.0` and row-position match `0.99988`.
- Adds FastAPI `/search`, `/batch-search`, and `/health` support for optional fixed Top-B RARS correction.
- Retains negative results rather than hiding them: higher-rate `M=48` remains stronger when rebuilding is allowed, RARS-vs-PCA confidence intervals cross zero, FiQA transfer is model-sensitive, learned routers do not beat fixed Top20, and the evaluated cross-encoder reranker does not justify its latency.
- Generates reproducible paper-ready CSV and LaTeX tables directly from committed result artifacts.

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
| IVF configuration | FiQA / SciFact reference benchmarks: `nlist=256`; MS MARCO high-rate sweep: `nlist=4096`; frozen-index sidecar / RARS study: `nlist=512`; representative `nprobe=16` |

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


## Cross-Setting Validation: Frozen PQ-Residual Sidecar

The 1M MS MARCO result establishes the main system-scale result. To test whether
its direction transfers beyond a single corpus-model setting, the same frozen
sidecar protocol was evaluated on FiQA / BEIR with both BGE-small and MiniLM.
The FiQA runs use the **same sidecar hyperparameters selected on MS MARCO**;
the only corpus-scale adjustment is `nlist=256` for the smaller FiQA corpus.

### Shared protocol

| Item | MS MARCO 1M | FiQA / BEIR |
|---|---:|---:|
| Base ANN index | IVF-PQ `M=32`, `nbits=8` | IVF-PQ `M=32`, `nbits=8` |
| IVF setting | `nlist=512`, `nprobe=16` | `nlist=256`, `nprobe=16` |
| Candidate pool | Top-100 | Top-100 |
| Sidecar | rank-16 PCA basis + int8 coefficients | rank-16 PCA basis + int8 coefficients |
| Sidecar payload | 16 B/document | 16 B/document |
| Activated candidates | Top-40 | Top-40 |
| Score fusion | `alpha=1.0` | `alpha=1.0` |
| Final cutoff | Top-10 | Top-10 |
| Statistical test | paired bootstrap, 20,000 resamples | paired bootstrap, 20,000 resamples |

The FiQA transfer experiments do **not** retune rank, code width, candidate
depth, or `alpha` after observing FiQA results.

### Cross-setting Recall@10 results

| Setting | Frozen IVF-PQ `M=32` | `M=32` + rank-16 sidecar | IVF-PQ `M=48` | Sidecar gain |
|:--|--:|--:|--:|--:|
| MS MARCO 1M × BGE-small | 0.6628 | **0.6914** | **0.7311** | **+0.0287** |
| FiQA × BGE-small | 0.3287 | **0.3418** | **0.3896** | **+0.0131** |
| FiQA × MiniLM | 0.3358 | **0.3454** | **0.3723** | **+0.0096** |

![PQ-residual sidecar Recall@10 across settings](results/pq_residual_sidecar_cross_setting/figures/recall_cross_setting.png)

Across all three settings, the sidecar improves the frozen `M=32` index. The
same-storage `M=48` baseline remains stronger in all three cases, reinforcing
the intended positioning: the method is a **retrofit enhancement for an
existing index**, not a replacement for retraining a higher-rate PQ index when
a full rebuild is acceptable.

### Paired-bootstrap interpretation

| Setting | Recall@10 gain | 95% paired-bootstrap CI | Bootstrap fraction with gain ≤ 0 | Interpretation |
|:--|--:|--:|--:|:--|
| MS MARCO 1M × BGE-small | +0.0287 | [+0.0147, +0.0430] | 0.0000 | Statistically positive on the frozen 1,000-query evaluation split |
| FiQA × BGE-small | +0.0131 | [-0.0006, +0.0268] | 0.0303 | Positive zero-retuning point estimate; interval marginally crosses zero |
| FiQA × MiniLM | +0.0096 | [-0.0056, +0.0252] | 0.1129 | Positive zero-retuning point estimate; interval crosses zero |

![PQ-residual sidecar gain with paired-bootstrap intervals](results/pq_residual_sidecar_cross_setting/figures/sidecar_gain_bootstrap.png)

The correct interpretation is deliberately conservative: the sidecar's
improvement direction is consistent across corpus and embedding-model settings,
while the strongest statistical evidence currently comes from the 1M MS MARCO
experiment. The smaller 648-query FiQA evaluations provide transfer evidence,
but do not establish a universally significant gain at the 95% confidence level.

### Candidate-pool recovery

The exact-rescoring oracle reranks only documents already present in the frozen
`M=32` Top-100 candidate pool. It measures recoverable ranking error within the
candidate pool, not full-corpus exact-search quality.

| Setting | Frozen `M=32` | Exact Top-100 candidate oracle | Recoverable Recall@10 gap | Sidecar recovery of oracle gap |
|:--|--:|--:|--:|--:|
| MS MARCO 1M × BGE-small | 0.6628 | 0.7792 | +0.1164 | 24.62% |
| FiQA × BGE-small | 0.3287 | 0.4306 | +0.1020 | 12.86% |
| FiQA × MiniLM | 0.3358 | 0.4114 | +0.0756 | 12.72% |

### Reproducible result package

The committed result package contains the flat table, structured per-setting
metadata, bootstrap summaries, figures, and a SHA-256 manifest:

- [cross-setting README](results/pq_residual_sidecar_cross_setting/README.md)
- [summary CSV](results/pq_residual_sidecar_cross_setting/cross_setting_summary.csv)
- [summary JSON](results/pq_residual_sidecar_cross_setting/cross_setting_summary.json)
- [per-setting metadata](results/pq_residual_sidecar_cross_setting/setting_details/)
- [integrity manifest](results/pq_residual_sidecar_cross_setting/manifest.json)



## Retrieval-Aware Residual Subspace (RARS)

The frozen sidecar study above uses a PCA residual basis. PCA is reconstruction-oriented: it captures the largest residual variance in

```text
r_PQ(x) = x - x_hat_PQ
```

but it does not directly optimize the score errors that affect Top-K retrieval. RARS evaluates whether a retrieval-aware residual basis can improve the same frozen IVF-PQ sidecar under the same storage and correction budget.

The correction form remains unchanged:

```text
s_corr(q, x)
= s_IVFPQ(q, x)
+ alpha · q^T B a_x
```

where `B` is a rank-16 residual basis and `a_x` is the per-document int8 coefficient vector. The base IVF-PQ index, PQ codes, sidecar rank, int8 coefficient budget, and candidate correction depth are kept fixed.

### Protocol

| Item | Configuration |
|---|---|
| Corpus | 1,000,000 deterministic MS MARCO passages |
| Evaluation queries | 1,000 held-out MS MARCO dev queries |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Base index | Frozen IVF-PQ `M=32`, `nbits=8`, `nlist=512`, `nprobe=16` |
| Candidate pool | Top-100 |
| Sidecar rank | 16 |
| Sidecar coefficient format | Per-dimension int8 |
| Main correction depth | Top-40 |
| Main RARS-Score alpha | `0.75` |
| Metrics | Recall@10, Success@10, MRR@10, nDCG@10 |
| Statistical test | Paired bootstrap, 20,000 resamples |

### Basis variants

| Basis | Description |
|---|---|
| PCA sidecar | Reconstruction-oriented residual PCA basis |
| RARS-Score | Score-error weighted residual basis |
| RARS-Boundary | Top-k boundary weighted residual basis |

### Main qrels results

| Method | Alpha | Top-B | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Δ Recall vs base |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Frozen IVF-PQ `M=32` | 0.00 | 0 | 0.6628 | 0.6740 | 0.4659 | 0.5099 | 0.0000 |
| PCA residual sidecar | 1.00 | 40 | 0.6914 | 0.7010 | 0.4792 | 0.5275 | +0.0287 |
| RARS-Boundary | 0.75 | 40 | 0.6949 | 0.7050 | 0.4793 | 0.5277 | +0.0322 |
| **RARS-Score** | **0.75** | **40** | **0.6999** | **0.7100** | **0.4845** | **0.5325** | **+0.0372** |

RARS-Score improves the frozen `M=32` sidecar result from Recall@10 `0.6914` with a PCA residual basis to `0.6999` under the same rank-16 int8 Top-40 correction budget.

### Paired-bootstrap interpretation

Compared with the PCA residual sidecar, RARS-Score gives positive differences across all reported metrics:

| Metric | Mean difference | 95% paired-bootstrap CI | Bootstrap fraction with gain ≤ 0 |
|:--|--:|--:|--:|
| Recall@10 | +0.0085 | [-0.0020, +0.0195] | 0.0662 |
| Success@10 | +0.0090 | [-0.0020, +0.0200] | 0.0621 |
| MRR@10 | +0.0053 | [-0.0036, +0.0142] | 0.1224 |
| nDCG@10 | +0.0050 | [-0.0023, +0.0124] | 0.0914 |

The correct interpretation is conservative: RARS-Score is directionally positive over PCA across Recall@10, Success@10, MRR@10, and nDCG@10, but the 95% paired-bootstrap intervals narrowly cross zero on the 1,000-query held-out split. It is therefore reported as a promising retrieval-aware basis improvement rather than a fully established statistically significant gain.

### RARS-Score alpha sweep

| Alpha | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|
| 0.25 | 0.6798 | 0.6910 | 0.4765 | 0.5220 |
| 0.50 | 0.6919 | 0.7020 | 0.4788 | 0.5269 |
| **0.75** | **0.6999** | **0.7100** | **0.4845** | **0.5325** |
| 1.00 | 0.6979 | 0.7080 | 0.4805 | 0.5291 |
| 1.25 | 0.6953 | 0.7060 | 0.4740 | 0.5236 |
| 1.50 | 0.6901 | 0.7000 | 0.4635 | 0.5147 |
| 1.75 | 0.6821 | 0.6930 | 0.4537 | 0.5050 |
| 2.00 | 0.6778 | 0.6890 | 0.4451 | 0.4971 |

The best qrels setting in this sweep is `alpha=0.75`.

### RARS-Score Top-B depth ablation

| Corrected candidates | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|
| 0 | 0.6628 | 0.6740 | 0.4659 | 0.5099 |
| 5 | 0.6648 | 0.6760 | 0.4776 | 0.5194 |
| 10 | 0.6749 | 0.6860 | 0.4803 | 0.5237 |
| 20 | 0.6989 | 0.7090 | 0.4845 | 0.5324 |
| **40** | **0.6999** | **0.7100** | **0.4845** | **0.5325** |
| 60 | 0.6989 | 0.7090 | 0.4843 | 0.5322 |
| 80 | 0.6989 | 0.7090 | 0.4843 | 0.5322 |
| 100 | 0.6989 | 0.7090 | 0.4843 | 0.5322 |

Top-20 captures nearly all of the RARS-Score gain, while Top-40 is the best observed operating point. Correcting deeper than Top-40 does not improve qrels metrics on this split. This motivates future query-adaptive correction-depth selection.

### Query-adaptive correction diagnostics

A follow-up diagnostic evaluates whether RARS correction depth can be selected per query from ANN score uncertainty and correction-magnitude features. The strongest currently validated cost-aware operating point remains fixed Top20 correction: it reaches Recall@10 `0.6989`, within `0.0010` of fixed Top40 (`0.6999`), while halving correction depth from 40 to 20 candidates/query.

#### Fixed-depth and oracle routing

| Strategy | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|:--|--:|--:|--:|--:|--:|
| Always Top0 | 0.6628 | 0.6740 | 0.4659 | 0.5099 | 0.0 |
| Always Top20 | 0.6989 | 0.7090 | **0.4845** | 0.5324 | 20.0 |
| Always Top40 | 0.6999 | 0.7100 | 0.4845 | **0.5325** | 40.0 |
| Oracle cheapest Top0/Top20/Top40 | **0.7103** | **0.7200** | 0.4731 | 0.5262 | **1.1** |

The oracle chooses the cheapest correction depth among Top0, Top20, and Top40 that attains the best per-query Recall@10. Its label distribution is highly imbalanced:

| Oracle depth | Queries | Fraction |
|---:|---:|---:|
| 0 | 950 | 0.950 |
| 20 | 45 | 0.045 |
| 40 | 5 | 0.005 |

This shows substantial query-adaptive headroom in principle: only a small minority of queries need deeper correction. However, that headroom is hard to recover with simple learned routers because the useful correction cases are rare.

#### Simple gate and learned-router results

Earlier random 5-fold threshold routing showed that simple one-dimensional score-margin gates do not robustly outperform fixed-depth activation:

| Strategy | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|:--|--:|--:|--:|--:|--:|
| Best train-recall gate, target Top20 | 0.6979 | 0.7080 | 0.4840 | 0.5317 | 19.38 |
| Best train-recall gate, target Top40 | 0.6989 | 0.7090 | 0.4839 | 0.5318 | 38.24 |
| Cheapest within 0.001 train Recall, target Top20 | 0.6944 | 0.7040 | 0.4815 | 0.5289 | 15.32 |
| Cheapest within 0.001 train Recall, target Top40 | 0.6949 | 0.7050 | 0.4819 | 0.5292 | 27.56 |

A stronger learned-router diagnostic then trained 5-fold query-level routers over ANN score features, correction-magnitude features, and query-vector summary features. The router predicts one of Top0, Top20, or Top40 correction depth.

| Strategy | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|:--|--:|--:|--:|--:|--:|
| Offline exact-proxy features + logistic regression | 0.6813 | 0.6920 | 0.4718 | 0.5185 | 4.56 |
| Deployable features + logistic regression | 0.6774 | 0.6870 | 0.4760 | 0.5214 | 5.94 |
| Deployable features + random forest | 0.6628 | 0.6740 | 0.4671 | 0.5109 | 0.70 |
| Deployable features + histogram gradient boosting | 0.6628 | 0.6740 | 0.4659 | 0.5099 | 0.02 |

The learned routers do not approach fixed Top20 / Top40 quality. The best learned result reaches Recall@10 `0.6813` with `4.56` corrected candidates/query, below fixed Top20 at `0.6989`. Most learned models either collapse toward Top0 because the oracle labels are highly imbalanced, or over-correct without recovering enough retrieval quality.

The correct interpretation is conservative. Fixed Top20 remains the strongest validated deployable cost-aware RARS setting. The oracle result shows routing headroom, but the current handcrafted score, correction, and query-vector features are insufficient for robust learned query-adaptive activation. The learned-router experiment is therefore retained as a negative diagnostic result, not as a completed adaptive-routing method.

See [`query_adaptive_rars_gate_diagnostics.md`](results/retrieval_aware_residual_basis/query_adaptive_rars_gate_diagnostics.md) and the [learned RARS router diagnostics](results/retrieval_aware_residual_basis/learned_rars_router/README.md).

### Cross-setting validation: FiQA BGE-small and MiniLM

To check whether retrieval-aware residual correction transfers beyond the
MS MARCO 1M setting, the RARS transfer notebooks were run on FiQA / BEIR with
both `BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`.

Each transfer experiment builds a FiQA-specific frozen IVF-PQ index and Top-100
candidate cache, then compares current-setting PCA, score-error weighted RARS,
and top-k-boundary weighted RARS under the same rank-16 int8 sidecar form.

| Item | Configuration |
|---|---|
| Dataset | FiQA / BEIR |
| Documents | 57,638 |
| Evaluation queries | 648 |
| Embedding models | `BAAI/bge-small-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2` |
| Base index | Frozen IVF-PQ `M=32`, `nbits=8`, `nlist=256`, `nprobe=16` |
| Candidate pool | Top-100 |
| Sidecar rank | 16 |
| Final cutoff | Top-10 |

#### FiQA BGE-small qrels-based final metrics

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|:--|--:|--:|--:|--:|
| Frozen IVF-PQ `M=32` | 0.2935 | 0.4784 | 0.2964 | 0.2373 |
| Score-error RARS Top10 | 0.2967 | 0.4892 | 0.3101 | 0.2435 |
| Score-error RARS Top20 | 0.3184 | 0.5062 | 0.3180 | 0.2559 |
| **Score-error RARS Top40** | 0.3235 | 0.5201 | **0.3220** | 0.2587 |
| Score-error RARS Top100 | 0.3232 | 0.5185 | 0.3213 | 0.2583 |
| PCA-current Top20 | 0.3203 | 0.5123 | 0.3137 | 0.2557 |
| **PCA-current Top40** | **0.3282** | **0.5231** | 0.3181 | **0.2600** |

On FiQA BGE-small, score-error weighted RARS Top40 improves over the frozen
IVF-PQ baseline:

- Recall@10: `0.2935 → 0.3235`
- Success@10: `0.4784 → 0.5201`
- MRR@10: `0.2964 → 0.3220`
- nDCG@10: `0.2373 → 0.2587`

The proxy diagnostics also support the intended mechanism: score-error weighted
RARS has the highest correlation with the exact-minus-ANN score error
(`0.5209`) and the best Top-10 overlap gain among the evaluated bases
(`0.4676 → 0.5290`). Current-setting PCA remains highly competitive and slightly
leads Recall@10, Success@10, and nDCG@10 in this setting.

#### FiQA MiniLM qrels-based final metrics

MiniLM is a more mixed transfer setting. With fixed `alpha=1.0`, most qrels
metrics do not improve over the frozen IVF-PQ baseline, even though proxy
score-error alignment improves clearly.

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|:--|--:|--:|--:|--:|
| Frozen IVF-PQ `M=32` | 0.3446 | 0.5494 | **0.3603** | 0.2862 |
| Score-error RARS Top10, alpha=1.0 | 0.3333 | 0.5386 | 0.3495 | 0.2757 |
| Score-error RARS Top20, alpha=1.0 | 0.3428 | 0.5478 | 0.3490 | 0.2789 |
| Score-error RARS Top40, alpha=1.0 | 0.3440 | 0.5494 | 0.3489 | 0.2795 |
| Score-error RARS Top100, alpha=1.0 | 0.3500 | **0.5586** | 0.3513 | 0.2822 |
| PCA-current Top40, alpha=1.0 | 0.3389 | 0.5432 | 0.3469 | 0.2795 |

The proxy-selected `alpha=0.75` gives a stronger MiniLM transfer point:

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|:--|--:|--:|--:|--:|
| Frozen IVF-PQ `M=32` | 0.3446 | 0.5494 | **0.3603** | 0.2862 |
| PCA-current Top40, alpha=0.75 | 0.3484 | 0.5525 | 0.3550 | 0.2857 |
| Score-error RARS Top40, alpha=0.75 | 0.3502 | 0.5540 | 0.3577 | 0.2857 |
| **Top10-boundary RARS Top40, alpha=0.75** | **0.3506** | **0.5540** | 0.3595 | **0.2880** |

MiniLM proxy diagnostics still favor retrieval-aware bases: score-error weighted
RARS has the highest exact-minus-ANN correlation (`0.5342`) and sign agreement
(`0.7054`). However, qrels gains are small and alpha-sensitive. The best
proxy-selected boundary basis improves Recall@10, Success@10, and nDCG@10
slightly, while MRR@10 remains essentially flat/slightly below the frozen base.

The cross-setting interpretation is intentionally conservative. BGE-small
provides a clear positive qrels transfer point; MiniLM shows strong proxy
alignment but only modest, alpha-sensitive qrels gains. This supports the
residual-correction mechanism while showing that RARS transfer depends on the
embedding model and score geometry, rather than universally dominating PCA.

### Reproducible result package

The committed RARS package contains qrels summaries, proxy diagnostics, alpha and Top-B ablations, query-adaptive gate diagnostics, paired-bootstrap results, basis metadata, and a SHA-256 manifest:

- [RARS result README](results/retrieval_aware_residual_basis/README.md)
- [main qrels table](results/retrieval_aware_residual_basis/basis_qrels_eval_main.csv)
- [RARS-Score alpha sweep](results/retrieval_aware_residual_basis/score_error_weighted_alpha_qrels_sweep.csv)
- [RARS-Score Top-B ablation](results/retrieval_aware_residual_basis/score_error_weighted_topb_qrels_ablation.csv)
- [query-adaptive gate diagnostics](results/retrieval_aware_residual_basis/query_adaptive_rars_gate_diagnostics.md)
- [learned RARS router diagnostics](results/retrieval_aware_residual_basis/learned_rars_router/README.md)
- [FiQA BGE-small RARS transfer validation](results/retrieval_aware_residual_basis/fiqa_bge_small_transfer/README.md)
- [FiQA MiniLM RARS transfer validation](results/retrieval_aware_residual_basis/fiqa_minilm_transfer/README.md)
- [Gate 1b 5-fold strategy summary](results/retrieval_aware_residual_basis/gate1b_5fold_strategy_summary.csv)
- [Gate 1b 5-fold oracle summary](results/retrieval_aware_residual_basis/gate1b_5fold_oracle_router_summary.csv)
- [final comparison table](results/retrieval_aware_residual_basis/rars_final_comparison_qrels.csv)
- [paired-bootstrap comparison](results/retrieval_aware_residual_basis/paired_bootstrap_best_rars_score_vs_pca.json)
- [integrity manifest](results/retrieval_aware_residual_basis/manifest.json)



## Deployable RARS Artifact and Live Faiss Benchmark

The final rank-16 int8 RARS sidecar is exported as a versioned artifact for the
1,000,000-document frozen IVF-PQ `M=32` index.

```text
artifacts/msmarco_rars_sidecar_m32_rank16/
├── basis.npy
├── scales.npy
├── codes.int8.npy
├── doc_ids.npy
├── sidecar_config.json
└── manifest.json
```

### Storage accounting

| Component | Total bytes | Bytes/document |
|:--|--:|--:|
| Int8 coefficient codes | 16,000,128 | 16.000128 |
| Shared basis and scales amortized with codes | 16,025,024 | 16.025024 |
| External document IDs | 8,000,128 | 8.000128 |
| Complete deployable artifact | 24,027,749 | 24.027749 |

The correct paper wording is:

> The rank-16 int8 RARS representation requires 16.03 bytes per document, while the complete deployable artifact including external document-ID metadata requires 24.03 bytes per document.

The external document IDs are reported separately because they are serving
metadata rather than part of the residual representation itself.

### Artifact-backed cached-candidate benchmark

The artifact loader is tested against the committed Top-100 candidate cache and
held-out qrels. It reproduces the expected retrieval metrics:

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|:--|--:|--:|--:|--:|
| Frozen IVF-PQ `M=32` | 0.66275 | 0.674 | 0.46588 | 0.50991 |
| RARS Top20 | 0.69892 | 0.709 | 0.48455 | 0.53236 |
| RARS Top40 | 0.69992 | 0.710 | 0.48450 | 0.53251 |

Detailed outputs are stored under
[`results/retrieval_aware_residual_basis/sidecar_artifact_benchmark/`](results/retrieval_aware_residual_basis/sidecar_artifact_benchmark/).

### Live Faiss benchmark

The live benchmark executes the actual frozen Faiss `index.search()` call and
then applies the artifact-backed RARS correction. It compares the original
Python loop with a vectorized implementation and records search-only,
correction-only, estimated-combined, and alternating paired end-to-end timing.

The vectorized correction computes:

```text
q_proj = queries @ basis
coeff  = int8_codes[candidate_rows] * scales
delta  = einsum(coeff, q_proj)
corrected_scores = ann_scores + alpha * delta
```

Recorded 1,000-query results:

| Threads | Method | Recall@10 | Correction | Correction / Faiss | Paired E2E overhead |
|--:|:--|--:|--:|--:|--:|
| 1 | RARS Top20 | 0.69892 | 0.613 µs/query | 0.23% | 0.49% |
| 1 | RARS Top40 | 0.69992 | 1.058 µs/query | 0.39% | 0.60% |
| 14 | RARS Top20 | 0.69892 | 0.816 µs/query | 2.72% | 5.41% |
| 14 | RARS Top40 | 0.69992 | 1.325 µs/query | 4.41% | 5.51% |

Vectorization accelerates correction by approximately `8.9×` for Top20 and
`5.9×` for Top40 relative to the recorded Python loop.

The two overhead measures answer different questions:

- **Correction / Faiss** compares independently timed correction work with
  independently timed Faiss search and is the cleanest estimate of the
  incremental computation.
- **Paired E2E overhead** alternates baseline and corrected runs to reduce drift,
  but remains sensitive to operating-system and multi-thread scheduling noise.

Small non-zero or negative Top0 paired deltas are measurement noise and do not
represent acceleration from a no-op correction.

Detailed outputs are stored under
[`results/retrieval_aware_residual_basis/live_faiss_benchmark/`](results/retrieval_aware_residual_basis/live_faiss_benchmark/).

## Paper-Ready Result Tables

The repository includes a reproducible table-generation pipeline:

```bash
python scripts/build_rars_paper_tables.py
```

It generates CSV and LaTeX tables under `results/paper_tables/`:

```text
paper_main_table.*
paper_rars_cross_setting_table.*
paper_pca_transfer_table.*
paper_system_table.*
paper_ablation_table.*
paper_significance_table.*
paper_storage_table.*
```

The generated tables deliberately preserve unavailable metrics as blank rather
than mixing incompatible protocols or imputing values. They also separate:

- RARS cross-setting results from the legacy PCA-only transfer package;
- frozen-index retrofit methods from higher-rate indexes that require
  re-encoding;
- residual representation bytes from complete deployable artifact bytes;
- independent correction cost from paired end-to-end overhead.

The main paper interpretation is:

- RARS is strongest on the held-out MS MARCO 1M setting.
- Top20 captures nearly all Top40 gain at lower serving cost.
- PCA remains competitive on FiQA BGE-small.
- MiniLM transfer is mixed and alpha-sensitive.
- RARS has a positive point estimate over PCA on MS MARCO, but the paired
  bootstrap confidence intervals cross zero.
- Higher-rate `M=48` remains stronger when a full rebuild and re-encoding are
  operationally acceptable.


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
- **Retrieval-Aware Residual Subspace (RARS):** replacing the PCA residual basis with a score-error weighted basis improves the same frozen `M=32` sidecar setting from Recall@10 `0.6914` to `0.6999` under the same rank-16 int8 Top-40 correction budget. The paired-bootstrap difference over PCA is positive but narrowly crosses zero at the 95% level.
- **FiQA RARS transfer:** on FiQA / BEIR with BGE-small, score-error weighted RARS Top40 improves frozen IVF-PQ from Recall@10 `0.2935` to `0.3235`, MRR@10 `0.2964` to `0.3220`, and nDCG@10 `0.2373` to `0.2587`. On MiniLM, proxy score-error alignment improves strongly, while qrels gains are smaller and alpha-sensitive: the best proxy-selected boundary basis improves Recall@10 `0.3446 → 0.3506` and nDCG@10 `0.2862 → 0.2880`, with MRR@10 essentially flat. The transfer result is therefore framed as model-sensitive validation rather than universal RARS superiority.
- **Query-adaptive RARS diagnostics:** fixed Top20 correction reaches Recall@10 `0.6989`, within `0.0010` of Top40 while halving the correction depth. Oracle Top0/Top20/Top40 routing reaches Recall@10 `0.7103` with only `1.1` corrected candidates/query, showing headroom, but 5-fold learned routers do not recover it: the best learned router reaches Recall@10 `0.6813` with `4.56` corrected candidates/query, below fixed Top20. This is retained as a negative diagnostic result.
- **Cross-setting frozen-sidecar transfer:** the same rank-16 int8 Top-40 protocol produces positive Recall@10 point estimates on FiQA × BGE-small (`0.3287 → 0.3418`) and FiQA × MiniLM (`0.3358 → 0.3454`) without retuning. Their 95% paired-bootstrap intervals cross zero, so the transfer result is directional consistency rather than a universal significance claim.
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

### Optional RARS sidecar correction

The service also exposes an optional RARS / PQ-residual sidecar correction path
when a compatible sidecar artifact is configured in `service_config.json`.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What is a dividend stock?",
    "top_k":5,
    "candidate_k":100,
    "nprobe":16,
    "sidecar":true,
    "sidecar_top_b":20
  }'
```

The sidecar path performs:

```text
query embedding
→ IVF-PQ candidate retrieval
→ Top-B residual sidecar correction
→ corrected reranking
→ final Top-K response
```

The returned results include `ann_score`, `sidecar_correction`, and
`corrected_score`. The `/health` endpoint reports sidecar readiness metadata.
If `sidecar=true` is requested without a configured artifact, the service
returns a clear runtime error. The current deployable default remains fixed
Top20 correction, because learned query-adaptive routing did not beat fixed
Top20 in the recorded diagnostics.

This API path is documented in
[`docs/rars_sidecar_serving.md`](docs/rars_sidecar_serving.md).

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
  msmarco_rars_sidecar_m32_rank16/
docker/
  entrypoint.sh
docs/
  api_benchmark.md
  benchmark_methodology.md
  docker_api.md
  rars_sidecar_serving.md
  residual_pq_scale_limitations.md
  retrieval_api.md
  selective_residual_pq_protocol.md
  testing_ci.md
figures/
notebooks/
results/
  api_benchmark/
  compact_residual_pq_sidecar/
  fixed_budget_residual_pq/
  fiqa_gpu_benchmark/
  msmarco_low_rate_pareto/
  paper_tables/
  pq_residual_sidecar_cross_setting/
  retrieval_aware_residual_basis/
    fiqa_bge_small_transfer/
    fiqa_minilm_transfer/
    learned_rars_router/
    live_faiss_benchmark/
    sidecar_artifact_benchmark/
    sidecar_serving/
  rerank_fiqa_benchmark/
scripts/
  benchmark_api.py
  benchmark_rars_live_faiss.py
  benchmark_rars_sidecar_artifact.py
  build_rars_paper_tables.py
  export_rars_sidecar_artifact.py
  export_service_artifacts.py
  prepare_fiqa_documents.py
tests/
  test_api.py
  test_artifact_contract.py
  test_reranker.py
  test_retriever.py
  test_sidecar.py
  test_sidecar_api_contract.py
Dockerfile
docker-compose.yml
environment.yml
environment-ci.yml
requirements-api.txt
requirements-dev.txt
requirements-ci.txt
```

The tree above is intentionally abbreviated. Large embedding memmaps, transient
candidate caches, generated document metadata, and other reproducible heavy
artifacts are excluded from Git.

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


### Retrieval-Aware Residual Subspace: MS MARCO 1M

1. Open `notebooks/MSMARCO_1M_Retrieval_Aware_Residual_Basis.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime and mount the Google Drive cache used by the frozen MS MARCO 1M sidecar experiment.
3. Load the frozen `M=32`, `nlist=512`, `nprobe=16` IVF-PQ index, the 1M BGE-small document embedding memmap, held-out query split, qids, and qrels.
4. Reconstruct the current 1M candidate cache from the frozen index; do not reuse earlier Gate1 candidate caches from incompatible index/corpus states.
5. Train the PCA, RARS-Score, and RARS-Boundary residual bases under the same rank-16 sidecar budget.
6. Build int8 coefficients, run candidate-score proxy diagnostics, qrels evaluation, alpha and Top-B ablations, and paired-bootstrap comparisons.
7. Run query-adaptive correction diagnostics: Gate 0 same-split oracle analysis, Gate 1 sorted split diagnostic, and Gate 1b random 5-fold threshold routing.
8. Export the summary artifacts into `results/retrieval_aware_residual_basis/` and regenerate the package manifest after removing large memmaps, sidecar code arrays, residual caches, and candidate caches.


### Retrieval-Aware Residual Subspace: FiQA transfer

1. Open `notebooks/FiQA_BGE_Small_RARS_Transfer.ipynb` or `notebooks/FiQA_MiniLM_RARS_Transfer.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime and mount Google Drive.
3. Download / load FiQA, encode document and query embeddings with the selected model, and build the frozen `M=32`, `nlist=256`, `nprobe=16` IVF-PQ index.
4. Build the Top-100 candidate cache from the current FiQA index.
5. Reconstruct residuals, train `pca_current`, `score_error_weighted`, and `top10_boundary_weighted` bases under the same rank-16 int8 sidecar form.
6. Run proxy diagnostics, alpha sweep, Top-B ablation, fixed-alpha qrels metrics, and extended proxy-selected-alpha qrels metrics.
7. Export compact CSV summaries into:
   - `results/retrieval_aware_residual_basis/fiqa_bge_small_transfer/`
   - `results/retrieval_aware_residual_basis/fiqa_minilm_transfer/`


### Frozen IVF-PQ PQ-residual sidecar: cross-setting transfer

1. Open `notebooks/FiQA_BGE_Small_PQ_Residual_Sidecar_Transfer.ipynb` in Google Colab.
2. Enable an NVIDIA GPU runtime and run the notebook from top to bottom.
3. The notebook encodes FiQA with `BAAI/bge-small-en-v1.5`, builds a frozen
   `M=32` IVF-PQ index, and evaluates the fixed rank-16 int8 Top-40 sidecar
   protocol without FiQA-specific hyperparameter retuning.
4. Run `notebooks/FiQA_MiniLM_PQ_Residual_Sidecar_Transfer.ipynb` with the
   same fixed protocol for `sentence-transformers/all-MiniLM-L6-v2`.
5. Export the result tables, figures, per-setting metadata, and package
   README into `results/pq_residual_sidecar_cross_setting/`, then regenerate
   the SHA-256 manifest after finalizing the files.



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


### RARS artifact-backed benchmark

```bash
python scripts/benchmark_rars_sidecar_artifact.py
```

This validates the serialized sidecar against the committed held-out candidate
cache and regenerates:

```text
results/retrieval_aware_residual_basis/sidecar_artifact_benchmark/
```

### Live Faiss RARS benchmark

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

This requires the aligned local benchmark inputs and regenerates:

```text
results/retrieval_aware_residual_basis/live_faiss_benchmark/
```

The script verifies live/cached candidate alignment and numerical equivalence
between loop and vectorized correction before writing results.

### Paper-ready tables

```bash
python scripts/build_rars_paper_tables.py
```

This regenerates all CSV and LaTeX outputs under `results/paper_tables/` from
committed benchmark artifacts.

### Tests

```bash
python -m pytest -q
```

The current recorded suite contains 19 passing tests covering retrieval,
artifact contracts, optional reranking, sidecar loading, API contracts, and
fake-retriever request behavior.


## Limitations and Next Steps

- The strongest RARS result is the held-out MS MARCO 1M setting. FiQA transfer
  is model-sensitive: BGE-small improves, while MiniLM fixed-transfer results
  are nearly flat and some ranking metrics decline.
- RARS-Score has a positive point estimate over PCA on MS MARCO, but the
  paired-bootstrap 95% confidence intervals cross zero. The repository does not
  claim statistically significant superiority over PCA.
- IVF-PQ `M=48` produces higher absolute quality at a comparable total
  representation budget. RARS is positioned as a frozen-index retrofit for
  environments where rebuilding and re-encoding the corpus are undesirable,
  not as a globally storage-optimal replacement for higher-rate PQ.
- The RARS representation costs `16.025 B/document`; the complete deployable
  artifact costs `24.028 B/document` when external document IDs are included.
  These values must not be conflated.
- Live-Faiss timings cover index search and residual correction. They exclude
  query encoding, HTTP transport, JSON serialization, document lookup,
  process startup, and artifact loading.
- The paired end-to-end benchmark reduces order drift but remains sensitive to
  OS scheduling and Faiss multi-thread variability. Independent correction cost
  and paired end-to-end overhead are therefore reported separately.
- The current sidecar implementation is vectorized NumPy around Faiss output,
  not a fused Faiss/C++/CUDA kernel.
- Learned query-adaptive routers do not recover the oracle routing headroom
  under the current feature set. Fixed Top20 remains the strongest validated
  deployable cost-aware setting.
- The evaluated BGE cross-encoder reranker does not improve the recorded FiQA
  subset and adds substantial CPU latency; it remains disabled by default.
- The benchmark covers English dense retrieval with MiniLM and BGE-small. It
  does not yet establish multilingual, Traditional Chinese, hybrid
  sparse-dense, multi-node, billion-vector, or production-online behavior.
- Some historical result packages use different index configurations and
  evaluation splits. Paper tables explicitly separate these protocols rather
  than combining them into a single universal comparison.

The immediate research priority is manuscript preparation rather than adding
more serving features:

1. finalize the SIGIR short-paper narrative and novelty boundary;
2. document train/validation/held-out selection protocol and eliminate any
   ambiguity about hyperparameter tuning;
3. add a compact method diagram and quality-storage-latency trade-off figure;
4. benchmark PCA and RARS under the same vectorized serving implementation;
5. consider one additional medium-to-large retrieval dataset only if it
   materially strengthens the generalization claim.

## Release Readiness

The repository now represents a complete research and engineering workflow:

```text
FiQA / SciFact compression benchmarks
→ MS MARCO 1M PQ / OPQ low-rate sweep
→ frozen IVF-PQ residual sidecar
→ Retrieval-Aware Residual Subspace (RARS)
→ deployable rank-16 int8 sidecar artifact
→ artifact-backed correctness benchmark
→ FastAPI optional sidecar serving
→ vectorized live-Faiss benchmark
→ reproducible CSV / LaTeX paper tables
→ automated tests and CI
```

Current validated headline result:

> On the held-out MS MARCO 1M evaluation, RARS Top20 improves Recall@10 from `0.66275` to `0.69892` on a frozen IVF-PQ `M=32` index. The rank-16 int8 representation costs `16.025 B/document`, and the recorded vectorized 14-thread live-Faiss correction costs `0.816 µs/query` in a 1,000-query batch.

The project is ready for research-paper drafting and reproducible artifact
release. It should still be described as a research prototype rather than a
production vector database: operational hardening, fused-kernel integration,
broader generalization, and full request-level load testing remain future work.
