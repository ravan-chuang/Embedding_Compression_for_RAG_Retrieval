# RARS Sidecar Serving

This document describes the deployable serving path for the Retrieval-Aware
Residual Subspace (RARS) / PQ-residual sidecar.

The sidecar is a frozen-index retrofit layer. It does not retrain the IVF-PQ
index and does not rewrite existing PQ codes. Instead, it attaches a compact
low-rank residual correction payload and reranks only the highest-ranked ANN
candidates.

## Goal

The serving path evaluates whether the notebook-level RARS result can be moved
toward a deployable retrieval system:

```text
query
→ embedding
→ frozen IVF-PQ ANN search
→ Top-B sidecar correction
→ corrected reranking
→ final Top-K results
```

The main validated operating points are:

| Mode | Corrected candidates/query | Recall@10 | Notes |
|---|---:|---:|---|
| IVF-PQ only | 0 | 0.6628 | Frozen `M=32` baseline |
| RARS Top20 | 20 | 0.6989 | Strongest cost-aware point |
| RARS Top40 | 40 | 0.6999 | Best observed fixed-depth point |

Top20 captures nearly all of the Top40 quality while halving correction depth.

## Score correction

For query embedding `q`, ANN candidate document `x`, frozen IVF-PQ score
`s_IVFPQ(q, x)`, shared sidecar basis `B`, and document coefficient vector
`a_x`, the corrected score is:

```text
s_corr(q, x)
= s_IVFPQ(q, x) + alpha · q^T B a_x
```

Only the first `Top-B` ANN candidates are corrected. Candidates below that
depth keep their original ANN scores.

## Artifact layout

A deployable sidecar artifact contains:

```text
sidecar_config.json
basis.npy
scales.npy
codes.int8.npy
doc_ids.npy
manifest.json
```

### Files

| File | Description |
|---|---|
| `sidecar_config.json` | Artifact metadata and serving configuration |
| `basis.npy` | Shared residual basis, shape `[dim, rank]`, float32 |
| `scales.npy` | Per-sidecar-dimension dequantization scales, shape `[rank]`, float32 |
| `codes.int8.npy` | Per-document int8 sidecar coefficients, shape `[num_docs, rank]` |
| `doc_ids.npy` | External document ids aligned with corpus-internal row ids |
| `manifest.json` | Shape, dtype, byte size, and SHA-256 metadata |

The current RARS serving artifact is expected to use:

```text
dim = 384
rank = 16
code dtype = int8
default_top_b = 20
max_top_b = 40
```

## Exporting an artifact

Use:

```bash
python scripts/export_rars_sidecar_artifact.py \
  --basis /path/to/basis.npy \
  --scales /path/to/scales.npy \
  --codes /path/to/codes.int8.npy \
  --doc-ids /path/to/doc_ids.npy \
  --output-dir artifacts/msmarco_rars_sidecar_m32_rank16 \
  --alpha 0.75 \
  --default-top-b 20 \
  --max-top-b 40 \
  --force
```

The exporter validates array shapes and dtypes, writes a serving config, and
produces a SHA-256 manifest.

## Python serving API

The artifact is loaded through `app.sidecar.RARSSidecar`:

```python
from app.sidecar import RARSSidecar

sidecar = RARSSidecar("artifacts/msmarco_rars_sidecar_m32_rank16")

result = sidecar.rerank(
    query_embedding=query_embedding,
    candidate_rows=candidate_rows,
    ann_scores=ann_scores,
    top_k=10,
    top_b=20,
)
```

The returned dictionary contains:

```text
candidate_rows
doc_ids
ann_scores
corrected_scores
corrections
rerank_order
top_k
top_b
actual_top_b
alpha
sidecar_enabled
```

## Candidate id contract

`RARSSidecar.rerank()` expects `candidate_rows`, meaning corpus-internal row ids.
These row ids must directly index `codes.int8.npy`.

If the retrieval path returns external document ids instead, map them first:

```python
candidate_rows = sidecar.rows_from_doc_ids(candidate_doc_ids)
```

This distinction is important because ANN libraries often expose internal row
ids, while API responses usually expose external document ids.

## Batch reranking

`RARSSidecar.rerank_batch()` applies the same correction independently to a
batch of queries:

```python
results = sidecar.rerank_batch(
    query_embeddings=query_embeddings,
    candidate_rows=candidate_rows,
    ann_scores=ann_scores,
    top_k=10,
    top_b=20,
)
```

The initial implementation favors correctness and easy integration over fused
GPU execution. It is suitable for local API integration and benchmark
instrumentation, not yet a fused production CUDA kernel.

## Benchmark protocol

The serving benchmark should report both retrieval quality and API/runtime cost.

Recommended settings:

| Setting | Description |
|---|---|
| IVF-PQ only | No sidecar correction |
| RARS Top20 | Correct first 20 ANN candidates |
| RARS Top40 | Correct first 40 ANN candidates |

Recommended metrics:

| Metric | Reason |
|---|---|
| Recall@10 | Main retrieval quality metric |
| MRR@10 | Ranking quality near the top |
| nDCG@10 | Graded ranking quality |
| P50 latency | Typical request cost |
| P95 latency | Tail behavior |
| Avg corrected candidates/query | Sidecar compute budget |
| Artifact memory overhead | Deployment cost |

The target serving claim is conservative:

```text
RARS Top20 retains almost all Top40 quality while halving correction depth.
```

## Current limitation

This serving path does not claim that learned query-adaptive routing is solved.
The learned router diagnostics show that oracle routing has headroom, but
current handcrafted features do not reliably beat fixed Top20 correction.

Therefore, the deployable default remains fixed Top20.

## Not a replacement for higher-rate PQ

The sidecar is a frozen-index retrofit mechanism. It can recover part of the
ranking loss of an already deployed low-rate IVF-PQ index, but it is not claimed
to beat a newly rebuilt higher-rate PQ index under equal storage.

The correct positioning is:

```text
frozen-index enhancement / retrofit path
```

not:

```text
globally storage-optimal PQ replacement
```
