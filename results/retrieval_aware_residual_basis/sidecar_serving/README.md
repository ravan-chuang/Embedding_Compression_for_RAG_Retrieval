# RARS Sidecar Serving Foundation

This package records the serving foundation and first FastAPI API-path
integration for optional Retrieval-Aware Residual Subspace (RARS) /
PQ-residual sidecar correction.

The integration exposes the contract needed for deployment, while full
artifact-backed local serving and latency-quality benchmarking remain the next
milestone.

## Scope

This milestone adds:

- deployable sidecar artifact exporter
- sidecar artifact config contract
- sidecar loader and validator
- Top-B residual score correction
- single-query and batch reranking APIs
- optional FastAPI `/search` and `/batch-search` sidecar request path
- API health metadata for sidecar readiness
- unit tests for correction, reranking behavior, and API contract behavior
- serving documentation

This milestone does **not** yet commit a production MS MARCO sidecar artifact,
nor does it report a full artifact-backed serving benchmark.

## Added files

| File | Purpose |
|---|---|
| `app/sidecar.py` | Loads a sidecar artifact and applies Top-B residual correction |
| `scripts/export_rars_sidecar_artifact.py` | Packages notebook outputs into a deployable artifact |
| `docs/rars_sidecar_serving.md` | Documents artifact layout, serving path, API request format, and benchmark protocol |
| `tests/test_sidecar.py` | Unit tests for loading, correction, reranking, and batch reranking |
| `tests/test_sidecar_api_contract.py` | API contract tests for sidecar request fields and health metadata |
| `results/retrieval_aware_residual_basis/sidecar_serving/README.md` | This milestone summary |

## Current validated behavior

The sidecar unit test suite validates:

- sidecar artifact loading
- shape and dtype validation
- document-id to row-id mapping
- correction computation
- Top-B-only correction behavior
- single-query reranking
- batch reranking

Test command:

```bash
python -m pytest tests/test_sidecar.py -q
```

Expected result:

```text
5 passed
```

The API contract tests validate:

- `/health` exposes sidecar metadata
- `/search` accepts `sidecar` and `sidecar_top_b`
- `/batch-search` accepts `sidecar` and `sidecar_top_b`
- invalid `sidecar_top_b` is rejected by schema validation
- unloaded services fail at runtime readiness, not request parsing

API contract test command:

```bash
python -m pytest tests/test_sidecar.py tests/test_sidecar_api_contract.py -q
```

Expected result:

```text
9 passed
```

## Intended operating points

The serving layer is designed to support the already validated RARS fixed-depth
settings:

| Mode | Corrected candidates/query | Recall@10 | Notes |
|---|---:|---:|---|
| IVF-PQ only | 0 | 0.6628 | Frozen `M=32` baseline |
| RARS Top20 | 20 | 0.6989 | Strongest cost-aware point |
| RARS Top40 | 40 | 0.6999 | Best observed fixed-depth point |

The deployable default remains fixed Top20 because learned query-adaptive
routing did not beat fixed Top20 in the current diagnostics.

## API integration

The first FastAPI integration path is now present:

```text
/search or /batch-search
→ IVF-PQ Top100 candidates
→ optional RARS Top20 / Top40 sidecar correction
→ corrected Top-K response
```

The path is controlled by request fields:

```text
sidecar: bool
sidecar_top_b: int | null
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a dividend stock?",
    "top_k": 5,
    "candidate_k": 100,
    "nprobe": 16,
    "sidecar": true,
    "sidecar_top_b": 20
  }'
```

When `sidecar=true`, the service applies Top-B sidecar correction before final
Top-K selection. The response includes sidecar metadata such as:

```text
sidecar_enabled
sidecar_top_b
sidecar_actual_top_b
sidecar_alpha
sidecar_latency_ms
```

Each corrected result can include:

```text
score
ann_score
sidecar_correction
corrected_score
```

If `sidecar=true` is requested but no sidecar artifact is configured or loaded,
the service returns a clear runtime error instead of silently falling back to
uncorrected retrieval.

The `/health` endpoint exposes:

```text
sidecar_enabled
sidecar_artifact_dir
sidecar_default_top_b
sidecar_max_top_b
```

## Artifact configuration contract

A future artifact-backed serving run should add a `sidecar` block to
`service_config.json`, for example:

```json
{
  "sidecar": {
    "enabled": true,
    "artifact_dir": "../msmarco_rars_sidecar_m32_rank16",
    "config_file": "sidecar_config.json"
  }
}
```

Relative artifact paths are resolved relative to the retrieval artifact
directory.

## Next step

The next milestone is artifact-backed local serving and benchmarking, comparing:

- IVF-PQ only
- RARS Top20
- RARS Top40

with:

- Recall@10
- MRR@10
- nDCG@10
- P50 latency
- P95 latency
- sidecar latency
- memory overhead

The target serving claim should remain conservative:

```text
RARS Top20 retains almost all Top40 quality while halving correction depth.
```

This package should not be used to claim that learned query-adaptive routing is
solved. The learned-router diagnostics remain a negative result: oracle routing
has headroom, but current handcrafted features do not reliably beat fixed Top20.
