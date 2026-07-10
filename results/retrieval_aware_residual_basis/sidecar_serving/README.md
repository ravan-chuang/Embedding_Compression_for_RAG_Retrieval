# RARS Sidecar Serving Foundation

This package documents the first deployable serving foundation for the
Retrieval-Aware Residual Subspace (RARS) / PQ-residual sidecar.

## Scope

This milestone does not yet integrate the sidecar into the FastAPI retrieval
endpoint. It adds the serving foundation required for that integration:

- deployable sidecar artifact exporter
- sidecar artifact config contract
- sidecar loader and validator
- Top-B residual score correction
- single-query and batch reranking APIs
- unit tests for correction and reranking behavior
- serving documentation

## Added files

| File | Purpose |
|---|---|
| `app/sidecar.py` | Loads a sidecar artifact and applies Top-B residual correction |
| `scripts/export_rars_sidecar_artifact.py` | Packages notebook outputs into a deployable artifact |
| `docs/rars_sidecar_serving.md` | Documents artifact layout, serving path, and benchmark protocol |
| `tests/test_sidecar.py` | Unit tests for loading, correction, reranking, and batch reranking |

## Current validated behavior

The unit test suite validates:

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

## Next step

The next milestone is FastAPI integration:

```text
/search
→ IVF-PQ Top100 candidates
→ optional RARS Top20 / Top40 sidecar correction
→ corrected Top-K response
```

This should be followed by a serving benchmark comparing:

- IVF-PQ only
- RARS Top20
- RARS Top40

with Recall@10, MRR@10, nDCG@10, P50 latency, P95 latency, and memory overhead.
