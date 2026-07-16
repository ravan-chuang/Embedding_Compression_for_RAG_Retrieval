from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/MSMARCO_RARS_v2_2_FP32_Replication.ipynb"
)
TRAINING_COMMIT = "bb9b106e69b9a453756fd800665f701614ce67b3"
CONTROL_COMMIT = "00a0dee30767b04b8c650c28d63f4f662ef61517"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell_source(notebook: dict, cell_id: str) -> str:
    return "".join(next(
        cell["source"] for cell in notebook["cells"] if cell["id"] == cell_id
    ))


def test_replication_notebook_is_clean_ordered_and_dual_pinned() -> None:
    notebook = _notebook()
    cells = notebook["cells"]
    assert len({cell["id"] for cell in cells}) == len(cells)
    assert all(
        cell.get("execution_count") is None
        for cell in cells
        if cell["cell_type"] == "code"
    )
    assert all(
        not cell.get("outputs") for cell in cells if cell["cell_type"] == "code"
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "markdown"
    )
    assert markdown.index("## tl;dr") < markdown.index("## Context & Methods")
    assert markdown.index("## Context & Methods") < markdown.index("## Data")
    assert markdown.index("## Data") < markdown.index("## Results")
    assert markdown.index("## Results") < markdown.index("## Takeaways")
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert TRAINING_COMMIT in source
    assert CONTROL_COMMIT in source
    checkout = _cell_source(notebook, "v22r-checkout")
    assert "tests/test_aggregate_rars_v2_2_fp32_replication.py" in checkout
    assert "'-m', 'pytest', '-q']" not in checkout


def test_replication_notebook_trains_only_sealed_heldout_seeds() -> None:
    notebook = _notebook()
    train = _cell_source(notebook, "v22r-run-heldout")
    assert "HELDOUT_SEEDS = (43, 44)" in train
    assert "for seed in HELDOUT_SEEDS" in train
    assert "capture_output=True" in train
    assert "interim_metrics_revealed': False" in train
    assert "'--seed', '42'" not in train
    assert "SEED_DIRS[42]" not in train
    assert "seed42-fp32-stage-a" not in train
    assert "break" not in train
    assert "train_boundary_loss_sidecar_v2_2.py" in train
    assert "--reuse-complete" in train
    assert "query_gate" not in train
    assert "batch_attempt_id" in train
    assert "preserved partial attempt" in train
    assert "BATCH_HISTORY" in train


def test_replication_notebook_rebuilds_only_inner_and_requires_exact_hashes() -> None:
    notebook = _notebook()
    build = _cell_source(notebook, "v22r-build-freeze")
    assert "'--inner-only'" in build
    assert "--cached-full-residuals" not in build
    assert "outer_validation_built'] is False" in build
    assert "exact_inner_train_manifest_sha256_required" in build
    assert "exact_inner_validation_manifest_sha256_required" in build
    assert "exact_split_audit_sha256_required" in build
    assert "closed_test_relevance_values_used'] is False" in build
    assert "outer_outcomes_used'] is False" in build


def test_replication_notebook_preserves_seed42_and_uses_frozen_aggregator() -> None:
    notebook = _notebook()
    preflight = _cell_source(notebook, "v22r-preflight")
    aggregate = _cell_source(notebook, "v22r-aggregate")
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "Audited seed 42 verified without rerunning or modifying it" in preflight
    assert "aggregate_rars_v2_2_fp32_replication.py" in aggregate
    assert "--replication-protocol" in aggregate
    assert "--environment-manifest" in aggregate
    assert "replication_runner_manifest.json" in aggregate
    assert "BATCH_PATH" in aggregate
    assert "selection_pca_fp32_per_query" not in preflight
    assert "QAT" not in source
    assert "--validation-bundle-dir" not in source


def test_replication_notebook_persists_environment_and_input_audit() -> None:
    notebook = _notebook()
    audit = _cell_source(notebook, "v22r-audit-env")
    for field in (
        "aggregator_sha256",
        "metric_helper_sha256",
        "faiss_version",
        "gpu_name",
        "compute_capability",
        "cudnn_version",
        "cuda_driver_version",
        "deterministic_algorithms_supported_and_enabled",
        "cublas_workspace_config",
        "pip_freeze_sha256",
        "protocol_sha256",
    ):
        assert field in audit
    assert "input_audit_manifest.json" in audit
    assert "replication_environment.json" not in audit
    setup = _cell_source(notebook, "v22r-setup")
    assert "replication_environment.json" in setup
    preflight = _cell_source(notebook, "v22r-preflight")
    assert "current_training_environment == seed42_started['environment']" in preflight
    assert "gpu_name_must_contain" in preflight
    assert "compute_capability" in preflight
    assert "cudnn.benchmark" in preflight
    assert "Deterministic CUDA matmul preflight failed" in preflight
