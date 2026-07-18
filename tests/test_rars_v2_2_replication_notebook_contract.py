from __future__ import annotations

import ast
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


def _imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", maxsplit=1)[0])
    return modules


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
    setup = _cell_source(notebook, "v22r-setup")
    assert "'numpy==1.26.4'" in setup
    assert "installed_numpy_version == '1.26.4'" in setup
    assert "EXPERIMENT_PYTHON = sys.executable" in setup
    assert "NUMPY_TARGET = Path('/content/rars-v2.2-numpy126')" in setup
    assert "'--target', str(NUMPY_TARGET), 'numpy==1.26.4'" in setup
    assert "EXPERIMENT_ENV['PYTHONPATH']" in setup
    assert "env=EXPERIMENT_ENV" in setup
    assert "probe_experiment_environment" in setup
    assert "RARS_V22_EXPERIMENT_ENV=" in setup
    assert "Fresh experiment-subprocess NumPy" in setup
    assert "Restart the Colab runtime" not in setup
    assert "sys.modules['numpy']" not in setup
    assert setup.index("'numpy==1.26.4'") < setup.index("from google.colab import drive")
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        assert not {"faiss", "numpy", "torch"} & _imported_modules(
            "".join(cell["source"])
        )


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
        "fresh_child_process",
        "base_plus_target_numpy_override",
        "python_executable",
        "numpy_module_path",
    ):
        assert field in audit
    assert "input_audit_manifest.json" in audit
    assert "replication_environment.json" not in audit
    setup = _cell_source(notebook, "v22r-setup")
    assert "replication_environment.json" in setup
    preflight = _cell_source(notebook, "v22r-preflight")
    assert "current_training_environment == seed42_started['environment']" in preflight
    assert "probe_experiment_environment()" in preflight
    assert "gpu_name_must_contain" in preflight
    assert "compute_capability" in preflight
    assert "cudnn_benchmark" in preflight
    setup = _cell_source(notebook, "v22r-setup")
    assert "Deterministic CUDA matmul preflight failed" in setup


def test_replication_notebook_routes_numerical_work_to_pinned_subprocess() -> None:
    notebook = _notebook()
    expected_cells = {
        "v22r-checkout": "EXPERIMENT_PYTHON, '-m', 'pytest'",
        "v22r-build-freeze": "EXPERIMENT_PYTHON, str(TRAIN_REPO",
        "v22r-run-heldout": "EXPERIMENT_PYTHON, str(TRAIN_REPO",
        "v22r-aggregate": "EXPERIMENT_PYTHON",
    }
    for cell_id, marker in expected_cells.items():
        source = _cell_source(notebook, cell_id)
        assert marker in source
        assert "env=EXPERIMENT_ENV" in source
    build = _cell_source(notebook, "v22r-build-freeze")
    assert build.count("env=EXPERIMENT_ENV") == 2
    audit = _cell_source(notebook, "v22r-audit-env")
    assert "[EXPERIMENT_PYTHON, '-m', 'pip', 'freeze']" in audit
    assert "'pip', 'freeze', '--path', str(NUMPY_TARGET)" in audit
    assert "text=True, env=EXPERIMENT_ENV" in audit
    assert "target_lines == ['numpy==1.26.4']" in audit
    assert "count('numpy==1.26.4') == 1" in audit
    assert "audit_experiment_environment == current_experiment_environment" in audit
