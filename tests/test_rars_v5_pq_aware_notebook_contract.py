from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/MSMARCO_RARS_v5_PQ_Aware_100K_Pilot.ipynb"
IMPLEMENTATION_COMMIT = "93105ae1895974e28b34952c2dd777f037c6e0bf"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _cells_by_id() -> dict[str, dict]:
    return {cell["id"]: cell for cell in _notebook()["cells"]}


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def test_v5_notebook_is_clean_and_python_cells_compile() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse(_source(cell), filename=f"v5-notebook-cell-{index}")


def test_v5_notebook_pins_exact_implementation_and_environment() -> None:
    setup = _source(_cells_by_id()["v5-setup"])
    checkout = _source(_cells_by_id()["v5-checkout-tests"])
    preflight = _source(_cells_by_id()["v5-preflight"])
    assert f"V5_IMPLEMENTATION_COMMIT = '{IMPLEMENTATION_COMMIT}'" in setup
    assert "numpy==1.26.4" in setup
    assert "faiss-gpu-cu12==1.12.0" in setup
    assert "CUBLAS_WORKSPACE_CONFIG" in setup
    assert "clone_exact(V5_REPO, V5_IMPLEMENTATION_COMMIT)" in checkout
    assert (
        "V3_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v3_oracle')"
        in setup
    )
    assert "Embedding_Compression_for_RAG_Retrieval_rars_v3')" not in setup
    assert "torch_version" in preflight
    assert "gpu_name_must_contain" in preflight


def test_v5_builder_never_receives_qrels_or_future_role() -> None:
    build = _source(_cells_by_id()["v5-build-100k"])
    assert "build_rars_v5_pq_aware_100k_bundle.py" in build
    assert "--qrels" not in build
    assert "--future" not in build
    assert "V3_BUNDLES / 'future_method_holdout'" not in build
    assert "--design-role-dir" in build
    assert "--audit-role-dir" in build


def test_v5_future_role_is_identity_only_and_training_uses_frozen_defaults() -> None:
    roles = _source(_cells_by_id()["v5-v3-observed-roles"])
    training = _source(_cells_by_id()["v5-train-seed42"])
    assert "future_files == {'query_manifest.json', 'v3_identity_manifest.json'}" in roles
    assert "future_method_holdout" in roles
    assert "train_rars_v5_pq_aware_adapter.py" in training
    assert "--source-commit', V5_IMPLEMENTATION_COMMIT" in training
    for override in (
        "--epochs",
        "--learning-rate",
        "--rank-margin",
        "--minimum-recall-gain",
    ):
        assert override not in training
