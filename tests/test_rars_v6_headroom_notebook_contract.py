from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/MSMARCO_RARS_v6_1M_Headroom.ipynb"
IMPLEMENTATION_COMMIT = "9abc24af7f2f8a6eb7a4d1416036c3151c51c924"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v6_notebook_is_clean_and_every_code_cell_compiles() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_v6_notebook_pins_exact_lineage_and_environment() -> None:
    code = _code()
    assert f"V6_IMPLEMENTATION_COMMIT = '{IMPLEMENTATION_COMMIT}'" in code
    assert "bb9b106e69b9a453756fd800665f701614ce67b3" in code
    assert "05c2ae43b7d11783460822d10c590240dab1a399" in code
    assert "numpy==1.26.4" in code
    assert "faiss-gpu-cu12==1.12.0" in code
    assert "CUBLAS_WORKSPACE_CONFIG" in code
    assert "V6_REPO, V6_IMPLEMENTATION_COMMIT" in code


def test_v6_notebook_runs_only_the_frozen_no_training_diagnostic() -> None:
    code = _code()
    assert "evaluate_rars_v6_1m_headroom.py" in code
    assert "--design-role-dir" in code
    assert "V3_BUNDLES / 'oracle_design'" in code
    assert "materialize_rars_v3_role_labels.py" not in code
    assert "train_rars_v5_pq_aware_adapter.py" not in code
    assert "train_boundary_loss_sidecar" not in code
    assert "--audit-role-dir" not in code
    assert "--query-batch-size" not in code
    assert "--candidate-batch-size" not in code


def test_v6_notebook_guards_future_identity_and_durable_output() -> None:
    code = _code()
    assert "future_files == {'query_manifest.json', 'v3_identity_manifest.json'}" in code
    assert "not OUTPUT.exists() or not any(OUTPUT.iterdir())" in code
    assert "future_or_audit_role_opened" in code
    assert "training_performed" in code
    assert "training_authorized" in code


def test_v6_notebook_verifies_complete_packet_and_formal_decision() -> None:
    code = _code()
    assert "headroom_complete.json" in code
    assert "headroom_result.json" in code
    assert "GO_TO_V6_LOSS_IMPLEMENTATION" in code
    assert "STOP_NO_DISTRIBUTED_PQ_HEADROOM" in code
    assert "protocol['required_outputs']" in code
    assert "verify_record(result_path, complete['result'])" in code

