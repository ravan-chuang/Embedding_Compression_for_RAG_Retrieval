from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/MSMARCO_RARS_v7_Query_Adapter_Pilot.ipynb"
IMPLEMENTATION_COMMIT = "303c31b7ac3264a0af386d04d1f00b87385f056e"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v7_notebook_is_clean_and_every_code_cell_compiles() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_v7_notebook_pins_exact_lineage_and_environment() -> None:
    code = _code()
    assert f"V7_IMPLEMENTATION_COMMIT = '{IMPLEMENTATION_COMMIT}'" in code
    assert "26a7717b964eed979b3bf7a3149d0d24e9bce3f1" in code
    assert "05c2ae43b7d11783460822d10c590240dab1a399" in code
    assert "bb9b106e69b9a453756fd800665f701614ce67b3" in code
    assert "numpy==1.26.4" in code
    assert "faiss-gpu-cu12==1.12.0" in code
    assert "CUBLAS_WORKSPACE_CONFIG" in code


def test_v7_notebook_verifies_v6_before_training() -> None:
    code = _code()
    verification = code.index("verify_rars_v6_1m_headroom_packet.py")
    training = code.index("train_rars_v7_query_adapter.py")
    assert verification < training
    assert "RARS_V6_1M_HEADROOM_PACKET_VERIFIED" in code
    assert "GO_TO_V6_LOSS_IMPLEMENTATION" in code


def test_v7_notebook_materializes_no_audit_or_future_labels() -> None:
    code = _code()
    assert "materialize_rars_v3_role_labels.py" not in code
    assert "V3_BUNDLES / 'oracle_design'" in code
    assert "--audit-role-dir" not in code
    assert "future_files == {'query_manifest.json', 'v3_identity_manifest.json'}" in code


def test_v7_notebook_guards_durable_output_and_frozen_index() -> None:
    code = _code()
    assert "not OUTPUT.exists() or not any(OUTPUT.iterdir())" in code
    assert "complete['index_before'] == complete['index_after']" in code
    assert "complete['document_reencoding_performed'] is False" in code
    assert "complete['rars_used'] is False" in code
    assert "complete['oracle_audit_opened'] is False" in code
    assert "complete['future_method_holdout_opened'] is False" in code

