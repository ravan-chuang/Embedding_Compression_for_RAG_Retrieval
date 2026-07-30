from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/MSMARCO_RARS_v8_Cutoff_Sidecar_Development.ipynb"
IMPLEMENTATION_COMMIT = "c9d95f15d55e7700db069da69567157f2eed469e"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v8_notebook_is_clean_and_every_code_cell_compiles() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_v8_notebook_pins_exact_lineage_and_isolated_numpy() -> None:
    code = _code()
    assert f"V8_IMPLEMENTATION_COMMIT = '{IMPLEMENTATION_COMMIT}'" in code
    assert "26a7717b964eed979b3bf7a3149d0d24e9bce3f1" in code
    assert "05c2ae43b7d11783460822d10c590240dab1a399" in code
    assert "bb9b106e69b9a453756fd800665f701614ce67b3" in code
    assert "numpy==1.26.4" in code
    assert "faiss-gpu-cu12==1.12.0" in code
    assert "CUBLAS_WORKSPACE_CONFIG" in code
    assert "clone_exact(V8_REPO, V8_IMPLEMENTATION_COMMIT)" in code


def test_v8_notebook_materializes_only_design_labels() -> None:
    code = _code()
    materializer = code.index("materialize_rars_v3_role_labels.py")
    trainer = code.index("train_rars_v8_cutoff_sidecar.py")
    assert materializer < trainer
    assert "'--role', 'oracle_design'" in code
    assert "'--role', 'oracle_audit'" not in code
    assert "future_files == {'query_manifest.json', 'v3_identity_manifest.json'}" in code
    assert "--audit-role-dir" not in code
    assert "--future" not in code


def test_v8_notebook_verifies_v6_and_runs_int8_oof_development() -> None:
    code = _code()
    verification = code.index("verify_rars_v6_1m_headroom_packet.py")
    training = code.index("train_rars_v8_cutoff_sidecar.py")
    assert verification < training
    assert "RARS_V6_1M_HEADROOM_PACKET_VERIFIED" in code
    assert "GO_TO_V6_LOSS_IMPLEMENTATION" in code
    assert "development_complete.json" in code
    assert "method_freeze.json" in code
    assert "STOP_V8_CUTOFF_SIDECAR" in code


def test_v8_notebook_builds_full_sidecars_only_after_go() -> None:
    code = _code()
    assert "if result['formal_decision'] in go_decisions:" in code
    assert "build_rars_v8_frozen_sidecars.py" in code
    assert "sidecar_complete['index_before'] == sidecar_complete['index_after']" in code
    assert "sidecar_complete['qrels_opened'] is False" in code
    assert "sidecar_complete['future_method_holdout_opened'] is False" in code
    assert "STOP decision: full-corpus encoding correctly skipped." in code
