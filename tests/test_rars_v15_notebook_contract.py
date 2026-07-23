from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MSMARCO_RARS_v15_Selective_RPQ_Gate_Development.ipynb"
GENERATOR = ROOT / "scripts/generate_rars_v15_selective_gate_notebook.py"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"v15-cell-{index}", "exec")
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_v15_notebook_is_source_pinned_and_uses_isolated_numpy() -> None:
    source = _source()
    assert "SOURCE_HASHES" in source
    assert "clone_exact" in source
    assert "--without-pip" in source
    assert "--system-site-packages" in source
    assert "numpy==1.26.4" in source
    assert "faiss-gpu-cu12==1.12.0" in source


def test_v15_notebook_is_exact_generator_output() -> None:
    spec = importlib.util.spec_from_file_location("v15_notebook_generator", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert json.loads(NOTEBOOK.read_text()) == module.build()


def test_v15_notebook_runs_parent_audits_evaluator_and_verifier() -> None:
    source = _source()
    assert "verify_rars_v13_committed_closure.py" in source
    assert "verify_rars_v14_committed_closure.py" in source
    assert "evaluate_rars_v15_selective_gate.py" in source
    assert "verify_rars_v15_selective_gate_packet.py" in source
    assert "additional_document_bytes" in source


def test_v15_notebook_discloses_reused_query_boundary() -> None:
    source = _source().lower()
    assert "outcome-informed development on the already opened v13 query role" in source
    assert "not fresh evidence or confirmation" in source
    assert "fresh_query_access_authorized" in source
