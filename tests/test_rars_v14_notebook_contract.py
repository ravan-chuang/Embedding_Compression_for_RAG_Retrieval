from __future__ import annotations

import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MSMARCO_RARS_v14_Anisotropic_Rate_RPQ_Diagnostic.ipynb"
GENERATOR = ROOT / "scripts/generate_rars_v14_anisotropic_rate_notebook.py"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"v14-cell-{index}", "exec")
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_v14_notebook_is_source_pinned_and_uses_isolated_numpy() -> None:
    source = _source()
    assert "SOURCE_HASHES" in source
    assert "clone_exact" in source
    assert "--without-pip" in source
    assert "--system-site-packages" in source
    assert "numpy==1.26.4" in source
    assert "faiss-gpu-cu12==1.12.0" in source


def test_v14_notebook_is_exact_generator_output() -> None:
    spec = importlib.util.spec_from_file_location("v14_notebook_generator", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert json.loads(NOTEBOOK.read_text()) == module.build()


def test_v14_notebook_runs_parent_audit_tests_evaluator_and_verifier() -> None:
    source = _source()
    assert "verify_rars_v13_committed_closure.py" in source
    assert "test_rars_v14_anisotropic_rate_core.py" in source
    assert "evaluate_rars_v14_anisotropic_rate_rpq.py" in source
    assert "verify_rars_v14_anisotropic_rate_rpq_packet.py" in source
    assert "full_corpus_qw_ar_rpq_codes.uint8.memmap" in source


def test_v14_notebook_discloses_diagnostic_boundary() -> None:
    source = _source()
    assert "outcome-informed architecture diagnostic" in source.lower()
    assert "not fresh evidence or confirmation" in source.lower()
    assert "fresh_query_access_authorized" in source
