from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "RARS_V17_Million_Scale_Setting_Transfer_Diagnostic.ipynb"
)
GENERATOR = ROOT / "scripts" / "generate_rars_v17_million_scale_notebook.py"


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )


def test_v17_notebook_is_valid_and_source_generated() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert len(payload["cells"]) == 13
    assert GENERATOR.is_file()
    assert all(cell["outputs"] == [] for cell in payload["cells"] if cell["cell_type"] == "code")


def test_v17_notebook_uses_only_million_scale_settings() -> None:
    source = _source()
    assert "msmarco_1m_bge_opened_development" in source
    assert "beir_nq_2_68m_bge_opened_test_diagnostic" in source
    assert 'nq_preparation["document_count"] >= 2_000_000' in source
    assert "fiqa_bge_same_encoder" not in source
    assert "scifact_bge_same_encoder" not in source
    assert "--msmarco-fit" in source
    assert "--nq-fit" in source


def test_v17_notebook_preserves_evidence_boundary() -> None:
    source = _source()
    assert "prior_confirmation_outcomes_known" in source
    assert "confirmatory_claim_allowed" in source
    assert "NQ official test role was" in source
    assert "opened-development" not in source
    assert "closed test" not in source.casefold()
    assert "MSMARCO_PARENT_COMMIT" in source
    assert "V17_IMPLEMENTATION_COMMIT" in source
