from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MSMARCO_RARS_v13_Signed_Score_RPQ_Development.ipynb"


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_v13_notebook_is_clean_source_pinned_and_one_shot() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = _code(notebook)
    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)
    assert "SOURCE_HASHES" in code
    assert "V13_IMPLEMENTATION_COMMIT" in code
    assert "freeze_rars_v13_fresh_queries.py" in code
    assert "build_rars_v13_fresh_bundle.py" in code
    assert "train_rars_v13_signed_score_rpq.py" in code
    assert "verify_rars_v13_signed_score_rpq_packet.py" in code
    assert "results/rars_v12_ca_rpq/development/query_ids.utf8.txt" in code
    assert "--target-query-count', '5000'" in code
    assert "full_corpus_signed_score_assignments.uint8.memmap" in code
    assert "fresh_confirmation_access_authorized" in code
    assert "oracle_audit" not in code
    assert "future_method_holdout" not in code

    generator = ROOT / "scripts/generate_rars_v13_signed_score_notebook.py"
    spec = importlib.util.spec_from_file_location("v13_notebook_generator", generator)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert NOTEBOOK.read_text(encoding="utf-8") == (
        json.dumps(module.build(), indent=1) + "\n"
    )


def test_v13_embedded_source_hashes_match_repository() -> None:
    generator_path = ROOT / "scripts/generate_rars_v13_signed_score_notebook.py"
    spec = importlib.util.spec_from_file_location("v13_hash_generator", generator_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in module.PINNED_SOURCES
    }
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for relative, digest in expected.items():
        assert f'"{relative}": "{digest}"' in setup
