from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/MSMARCO_RARS_v12_Anchored_Cutoff_RPQ_Development.ipynb"
GENERATOR_PATH = ROOT / "scripts/generate_rars_v12_ca_rpq_notebook.py"


def _code() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_clean_and_generated_exactly() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    spec = importlib.util.spec_from_file_location("v12_notebook_generator", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.dumps(module.build(), indent=1) + "\n"
    assert NOTEBOOK_PATH.read_text() == expected


def test_notebook_pins_every_runtime_source_hash() -> None:
    code = _code()
    for relative in (
        "protocols/rars_v12_anchored_cutoff_rpq_v1.json",
        "scripts/rars_v12_ca_rpq_core.py",
        "scripts/freeze_rars_v12_fresh_queries.py",
        "scripts/build_rars_v12_fresh_bundle.py",
        "scripts/train_rars_v12_ca_rpq.py",
        "scripts/verify_rars_v12_ca_rpq_packet.py",
    ):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert relative in code
        assert digest in code


def test_notebook_freezes_fresh_queries_before_candidates_and_metrics() -> None:
    code = _code()
    freeze_position = code.index("freeze_rars_v12_fresh_queries.py")
    bundle_position = code.index("build_rars_v12_fresh_bundle.py")
    train_position = code.index("train_rars_v12_ca_rpq.py")
    assert freeze_position < bundle_position < train_position
    assert "queries.train.tsv" in code
    assert "qrels.train.tsv" in code
    assert code.count("--prior-qids") == 3
    assert "historical_qid_overlap'] == []" in code


def test_notebook_runs_verifier_and_exports_real_full_corpus_codes() -> None:
    code = _code()
    assert "verify_rars_v12_ca_rpq_packet.py" in code
    assert "full_corpus_ca_rpq_codes.uint8.memmap" in code
    assert "16000000" in code
    assert "result['v11_packet_opened'] is False" in code
    assert "result['old_holdout_opened'] is False" in code
    assert "fresh_confirmation_access_authorized'] is False" in code
