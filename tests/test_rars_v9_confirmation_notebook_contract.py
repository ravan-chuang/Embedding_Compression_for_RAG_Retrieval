from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/MSMARCO_RARS_v9_Locked_Confirmation.ipynb"


def _notebook() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v9_notebook_is_clean_compilable_and_gpu_ready() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"v9-cell-{index}", "exec")


def test_v9_notebook_uses_isolated_numpy_and_source_hashes() -> None:
    code = _code()
    assert "'venv', '--without-pip'" in code
    assert "'-m', 'pip', '--python', EXPERIMENT_PYTHON" in code
    assert "numpy==1.26.4" in code
    assert "faiss-gpu-cu12==1.12.0" in code
    assert "SOURCE_HASHES =" in code
    assert "V9_IMPLEMENTATION_COMMIT == resolved" in code
    assert "All registered source hashes verified before any outcome access." in code


def test_v9_notebook_builds_identity_and_m48_without_qrels_first() -> None:
    code = _code()
    # Each filename also appears in the early source-hash registry.  The last
    # occurrence is the actual subprocess invocation.
    identity = code.rindex("build_rars_v9_future_identity.py")
    m48 = code.rindex("build_rars_v9_m48_baseline.py")
    evaluator = code.rindex("evaluate_rars_v9_locked_confirmation.py")
    assert identity < m48 < evaluator
    pre_evaluator = code[:evaluator]
    assert "build_msmarco_rars_v2_boundary_bundles.py" not in pre_evaluator
    assert "RARS_V9_QRELS_FREE_FUTURE_IDENTITY_COMPLETE" in pre_evaluator
    assert "RARS_V9_QRELS_FREE_M48_BASELINE_COMPLETE" in pre_evaluator


def test_v9_notebook_is_explicitly_one_shot_and_not_independent() -> None:
    content = PATH.read_text(encoding="utf-8")
    assert "within-program prospective" in content
    assert "confirmation" in content
    assert "not independent" in content
    assert "ONE-SHOT OUTCOME ACCESS; NO RETUNING OR RERUN" in content
    code = _code()
    assert "method_or_threshold_tuning_authorized" in code
    assert "independent_confirmation_claim_allowed" in code
