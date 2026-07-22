from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/MSMARCO_RARS_v10_Stable_Sidecar_Development.ipynb"


def _notebook() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v10_notebook_is_clean_compilable_and_gpu_ready() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"v10-cell-{index}", "exec")


def test_v10_notebook_uses_isolated_environment_and_source_hashes() -> None:
    code = _code()
    assert "'venv', '--without-pip', '--system-site-packages'" in code
    assert "'-m', 'pip', '--python', EXPERIMENT_PYTHON" in code
    assert "numpy==1.26.4" in code
    assert "faiss-gpu-cu12==1.12.0" in code
    assert "SOURCE_HASHES =" in code
    assert "clone_exact(V10_REPO, resolved)" in code
    assert "assert actual == expected" in code


def test_v10_notebook_materializes_only_historical_design_labels() -> None:
    code = _code()
    materializer = "materialize_rars_v3_role_labels.py"
    assert code.count(materializer) == 1
    invocation = code.index(materializer)
    tail = code[invocation : invocation + 1000]
    assert "'--role', 'oracle_design'" in tail
    assert "'--role', 'oracle_audit'" not in code
    assert "future_method_holdout" in code
    assert "identity-only" in PATH.read_text(encoding="utf-8")


def test_v10_notebook_runs_exactly_one_configuration_without_v9_inputs() -> None:
    code = _code()
    assert code.count("str(V10_REPO / 'scripts/train_rars_v10_stable_sidecar.py')") == 1
    assert "configuration_count'] == 1" in code
    assert "--v9" not in code.lower()
    assert "rars-v9-confirmation" not in code.lower()
    assert "v9_files_opened'] is False" in code
    assert "fresh_external_access_authorized'] is False" in code


def test_v10_notebook_never_authorizes_reuse_or_retuning() -> None:
    content = PATH.read_text(encoding="utf-8")
    assert "does not read any V9 file" in content
    assert "Do not tune V10 from this output" in content
    assert "A GO only permits writing a" in content
    assert "new protocol for a genuinely fresh dataset/model" in content


def test_v10_notebook_reports_avq_headroom_without_training_codebook() -> None:
    code = _code()
    assert "avq_scalar_headroom_diagnostic" in code
    assert "codebook_training_performed'] is False" in code
    assert "fit_score_aware_codebook" not in code
