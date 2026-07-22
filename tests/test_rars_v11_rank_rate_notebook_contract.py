from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/MSMARCO_RARS_v11_Rank_Rate_Diagnostic.ipynb"


def _notebook() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def _code() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v11_notebook_is_clean_compilable_and_gpu_ready() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"v11-cell-{index}", "exec")


def test_v11_notebook_uses_isolated_pinned_environment() -> None:
    code = _code()
    assert "'venv', '--without-pip', '--system-site-packages'" in code
    assert "'-m', 'pip', '--python', EXPERIMENT_PYTHON" in code
    assert "numpy==1.26.4" in code
    assert "faiss-gpu-cu12==1.12.0" in code
    assert "SOURCE_HASHES =" in code
    assert "clone_exact(V11_REPO, resolved)" in code
    assert "assert actual == expected" in code


def test_v11_notebook_materializes_only_historical_design_labels() -> None:
    code = _code()
    materializer = "materialize_rars_v3_role_labels.py"
    assert code.count(materializer) == 1
    invocation = code.index(materializer)
    tail = code[invocation : invocation + 1000]
    assert "'--role', 'oracle_design'" in tail
    assert "'--role', 'oracle_audit'" not in code
    assert "future_method_holdout" in code
    assert "identity-only" in PATH.read_text(encoding="utf-8")


def test_v11_notebook_runs_one_fixed_screen_without_old_packets() -> None:
    code = _code()
    assert code.count("str(V11_REPO / 'scripts/evaluate_rars_v11_rank_rate.py')") == 1
    assert "fixed_sidecar_screen_count'] == 7" in code
    assert "--v9" not in code.lower()
    assert "--v10" not in code.lower()
    assert "rars-v9-confirmation" not in code.lower()
    assert "rars-v10-development" not in code.lower()
    assert "cutoff_training_performed'] is False" in code


def test_v11_notebook_accepts_only_registered_hierarchical_decisions() -> None:
    code = _code()
    assert "STOP_LINEAR_RANK_EXPANSION_NO_HEADROOM" in code
    assert "STOP_RPQ_16B_CANNOT_RETAIN_HEADROOM" in code
    assert "GO_TO_SEPARATE_CA_RPQ_CUTOFF_PROTOCOL" in code
    assert "fresh_confirmation_access_authorized'] is False" in code
    assert "failed_gates" in code


def test_v11_notebook_persists_runner_logs_before_raising() -> None:
    code = _code()
    assert "RUNNER_LOGS = V11_ROOT / 'runner-logs'" in code
    assert "'diagnostic_stdout.log'" in code
    assert "'diagnostic_stderr.log'" in code
    assert "text=True, capture_output=True" in code
    assert "runner.check_returncode()" in code
