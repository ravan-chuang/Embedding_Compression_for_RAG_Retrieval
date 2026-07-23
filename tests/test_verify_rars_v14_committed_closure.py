from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_rars_v14_committed_closure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v14_committed_closure", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v14_committed_closure_recomputes_the_frozen_stop_decision() -> None:
    module = _load_module()
    summary = module.verify_closure(
        ROOT / "results/rars_v14_anisotropic_rate_rpq", ROOT
    )
    assert summary["status"] == "RARS_V14_COMMITTED_CLOSURE_VERIFIED"
    assert summary["source_commit"] == (
        "a3895e8d2ba298b155ac7f866453af134fd3c222"
    )
    assert summary["formal_decision"] == "STOP_V14_NO_ANISOTROPIC_RATE_SIGNAL"
    assert summary["query_count"] == 5000
    assert summary["primary_gain"] == -0.0003
    assert summary["primary_ci95"] == [-0.0044, 0.0038]
    assert summary["seed_gains"] == [-0.0003, 0.0003, 0.0027]
    assert summary["multi_seed_consensus"] == {
        "improved_in_at_least_two_seeds": 30,
        "harmed_in_at_least_two_seeds": 25,
        "improved_in_all_three_seeds": 2,
        "harmed_in_all_three_seeds": 3,
    }
    assert summary["verified_committed_outputs"] == 40


def test_clean_notebook_prefers_checkout_without_git_history() -> None:
    module = _load_module()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / module.CLEAN_NOTEBOOK
        path.parent.mkdir(parents=True)
        path.write_text('{"cells": []}\n', encoding="utf-8")
        with mock.patch.object(
            module.subprocess,
            "check_output",
            side_effect=AssertionError("git history must not be read"),
        ):
            assert module._load_clean_notebook(root, "0" * 40) == {"cells": []}
