from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_rars_v13_committed_closure.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v13_committed_closure", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v13_committed_closure_recomputes_the_frozen_stop_decision() -> None:
    module = _load_module()
    summary = module.verify_closure(ROOT / "results/rars_v13_signed_score_rpq", ROOT)
    assert summary["status"] == "RARS_V13_COMMITTED_CLOSURE_VERIFIED"
    assert summary["source_commit"] == (
        "d8cb761c289fe17ea2c2bfb92059e8b5553cfd74"
    )
    assert summary["formal_decision"] == "STOP_SIGNED_SCORE_RPQ_NO_STABLE_ADVANTAGE"
    assert summary["query_count"] == 5000
    assert summary["primary_gain"] == 0.0006
    assert summary["primary_ci95"] == [-0.0006, 0.0018]
    assert summary["seed_changed_query_stability"] == {
        "changed_counts": [9, 11, 9],
        "union_count": 29,
        "pairwise_overlap_counts": [0, 0, 0],
        "three_way_overlap_count": 0,
    }
    assert summary["verified_committed_outputs"] == 37


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
