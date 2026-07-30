from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/rars_v17_million_scale_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v17_million_scale_core", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _observed(**changes: float | int) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "n_queries": 2200,
        "headroom": 0.03,
        "capacity_gain": 0.0,
        "coding_gap": 0.0,
        "objective_gain": 0.0,
        "setting_interaction": 0.0,
        "pooled_recovery": 0.0,
        "pooled_gain": 0.0,
        "improved_queries": 40,
        "harmed_queries": 10,
        "gap_recovery": 0.2,
        "worst_domain_gain": 0.0,
    }
    values.update(changes)
    return values


def test_v17_decision_uses_setting_not_domain_causal_vocabulary() -> None:
    result = MODULE.setting_transfer_decision(
        _observed(
            objective_gain=0.006,
            setting_interaction=0.006,
            worst_domain_gain=-0.003,
        ),
        MODULE.DEFAULT_SETTING_THRESHOLDS,
    )
    assert result["decision"] == "SETTING_INTERACTION_SUPPORTED"
    assert result["setting_interaction_supported"] is True
    assert "domain_shift_supported" not in result


def test_v17_core_retains_deterministic_query_level_inference() -> None:
    treatment = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0])
    baseline = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0])
    controls = {
        "bootstrap_replicates": 200,
        "bootstrap_seed": 17,
        "randomization_replicates": 400,
        "randomization_seed": 18,
    }
    first = MODULE.paired_query_inference(treatment, baseline, **controls)
    second = MODULE.paired_query_inference(treatment, baseline, **controls)
    assert first == second
    assert first["improved_queries"] == 2
    assert first["harmed_queries"] == 1
