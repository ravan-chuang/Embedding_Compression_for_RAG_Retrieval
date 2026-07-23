from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rars_v15_selective_gate_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v15_selective_gate_core", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_query_features_are_label_free_finite_and_deterministic() -> None:
    rows = np.tile(np.arange(12, dtype=np.int64), (3, 1))
    base = np.tile(np.linspace(1.0, 0.0, 12, dtype=np.float64), (3, 1))
    sidecar = base.copy()
    # Force row 10 into the head so the fixture genuinely exercises disagreement.
    sidecar[0, 10] = 2.0
    sidecar[1, :4] += np.asarray([0.01, -0.02, 0.03, -0.04])
    first = MODULE.query_gate_features(base, sidecar, rows, final_k=5, top_b=10)
    second = MODULE.query_gate_features(base, sidecar, rows, final_k=5, top_b=10)
    assert first.shape == (3, len(MODULE.FEATURE_NAMES))
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    assert first[0, MODULE.FEATURE_NAMES.index("top10_disagreement_fraction")] > 0


def test_weighted_gate_fits_and_scores_a_signed_utility_signal() -> None:
    x = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    y = np.asarray([-1.0, -1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    model = MODULE.fit_weighted_ridge_gate(
        x,
        y,
        ridge=0.1,
        neutral_weight=0.05,
        harm_weight=2.0,
    )
    scores = MODULE.gate_utility_scores(x, model)
    assert scores[0] < scores[-1]
    assert model["harm_rows"] == 2
    assert model["positive_rows"] == 2


def test_threshold_selection_uses_calibration_metrics_and_can_fallback() -> None:
    scores = np.asarray([-2.0, -1.0, 1.0, 2.0])
    base = {
        "recall": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "mrr": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "ndcg": np.asarray([1.0, 1.0, 0.0, 0.0]),
    }
    sidecar = {
        "recall": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "mrr": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "ndcg": np.asarray([0.0, 0.0, 1.0, 1.0]),
    }
    selected = MODULE.select_calibrated_threshold(
        scores,
        base,
        sidecar,
        quantile_grid=[0.0, 0.5, 1.0],
        minimum_coverage=0.25,
        maximum_coverage=0.75,
        minimum_mrr_change=0.0,
        minimum_ndcg_change=0.0,
    )
    assert selected["fallback_always_on"] is False
    assert selected["coverage"] == 0.5
    assert np.array_equal(MODULE.apply_query_gate(scores, selected["threshold"]), [False, False, True, True])

    fallback = MODULE.select_calibrated_threshold(
        scores,
        sidecar,
        sidecar,
        quantile_grid=[0.0, 0.5, 1.0],
        minimum_coverage=0.25,
        maximum_coverage=0.75,
        minimum_mrr_change=0.0,
        minimum_ndcg_change=0.0,
    )
    assert fallback["fallback_always_on"] is True
    assert np.all(MODULE.apply_query_gate(scores, fallback["threshold"]))


def test_decision_requires_every_statistical_stability_and_storage_gate() -> None:
    comparison = {
        "mean_difference": 0.004,
        "lower": 0.001,
        "randomization_p_value_one_sided": 0.01,
    }
    thresholds = {
        "minimum_recall_gain_over_uniform_rpq": 0.003,
        "bootstrap_lower_must_exceed": 0.0,
        "maximum_randomization_p_value": 0.05,
        "minimum_recall_gain_over_base": 0.01,
        "minimum_improved_queries": 30,
        "maximum_harmed_queries": 15,
        "minimum_net_improved_queries": 15,
        "minimum_each_seed_gain": 0.0,
        "minimum_median_seed_gain": 0.002,
        "minimum_worst_fold_gain": 0.0,
        "minimum_mrr_change": -0.001,
        "minimum_ndcg_change": -0.001,
        "minimum_primary_coverage": 0.2,
        "maximum_primary_coverage": 0.95,
        "maximum_global_gate_bytes": 4096,
        "go_decision": "GO",
        "stop_decision": "STOP",
    }
    decision = MODULE.selective_gate_decision(
        primary_vs_uniform=comparison,
        primary_vs_base={**comparison, "mean_difference": 0.02},
        seed_gains=[0.003, 0.004, 0.005],
        fold_gains=[0.001] * 5,
        uniform_mrr=0.1,
        selective_mrr=0.1,
        uniform_ndcg=0.1,
        selective_ndcg=0.1,
        applied_coverages=[0.7, 0.6, 0.8],
        improved_queries=40,
        harmed_queries=10,
        parent_payload_bytes_per_document=16,
        extra_document_bytes=0,
        global_model_bytes=1024,
        thresholds=thresholds,
    )
    assert decision["decision"] == "GO"
    assert decision["all_gates_passed"] is True
    thresholds["minimum_median_seed_gain"] = 0.01
    stopped = MODULE.selective_gate_decision(
        primary_vs_uniform=comparison,
        primary_vs_base={**comparison, "mean_difference": 0.02},
        seed_gains=[0.003, 0.004, 0.005],
        fold_gains=[0.001] * 5,
        uniform_mrr=0.1,
        selective_mrr=0.1,
        uniform_ndcg=0.1,
        selective_ndcg=0.1,
        applied_coverages=[0.7, 0.6, 0.8],
        improved_queries=40,
        harmed_queries=10,
        parent_payload_bytes_per_document=16,
        extra_document_bytes=0,
        global_model_bytes=1024,
        thresholds=thresholds,
    )
    assert stopped["decision"] == "STOP"
    assert stopped["failed_gates"] == ["median_seed_gain"]
