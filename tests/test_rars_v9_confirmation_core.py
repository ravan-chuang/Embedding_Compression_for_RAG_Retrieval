from __future__ import annotations

import numpy as np
import pytest

from scripts import rars_v9_confirmation_core as MODULE


THRESHOLDS = {
    "minimum_rars_recall_at_10_gain_over_base": 0.01,
    "minimum_rars_recall_at_10_gain_over_pca": 0.005,
    "minimum_generic_sidecar_gain_over_base": 0.01,
    "bootstrap_lower_must_exceed": 0.0,
    "maximum_primary_randomization_p_value": 0.025,
    "minimum_candidate_gap_recovery_fraction": 0.15,
    "minimum_improved_queries_over_base": 20,
    "minimum_net_improved_queries_over_base": 10,
    "minimum_improved_queries_over_pca": 15,
    "minimum_net_improved_queries_over_pca": 8,
    "algorithm_confirmation_decision": "CONFIRM_RARS_V8_ALGORITHM_WITHIN_PROGRAM",
    "generic_sidecar_confirmation_decision": "CONFIRM_GENERIC_FROZEN_SIDECAR_WITHIN_PROGRAM",
    "stop_decision": "STOP_RARS_V8_AFTER_LOCKED_CONFIRMATION",
}


def test_metrics_use_stable_row_ties_and_all_positive_denominators() -> None:
    rows = np.asarray([[12, 10, 11, 13], [20, 21, 22, 23]])
    scores = np.asarray([[0.8, 0.9, 0.9, 0.1], [0.7, 0.6, 0.5, 0.4]])
    positive_rows = np.asarray([[10, 13], [22, -1]])
    positive_valid = np.asarray([[True, True], [True, False]])
    result = MODULE.per_query_metrics(
        scores, rows, positive_rows, positive_valid, k=2
    )
    # Row 10 precedes row 11 under an exact score tie.
    assert result["recall"].tolist() == [0.5, 0.0]
    assert result["success"].tolist() == [1.0, 0.0]
    assert result["mrr"].tolist() == [1.0, 0.0]
    assert 0.0 < result["ndcg"][0] <= 1.0


def test_paired_inference_is_deterministic_and_tracks_query_support() -> None:
    baseline = np.zeros(80, dtype=np.float64)
    treatment = np.r_[np.ones(50), np.zeros(20), -np.ones(10)]
    first = MODULE.comparison(
        treatment,
        baseline,
        bootstrap_replicates=2000,
        randomization_replicates=5000,
        seed=42,
    )
    second = MODULE.comparison(
        treatment,
        baseline,
        bootstrap_replicates=2000,
        randomization_replicates=5000,
        seed=42,
    )
    assert first == second
    assert first["mean_difference"] == 0.5
    assert first["improved_queries"] == 50
    assert first["harmed_queries"] == 10
    assert first["lower"] > 0
    assert first["randomization_p_value_one_sided"] < 0.025


def test_candidate_gap_recovery_validates_inputs() -> None:
    value = MODULE.candidate_gap_recovery(
        [0.7, 0.8], [0.6, 0.7], [0.8, 0.9]
    )
    assert np.isclose(value, 0.5)
    with pytest.raises(ValueError, match="matching and finite"):
        MODULE.candidate_gap_recovery([0.7], [0.6, 0.7], [0.9])


def _comparison(gain: float, improved: int, harmed: int, *, p: float = 0.001):
    return {
        "mean_difference": gain,
        "lower": gain / 2,
        "upper": gain * 1.5,
        "improved_queries": improved,
        "harmed_queries": harmed,
        "unchanged_queries": 803 - improved - harmed,
        "randomization_p_value_one_sided": p,
    }


def test_decision_separates_algorithm_generic_and_stop_claims() -> None:
    algorithm = MODULE.confirmation_decision(
        rars_vs_base=_comparison(0.02, 35, 10),
        pca_vs_base=_comparison(0.012, 25, 8),
        rars_vs_pca=_comparison(0.008, 21, 9),
        gap_recovery=0.20,
        thresholds=THRESHOLDS,
    )
    assert algorithm["decision"] == "CONFIRM_RARS_V8_ALGORITHM_WITHIN_PROGRAM"
    assert algorithm["selected_path"] == "algorithm"
    assert algorithm["failed_gates_for_selected_or_primary_path"] == []
    assert algorithm["method_or_threshold_tuning_authorized"] is False

    generic = MODULE.confirmation_decision(
        rars_vs_base=_comparison(0.02, 35, 10),
        pca_vs_base=_comparison(0.012, 25, 8),
        rars_vs_pca=_comparison(0.003, 12, 8, p=0.2),
        gap_recovery=0.20,
        thresholds=THRESHOLDS,
    )
    assert generic["decision"] == "CONFIRM_GENERIC_FROZEN_SIDECAR_WITHIN_PROGRAM"
    assert generic["selected_path"] == "generic_sidecar"

    stopped = MODULE.confirmation_decision(
        rars_vs_base=_comparison(0.004, 8, 7, p=0.2),
        pca_vs_base=_comparison(0.002, 7, 7, p=0.2),
        rars_vs_pca=_comparison(0.002, 6, 6, p=0.2),
        gap_recovery=0.05,
        thresholds=THRESHOLDS,
    )
    assert stopped["decision"] == "STOP_RARS_V8_AFTER_LOCKED_CONFIRMATION"
    assert stopped["selected_path"] == "stop"
    assert stopped["failed_gates_for_selected_or_primary_path"]
