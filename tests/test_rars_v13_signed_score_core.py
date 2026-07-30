from __future__ import annotations

import numpy as np
import pytest

from scripts.rars_v13_signed_score_core import (
    assign_product_codes,
    build_signed_score_statistics,
    deterministic_fold_ids,
    deterministic_query_priority,
    fit_signed_score_codebooks,
    signed_score_decision,
)


def test_v13_query_selection_and_folds_are_stable_and_new() -> None:
    qids = ["11", "12", "13", "14", "15"]
    assert deterministic_query_priority("11") == deterministic_query_priority("11")
    assert deterministic_query_priority("11") != deterministic_query_priority("12")
    assert np.array_equal(
        deterministic_fold_ids(qids), deterministic_fold_ids(qids)
    )
    with pytest.raises(ValueError, match="unique"):
        deterministic_fold_ids(["11", "11"])


def test_fixed_product_assignment_has_expected_shape() -> None:
    values = np.zeros((3, 64), dtype=np.float32)
    books = np.zeros((16, 256, 4), dtype=np.float32)
    for block in range(16):
        books[block, :, 0] = np.arange(256, dtype=np.float32)
    values[1, ::4] = 5.1
    values[2, ::4] = 200.0
    codes = assign_product_codes(values, books, batch_size=2)
    assert codes.shape == (3, 16)
    assert codes.dtype == np.uint8
    assert np.all(codes[0] == 0)
    assert np.all(codes[1] == 5)
    assert np.all(codes[2] == 200)


def _toy_statistics() -> tuple[dict[str, np.ndarray], np.ndarray]:
    queries = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    basis = np.eye(2, dtype=np.float32)
    coefficients = np.array([[2.0, 0.0], [-3.0, 0.0]], dtype=np.float32)
    codes = np.zeros((2, 1), dtype=np.uint8)
    rows = np.array([[0, 1], [0, 1]], dtype=np.int64)
    lookup = rows.copy()
    base = np.array([[0.9, 0.8], [0.9, 0.8]], dtype=np.float32)
    labels = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    stats, summary = build_signed_score_statistics(
        queries,
        basis,
        coefficients,
        codes,
        rows,
        lookup,
        base,
        labels,
        top_b=2,
        final_k=1,
        cutoff_boost=0.0,
        margin_temperature=0.02,
        known_positive_multiplier=1.0,
    )
    assert summary["candidate_observations"] == 4
    assert summary["block_observations"] == 4
    return stats, coefficients


def test_signed_targets_preserve_direction() -> None:
    stats, _ = _toy_statistics()
    # q=+1 observes targets +2 and -3; q=-1 observes -2 and +3.
    # Both yield q*y contributions +2/-3 +2/-3 = -2 in the RHS.
    assert stats["rhs"][0, 0, 0] == pytest.approx(-2.0)
    assert stats["normal"][0, 0, 0, 0] == pytest.approx(4.0)


def test_signed_score_fit_reduces_anchored_objective_and_clips() -> None:
    stats, coefficients = _toy_statistics()
    books = np.zeros((1, 256, 2), dtype=np.float32)
    updated, summary = fit_signed_score_codebooks(
        coefficients,
        books,
        stats,
        anchor_ratio=0.0,
        maximum_drift_fraction=0.15,
        jitter=1e-8,
    )
    assert updated.shape == books.shape
    assert summary["objective_nonincreasing"] is True
    assert summary["assignment_changes"] == 0
    assert summary["maximum_centroid_drift_fraction"] <= 0.15 + 1e-7
    assert np.array_equal(updated[:, 1:], books[:, 1:])


def _comparison(gain: float, *, improved: int = 50, harmed: int = 10) -> dict:
    return {
        "mean_difference": gain,
        "lower": gain / 2,
        "upper": gain * 1.5,
        "randomization_p_value_one_sided": 0.01,
        "improved_queries": improved,
        "harmed_queries": harmed,
    }


def test_v13_decision_requires_every_stability_gate() -> None:
    thresholds = {
        "minimum_recall_gain_over_unsupervised": 0.003,
        "minimum_recall_gain_over_pca16": 0.003,
        "minimum_recall_gain_over_base": 0.01,
        "bootstrap_lower_must_exceed": 0.0,
        "maximum_randomization_p_value": 0.05,
        "minimum_improved_queries": 30,
        "minimum_net_improved_queries": 15,
        "minimum_each_seed_gain": 0.0,
        "minimum_median_seed_gain": 0.002,
        "minimum_worst_fold_gain": 0.0,
        "minimum_candidate_gap_recovery_fraction": 0.2,
        "minimum_mrr_change": -0.002,
        "minimum_ndcg_change": -0.002,
        "maximum_centroid_drift_fraction": 0.15,
        "maximum_assignment_changes": 0,
        "go_decision": "GO",
        "stop_decision": "STOP",
    }
    result = signed_score_decision(
        primary_vs_unsupervised=_comparison(0.004),
        primary_vs_pca16=_comparison(0.004),
        primary_vs_base=_comparison(0.02),
        seed_gains=[0.003, 0.004, 0.005],
        fold_gains=[0.001] * 5,
        candidate_gap_recovery=0.3,
        unsupervised_mrr=0.1,
        challenger_mrr=0.101,
        unsupervised_ndcg=0.2,
        challenger_ndcg=0.201,
        payload_bytes_per_document=16,
        full_corpus_codes_materialized=True,
        all_objectives_nonincreasing=True,
        maximum_centroid_drift_fraction=0.15,
        assignment_changes=0,
        thresholds=thresholds,
    )
    assert result["decision"] == "GO"
    stopped = signed_score_decision(
        primary_vs_unsupervised=_comparison(0.004),
        primary_vs_pca16=_comparison(0.004),
        primary_vs_base=_comparison(0.02),
        seed_gains=[0.003, -0.001, 0.005],
        fold_gains=[0.001] * 5,
        candidate_gap_recovery=0.3,
        unsupervised_mrr=0.1,
        challenger_mrr=0.101,
        unsupervised_ndcg=0.2,
        challenger_ndcg=0.201,
        payload_bytes_per_document=16,
        full_corpus_codes_materialized=True,
        all_objectives_nonincreasing=True,
        maximum_centroid_drift_fraction=0.15,
        assignment_changes=0,
        thresholds=thresholds,
    )
    assert stopped["decision"] == "STOP"
    assert "all_seed_gains_nonnegative" in stopped["failed_gates"]
