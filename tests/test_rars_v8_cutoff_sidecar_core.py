from __future__ import annotations

import numpy as np
import pytest

from scripts import rars_v8_cutoff_sidecar_core as MODULE


def _toy_candidates() -> dict[str, np.ndarray]:
    # Base order is the stored order.  Query 0 has a relevant item at rank 3
    # that the exact teacher promotes above rank 1.  Query 1 protects rank 1.
    rows = np.asarray([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=np.int64)
    lookup = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    base = np.asarray(
        [[0.90, 0.80, 0.70, 0.60], [0.90, 0.80, 0.70, 0.60]],
        dtype=np.float32,
    )
    exact = np.asarray(
        [[0.70, 0.69, 0.95, 0.50], [0.92, 0.75, 0.70, 0.60]],
        dtype=np.float32,
    )
    labels = np.asarray([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.uint8)
    return {"rows": rows, "lookup": lookup, "base": base, "exact": exact, "labels": labels}


def test_query_role_weights_equalise_role_mass_and_queries() -> None:
    query = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    kind = np.asarray(
        [MODULE.PROMOTION, MODULE.PROMOTION, MODULE.PROMOTION,
         MODULE.PROTECTION, MODULE.PROTECTION, MODULE.PROTECTION],
        dtype=np.uint8,
    )
    raw = np.asarray([1, 3, 2, 1, 1, 2], dtype=np.float32)
    weights = MODULE.query_role_balanced_weights(
        query, kind, raw, promotion_mass=0.5
    )
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[kind == MODULE.PROMOTION].sum(), 0.5)
    assert np.isclose(weights[kind == MODULE.PROTECTION].sum(), 0.5)
    assert np.isclose(weights[(kind == MODULE.PROMOTION) & (query == 0)].sum(), 0.25)
    assert np.isclose(weights[(kind == MODULE.PROMOTION) & (query == 1)].sum(), 0.25)


def test_cutoff_miner_uses_positive_vs_unjudged_semantics() -> None:
    arrays = _toy_candidates()
    pairs = MODULE.mine_cutoff_pairs(
        arrays["rows"], arrays["lookup"], arrays["base"], arrays["exact"],
        arrays["labels"], final_k=2, top_b=4, protection_window=2,
        max_challengers_per_positive=1,
    )
    assert len(pairs) == 2
    promotion = np.flatnonzero(pairs.kind == MODULE.PROMOTION)
    protection = np.flatnonzero(pairs.kind == MODULE.PROTECTION)
    assert promotion.tolist() == [0]
    assert protection.tolist() == [1]
    assert pairs.positive_position[promotion[0]] == 2
    assert pairs.challenger_position[promotion[0]] == 0
    assert pairs.positive_position[protection[0]] == 0
    assert pairs.teacher_margin[promotion[0]] > 0
    assert np.isclose(
        pairs.target_residual_margin[promotion[0]],
        pairs.teacher_margin[promotion[0]] - pairs.base_margin[promotion[0]],
    )


def test_cutoff_basis_is_deterministic_orthonormal_and_reduces_pair_error() -> None:
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    residuals = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [0.8, 0.0], [0.0, 0.0],
         [0.0, 0.8], [0.0, 0.0]],
        dtype=np.float32,
    )
    records = dict(
        query=np.asarray([0, 1], dtype=np.int64),
        positive_position=np.asarray([0, 0], dtype=np.int64),
        challenger_position=np.asarray([1, 1], dtype=np.int64),
        positive_residual_row=np.asarray([2, 4], dtype=np.int64),
        challenger_residual_row=np.asarray([3, 5], dtype=np.int64),
        base_margin=np.asarray([-0.8, -0.8], dtype=np.float32),
        teacher_margin=np.asarray([0.0, 0.0], dtype=np.float32),
        target_residual_margin=np.asarray([0.8, 0.8], dtype=np.float32),
        raw_weight=np.ones(2, dtype=np.float32),
        balanced_weight=np.asarray([0.5, 0.5], dtype=np.float32),
        kind=np.asarray([MODULE.PROMOTION, MODULE.PROTECTION], dtype=np.uint8),
    )
    # Dataclass correctly rejects non-positive teacher margins.
    with pytest.raises(ValueError, match="Teacher margins"):
        MODULE.CutoffPairBatch(**records)
    records["teacher_margin"] = np.asarray([0.1, 0.1], dtype=np.float32)
    records["base_margin"] = np.asarray([-0.7, -0.7], dtype=np.float32)
    pairs = MODULE.CutoffPairBatch(**records)
    anchor = np.asarray([[1.0], [0.0]], dtype=np.float32)

    fitted_a, history_a = MODULE.fit_cutoff_aware_basis(
        queries, residuals, pairs, anchor, steps=80, learning_rate=0.01,
        anchor_weight=0.0,
    )
    fitted_b, history_b = MODULE.fit_cutoff_aware_basis(
        queries, residuals, pairs, anchor, steps=80, learning_rate=0.01,
        anchor_weight=0.0,
    )
    assert np.array_equal(fitted_a, fitted_b)
    assert history_a == history_b
    assert np.allclose(fitted_a.T @ fitted_a, np.eye(1), atol=2e-4)
    assert history_a[-1]["pair_huber_loss"] <= history_a[0]["pair_huber_loss"]


def test_int8_sidecar_scores_only_base_top_b_and_preserves_tail() -> None:
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    rows = np.asarray([[10, 11, 12]], dtype=np.int64)
    lookup = np.asarray([[0, 1, 2]], dtype=np.int64)
    base = np.asarray([[0.3, 0.2, 0.1]], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0]], dtype=np.float32)
    residuals = np.asarray([[0.1, 0.0], [0.4, 0.0], [0.9, 0.0]], dtype=np.float32)
    scales = MODULE.fit_int8_scales(residuals, basis)
    codes, audit = MODULE.encode_residuals_int8(residuals, basis, scales)
    corrected = MODULE.score_sidecar_candidates(
        queries, rows, lookup, base, basis, codes, scales, alpha=1.0, top_b=2
    )
    assert audit["saturation_fraction"] == 0.0
    assert corrected[0, 0] > base[0, 0]
    assert corrected[0, 1] > base[0, 1]
    assert corrected[0, 2] == base[0, 2]


def test_metrics_use_known_positive_denominator_and_deterministic_ties() -> None:
    scores = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
    rows = np.asarray([[11, 10, 12]], dtype=np.int64)
    labels = np.asarray([[0, 1, 1]], dtype=np.uint8)
    metrics = MODULE.per_query_metrics(
        scores, rows, labels, np.asarray([2]), k=2
    )
    # Equal scores are broken by corpus row, so relevant row 10 is first.
    assert metrics["recall"][0] == 0.5
    assert metrics["mrr"][0] == 1.0
    assert metrics["ndcg"][0] > 0.0


def test_development_decision_separates_algorithm_and_generic_claims() -> None:
    thresholds = {
        "minimum_rars_recall_at_10_gain_over_base": 0.01,
        "minimum_rars_recall_at_10_gain_over_pca": 0.002,
        "minimum_generic_sidecar_gain_over_base": 0.01,
        "bootstrap_lower_must_exceed": 0.0,
        "minimum_candidate_gap_recovery_fraction": 0.15,
        "minimum_improved_queries": 20,
        "minimum_net_improved_queries": 5,
        "minimum_promotion_queries": 100,
        "minimum_protection_queries": 100,
        "algorithm_go_decision": "GO_TO_RARS_ALGORITHM_CONFIRMATION",
        "generic_sidecar_go_decision": "GO_TO_GENERIC_SIDECAR_CONFIRMATION",
        "stop_decision": "STOP_V8",
    }
    rars_base = {"mean_difference": 0.02, "lower": 0.01,
                 "improved_queries": 30, "harmed_queries": 10}
    pca_base = {"mean_difference": 0.018, "lower": 0.008,
                "improved_queries": 28, "harmed_queries": 10}
    rars_pca = {"mean_difference": 0.0025, "lower": -0.001,
                "improved_queries": 12, "harmed_queries": 10}
    support = {"promotion": {"queries": 120}, "protection": {"queries": 400}}
    decision = MODULE.development_decision(
        rars_vs_base=rars_base, pca_vs_base=pca_base,
        rars_vs_pca=rars_pca, gap_recovery=0.2,
        pair_support=support, thresholds=thresholds,
    )
    assert decision["decision"] == "GO_TO_GENERIC_SIDECAR_CONFIRMATION"
    assert not decision["future_access_authorized"]
