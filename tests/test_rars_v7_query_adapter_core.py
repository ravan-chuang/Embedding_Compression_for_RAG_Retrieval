from __future__ import annotations

import numpy as np

from scripts import rars_v7_query_adapter_core as MODULE


def test_label_blind_query_split_is_exact_disjoint_and_stable() -> None:
    qids = [str(value) for value in range(30)]
    left = MODULE.deterministic_query_split(qids, selection_count=6, salt="fixed")
    right = MODULE.deterministic_query_split(qids, selection_count=6, salt="fixed")
    assert len(left.training) == 24
    assert len(left.selection) == 6
    assert not set(left.training).intersection(left.selection)
    assert np.array_equal(left.training, right.training)
    assert left.selection_qids == right.selection_qids


def test_query_balancing_gives_each_query_unit_weight_per_pair_type() -> None:
    query = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.int64)
    kind = np.asarray(
        [MODULE.PROMOTION, MODULE.PROMOTION, MODULE.PROMOTION,
         MODULE.PROTECTION, MODULE.PROTECTION, MODULE.PROTECTION],
        dtype=np.uint8,
    )
    raw = np.asarray([1, 3, 7, 2, 1, 3], dtype=np.float32)
    balanced = MODULE.query_balanced_weights(query, kind, raw)
    for pair_kind in (MODULE.PROMOTION, MODULE.PROTECTION):
        for query_index in np.unique(query[kind == pair_kind]):
            mask = (kind == pair_kind) & (query == query_index)
            assert np.isclose(balanced[mask].sum(), 1.0)


def test_top10_protection_uses_only_known_positive_and_unjudged_challenger() -> None:
    base = np.asarray([[10, 20, 30, 40, 50, 60]], dtype=np.int64)
    candidates = np.asarray([[10, 20, 30, 40, 50, 60, 70]], dtype=np.int64)
    exact = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]], dtype=np.float32)
    pq = np.asarray([[0.7, 0.8, 0.75, 0.6, 0.5, 0.4, 0.3]], dtype=np.float32)
    positives = np.asarray([[10]], dtype=np.int64)
    valid = np.asarray([[True]])
    pairs = MODULE.mine_top10_protection_pairs(
        base,
        candidates,
        exact,
        pq,
        positives,
        valid,
        negative_window=3,
        max_challengers_per_positive=2,
    )
    assert len(pairs) == 0  # no rank-11 challenger exists in this small Top-6 fixture

    wide_base = np.arange(100, 130, dtype=np.int64)[None, :]
    wide_candidates = np.arange(100, 131, dtype=np.int64)[None, :]
    wide_exact = np.linspace(1.0, 0.1, 31, dtype=np.float32)[None, :]
    wide_pq = wide_exact.copy()
    protected = MODULE.mine_top10_protection_pairs(
        wide_base,
        wide_candidates,
        wide_exact,
        wide_pq,
        np.asarray([[100]], dtype=np.int64),
        valid,
        negative_window=16,
        max_challengers_per_positive=4,
    )
    assert len(protected) == 4
    assert np.all(protected.positive_row == 100)
    assert set(protected.challenger_row).isdisjoint({100})
    assert np.all(protected.kind == MODULE.PROTECTION)


def test_checkpoint_selection_applies_guards_before_recall_order() -> None:
    history = [
        {
            "epoch": 0,
            "hard_pq_recall_at_10": 0.70,
            "hard_pq_recall_at_100": 0.84,
            "adapted_same_ivf_fp32_recall_at_100": 0.89,
            "mean_query_cosine": 1.0,
        },
        {
            "epoch": 1,
            "hard_pq_recall_at_10": 0.69,
            "hard_pq_recall_at_100": 0.90,
            "adapted_same_ivf_fp32_recall_at_100": 0.89,
            "mean_query_cosine": 0.99,
        },
        {
            "epoch": 2,
            "hard_pq_recall_at_10": 0.70,
            "hard_pq_recall_at_100": 0.85,
            "adapted_same_ivf_fp32_recall_at_100": 0.89,
            "mean_query_cosine": 0.999,
        },
    ]
    selected = MODULE.select_checkpoint(
        history,
        base_r10=0.70,
        teacher_r100=0.89,
        maximum_r10_drop=0.0025,
        maximum_teacher_drop=0.0025,
    )
    assert selected == 2


def test_v7_gate_requires_nonidentity_effect_guardrails_and_support() -> None:
    thresholds = {
        "minimum_hard_pq_recall_at_100_gain": 0.005,
        "paired_bootstrap_95_lower_must_exceed": 0.0,
        "minimum_same_route_teacher_gap_recovery_fraction": 0.15,
        "minimum_improved_selection_queries": 5,
        "minimum_net_improved_selection_queries": 3,
        "maximum_hard_pq_recall_at_10_drop": 0.0025,
        "maximum_adapted_same_route_fp32_recall_at_100_drop": 0.0025,
        "minimum_mean_query_cosine": 0.995,
        "go_decision": "GO_TO_V7_DEVELOPMENT_AUDIT",
        "stop_decision": "STOP_V7_QUERY_ADAPTER_PILOT",
    }
    decision = MODULE.pilot_gate_decision(
        selected_epoch=2,
        base_r10=0.68,
        adapted_r10=0.68,
        base_r100=0.84,
        adapted_r100=0.85,
        teacher_r100=0.89,
        adapted_teacher_r100=0.89,
        bootstrap_lower=0.001,
        improved_queries=8,
        harmed_queries=2,
        mean_query_cosine=0.999,
        thresholds=thresholds,
    )
    assert decision["decision"] == "GO_TO_V7_DEVELOPMENT_AUDIT"
    assert decision["future_access_authorized"] is False
    assert decision["rars_combination_authorized"] is False

