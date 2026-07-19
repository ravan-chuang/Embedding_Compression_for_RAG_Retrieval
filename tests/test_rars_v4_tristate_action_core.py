from __future__ import annotations

import numpy as np

from scripts import rars_v4_tristate_action_core as MODULE


def test_tristate_contract_rejects_binary_dtype_and_unknown_values() -> None:
    counts = np.asarray([1], dtype=np.int32)
    try:
        MODULE.validate_tristate_labels(
            np.asarray([[1, 0]], dtype=np.uint8), counts
        )
    except ValueError as error:
        assert "int8" in str(error)
    else:
        raise AssertionError("Binary uint8 labels were accepted as tri-state")
    try:
        MODULE.validate_tristate_labels(
            np.asarray([[1, 2]], dtype=np.int8), counts
        )
    except ValueError as error:
        assert "{-1, 0, +1}" in str(error)
    else:
        raise AssertionError("Unknown judgment state was accepted")


def test_label_support_never_counts_unjudged_as_explicit_negative() -> None:
    scores = np.asarray(
        [[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]], np.float32
    )
    docids = np.asarray([[1, 2, 3, 4], [11, 12, 13, 14]], np.int64)
    labels = np.asarray(
        [[-1, 0, 1, 0], [1, -1, 0, 0]], dtype=np.int8
    )
    counts = np.asarray([1, 1], dtype=np.int32)
    result = MODULE.label_support_diagnostics(
        scores,
        docids,
        labels,
        counts,
        final_k=1,
        correction_depth=3,
    )
    assert result.summary["coverage"]["penalty_topk_explicit_negative"]["count"] == 1
    assert result.summary["coverage"]["promotion_pair"]["count"] == 1
    assert result.summary["coverage"]["protection_pair"]["count"] == 1
    assert result.summary["state_counts"]["candidate_100"] == {
        "positive": 2,
        "explicit_negative": 2,
        "unjudged": 4,
        "total": 8,
    }
    np.testing.assert_allclose(result.label_swap_gain, [1.0, 0.0])


def test_action_reachability_separates_label_support_from_score_actions() -> None:
    base = np.asarray([[0.9, 0.8, 0.7, 0.6]], np.float32)
    docids = np.asarray([[1, 2, 3, 4]], np.int64)
    labels = np.asarray([[-1, 0, 1, 0]], np.int8)
    counts = np.asarray([1], np.int32)
    support = MODULE.label_support_diagnostics(
        base,
        docids,
        labels,
        counts,
        final_k=1,
        correction_depth=3,
    )
    unreachable = np.stack([base, base], axis=1)
    result = MODULE.action_reachability_diagnostics(
        unreachable, docids, support, correction_depth=3
    )
    assert result.summary["joint_swap_reachable"]["count"] == 0

    reachable = unreachable.copy()
    reachable[0, 1, 0] = 0.1
    reachable[0, 1, 2] = 1.0
    result = MODULE.action_reachability_diagnostics(
        reachable, docids, support, correction_depth=3
    )
    assert result.summary["downward_feasible"]["count"] == 1
    assert result.summary["upward_feasible"]["count"] == 1
    assert result.summary["joint_swap_reachable"]["count"] == 1


def test_exact_triage_oracle_promotes_positive_and_demotes_explicit_negative() -> None:
    base = np.asarray([[0.9, 0.8, 0.7, 0.6]], np.float32)
    action = np.asarray([[0.1, 0.8, 1.0, 0.6]], np.float32)
    tiers = np.stack([base, action], axis=1)
    docids = np.asarray([[1, 2, 3, 4]], np.int64)
    labels = np.asarray([[-1, 0, 1, 0]], np.int8)
    counts = np.asarray([1], np.int32)
    result = MODULE.exact_triage_action_oracle(
        tiers,
        (0, 1),
        labels,
        docids,
        counts,
        final_k=1,
        correction_depth=3,
        budget=2,
    )
    assert result.positive_hits_at_k.tolist() == [1]
    assert result.explicit_negative_hits_at_k.tolist() == [0]
    assert result.recall_at_k.tolist() == [1.0]
    assert result.action_cost[0] <= 2
    assert result.topk_membership.tolist() == [[False, False, True, False]]


def test_triage_oracle_secondary_objective_avoids_explicit_negative() -> None:
    base = np.asarray([[0.9, 0.8, 0.1, 0.0]], np.float32)
    action = np.asarray([[0.9, 1.0, 0.1, 0.0]], np.float32)
    tiers = np.stack([base, action], axis=1)
    docids = np.asarray([[1, 2, 3, 4]], np.int64)
    labels = np.asarray([[-1, 0, 1, 0]], np.int8)
    counts = np.asarray([1], np.int32)
    result = MODULE.exact_triage_action_oracle(
        tiers,
        (0, 1),
        labels,
        docids,
        counts,
        final_k=1,
        correction_depth=3,
        budget=1,
    )
    assert result.positive_hits_at_k.tolist() == [0]
    assert result.explicit_negative_hits_at_k.tolist() == [0]
    assert result.topk_membership.tolist() == [[False, True, False, False]]


def test_pre_action_gate_stops_positive_only_sources_before_oracle() -> None:
    support = {
        "query_count": 2307,
        "coverage": {},
        "label_swap_ceiling": {},
    }
    result = MODULE.pre_action_decision(
        role_id=MODULE.DESIGN_ROLE_ID,
        explicit_negative_semantics_preserved=False,
        support_summary=support,
        label_bootstrap={},
        thresholds={"minimum_design_queries": 2000, "minimum_audit_queries": 800},
    )
    assert result["decision"] == "STOP_NO_EXPLICIT_NEGATIVE_SEMANTICS"
    assert result["action_oracle_authorized"] is False


def test_wilson_interval_is_bounded_and_uses_all_queries() -> None:
    result = MODULE.wilson_interval(10, 100)
    assert result["count"] == 10
    assert result["total"] == 100
    assert 0.0 < result["lower"] < result["fraction"] < result["upper"] < 1.0

