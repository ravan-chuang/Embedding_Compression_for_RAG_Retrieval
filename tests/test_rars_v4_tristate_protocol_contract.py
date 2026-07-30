from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rars_v4_tristate_action_feasibility_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v4_protocol_is_frozen_outcome_informed_and_parent_bound() -> None:
    protocol = _protocol()
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_TRISTATE_LABEL_AUDIT"
    assert protocol["method_rationale_is_outcome_informed"] is True
    assert protocol["method_revision_allowed"] is False
    assert protocol["outcome_informed_revision_allowed"] is False
    lineage = protocol["parent_lineage"]
    assert lineage["v3_implementation_commit"] == (
        "05c2ae43b7d11783460822d10c590240dab1a399"
    )
    assert lineage["v3_observed_formal_decision"] == "STOP_NO_HEADROOM"
    for field in ("v3_protocol_sha256", "v3_core_sha256"):
        assert len(lineage[field]) == 64
        int(lineage[field], 16)


def test_v4_judgment_contract_never_conflates_unjudged_and_negative() -> None:
    contract = _protocol()["judgment_contract"]
    assert contract == {
        **contract,
        "POSITIVE": 1,
        "EXPLICIT_NEGATIVE": -1,
        "UNJUDGED": 0,
        "binary_candidate_relevance_is_primary_eligible": False,
        "unjudged_as_negative_primary_eligible": False,
        "sensitivity_can_unlock_next_stage": False,
    }
    assert "no source judgment row" in contract["unjudged_rule"]


def test_v4_keeps_future_role_identity_only_and_claims_development_only() -> None:
    protocol = _protocol()
    future = protocol["data_policy"]["roles"]["future_fp32_holdout"]
    assert future["query_count"] == 803
    assert future["candidate_arrays_allowed_in_phase0"] is False
    assert future["labels_or_support_counts_allowed_in_phase0"] is False
    assert future["metrics_allowed_in_phase0"] is False
    assert len(future["source_order_newline_qid_sha256"]) == 64
    assert "NOT_INDEPENDENT_CONFIRMATION" in protocol["data_policy"][
        "evidence_status"
    ]


def test_v4_action_spaces_separate_bytes_from_exact_action_count() -> None:
    spaces = _protocol()["frozen_action_spaces"]
    assert spaces["progressive_primary"]["tiers_code_bytes"] == [0, 8, 16, 32]
    assert spaces["progressive_primary"]["budget_bytes_per_query"] == 640
    assert spaces["progressive_primary"]["persistent_storage_claim_allowed"] is False
    assert spaces["selective_exact_diagnostic"]["maximum_exact_actions_per_query"] == 40
    assert "not bytes" in spaces["selective_exact_diagnostic"]["cost_unit"]


def test_v4_go_cannot_start_training_qat_or_external_evaluation() -> None:
    guards = _protocol()["later_stage_guards"]
    assert guards["training_allowed"] is False
    assert guards["qat_allowed"] is False
    assert guards["external_evaluation_allowed"] is False
    assert guards["go_is_method_success"] is False
    assert guards["future_holdout_open_allowed"] is False
    policy = _protocol()["artifact_policy"]
    for field in ("label_materializer", "core", "evaluator"):
        assert (ROOT / policy[field]).is_file()


def test_v4_novelty_guard_does_not_rebrand_existing_pair_loss() -> None:
    guard = _protocol()["novelty_guard"]
    assert guard["protect_promote_penalize_is_claimed_as_a_new_loss"] is False
    assert guard["binary_unjudged_as_negative_is_existing_v2_2_behavior"] is True
    assert "full v4 without unary negative penalty" in guard[
        "required_future_ablations"
    ]

