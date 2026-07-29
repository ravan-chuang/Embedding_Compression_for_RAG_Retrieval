from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (
        ROOT / "protocols/rars_v16_causal_generalization_diagnostic_v1.json"
    ).read_text(encoding="utf-8")
)


def test_v16_is_frozen_outcome_informed_and_nonconfirmatory() -> None:
    assert PROTOCOL["protocol_id"] == (
        "rars_v16_causal_generalization_diagnostic_v1"
    )
    assert PROTOCOL["status"] == (
        "FROZEN_BEFORE_FIRST_V16_MECHANISM_DIAGNOSTIC_RUN"
    )
    assert PROTOCOL["method_revision_allowed"] is False
    assert PROTOCOL["parent_evidence"]["post_outcome_status"] is True
    assert PROTOCOL["parent_evidence"]["hypothesis_selected_after_prior_outcomes"]
    assert PROTOCOL["evidence_boundary"]["tier"] == (
        "OUTCOME_INFORMED_PREPARED_BUNDLE_MECHANISM_DIAGNOSTIC"
    )
    assert "independent causal confirmation" in (
        PROTOCOL["evidence_boundary"]["forbidden_claims"]
    )


def test_v16_uses_exactly_two_named_same_encoder_development_domains() -> None:
    policy = PROTOCOL["data_policy"]
    assert policy["domain_count"] == 2
    assert policy["allowed_development_domains"] == [
        "fiqa_bge_same_encoder",
        "scifact_bge_same_encoder",
    ]
    assert policy["required_roles"] == ["fit", "evaluation"]
    assert policy["fit_and_evaluation_query_ids_must_be_disjoint"] is True
    assert (
        policy[
            "same_encoder_id_revision_dimension_pooling_and_normalization_required"
        ]
        is True
    )
    assert policy["evaluation_used_for_method_selection"] is False


def test_v16_factor_matrix_separates_headroom_rank_coding_objective_and_domain() -> None:
    factors = PROTOCOL["factor_matrix"]
    assert set(factors).issuperset(
        {
            "candidate_headroom",
            "rank_capacity",
            "int8_coding",
            "objective_value",
            "fit_domain_interaction",
            "pooled_repair",
        }
    )
    assert factors["only_one_factor_changes_within_each_named_contrast"] is True
    configuration = PROTOCOL["diagnostic_configuration"]
    assert configuration["pca_ranks"] == [16, 64]
    assert configuration["rars_rank"] == 16
    assert configuration["coefficient_dtype"] == "int8"
    assert configuration["payload_bytes_per_document_at_rank16"] == 16
    assert configuration["alpha"] == 0.75
    assert configuration["top_b"] == 40


def test_v16_uses_v8_cutoff_objective_and_query_level_inference() -> None:
    assert PROTOCOL["pair_mining"]["promotion_total_loss_mass"] == 0.5
    assert PROTOCOL["basis_optimization"]["pca_anchor_weight"] == 0.1
    inference = PROTOCOL["inference"]
    assert inference["resampling_unit"] == "query"
    assert inference["bootstrap_replicates"] == 20_000
    assert inference["randomization_replicates"] == 50_000
    assert inference["domain_rows_reported_separately"] is True


def test_v16_decision_cannot_be_rewritten_or_claim_universality() -> None:
    assert PROTOCOL["decision_order"] == [
        "STOP_FROZEN_CANDIDATE_METHOD",
        "OBJECTIVE_REPAIR_SUPPORTED",
        "DOMAIN_SHIFT_SUPPORTED",
        "CODING_BOTTLENECK_SUPPORTED",
        "CAPACITY_BOTTLENECK_SUPPORTED",
        "STOP_LEARNING_CLAIM_KEEP_UNIFORM_RPQ",
        "STOP_RARS_METHOD_EXPANSION",
    ]
    prohibited = PROTOCOL["prohibited_actions"]
    assert any("different encoders" in action for action in prohibited)
    assert any("confirmatory causal evidence" in action for action in prohibited)
    assert any("evaluation outcomes" in action for action in prohibited)
