from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "protocols/rars_v10_pca_anchored_harm_constrained_v1.json"


def _protocol() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_v10_is_single_configuration_post_confirmation_development() -> None:
    protocol = _protocol()
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_V10_DEVELOPMENT_RUN"
    assert protocol["evidence_boundary"]["tier"] == (
        "POST_CONFIRMATION_OUTCOME_INFORMED_METHOD_DEVELOPMENT_ONLY"
    )
    assert protocol["data_policy"]["cross_validation"]["configuration_count"] == 1
    assert protocol["data_policy"]["cross_validation"][
        "outcome_based_hyperparameter_selection"
    ] is False
    assert protocol["development_gate"]["v9_reuse_authorized"] is False


def test_v10_preserves_frozen_index_and_one_sidecar_budget() -> None:
    protocol = _protocol()
    method = protocol["method"]
    assert method["rank"] == 16
    assert method["representation_payload_bytes_per_document"] == 16
    assert method["single_sidecar_only"] is True
    assert method["pca_and_learned_codes_stacked"] is False
    assert method["query_adapter_used"] is False
    assert len(protocol["frozen_index_contract"]["immutable_components"]) == 6


def test_v10_freezes_stability_objective_and_monotone_optimizer() -> None:
    protocol = _protocol()
    objective = protocol["objective"]
    assert objective["tail_harm_weight"] > 0
    assert objective["pca_anchor_weight"] > 0
    assert 0 < objective["cvar_fraction"] <= 1
    optimization = protocol["optimization"]
    assert "Riemannian" in optimization["optimizer"]
    assert optimization["accepted_objective_must_be_monotone"] is True
    assert optimization["maximum_principal_angle_degrees_from_pca"] == 20.0
    assert protocol["optimizer_audit"]["pair_limit"] == 512
    assert protocol["optimizer_audit"]["required_for_every_fold_and_final_fit"] is True


def test_v10_requires_pca_superiority_and_worst_fold_stability() -> None:
    gate = _protocol()["development_gate"]
    assert gate["minimum_recall_at_10_gain_over_pca"] == 0.005
    assert gate["maximum_randomization_p_value"] == 0.025
    assert gate["minimum_improved_queries_over_pca"] == 30
    assert gate["minimum_net_improved_queries_over_pca"] == 15
    assert gate["minimum_worst_fold_gain_over_pca"] == 0.0
    assert gate["fresh_external_access_authorized_by_development_result"] is False


def test_v10_registers_primary_inference_seeds_without_offsets() -> None:
    inference = _protocol()["inference"]
    assert inference["bootstrap_seed"] == 20260730
    assert inference["randomization_seed"] == 20260731
    assert inference["seed_offsets"].startswith("none")


def test_v10_registers_scann_inspired_scalar_headroom_without_training() -> None:
    diagnostic = _protocol()["avq_scalar_headroom_diagnostic"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["codebook_training_performed"] is False
    assert diagnostic["not_a_scann_implementation"] is True
    assert diagnostic["minimum_recall_at_10_gain"] == 0.003
    assert diagnostic["go_does_not_authorize_training_in_this_run"] is True
    assert diagnostic["bootstrap_seed"] != diagnostic["randomization_seed"]
