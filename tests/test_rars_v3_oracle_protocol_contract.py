from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rars_v3_oracle_first_feasibility_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v3_protocol_is_preexecution_and_parent_bound() -> None:
    protocol = _protocol()
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_ORACLE_RUN"
    assert protocol["method_revision_allowed"] is False
    assert protocol["outcome_informed_revision_allowed"] is False
    lineage = protocol["parent_lineage"]
    assert lineage["closed_v2_2_commit"] == (
        "32291e2aed75a99999e83e5d168b1133b70cc866"
    )
    expected_hash_fields = {
        "parent_inner_train_manifest_sha256",
        "parent_inner_train_source_manifest_sha256",
        "parent_inner_train_query_manifest_sha256",
        "closed_inner_validation_query_manifest_sha256",
        "parent_v2_2_split_audit_sha256",
        "outer_validation_split_sha256",
        "clean_test_split_sha256",
        "frozen_doc_ids_sha256",
    }
    for field in expected_hash_fields:
        value = lineage[field]
        assert len(value) == 64
        int(value, 16)


def test_v3_roles_and_candidate_source_are_frozen() -> None:
    protocol = _protocol()
    roles = protocol["data_policy"]["roles"]
    assert {name: role["query_count"] for name, role in roles.items()} == {
        "oracle_design": 2307,
        "oracle_audit": 851,
        "future_method_holdout": 803,
    }
    assert roles["oracle_audit"][
        "role_labels_materialization_allowed_only_after_design_freeze"
    ] is True
    assert roles["future_method_holdout"][
        "candidate_retrieval_allowed_in_this_gate"
    ] is False
    retrieval = protocol["frozen_retrieval"]
    assert retrieval["candidate_retrieval_performed_in_v3"] is False
    assert retrieval["candidate_reordering_allowed"] is False
    assert retrieval["candidate_k"] == 100
    assert retrieval["correction_depth"] == 40
    assert retrieval["final_k"] == 10


def test_v3_primary_gate_uses_complete_curve_and_comparator_recovery() -> None:
    protocol = _protocol()
    oracle = protocol["matched_access_oracle"]
    assert oracle["action_tiers_code_bytes"] == [0, 8, 16, 32]
    assert oracle["budget_curve_bytes_per_query"] == [0, 320, 640, 1280]
    assert oracle["primary_budget_bytes_per_query"] == 640
    recovery = protocol["metric_contract"]["counterfactual_recovery"]
    assert "Exact40-minus-design-frozen-primary-comparator" in recovery[
        "primary_reference"
    ]
    gate = protocol["access_gate"]
    assert gate["minimum_comparator_counterfactual_recovery_fraction_8b"] == 0.2
    assert gate["minimum_comparator_counterfactual_recovery_fraction_16b"] == 0.35
    assert (
        gate[
            "minimum_comparator_positive_gain_mass_with_exact_distance_reduction_fraction"
        ]
        == 0.7
    )


def test_v3_artifact_policy_requires_two_stage_freeze_and_full_reuse_audit() -> None:
    protocol = _protocol()
    policy = protocol["artifact_policy"]
    assert policy["candidate_freeze_must_be_qrels_and_label_free"] is True
    assert policy[
        "design_freeze_required_before_audit_role_labels_are_materialized_or_audit_arrays_are_loaded"
    ] is True
    registered = " ".join(policy["design_freeze_must_register"])
    assert "all registered baseline per-query" in registered
    assert "exact-solver preflight" in registered
    assert "execution environment" in registered
    assert "recursively" in policy["complete_run_reuse"]
    for field in ("source_builder", "role_label_materializer", "core", "evaluator"):
        assert (ROOT / policy[field]).is_file()


def test_v3_environment_contract_is_explicit() -> None:
    environment = _protocol()["execution_environment_contract"]
    assert environment["python_version"] == "3.12.13"
    assert environment["numpy_version"] == "1.26.4"
    assert environment["numpy_quantile_method"] == "linear"
    assert environment["blas_configuration_must_be_recorded"] is True
    assert environment["singular_values_and_adjacent_gaps_must_be_recorded"] is True
    assert environment["faiss_required_by_v3_candidate_builder_or_evaluator"] is False
