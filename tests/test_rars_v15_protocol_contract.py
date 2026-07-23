from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "protocols/rars_v15_cross_fitted_selective_rpq_gate_v1.json").read_text()
)


def test_v15_is_outcome_informed_development_not_confirmation() -> None:
    assert PROTOCOL["status"] == "FROZEN_BEFORE_FIRST_V15_DEVELOPMENT_RUN"
    assert PROTOCOL["evidence_boundary"]["tier"] == (
        "POST_V14_OUTCOME_INFORMED_QUERY_GATING_DEVELOPMENT_ON_V13_QUERIES"
    )
    assert "independent confirmation" in PROTOCOL["evidence_boundary"]["forbidden_claims"]
    assert PROTOCOL["development_gate"]["go_authorizes_only_protocol_writing"] is True
    assert PROTOCOL["development_gate"]["fresh_query_access_authorized"] is False


def test_v15_freezes_document_representation_and_adds_zero_document_bytes() -> None:
    sidecar = PROTOCOL["uniform_sidecar"]
    assert sidecar["rank"] == 64
    assert sidecar["subquantizers"] == 16
    assert sidecar["bits_per_subquantizer"] == 8
    assert sidecar["payload_bytes_per_document"] == 16
    assert sidecar["document_side_changes_from_v13_uniform"] is False
    assert PROTOCOL["storage_contract"]["additional_document_bytes"] == 0
    assert PROTOCOL["storage_contract"]["maximum_global_gate_bytes"] == 4096


def test_v15_gate_and_calibration_are_single_preregistered_configuration() -> None:
    gate = PROTOCOL["gate"]
    assert gate["feature_count"] == 12
    assert len(gate["features"]) == 12
    assert gate["features_use_relevance_labels"] is False
    assert gate["configuration_count"] == 1
    assert gate["ridge"] == 1.0
    assert gate["harm_weight"] == 2.0
    assert gate["threshold_quantiles"] == [index / 20 for index in range(21)]
    assert gate["calibration_minimum_coverage"] == 0.2
    assert gate["calibration_maximum_coverage"] == 0.95


def test_v15_uses_disjoint_gate_fit_calibration_and_outer_test_roles() -> None:
    cv = PROTOCOL["cross_validation"]
    assert cv["outer_fold_count"] == 5
    assert cv["gate_fit_folds_per_run"] == 3
    assert cv["calibration_folds_per_run"] == 1
    assert cv["outer_heldout_labels_not_used_until_final_scoring"] is True
    assert cv["out_of_fold_primary_endpoint"] is True
    assert cv["final_all-development_gate_is_export_only"] is True
