from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rars_v8_cutoff_sidecar_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v8_protocol_freezes_deployable_symmetric_int8_sidecar() -> None:
    protocol = _protocol()
    assert protocol["protocol_id"] == "rars_v8_cutoff_sidecar_v1"
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_V8_DEVELOPMENT_RUN"
    method = protocol["method"]
    assert method["rank"] == 16
    assert method["coefficient_dtype"] == "int8"
    assert method["representation_payload_bytes_per_document"] == 16
    assert method["top_b"] == 40
    assert method["basis_constraints"]["single_symmetric_basis"] is True
    assert method["basis_constraints"]["orthonormal_after_every_update"] is True
    assert method["int8_quantization"]["selection_and_reported_metrics_must_use_int8_codes"] is True


def test_v8_protocol_equalises_roles_and_queries() -> None:
    protocol = _protocol()
    mining = protocol["pair_mining"]
    assert mining["promotion_total_loss_mass"] == 0.5
    assert mining["protection_total_loss_mass"] == 0.5
    assert "equal total mass" in mining["query_balancing"]
    assert protocol["basis_optimization"]["random_seed_used"] is False
    assert protocol["basis_optimization"]["early_stopping_used"] is False


def test_v8_protocol_separates_algorithm_and_generic_claim_tiers() -> None:
    protocol = _protocol()
    gate = protocol["development_gate"]
    assert gate["algorithm_go_decision"] == "GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL"
    assert gate["generic_sidecar_go_decision"] == "GO_TO_GENERIC_SIDECAR_CONFIRMATION_PROTOCOL"
    assert gate["future_access_authorized_by_development_result"] is False
    assert "hiding the stronger same-storage M48 rebuild baseline" in protocol[
        "prohibited_actions"
    ]


def test_v8_protocol_forbids_all_non_design_outcomes() -> None:
    forbidden = _protocol()["data_policy"]["forbidden_roles"]
    assert set(forbidden) == {
        "oracle_audit",
        "future_method_holdout",
        "outer_validation",
        "clean_test",
        "external_collections",
    }
    assert "identity-only" in forbidden["future_method_holdout"]
