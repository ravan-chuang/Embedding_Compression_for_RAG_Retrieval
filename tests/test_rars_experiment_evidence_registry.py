from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/rars_experiment_evidence_registry_v1.json"


def test_registry_preserves_formal_decisions_and_evidence_boundaries() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiments = {item["id"]: item for item in payload["experiments"]}
    assert experiments["rars_v2_2_fp32_replication"]["decision"] == "UNSTABLE_NO_QAT"
    assert experiments["rars_v3_matched_access_oracle"]["decision"] == "STOP_NO_HEADROOM"
    assert experiments["rars_v5_pq_aware_100k"]["decision"] == "STOP_PQ_AWARE_100K_PILOT"
    assert experiments["rars_v7_query_adapter"]["decision"] == "STOP_V7_QUERY_ADAPTER_PILOT"
    assert experiments["rars_v8_cutoff_sidecar"]["decision"] == "GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL"
    assert experiments["rars_v9_locked_confirmation"]["decision"] == (
        "CONFIRM_GENERIC_FROZEN_SIDECAR_WITHIN_PROGRAM"
    )
    assert experiments["rars_v10_pca_anchored_harm_constrained"]["decision"] == (
        "PENDING_SINGLE_FROZEN_DEVELOPMENT_RUN"
    )


def test_registry_does_not_upgrade_v8_or_v9_to_independent_evidence() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiments = {item["id"]: item for item in payload["experiments"]}
    assert "development" in experiments["rars_v8_cutoff_sidecar"]["evidence_tier"]
    assert "within_program" in experiments["rars_v9_locked_confirmation"][
        "evidence_tier"
    ]
    assert "development" in experiments[
        "rars_v10_pca_anchored_harm_constrained"
    ]["evidence_tier"]
