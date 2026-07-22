from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "protocols/rars_v9_locked_confirmation_v1.json"


def _protocol() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_v9_is_one_shot_prospective_but_not_independent() -> None:
    protocol = _protocol()
    assert protocol["protocol_id"] == "rars_v9_locked_confirmation_v1"
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_V9_OUTCOME_ACCESS"
    assert (
        protocol["claim_boundary"]["evidence_tier"]
        == "WITHIN_PROGRAM_PROSPECTIVE_HOLDOUT_NOT_INDEPENDENT"
    )
    assert protocol["data_policy"]["confirmation_role"]["query_count"] == 803
    assert protocol["data_policy"]["confirmation_role"]["one_shot_opening"] is True
    assert "independent confirmation" in protocol["claim_boundary"]["forbidden_claims"]


def test_v9_freezes_exact_v8_artifacts_and_primary_endpoint() -> None:
    protocol = _protocol()
    lineage = protocol["frozen_v8_lineage"]
    assert lineage["source_commit"] == "c9d95f15d55e7700db069da69567157f2eed469e"
    assert len(lineage["rars_codes_sha256"]) == 64
    assert len(lineage["pca_codes_sha256"]) == 64
    method = protocol["frozen_methods"]["rars_v8_rank16_int8"]
    assert (method["rank"], method["alpha"], method["top_b"]) == (16, 0.75, 40)
    assert "RARS-v8 minus PCA" in protocol["metrics"]["primary_endpoint"]
    assert protocol["metrics"]["bootstrap"]["paired_query_replicates"] == 50000


def test_v9_registers_routing_and_rebuild_limitations() -> None:
    limitations = _protocol()["locked_limitation_baselines"]
    assert set(limitations) == {
        "m32_nprobe32",
        "m32_nprobe64",
        "m48_rebuild_nlist512_nprobe16",
    }
    m48 = limitations["m48_rebuild_nlist512_nprobe16"]
    assert m48["must_be_built_without_qrels"] is True
    assert m48["subquantizers"] == 48
    assert m48["nlist"] == 512


def test_v9_has_no_retuning_escape_hatch() -> None:
    protocol = _protocol()
    gate = protocol["confirmation_gate"]
    assert gate["method_or_threshold_tuning_authorized"] is False
    assert gate["algorithm_confirmation_decision"].endswith("WITHIN_PROGRAM")
    assert gate["stop_decision"] == "STOP_RARS_V8_AFTER_LOCKED_CONFIRMATION"
    prohibited = " ".join(protocol["prohibited_actions"])
    assert "changing rank, alpha, Top-B" in prohibited
    assert "calling the future role independent" in prohibited
