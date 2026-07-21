from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rars_v7_query_adapter_pilot_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v7_is_frozen_query_only_and_keeps_the_index_immutable() -> None:
    protocol = _protocol()
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_V7_TRAINING_RUN"
    assert protocol["adapter"]["side"] == "query_only"
    assert protocol["adapter"]["serialized_float32_bytes"] == 49_152
    frozen = protocol["frozen_index_contract"]
    assert frozen["document_embeddings_immutable"] is True
    assert frozen["pq_codebooks_immutable"] is True
    assert frozen["pq_codes_immutable"] is True
    assert frozen["routing_query"].startswith("original")
    assert frozen["scoring_query"].startswith("adapted")


def test_v7_has_label_blind_disjoint_design_split_and_forbids_other_roles() -> None:
    policy = _protocol()["data_policy"]
    split = policy["split"]
    assert policy["source_role"] == "oracle_design"
    assert split["training_query_count"] + split["selection_query_count"] == 2307
    assert split["query_disjoint"] is True
    assert "SHA-256" in split["method"]
    assert "no arrays" in policy["forbidden_roles"]["oracle_audit"]
    assert "identity-only" in policy["forbidden_roles"]["future_method_holdout"]


def test_v7_separates_top10_protection_and_top100_promotion() -> None:
    mining = _protocol()["pair_mining"]
    assert mining["promotion_top_100"]["pool_k"] == 100
    assert mining["protection_top_10"]["pool_k"] == 10
    assert "unjudged" in mining["penalty_interpretation"]
    objective = _protocol()["objective"]
    assert objective["protection_weight"] > objective["promotion_weight"]
    assert objective["recall_is_directly_differentiated"] is False


def test_v7_gate_is_constrained_and_does_not_unlock_future_or_rars() -> None:
    protocol = _protocol()
    gate = protocol["pilot_gate"]
    assert gate["selected_epoch_must_be_nonzero"] is True
    assert gate["minimum_hard_pq_recall_at_100_gain"] == 0.005
    assert gate["maximum_hard_pq_recall_at_10_drop"] == 0.0025
    assert gate["go_decision"] == "GO_TO_V7_DEVELOPMENT_AUDIT"
    assert protocol["positioning"]["rars_combination_locked"] is True
    prohibited = "\n".join(protocol["prohibited_actions"])
    assert "future_method_holdout" in prohibited
    assert "RARS" in prohibited
    assert "SIGIR readiness" in prohibited

