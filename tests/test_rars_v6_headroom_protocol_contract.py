from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rars_v6_1m_headroom_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v6_is_a_frozen_diagnostic_not_a_training_or_method_claim() -> None:
    protocol = _protocol()
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_1M_HEADROOM_RUN"
    assert "development diagnostic only" in protocol["scope"]
    assert protocol["positioning"]["method_contribution_claimed"] is False
    assert protocol["positioning"]["passing_is_method_success"] is False
    assert (
        protocol["signal_gate"]["go_decision"]
        == "GO_TO_V6_LOSS_IMPLEMENTATION"
    )


def test_v6_uses_only_design_queries_and_forbids_other_roles() -> None:
    policy = _protocol()["data_policy"]
    assert policy["diagnostic_role"]["role_id"] == "oracle_design"
    assert policy["diagnostic_role"]["source_query_count"] == 2307
    assert policy["shared_qrels_container"]["only_design_query_values_used"] is True
    assert policy["shared_qrels_container"]["non_design_metrics_computed"] is False
    forbidden = policy["forbidden_roles"]
    assert "no arrays" in forbidden["oracle_audit"]
    assert "no arrays" in forbidden["future_method_holdout"]
    assert forbidden["outer_validation"] == "forbidden"
    assert forbidden["clean_test"] == "forbidden"


def test_v6_decomposes_frozen_1m_m32_pq_headroom() -> None:
    protocol = _protocol()
    inputs = protocol["input_contract"]
    index = inputs["frozen_index"]
    config = protocol["diagnostic_configuration"]
    assert inputs["document_count"] == 1_000_000
    assert inputs["embedding_dimension"] == 384
    assert index["nlist"] == 512
    assert index["nprobe"] == 16
    assert index["subquantizers"] == 32
    assert index["bits_per_subquantizer"] == 8
    assert config["analysis_k"] == 200
    assert config["pool_k"] == 100
    assert "same nprobe=16 IVF lists" in config["ivf_exact_definition"]
    assert "ivf_exact_recall minus base_pq_recall" in config[
        "pq_specific_gap_definition"
    ]


def test_v6_signal_gate_requires_quantity_breadth_and_effective_support() -> None:
    gate = _protocol()["signal_gate"]
    assert gate["minimum_qrels_in_corpus_coverage"] == 1.0
    assert gate["minimum_pq_specific_recall_at_100_gap"] == 0.005
    assert gate["minimum_uncapped_flip_triplets"] == 500
    assert gate["minimum_distinct_flip_queries"] == 100
    assert gate["minimum_flip_weight_effective_sample_size"] == 250.0
    assert gate["maximum_single_query_flip_weight_share"] == 0.02
    assert "not a statistical-significance test" in gate["interpretation"]


def test_v6_does_not_authorize_training_or_rars() -> None:
    prohibited = "\n".join(_protocol()["prohibited_actions"])
    assert "adapter" in prohibited
    assert "encoder" in prohibited
    assert "PQ codebook" in prohibited
    assert "RARS sidecar" in prohibited
    assert "SIGIR readiness" in prohibited

