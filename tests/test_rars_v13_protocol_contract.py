from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "protocols/rars_v13_signed_score_distilled_rpq_v1.json").read_text()
)


def test_v13_is_frozen_fresh_query_development_only() -> None:
    assert PROTOCOL["status"] == "FROZEN_BEFORE_FIRST_V13_FRESH_DEVELOPMENT_RUN"
    fresh = PROTOCOL["fresh_query_freeze"]
    assert fresh["target_query_count"] == 5000
    assert fresh["expected_unique_excluded_qids"] == 9480
    assert "results/rars_v12_ca_rpq/development/query_ids.utf8.txt" in fresh[
        "prior_qid_sources"
    ]
    assert fresh["fold_count"] == 5
    assert fresh["minimum_queries_per_fold"] == 800
    assert PROTOCOL["parent_evidence"]["runtime_packet_read_allowed"] is False
    assert PROTOCOL["development_gate"]["fresh_confirmation_access_authorized"] is False


def test_v13_is_storage_matched_and_assignment_preserving() -> None:
    method = PROTOCOL["method"]
    assert method["rank"] == 64
    assert method["subquantizers"] == 16
    assert method["block_dimension"] == 4
    assert method["payload_bytes_per_document"] == 16
    assert method["post_update_reassignment_allowed"] is False
    assert PROTOCOL["pca16_comparator"]["payload_bytes_per_document"] == 16
    assert PROTOCOL["full_corpus_sidecar"]["payload_bytes"] == 16_000_000
    assert PROTOCOL["full_corpus_sidecar"][
        "assignments_identical_for_unsupervised_and_challenger"
    ] is True
    assert PROTOCOL["development_gate"]["maximum_assignment_changes"] == 0


def test_v13_score_objective_and_gates_are_single_configuration() -> None:
    score = PROTOCOL["score_distillation"]
    assert score["target_per_block"].startswith("signed FP32")
    assert score["cutoff_boost"] == 4.0
    assert score["known_positive_multiplier"] == 2.0
    assert score["anchor_ratio"] == 1.0
    assert score["maximum_centroid_drift_fraction_of_training_block_rms"] == 0.15
    assert score["post_update_assignments"] == "byte-identical to unsupervised assignments"
    assert PROTOCOL["method"]["configuration_count"] == 1
    gate = PROTOCOL["development_gate"]
    assert gate["minimum_recall_gain_over_unsupervised"] == 0.003
    assert gate["minimum_recall_gain_over_pca16"] == 0.003
    assert gate["minimum_each_seed_gain"] == 0.0
    assert gate["minimum_worst_fold_gain"] == 0.0
