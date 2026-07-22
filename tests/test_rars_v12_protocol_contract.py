from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "protocols/rars_v12_anchored_cutoff_rpq_v1.json").read_text()
)


def test_v12_uses_fresh_train_queries_and_excludes_all_old_dev_qids() -> None:
    fresh = PROTOCOL["fresh_query_freeze"]
    assert fresh["target_query_count"] == 2500
    assert "qrels.train.tsv" in fresh["official_qrels_url"]
    assert fresh["queries_member"] == "queries.train.tsv"
    assert set(fresh["prior_qid_sources"]) == {
        "splits/msmarco_rars_train_qids.json",
        "splits/msmarco_rars_validation_qids.json",
        "splits/msmarco_rars_test_qids.json",
    }
    assert fresh["freeze_must_precede_candidate_or_metric_computation"] is True
    assert fresh["fold_count"] == 5


def test_v12_method_is_storage_matched_and_single_configuration() -> None:
    method = PROTOCOL["method"]
    assert method["rank"] == 64
    assert method["subquantizers"] == 16
    assert method["block_dimension"] == 4
    assert method["bits_per_subquantizer"] == 8
    assert method["payload_bytes_per_document"] == 16
    assert method["basis_learning_allowed"] is False
    assert method["query_adapter_allowed"] is False
    assert method["hard_assignment_gradient_allowed"] is False
    assert method["configuration_count"] == 1


def test_v12_update_is_closed_form_anchored_and_drift_bounded() -> None:
    update = PROTOCOL["centroid_update"]
    assert update["updates"] == 1
    assert update["assignment_during_update"] == "frozen unsupervised RPQ assignment"
    assert update["anchor_pseudocount"] == 32.0
    assert update["maximum_centroid_drift_fraction_of_training_block_rms"] == 0.25
    assert update["objective_must_not_increase"] is True
    assert PROTOCOL["full_corpus_sidecar"]["code_shape"] == [1000000, 16]
    assert PROTOCOL["full_corpus_sidecar"]["payload_bytes"] == 16000000


def test_v12_gate_requires_effect_seed_fold_secondary_and_payload_stability() -> None:
    gate = PROTOCOL["development_gate"]
    assert gate["minimum_recall_gain_over_unsupervised"] == 0.003
    assert gate["minimum_each_seed_gain"] == 0.0
    assert gate["minimum_median_seed_gain"] == 0.002
    assert gate["minimum_worst_fold_gain"] == 0.0
    assert gate["minimum_mrr_change"] == -0.002
    assert gate["minimum_ndcg_change"] == -0.002
    assert gate["old_holdout_reuse_authorized"] is False
    assert gate["fresh_confirmation_access_authorized"] is False
    assert len(PROTOCOL["rpq_training"]["seeds"]) == 3
    assert PROTOCOL["rpq_training"]["all_seeds_run_in_all_folds"] is True


def test_v12_packet_repairs_v11_self_audit_gap() -> None:
    audit = PROTOCOL["self_audit_outputs"]
    assert audit["query_ids_required"] is True
    assert audit["fold_ids_required"] is True
    assert audit[
        "per_query_recall_mrr_ndcg_required_for_base_exact_unsupervised_and_challenger"
    ] is True
    assert audit["per_seed_arrays_required"] is True
    assert audit["packet_verifier_required"] is True

