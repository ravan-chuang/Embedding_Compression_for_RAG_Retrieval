from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rars_v5_pq_aware_100k_pilot_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v5_is_a_frozen_development_pilot_not_a_method_claim() -> None:
    protocol = _protocol()
    assert protocol["status"] == "IMPLEMENTATION_FROZEN_BEFORE_FIRST_100K_PILOT_OUTCOME"
    assert protocol["scope"].startswith("outcome-informed 100K development pilot")
    assert protocol["positioning"]["primary_contribution_claimed"] is False
    assert protocol["positioning"]["ranking_loss_through_pq_is_novel"] is False
    assert protocol["method_revision_allowed"] is False


def test_v5_never_opens_future_or_external_outcomes() -> None:
    policy = _protocol()["data_policy"]
    future = policy["future_method_holdout"]
    assert future["query_count"] == 803
    assert future["allowed_to_open"] is False
    assert future["candidate_arrays_allowed"] is False
    assert future["labels_allowed"] is False
    assert future["metrics_allowed"] is False
    assert "any new external collection" in policy["forbidden_outcome_sources"]
    assert policy["raw_qrels_read_by_v5_builder"] is False


def test_v5_uses_hard_fixed_pq_and_small_adapters_only() -> None:
    protocol = _protocol()
    index = protocol["pilot_index"]
    adapter = protocol["adapter"]
    assert index["hard_forward_required"] is True
    assert index["soft_assignment_primary_result_allowed"] is False
    assert index["coarse_assignments_fixed_during_adapter_training"] is True
    assert index["pq_codebooks_fixed_during_adapter_training"] is True
    assert "recomputed" in index["residual_pq_code_assignments"]
    assert adapter["rank"] == 8
    assert adapter["trainable_parameter_count"] == 12288
    assert adapter["full_encoder_updates_allowed"] is False


def test_v5_selection_is_end_to_end_and_never_injects_positives() -> None:
    protocol = _protocol()
    selection = protocol["data_policy"]["pilot_corpus"][
        "selection_candidate_generation"
    ]
    assert "label-independent" in selection
    assert "never appended" in selection
    assert "end-to-end 100K" in protocol["objective"]["selection_evaluation"]


def test_v5_single_seed_go_only_authorizes_replication() -> None:
    guards = _protocol()["later_stage_guards"]
    assert guards["go_is_method_success"] is False
    assert guards["one_million_document_rebuild_allowed"] is False
    assert guards["rars_training_allowed"] is False
    assert guards["future_holdout_open_allowed"] is False
    assert guards["external_evaluation_allowed"] is False
    assert "seeds 43 and 44" in guards["go_authorizes_only"]


def test_v5_artifact_paths_exist() -> None:
    policy = _protocol()["artifact_policy"]
    for name in ("builder", "core", "trainer"):
        assert (ROOT / policy[name]).is_file()


def test_v5_frozen_configuration_matches_declared_gate() -> None:
    protocol = _protocol()
    frozen = protocol["frozen_training_configuration"]
    gate = protocol["stage_a_gate"]
    assert frozen["seed"] == 42
    assert frozen["score_batch_size"] == 8192
    assert frozen["retrieval_query_batch_size"] == 32
    assert gate["minimum_hard_pq_recall_at_100_gain"] == 0.005
    assert gate["bootstrap_replicates"] == 20000
