from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (
        ROOT
        / "protocols/rars_v14_query_whitened_anisotropic_rate_rpq_diagnostic_v1.json"
    ).read_text()
)


def test_v14_is_an_outcome_informed_diagnostic_not_confirmation() -> None:
    assert PROTOCOL["status"] == "FROZEN_BEFORE_FIRST_V14_DIAGNOSTIC_RUN"
    assert PROTOCOL["evidence_boundary"]["tier"] == (
        "POST_OUTCOME_ARCHITECTURE_DIAGNOSTIC_ON_V13_DEVELOPMENT_QUERIES"
    )
    assert "independent confirmation" in PROTOCOL["evidence_boundary"]["forbidden_claims"]
    assert PROTOCOL["diagnostic_gate"]["go_authorizes_only_protocol_writing"] is True
    assert PROTOCOL["diagnostic_gate"]["fresh_query_access_authorized"] is False


def test_v14_preserves_frozen_index_and_exact_16_byte_budget() -> None:
    method = PROTOCOL["method"]
    assert method["rank"] == 64
    assert method["subquantizers"] == 16
    assert method["block_dimension"] == 4
    assert method["total_bits_per_document"] == 128
    assert method["payload_bytes_per_document"] == 16
    assert method["minimum_bits_per_block"] == 6
    assert method["maximum_bits_per_block"] == 10
    assert PROTOCOL["full_corpus_sidecar"]["payload_bytes"] == 16_000_000
    assert len(PROTOCOL["frozen_index_contract"]["immutable_components"]) == 6


def test_v14_metric_and_rate_learning_are_label_free() -> None:
    assert PROTOCOL["input_contract"]["labels_used_for_metric_or_rate_learning"] is False
    assert PROTOCOL["query_metric"]["uses_relevance_labels"] is False
    assert PROTOCOL["query_metric"]["known_positive_multiplier"] == 1.0
    assert any(
        "relevance labels" in item for item in PROTOCOL["prohibited_actions"]
    )


def test_v14_has_uniform_whitened_ablation_and_stability_gates() -> None:
    assert "uniform_whitened_ablation" in PROTOCOL["comparators"]
    gate = PROTOCOL["diagnostic_gate"]
    assert gate["minimum_recall_gain_over_v13_uniform_rpq"] == 0.003
    assert gate["minimum_recall_gain_over_uniform_whitened"] == 0.001
    assert gate["minimum_each_seed_gain"] == 0.0
    assert gate["minimum_worst_fold_gain"] == 0.0
    assert gate["minimum_queries_improved_in_at_least_two_seeds"] == 10
    assert gate["require_nonuniform_allocation"] is True
