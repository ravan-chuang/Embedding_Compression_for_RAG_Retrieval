from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "protocols/rars_v11_rank_rate_diagnostic_v1.json"


def _protocol() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_v11_is_architecture_diagnostic_not_algorithm_confirmation() -> None:
    protocol = _protocol()
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_V11_DIAGNOSTIC_RUN"
    assert "DIAGNOSTIC_ONLY" in protocol["evidence_boundary"]["tier"]
    assert protocol["data_policy"]["configuration_policy"][
        "cutoff_training_performed"
    ] is False
    assert protocol["diagnostic_gate"]["go_authorizes_only_protocol_writing"] is True
    assert protocol["diagnostic_gate"]["fresh_confirmation_access_authorized"] is False


def test_v11_primary_deployable_candidate_is_exactly_sixteen_bytes() -> None:
    protocol = _protocol()
    rows = {row["id"]: row for row in protocol["screen"]}
    assert rows["pca_rank64_rpq16x8"]["sidecar_payload_bytes_per_document"] == 16
    assert rows["pca_rank32_int4"]["sidecar_payload_bytes_per_document"] == 16
    assert protocol["rpq_training"]["subquantizers"] == 16
    assert protocol["rpq_training"]["bits_per_subquantizer"] == 8
    assert protocol["shared_scoring_contract"]["alpha"] == 0.75
    assert protocol["shared_scoring_contract"]["top_b"] == 40


def test_v11_freezes_rank_and_encoding_gates_before_execution() -> None:
    gate = _protocol()["diagnostic_gate"]
    assert gate["minimum_rank64_fp32_gain_over_pca"] == 0.005
    assert gate["minimum_rank64_rpq_gain_over_pca"] == 0.003
    assert gate["minimum_rank64_headroom_retention_fraction"] == 0.5
    assert gate["minimum_worst_fold_gain_over_pca"] == 0.0
    assert gate["old_holdout_reuse_authorized"] is False


def test_v11_uses_nested_fixed_pca_and_no_cutoff_loss() -> None:
    protocol = _protocol()
    assert "nested" in protocol["shared_scoring_contract"]["basis"]
    assert protocol["shared_scoring_contract"]["labels_used_only_for_metrics"] is True
    assert protocol["rpq_training"]["cutoff_or_relevance_loss_used"] is False
    assert protocol["rpq_training"]["basis_optimization_used"] is False
    assert protocol["rpq_training"]["redos"] == 1
    assert protocol["rpq_training"]["omp_threads"] == 1


def test_v11_registers_nonoverlapping_explicit_inference_seeds() -> None:
    inference = _protocol()["inference"]
    seeds = []
    for comparison in (
        "rank64_fp32_vs_pca",
        "rank64_rpq_vs_pca",
        "rank64_rpq_vs_base",
    ):
        seeds.extend(inference[comparison].values())
    assert len(seeds) == len(set(seeds))
    assert inference["seed_offsets"] == "none"
