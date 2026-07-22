from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_v13_freeze_precedes_candidates_and_excludes_v12() -> None:
    source = _source("freeze_rars_v13_fresh_queries.py")
    assert "expected_unique_excluded_qids" in source
    assert 'path.suffix == ".txt"' in source
    assert "candidate_retrieval_performed\": False" in source
    assert "metric_computation_performed\": False" in source
    assert "index.search" not in source


def test_v13_bundle_does_not_compute_metrics() -> None:
    source = _source("build_rars_v13_fresh_bundle.py")
    assert "candidate_retrieval_performed_after_query_freeze\": True" in source
    assert "metrics_computed\": False" in source
    assert "per_query_metrics" not in source
    assert "make_direct_map" in source


def test_v13_trainer_uses_signed_scores_and_frozen_assignments() -> None:
    source = _source("train_rars_v13_signed_score_rpq.py")
    assert "build_signed_score_statistics" in source
    assert "fit_signed_score_codebooks" in source
    assert "primary_vs_pca16" in source
    assert "full_corpus_signed_score_assignments.uint8.memmap" in source
    assert "updated_all_codes" not in source
    assert "mine_cutoff_pairs" not in source
    assert '"v12_packet_opened": False' in source


def test_v13_packet_has_independent_recomputation() -> None:
    source = _source("verify_rars_v13_signed_score_rpq_packet.py")
    assert "paired_inference" in source
    assert "signed_score_decision" in source
    assert "code_histograms" in source
    assert "expected_unique_excluded_qids" in source
    assert "assignment_changes" in source
