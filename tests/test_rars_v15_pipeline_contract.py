from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (ROOT / "scripts/evaluate_rars_v15_selective_gate.py").read_text()
VERIFIER = (ROOT / "scripts/verify_rars_v15_selective_gate_packet.py").read_text()


def test_v15_requires_clean_exact_source_and_empty_output() -> None:
    assert 'git", "rev-parse", "HEAD' in EVALUATOR
    assert 'git", "status", "--porcelain' in EVALUATOR
    assert "V15 development requires a clean exact checkout" in EVALUATOR
    assert "Refusing to reuse a non-empty V15 output directory" in EVALUATOR


def test_v15_reverifies_all_parent_evidence_and_never_opens_future_roles() -> None:
    for name in (
        "verify_v13_closure",
        "verify_v14_closure",
        "verify_v13_packet",
        '"future_method_holdout_opened": False',
        '"old_rars_holdout_opened": False',
        '"labels_used_for_representation_learning": False',
    ):
        assert name in EVALUATOR
    assert "verify_v13_closure" in VERIFIER
    assert "verify_v14_closure" in VERIFIER


def test_v15_is_cross_fitted_and_selects_whole_query_rankings() -> None:
    assert "calibration_fold = (outer_fold + 1) % fold_count" in EVALUATOR
    assert "gate_fit = np.flatnonzero" in EVALUATOR
    assert "apply_query_gate" in EVALUATOR
    assert "np.where" in EVALUATOR
    assert "gate_utility_scores" in VERIFIER
    assert "crossfit_gate_features.float64.npy" in EVALUATOR
    assert "crossfit_gate_features.float64.npy" in VERIFIER
    assert "fit_weighted_ridge_gate" in VERIFIER
    assert "select_calibrated_threshold" in VERIFIER
    assert "recomputed_applied" in VERIFIER
    assert "selective metric changed" in VERIFIER


def test_v15_reuses_parent_payload_and_exports_only_a_small_global_gate() -> None:
    assert "full_corpus_signed_score_assignments.uint8.memmap" in EVALUATOR
    assert '"additional_document_bytes": 0' in EVALUATOR
    assert "global_model_bytes" in EVALUATOR
    assert "full_corpus_signed_score_assignments.uint8.memmap" in VERIFIER
    assert "global_model_bytes" in VERIFIER
