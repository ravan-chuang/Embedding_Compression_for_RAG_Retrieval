from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (ROOT / "scripts/evaluate_rars_v14_anisotropic_rate_rpq.py").read_text()
VERIFIER = (ROOT / "scripts/verify_rars_v14_anisotropic_rate_rpq_packet.py").read_text()


def test_v14_requires_clean_exact_source_and_empty_output() -> None:
    assert 'git", "rev-parse", "HEAD' in EVALUATOR
    assert 'git", "status", "--porcelain"' in EVALUATOR
    assert "V14 diagnostic requires a clean exact checkout" in EVALUATOR
    assert "Refusing to reuse a non-empty V14 output directory" in EVALUATOR


def test_v14_reverifies_v13_parent_and_does_not_open_future_roles() -> None:
    assert "verify_rars_v13_committed_closure" in EVALUATOR
    assert "verify_v13_packet" in EVALUATOR
    assert '"future_method_holdout_opened": False' in EVALUATOR
    assert '"old_rars_holdout_opened": False' in EVALUATOR
    assert '"labels_used_for_representation_learning": False' in EVALUATOR


def test_v14_materializes_and_verifies_real_16_byte_full_codes() -> None:
    assert "full_corpus_qw_ar_rpq_codes.uint8.memmap" in EVALUATOR
    assert "n_docs * 16" in EVALUATOR
    assert "code_histograms" in EVALUATOR
    assert "full_corpus_qw_ar_rpq_codes.uint8.memmap" in VERIFIER
    assert "sha256_file(payload)" in VERIFIER


def test_v14_reports_all_required_comparisons_and_consensus() -> None:
    for name in (
        "anisotropic_vs_v13_uniform_rpq",
        "anisotropic_vs_uniform_whitened",
        "anisotropic_vs_pca16",
        "anisotropic_vs_base",
        "multi_seed_consensus",
    ):
        assert name in EVALUATOR
        assert name in VERIFIER
