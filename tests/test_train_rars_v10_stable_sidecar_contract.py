from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/train_rars_v10_stable_sidecar.py"
SOURCE = PATH.read_text(encoding="utf-8")


def test_v10_trainer_has_no_v9_or_future_input_argument() -> None:
    assert 'parser.add_argument("--design-role-dir"' in SOURCE
    assert 'parser.add_argument("--v6-packet-root"' in SOURCE
    assert "--v9" not in SOURCE
    assert "--future-method-holdout" not in SOURCE
    assert '_reject_forbidden_path(path, label)' in SOURCE
    assert '"v9_files_opened": False' in SOURCE
    assert '"future_method_holdout_opened": False' in SOURCE


def test_v10_trainer_runs_gradient_audit_before_each_fit() -> None:
    audit = SOURCE.index("audit = gradient_direction_audit(")
    fit = SOURCE.index("basis, history = fit_stable_basis(")
    assert audit < fit
    assert 'if audit["status"] != "PASS"' in SOURCE
    assert "all_accepted_losses_monotone" in SOURCE


def test_v10_trainer_uses_identical_int8_pca_comparator() -> None:
    assert "pca_basis = fit_uncentered_pca_basis" in SOURCE
    assert "pca_codes, pca_quantization = encode_residuals_int8" in SOURCE
    assert "pca_scores = score_sidecar_candidates" in SOURCE
    assert 'v10_metrics["recall"], pca_metrics["recall"]' in SOURCE


def test_v10_trainer_uses_explicit_registered_seeds_without_offsets() -> None:
    assert '"bootstrap_seed": int(inference["bootstrap_seed"])' in SOURCE
    assert '"randomization_seed": int(inference["randomization_seed"])' in SOURCE
    assert "bootstrap_seed +" not in SOURCE
    assert "randomization_seed +" not in SOURCE


def test_v10_trainer_never_authorizes_external_or_v9_reuse() -> None:
    assert '"fresh_external_access_authorized": False' in SOURCE
    assert '"v9_reuse_authorized": False' in SOURCE
    assert "If and only if every gate passes" in SOURCE


def test_v10_trainer_only_measures_avq_scalar_headroom() -> None:
    assert "pca_fp32_scores = score_float_sidecar_candidates" in SOURCE
    assert 'avq = protocol["avq_scalar_headroom_diagnostic"]' in SOURCE
    assert "scalar_quantization_headroom_decision" in SOURCE
    assert '"oof_pca_fp32_recall_at_10.float64.npy"' in SOURCE
    assert "fit_score_aware_codebook" not in SOURCE
    assert "train_avq" not in SOURCE
