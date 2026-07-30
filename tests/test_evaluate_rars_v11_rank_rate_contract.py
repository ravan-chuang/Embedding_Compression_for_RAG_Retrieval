from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/evaluate_rars_v11_rank_rate.py"
SOURCE = PATH.read_text(encoding="utf-8")


def test_v11_evaluator_has_no_old_result_or_holdout_argument() -> None:
    assert 'parser.add_argument("--design-role-dir"' in SOURCE
    assert 'parser.add_argument("--v6-packet-root"' in SOURCE
    assert "--v9" not in SOURCE
    assert "--v10" not in SOURCE
    assert "--future-method-holdout" not in SOURCE
    assert '"v9_packet_opened": False' in SOURCE
    assert '"v10_packet_opened": False' in SOURCE
    assert '"future_method_holdout_opened": False' in SOURCE


def test_v11_evaluator_fits_one_nested_pca_basis_and_fixed_rpq() -> None:
    assert "basis64 = fit_uncentered_pca_basis(residuals, rank=64)" in SOURCE
    assert "basis32 = basis64[:, :32]" in SOURCE
    assert "basis16 = basis64[:, :16]" in SOURCE
    assert "fit_faiss_product_quantizer(" in SOURCE
    assert 'subquantizers=int(rpq["subquantizers"])' in SOURCE
    assert "fit_stable_basis" not in SOURCE
    assert "mine_cutoff_pairs" not in SOURCE


def test_v11_evaluator_enforces_real_sixteen_byte_payloads() -> None:
    assert 'if int4_codes.shape[1] != 16:' in SOURCE
    assert 'if codes.shape[1] != 16:' in SOURCE
    assert "Packed rank-32 int4 payload is not 16 bytes" in SOURCE
    assert 'f"Rank-{rank} RPQ payload is not 16 bytes"' in SOURCE


def test_v11_evaluator_uses_only_registered_primary_decision() -> None:
    assert "rank_rate_decision(" in SOURCE
    assert '"rank64_fp32_vs_pca_rank16_int8": capacity' in SOURCE
    assert '"rank64_rpq_vs_pca_rank16_int8": encoding' in SOURCE
    assert '"cutoff_training_performed": False' in SOURCE
    assert '"fresh_confirmation_access_authorized": False' in SOURCE
