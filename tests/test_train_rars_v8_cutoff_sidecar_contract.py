from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_rars_v8_cutoff_sidecar.py").read_text(encoding="utf-8")


def test_v8_trainer_verifies_v6_and_uses_only_design_role() -> None:
    assert "verify_v6_packet(args.v6_packet_root)" in SOURCE
    assert 'role_dir.name != "oracle_design"' in SOURCE
    assert "--future" not in SOURCE
    assert "--audit" not in SOURCE
    assert '"future_method_holdout_opened": False' in SOURCE
    assert '"oracle_audit_opened": False' in SOURCE


def test_v8_trainer_reports_only_int8_out_of_fold_scores() -> None:
    assert "fit_int8_scales" in SOURCE
    assert "encode_residuals_int8" in SOURCE
    assert "score_sidecar_candidates" in SOURCE
    assert "oof_rars_scores[validation_queries] = fold_scores" in SOURCE
    assert "hyperparameter" not in SOURCE.casefold() or "selection" not in SOURCE.casefold()


def test_v8_trainer_freezes_claim_tier_without_authorizing_holdout() -> None:
    assert '"future_access_authorized": False' in SOURCE
    assert '"next_required_action"' in SOURCE
    assert "development_decision(" in SOURCE
    assert "rars_vs_pca" in SOURCE


def test_v8_trainer_never_mutates_or_even_opens_faiss_index() -> None:
    # V8 development consumes the already frozen V3 candidate bundle.  Full
    # corpus encoding is deliberately deferred to a separate pre-holdout stage.
    assert "faiss.read_index" not in SOURCE
    assert '"full_corpus_sidecar_encoded": False' in SOURCE
