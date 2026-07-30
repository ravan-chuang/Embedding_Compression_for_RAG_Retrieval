from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/evaluate_rars_v9_locked_confirmation.py").read_text(
    encoding="utf-8"
)


def test_v9_evaluator_opens_outcomes_only_after_durable_freeze() -> None:
    freeze = SOURCE.index('input_freeze_path = args.output_dir / "input_freeze.json"')
    started = SOURCE.index('started_path = args.output_dir / "confirmation_started.json"')
    qrels = SOURCE.index("qrels = load_positive_qrels(args.qrels)")
    assert freeze < started < qrels
    assert '"outcome_opened": False' in SOURCE
    assert '"qrels_opened": False' in SOURCE
    assert "FIRST OUTCOME ACCESS" in SOURCE


def test_v9_evaluator_verifies_v8_and_all_locked_comparators() -> None:
    assert "method_freeze_sha256" in SOURCE
    assert 'lineage[f"{name}_codes_sha256"]' in SOURCE
    assert 'for name in ("pca", "rars")' in SOURCE
    assert "validate_faiss_index(m32, faiss)" in SOURCE
    assert "validate_m48_packet(" in SOURCE
    assert '"m32_nprobe32"' in SOURCE
    assert '"m32_nprobe64"' in SOURCE
    assert '"same_candidate_exact"' in SOURCE
    assert '"m48_rebuild_nlist512_nprobe16"' in SOURCE


def test_v9_evaluator_uses_one_primary_comparison_and_no_retuning() -> None:
    assert '"rars_vs_pca_recall_at_10"' in SOURCE
    assert "paired_query_replicates" in SOURCE
    assert "paired_sign_replicates" in SOURCE
    assert '"method_or_threshold_tuning_authorized": False' in SOURCE
    assert '"independent_confirmation_claim_allowed": False' in SOURCE
    assert "fit_cutoff_aware_basis" not in SOURCE
    assert "mine_cutoff_pairs" not in SOURCE


def test_v9_evaluator_refuses_contaminated_identity_and_output() -> None:
    assert "Future role is not identity-only before confirmation" in SOURCE
    assert "RARS_V9_QRELS_FREE_FUTURE_IDENTITY_COMPLETE" in SOURCE
    assert "Refusing to reuse a non-empty confirmation output" in SOURCE
    assert "Confirmation requires a clean exact checkout" in SOURCE
    assert "future_method_holdout_opened_once" in SOURCE
