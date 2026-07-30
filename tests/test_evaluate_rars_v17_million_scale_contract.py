from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rars_v17_million_scale.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def test_v17_evaluator_imports_the_actual_v17_core_and_sources() -> None:
    assert "from rars_v17_million_scale_core import" in SOURCE
    assert "rars_v17_causal_generalization_core" not in SOURCE
    for relative in (
        "scripts/adapt_rars_v17_msmarco_bundle.py",
        "scripts/build_rars_v17_setting_bundle.py",
        "scripts/evaluate_rars_v17_million_scale.py",
        "scripts/freeze_rars_v17_setting_manifest.py",
        "scripts/prepare_rars_v17_nq_roles.py",
        "scripts/rars_v17_million_scale_core.py",
        "scripts/rars_v8_cutoff_sidecar_core.py",
    ):
        assert (ROOT / relative).is_file()
        assert f'Path("{relative}")' in SOURCE


def test_v17_evaluator_uses_setting_count_and_per_setting_recipes() -> None:
    assert 'protocol["data_policy"]["setting_count"]' in SOURCE
    assert 'protocol["data_policy"]["allowed_development_settings"]' in SOURCE
    assert "index_recipes[setting_id] = recipe" in SOURCE
    assert "recipe[0:1] + recipe[3:]" in SOURCE
    assert "len(set(index_recipes.values())) == 1" in SOURCE
    assert "settings must share dimension, M, nbits, and metric" in SOURCE


def test_v17_evaluator_does_not_block_the_preregistered_nq_artifact() -> None:
    assert '"beir_nq_confirmation"' not in SOURCE
    assert '"external_confirmation"' not in SOURCE
    assert '"nq_prior_opened_test_artifact_reused": True' in SOURCE
    assert '"nq_prior_confirmation_outcomes_known": True' in SOURCE
    assert '"closed_confirmation_role_opened": True' in SOURCE
    assert '"new_unseen_confirmation_role_opened": False' in SOURCE
    assert '"independent_confirmation_claim_allowed": False' in SOURCE
    assert SOURCE.count('"confirmatory_claim_allowed": False') >= 3


def test_v17_evaluator_reports_setting_not_pure_domain_claims() -> None:
    assert '"fit_setting_interaction"' in SOURCE
    assert "setting_transfer_decision(" in SOURCE
    assert '"equal_setting_factor_summary_descriptive_only"' in SOURCE
    assert '"different_nlist_and_nprobe_expected": True' in SOURCE
    assert '"fit_domain_interaction"' not in SOURCE
