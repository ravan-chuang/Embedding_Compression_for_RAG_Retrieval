from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_rars_v17_setting_bundle.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def test_v17_builder_uses_the_frozen_setting_specific_index_recipe() -> None:
    assert 'protocol["index_policy"].get(args.setting_id)' in SOURCE
    assert 'expected_nprobe = int(expected_index["nprobe"])' in SOURCE
    assert '"dimension": int(common_index["dimension"])' in SOURCE
    assert '"subquantizers": int(common_index["subquantizers"])' in SOURCE
    assert '"bits_per_subquantizer": int(common_index["bits_per_subquantizer"])' in SOURCE
    assert 'metric_by_name[expected_metric]' in SOURCE
    assert "args.nprobe is not None" in SOURCE


def test_v17_builder_enforces_million_scale_and_role_query_minima() -> None:
    assert '"minimum_document_count_per_setting"' in SOURCE
    assert '"minimum_fit_queries_per_domain"' in SOURCE
    assert '"minimum_evaluation_queries_per_domain"' in SOURCE
    assert "args.candidate_pool != int(common_index" in SOURCE


def test_v17_builder_allows_and_discloses_the_intended_opened_nq_paths() -> None:
    assert "beir_nq_confirmation" not in SOURCE
    assert "external_confirmation" not in SOURCE
    assert '"opened_nq_test_evidence": args.setting_id == NQ_SETTING' in SOURCE
    assert '"prior_confirmation_outcomes_known": args.setting_id == NQ_SETTING' in SOURCE
    assert '"independent_confirmation_claim_allowed": False' in SOURCE


def test_v17_builder_canonical_sources_all_exist() -> None:
    for name in (
        "adapt_rars_v17_msmarco_bundle.py",
        "build_rars_v17_setting_bundle.py",
        "evaluate_rars_v17_million_scale.py",
        "freeze_rars_v17_setting_manifest.py",
        "prepare_rars_v17_nq_roles.py",
        "rars_v17_million_scale_core.py",
    ):
        assert (ROOT / "scripts" / name).is_file()
        assert name in SOURCE
