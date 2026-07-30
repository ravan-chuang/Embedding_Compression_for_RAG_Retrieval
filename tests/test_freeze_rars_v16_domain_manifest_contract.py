from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_rars_v16_domain_manifest.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")


def test_v16_manifest_freezer_requires_all_four_roles() -> None:
    for argument in (
        "--fiqa-fit",
        "--fiqa-evaluation",
        "--scifact-fit",
        "--scifact-evaluation",
        "--source-commit",
    ):
        assert f'parser.add_argument("{argument}"' in SOURCE


def test_v16_manifest_freezes_source_encoder_index_and_no_metrics() -> None:
    assert '"source_domain_id": source_domain' in SOURCE
    assert "len(encoder_keys) != 1" in SOURCE
    assert "len(index_recipes) != 1" in SOURCE
    assert '"evaluation_used_for_selection": False' in SOURCE
    assert '"metrics_opened": False' in SOURCE
    assert "fit/evaluation query IDs overlap" in SOURCE
