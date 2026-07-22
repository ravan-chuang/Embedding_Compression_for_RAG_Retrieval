from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/build_rars_v9_m48_baseline.py").read_text(
    encoding="utf-8"
)


def test_m48_builder_is_qrels_free_and_storage_matched() -> None:
    assert '"subquantizers"' in SOURCE
    assert "faiss.IndexIVFPQ(" in SOURCE
    assert '"qrels_argument_accepted": False' in SOURCE
    assert '"qrels_opened": False' in SOURCE
    assert '"outcome_metric_computed": False' in SOURCE
    assert 'parser.add_argument("--qrels"' not in SOURCE
    assert "load_positive_qrels" not in SOURCE


def test_m48_builder_freezes_exact_source_and_refuses_reuse() -> None:
    assert "M48 build requires a clean exact checkout" in SOURCE
    assert "Refusing to reuse a non-empty M48 output directory" in SOURCE
    assert "training_rows.int64.npy" in SOURCE
    assert "m48_build_complete.json" in SOURCE
    assert "RARS_V9_QRELS_FREE_M48_BASELINE_COMPLETE" in SOURCE
