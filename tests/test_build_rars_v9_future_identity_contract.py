from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/build_rars_v9_future_identity.py").read_text(
    encoding="utf-8"
)


def test_future_identity_builder_is_qrels_free_and_identity_only() -> None:
    assert "rars-v2.1-inner:" in SOURCE
    assert 'V3_SPLIT_SALT = b"rars_v3_split_v1\\0"' in SOURCE
    assert '"candidate_arrays_created": False' in SOURCE
    assert '"labels_materialized": False' in SOURCE
    assert '"metrics_computed": False' in SOURCE
    assert '"qrels_argument_accepted": False' in SOURCE
    assert 'parser.add_argument("--qrels"' not in SOURCE
    assert "load_positive_qrels" not in SOURCE


def test_future_identity_builder_freezes_query_vectors_before_evaluation() -> None:
    assert "query_vectors.float32.npy" in SOURCE
    assert "RARS_V9_QRELS_FREE_FUTURE_IDENTITY_COMPLETE" in SOURCE
    assert "Refusing to reuse a non-empty future identity directory" in SOURCE
    assert "Future identity build requires a clean exact checkout" in SOURCE
