from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_rars_v2_intervention.py"
)
SPEC = importlib.util.spec_from_file_location("rars_v2_intervention", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_rows() -> list[dict[str, object]]:
    base = np.asarray([
        [0.9, 0.8, 0.7, 0.6],
        [0.9, 0.8, 0.7, 0.6],
    ], dtype=np.float32)
    corrected = np.asarray([
        [0.9, 0.8, 0.95, 0.6],
        [0.9, 0.8, 0.7, 0.6],
    ], dtype=np.float32)
    ungated = corrected.copy()
    labels = np.asarray([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ], dtype=np.uint8)
    correction = np.asarray([
        [0.0, 0.0, 0.05],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    return MODULE.per_query_diagnostics(
        base, corrected, ungated, labels, np.asarray([1, 1]),
        np.asarray([0.5, 0.5]), correction, correction, correction * 0.5,
        final_k=2, top_b=3, max_correction=0.05,
    )


def test_oracle_headroom_and_rank_flip_are_measured() -> None:
    rows = synthetic_rows()
    assert rows[0]["oracle_headroom"] is True
    assert rows[0]["base_hits_at_k"] == 0
    assert rows[0]["corrected_hits_at_k"] == 1
    assert rows[0]["top_k_membership_changes"] == 1
    assert rows[1]["oracle_headroom"] is False


def test_summary_decomposes_actual_ungated_and_oracle_gain() -> None:
    summary = MODULE.summarize(synthetic_rows())
    assert summary["recall_at_10"]["base"] == 0.5
    assert summary["recall_at_10"]["trained_gate"] == 1.0
    assert summary["recall_at_10"]["oracle_gain_over_base"] == 0.5
    assert summary["recall_at_10"]["fraction_of_oracle_gain_recovered"] == 1.0
    assert summary["query_outcomes"]["oracle_headroom_queries"] == 1


def test_outer_validation_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inner_validation"):
        MODULE.run(tmp_path / "outer_validation", tmp_path / "model", tmp_path, 10)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="correction shape"):
        MODULE.per_query_diagnostics(
            np.zeros((1, 4)), np.zeros((1, 4)), np.zeros((1, 4)),
            np.zeros((1, 4)), np.ones(1), np.ones(1),
            np.zeros((1, 2)), np.zeros((1, 3)), np.zeros((1, 3)),
            final_k=2, top_b=3, max_correction=0.05,
        )
