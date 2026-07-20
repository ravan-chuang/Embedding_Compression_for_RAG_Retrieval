from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_rars_v5_pq_aware_100k_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_rars_v5_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_pilot_rows_includes_every_observed_positive_and_is_deterministic() -> None:
    positives = [
        np.asarray([7, 2], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([9, 7], dtype=np.int64),
    ]
    first = MODULE.select_pilot_rows(positives, n_docs=20, pilot_docs=10, seed=4)
    second = MODULE.select_pilot_rows(positives, n_docs=20, pilot_docs=10, seed=4)
    assert np.array_equal(first, second)
    assert {2, 7, 9}.issubset(set(first.tolist()))
    assert len(np.unique(first)) == 10


def test_select_pilot_rows_rejects_too_small_corpus() -> None:
    positives = [np.asarray([0, 1, 2], dtype=np.int64)]
    with pytest.raises(ValueError, match="smaller than the required positive"):
        MODULE.select_pilot_rows(positives, n_docs=4, pilot_docs=2, seed=1)


def test_candidate_union_appends_missing_positives_without_duplicates() -> None:
    retrieved = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    positives = [
        np.asarray([1, 7], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
    ]
    rows, valid = MODULE._candidate_union(
        retrieved, positives, append_missing_positives=True
    )
    assert rows[0, valid[0]].tolist() == [0, 1, 2, 7]
    assert rows[1, valid[1]].tolist() == [3, 4, 5]
    assert len(rows[0, valid[0]]) == len(set(rows[0, valid[0]].tolist()))


def test_candidate_union_does_not_inject_positive_into_selection() -> None:
    retrieved = np.asarray([[0, 1, 2]], dtype=np.int64)
    positives = [np.asarray([1, 7], dtype=np.int64)]
    rows, valid = MODULE._candidate_union(
        retrieved, positives, append_missing_positives=False
    )
    assert rows[0, valid[0]].tolist() == [0, 1, 2]
    assert 7 not in rows[0, valid[0]]
