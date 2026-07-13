from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "trace_external_rank_flips.py"
)
SPEC = importlib.util.spec_from_file_location("trace", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ranks_from_scores() -> None:
    scores = np.asarray([0.2, 0.9, 0.4], dtype=np.float32)
    ranks = MODULE.ranks_from_scores(scores)
    assert ranks.tolist() == [3, 1, 2]


def test_corrections_only_apply_to_top_b() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    rows = np.asarray([0, 1, 2], dtype=np.int64)
    basis = np.eye(2, dtype=np.float32)
    scales = np.ones(2, dtype=np.float32)
    codes = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.int8)

    values = MODULE.corrections(
        query,
        rows,
        basis,
        scales,
        codes,
        top_b=2,
        alpha=1.0,
    )
    assert np.allclose(values, [1.0, 2.0, 0.0])
