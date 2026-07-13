from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_select_pca_comparator.py"
)
SPEC = importlib.util.spec_from_file_location("pca_comparator", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_orient_basis_deterministically() -> None:
    basis = np.asarray(
        [
            [-0.2, 0.1],
            [-0.9, -0.8],
            [0.1, 0.2],
        ],
        dtype=np.float32,
    )
    oriented = MODULE.orient_basis_deterministically(basis)
    assert oriented[1, 0] > 0
    assert oriented[1, 1] > 0


def test_registered_selection_prefers_smallest_eligible_top_b() -> None:
    df = pd.DataFrame(
        [
            {
                "alpha": 0.5,
                "top_b": 20,
                "corrected_top10_overlap": 0.58,
                "overlap_gain": 0.08,
                "mse_reduction_pct": 5.0,
            },
            {
                "alpha": 0.75,
                "top_b": 40,
                "corrected_top10_overlap": 0.59,
                "overlap_gain": 0.09,
                "mse_reduction_pct": 6.0,
            },
            {
                "alpha": 1.0,
                "top_b": 100,
                "corrected_top10_overlap": 0.60,
                "overlap_gain": 0.10,
                "mse_reduction_pct": 7.0,
            },
        ]
    )
    selected = MODULE.select_registered_configuration(df)
    assert selected["top_b"] == 40
    assert selected["alpha"] == 0.75


def test_registered_tie_break_prefers_smaller_absolute_alpha() -> None:
    df = pd.DataFrame(
        [
            {
                "alpha": -0.75,
                "top_b": 40,
                "corrected_top10_overlap": 0.60,
                "overlap_gain": 0.10,
                "mse_reduction_pct": 4.0,
            },
            {
                "alpha": 0.5,
                "top_b": 40,
                "corrected_top10_overlap": 0.60,
                "overlap_gain": 0.10,
                "mse_reduction_pct": 3.0,
            },
        ]
    )
    selected = MODULE.select_registered_configuration(df)
    assert selected["alpha"] == 0.5
