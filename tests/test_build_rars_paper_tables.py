from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_rars_paper_tables.py"
)
SPEC = importlib.util.spec_from_file_location("paper_tables", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_external_system_table_uses_frozen_42_query_result() -> None:
    table = MODULE.build_external_system_table().set_index("System")

    assert set(table.index) == {
        "Base IVF-PQ M32",
        "PCA rank-16 int8",
        "RARS rank-16 int8",
    }
    assert table["Queries"].eq(42).all()
    assert np.isclose(
        table.loc["RARS rank-16 int8", "Recall@10"],
        0.3263985439429755,
    )


def test_external_primary_contrast_remains_negative_and_unsupported() -> None:
    table = MODULE.build_external_contrast_table().set_index("Metric")
    primary = table.loc["Recall@10"]

    assert primary["Mean difference"] < 0.0
    assert primary["95% CI low"] < 0.0 < primary["95% CI high"]
    assert int(primary["Bootstrap samples"]) == 20_000
    assert int(primary["Seed"]) == 20_260_712
