from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_rars_pca_external.py"
)
SPEC = importlib.util.spec_from_file_location("external_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_bootstrap_is_deterministic() -> None:
    left = np.asarray([1.0, 0.5, 0.0, 1.0])
    right = np.asarray([0.0, 0.5, 0.0, 0.5])
    a = MODULE.paired_bootstrap(left, right, replicates=1000, seed=7)
    b = MODULE.paired_bootstrap(left, right, replicates=1000, seed=7)
    assert a == b
    assert a["difference"] == 0.375


def test_apply_sidecar_changes_only_top_b() -> None:
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    rows = np.asarray([[0, 1, 2]], dtype=np.int64)
    scores = np.asarray([[0.3, 0.2, 0.1]], dtype=np.float32)
    basis = np.eye(2, dtype=np.float32)
    scales = np.ones(2, dtype=np.float32)
    codes = np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.int8)

    corrected = MODULE.apply_sidecar(
        queries,
        rows,
        scores,
        basis,
        scales,
        codes,
        alpha=1.0,
        top_b=2,
    )
    assert np.allclose(corrected, [[1.3, 2.2, 0.1]])


def test_per_query_metrics() -> None:
    qids = ["q1"]
    ranked = np.asarray([[10, 20, 30]], dtype=np.int64)
    qrels = {"q1": {20: 2.0, 40: 1.0}}
    metrics = MODULE.per_query_metrics(qids, ranked, qrels, k=3)
    assert metrics["recall@10"][0] == 0.5
    assert metrics["success@10"][0] == 1.0
    assert metrics["mrr@10"][0] == 0.5
    assert 0.0 < metrics["ndcg@10"][0] < 1.0
