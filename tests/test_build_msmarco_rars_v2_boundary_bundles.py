from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_msmarco_rars_v2_boundary_bundles.py"
SPEC = importlib.util.spec_from_file_location("msmarco_boundary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qrels_list_format_and_candidate_labels(tmp_path: Path) -> None:
    path = tmp_path / "qrels_subset.json"
    path.write_text(json.dumps({"q1": ["10", "20"], "q2": ["30"]}))
    qrels = MODULE.load_qrels(path)
    assert qrels == {"q1": {10, 20}, "q2": {30}}
    doc_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    labels, counts = MODULE.labels_and_counts(
        ["q1", "q2"], np.asarray([[0, 2], [2, 3]]), doc_ids, qrels
    )
    assert labels.tolist() == [[1, 0], [1, 0]]
    assert counts.tolist() == [2, 1]


def test_sidecar_scoring_changes_only_top_b() -> None:
    result = MODULE.sidecar_scores(
        np.asarray([[1.0, 0.0]], np.float32),
        np.asarray([[0, 1, 2]], np.int64),
        np.asarray([[0.1, 0.2, 0.3]], np.float32),
        np.eye(2, dtype=np.float32),
        np.asarray([[1, 0], [2, 0], [3, 0]], np.int8),
        np.ones(2, np.float32), alpha=0.5, top_b=2,
    )
    assert np.allclose(result, [[0.6, 1.2, 0.3]])
