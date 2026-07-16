from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_rars_v2_boundary_bundles.py"
SPEC = importlib.util.spec_from_file_location("boundary_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_train_qrels_keeps_only_positive_and_deduplicates(tmp_path: Path) -> None:
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    path = qrels_dir / "train.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t1\n"
        "q1\td1\t1\n"
        "q1\td2\t0\n"
        "q2\td3\t2\n",
        encoding="utf-8",
    )
    assert MODULE.load_train_qrels(path) == {"q1": {"d1"}, "q2": {"d3"}}


def test_closed_and_non_train_qrels_are_rejected(tmp_path: Path) -> None:
    test_path = tmp_path / "qrels" / "test.tsv"
    test_path.parent.mkdir()
    test_path.write_text("q1\td1\t1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.load_train_qrels(test_path)
    with pytest.raises(ValueError, match="post-hoc"):
        MODULE.reject_closed_path(tmp_path / "stage3/posthoc_diagnosis/data.json")


def test_candidate_labels_use_full_relevant_denominator() -> None:
    doc_ids = np.asarray([b"d0", b"d1", b"d2", b"d3"], dtype="S2")
    rows = np.asarray([[0, 1, 2], [3, 2, 1]], dtype=np.int64)
    labels, counts = MODULE.candidate_labels(
        ["q1", "q2"],
        rows,
        doc_ids,
        {"q1": {"d1", "missing"}, "q2": {"d3"}},
    )
    assert labels.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert counts.tolist() == [2, 1]


def test_relevant_document_coverage_guard() -> None:
    doc_ids = np.asarray([b"d0", b"d1"], dtype="S2")
    MODULE.verify_relevant_documents_exist(doc_ids, {"q1": {"d1"}})
    with pytest.raises(ValueError, match="absent"):
        MODULE.verify_relevant_documents_exist(doc_ids, {"q1": {"d2"}})
