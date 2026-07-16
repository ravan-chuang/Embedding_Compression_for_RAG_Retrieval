from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_boundary_loss_sidecar.py"
SPEC = importlib.util.spec_from_file_location("boundary_loss_sidecar", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_accepts_development_train_split() -> None:
    MODULE.validate_development_manifest({
        "split_role": "train",
        "source": "BEIR NQ train archive",
        "test_qrels_accessed": False,
    })


@pytest.mark.parametrize("role", ["test", "evaluation", "posthoc"])
def test_manifest_rejects_non_development_roles(role: str) -> None:
    with pytest.raises(ValueError):
        MODULE.validate_development_manifest({
            "split_role": role,
            "source": "development",
            "test_qrels_accessed": False,
        })


def test_manifest_rejects_closed_test_marker() -> None:
    with pytest.raises(ValueError, match="Closed-test marker"):
        MODULE.validate_development_manifest({
            "split_role": "validation",
            "source": "stage3/evaluation/test_query_vectors.float32.npy",
            "test_qrels_accessed": False,
        })


def test_boundary_pairs_use_relevant_positive_and_boundary_negative() -> None:
    scores = np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    labels = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)
    pairs = MODULE.build_boundary_pairs(labels, scores, final_k=2, negative_window=1)
    assert pairs.tolist() == [[0, 2, 0], [0, 2, 1]]


def test_boundary_pairs_skip_queries_without_candidate_positive() -> None:
    scores = np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    labels = np.asarray([[0, 0], [1, 0]], dtype=np.uint8)
    pairs = MODULE.build_boundary_pairs(labels, scores, final_k=1, negative_window=1)
    assert pairs.tolist() == [[1, 0, 1]]
