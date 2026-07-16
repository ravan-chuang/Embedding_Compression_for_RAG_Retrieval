from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_boundary_aware_sidecar.py"
SPEC = importlib.util.spec_from_file_location("boundary_aware", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_beir_nq_is_rejected_as_development_data() -> None:
    for name in ["NQ", "BEIR-NQ", "beir_nq", "Natural Questions"]:
        with pytest.raises(ValueError, match="locked"):
            MODULE.validate_development_dataset(name)
    assert MODULE.validate_development_dataset("FiQA") == "FiQA"


def test_boundary_pairs_are_deterministic_and_cross_exact_topk() -> None:
    exact = np.asarray(
        [[0.90, 0.80, 0.70, 0.69, 0.68], [0.50, 0.40, 0.39, 0.20, 0.10]],
        dtype=np.float32,
    )
    # Candidate 2 incorrectly beats candidate 1 for the first query.
    ann = np.asarray(
        [[0.90, 0.60, 0.72, 0.69, 0.68], [0.50, 0.35, 0.36, 0.20, 0.10]],
        dtype=np.float32,
    )
    first = MODULE.build_boundary_pairs(
        exact, ann, final_k=2, top_b=5,
        positives_per_query=1, negatives_per_positive=2,
    )
    second = MODULE.build_boundary_pairs(
        exact, ann, final_k=2, top_b=5,
        positives_per_query=1, negatives_per_positive=2,
    )
    assert np.array_equal(first.query, second.query)
    assert np.array_equal(first.positive, second.positive)
    assert np.array_equal(first.negative, second.negative)
    assert np.allclose(first.weight, second.weight)
    assert first.positive[first.query.tolist().index(0)] == 1
    assert first.negative[first.query.tolist().index(0)] == 2
    assert np.all(
        exact[first.query, first.positive] >= exact[first.query, first.negative]
    )


def test_pairwise_loss_rewards_correct_boundary_order() -> None:
    exact = np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    ann = np.asarray([[0.9, 0.5, 0.75, 0.6]], dtype=np.float32)
    pairs = MODULE.build_boundary_pairs(
        exact, ann, final_k=2, top_b=4,
        positives_per_query=1, negatives_per_positive=2,
    )
    bad = MODULE.pairwise_softplus_numpy(ann, pairs)
    corrected = ann.copy()
    corrected[0, 1] = 0.85
    good = MODULE.pairwise_softplus_numpy(corrected, pairs)
    assert good < bad


def test_topk_overlap_uses_stable_candidate_ties() -> None:
    exact = np.asarray([[1.0, 0.9, 0.8]], dtype=np.float32)
    corrected = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
    assert MODULE.topk_overlap(corrected, exact, final_k=2) == 1.0
