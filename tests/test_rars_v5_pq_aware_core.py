from __future__ import annotations

import numpy as np
import pytest

from scripts import rars_v5_pq_aware_core as MODULE


def test_hard_residual_pq_numpy_uses_fixed_coarse_centroids() -> None:
    vectors = np.asarray([[1.1, -0.1, 2.9, 4.2]], dtype=np.float32)
    coarse = np.asarray([[1.0, 0.0, 3.0, 4.0]], dtype=np.float32)
    codebooks = np.asarray(
        [
            [[0.0, 0.0], [0.25, -0.25]],
            [[0.0, 0.0], [-0.25, 0.25]],
        ],
        dtype=np.float32,
    )
    reconstructed, codes = MODULE.hard_residual_pq_numpy(
        vectors, coarse, codebooks
    )
    assert codes.tolist() == [[0, 1]]
    assert np.allclose(reconstructed, [[1.0, 0.0, 2.75, 4.25]])


def test_hard_residual_pq_ste_has_hard_forward_and_identity_gradient() -> None:
    torch = pytest.importorskip("torch")
    vectors = torch.tensor([[0.2, 0.8]], dtype=torch.float32, requires_grad=True)
    coarse = torch.zeros_like(vectors)
    codebooks = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    ste, hard, codes = MODULE.hard_residual_pq_ste(vectors, coarse, codebooks)
    assert codes.tolist() == [[0]]
    assert torch.equal(ste, hard)
    ste.sum().backward()
    assert torch.equal(vectors.grad, torch.ones_like(vectors))


def test_pool_boundary_pairs_prioritize_pq_induced_flip() -> None:
    teacher = np.asarray([[1.0, 0.8, 0.7, 0.6]], dtype=np.float32)
    pq = np.asarray([[0.1, 0.9, 0.8, 0.0]], dtype=np.float32)
    relevance = np.asarray([[1, 0, 0, 0]], dtype=np.uint8)
    valid = np.ones_like(relevance, dtype=bool)
    pairs = MODULE.build_pool_boundary_pairs(
        teacher,
        pq,
        relevance,
        valid,
        pool_k=2,
        negative_window=2,
        negatives_per_positive=2,
    )
    assert len(pairs) == 2
    assert np.all(pairs.query == 0)
    assert np.all(pairs.positive == 0)
    assert np.all(pairs.teacher_margin > 0)
    assert np.all(pairs.pq_flip)
    assert np.mean(pairs.weight) == pytest.approx(1.0)


def test_pool_boundary_pairs_never_uses_invalid_or_positive_as_negative() -> None:
    teacher = np.asarray([[1.0, 0.95, 0.8, -np.inf]], dtype=np.float32)
    pq = np.asarray([[0.9, 1.0, 0.8, -np.inf]], dtype=np.float32)
    relevance = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
    valid = np.asarray([[1, 1, 1, 0]], dtype=bool)
    pairs = MODULE.build_pool_boundary_pairs(
        teacher,
        pq,
        relevance,
        valid,
        pool_k=2,
        negative_window=2,
        negatives_per_positive=1,
    )
    assert np.all(pairs.negative == 2)


def test_recall_and_paired_bootstrap_are_query_paired() -> None:
    scores = np.asarray([[0.9, 0.8, 0.1], [0.9, 0.2, 0.1]], dtype=np.float32)
    labels = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    counts = np.asarray([2, 1], dtype=np.int32)
    valid = np.ones_like(labels, dtype=bool)
    recall = MODULE.recall_at_k_per_query(scores, labels, counts, valid, k=1)
    assert recall.tolist() == [0.5, 0.0]
    result = MODULE.paired_bootstrap_mean_difference(
        np.asarray([1.0, 0.5]),
        np.asarray([0.5, 0.0]),
        replicates=100,
        seed=7,
    )
    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["lower"] == pytest.approx(0.5)
    assert result["upper"] == pytest.approx(0.5)


def test_known_positive_recall_counts_missing_retrieval_as_miss() -> None:
    retrieved = np.asarray([[4, 2, 8], [7, 3, 9]], dtype=np.int64)
    positives = np.asarray([[2, 6], [5, -1]], dtype=np.int64)
    positive_valid = np.asarray([[1, 1], [1, 0]], dtype=bool)
    recall = MODULE.known_positive_recall_at_k(
        retrieved, positives, positive_valid, k=2
    )
    assert recall.tolist() == [0.5, 0.0]


def test_recovery_fraction_stops_when_teacher_has_no_headroom() -> None:
    assert MODULE.recovery_fraction(base=0.8, treatment=0.9, teacher=0.8) == 0.0
    assert MODULE.recovery_fraction(base=0.5, treatment=0.6, teacher=0.9) == pytest.approx(0.25)
