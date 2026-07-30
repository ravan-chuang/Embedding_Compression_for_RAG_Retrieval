#!/usr/bin/env python3
"""Core utilities for the RARS-v5 100K PQ-aware adapter pilot.

The pilot deliberately keeps the encoder and IVF-PQ codebooks fixed.  It trains
small query/document residual adapters through a *hard-forward* residual-PQ
reconstruction.  This module contains only deterministic numerical helpers;
the experiment runner owns optimization, lineage, and artifact writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v5_pq_aware_100k_pilot_v1"


@dataclass(frozen=True)
class PairBatch:
    """Flattened query/positive/mined-negative candidate positions."""

    query: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    weight: np.ndarray
    teacher_margin: np.ndarray
    pq_margin: np.ndarray
    pq_flip: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.query),
            len(self.positive),
            len(self.negative),
            len(self.weight),
            len(self.teacher_margin),
            len(self.pq_margin),
            len(self.pq_flip),
        }
        if len(lengths) != 1:
            raise ValueError("Pair arrays must have identical lengths")

    def __len__(self) -> int:
        return len(self.query)


def stable_descending_order(scores: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid mask must match scores")
        values = np.where(mask, values, -np.inf)
    if not np.all(np.isfinite(values) | np.isneginf(values)):
        raise ValueError("scores contain unsupported non-finite values")
    return np.argsort(-values, axis=1, kind="stable")


def validate_pq_centroids(
    centroids: np.ndarray, *, dimension: int | None = None
) -> tuple[int, int, int]:
    codebooks = np.asarray(centroids, dtype=np.float32)
    if codebooks.ndim != 3:
        raise ValueError("PQ centroids must have shape [M, K, dsub]")
    subquantizers, codewords, subdimension = codebooks.shape
    if subquantizers <= 0 or codewords <= 1 or subdimension <= 0:
        raise ValueError("PQ centroid dimensions must be positive")
    if dimension is not None and subquantizers * subdimension != dimension:
        raise ValueError("PQ centroids do not match the embedding dimension")
    if not np.all(np.isfinite(codebooks)):
        raise ValueError("PQ centroids must be finite")
    return subquantizers, codewords, subdimension


def hard_residual_pq_numpy(
    vectors: np.ndarray,
    coarse_centroids: np.ndarray,
    pq_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference residual-PQ assignment and reconstruction.

    ``coarse_centroids`` contains the already-selected IVF centroid for every
    vector.  Assignment is Euclidean inside each PQ subspace, matching the
    residual-codebook objective.  Returned codes use uint8 when possible.
    """

    values = np.asarray(vectors, dtype=np.float32)
    coarse = np.asarray(coarse_centroids, dtype=np.float32)
    if values.ndim != 2 or coarse.shape != values.shape:
        raise ValueError("vectors and coarse centroids must be matching matrices")
    subquantizers, codewords, subdimension = validate_pq_centroids(
        pq_centroids, dimension=values.shape[1]
    )
    blocks = (values - coarse).reshape(len(values), subquantizers, subdimension)
    codebooks = np.asarray(pq_centroids, dtype=np.float32)
    distances = np.sum(
        (blocks[:, :, None, :] - codebooks[None, :, :, :]) ** 2,
        axis=-1,
    )
    codes64 = np.argmin(distances, axis=2).astype(np.int64)
    chosen = codebooks[
        np.arange(subquantizers, dtype=np.int64)[None, :], codes64
    ]
    reconstructed = coarse + chosen.reshape(values.shape)
    code_dtype = np.uint8 if codewords <= 256 else np.uint16
    return reconstructed.astype(np.float32), codes64.astype(code_dtype)


def hard_residual_pq_ste(
    vectors: Any,
    coarse_centroids: Any,
    pq_centroids: Any,
) -> tuple[Any, Any, Any]:
    """Torch hard-forward residual PQ with identity STE to ``vectors``.

    The codebooks and coarse assignments are fixed in the v5 pilot.  Therefore
    gradients intentionally flow only to the adapted document vector, not to
    the hard assignments or centroids.
    """

    import torch

    if vectors.ndim != 2 or coarse_centroids.shape != vectors.shape:
        raise ValueError("vectors and coarse centroids must be matching matrices")
    if pq_centroids.ndim != 3:
        raise ValueError("PQ centroids must have shape [M, K, dsub]")
    subquantizers, _, subdimension = pq_centroids.shape
    if subquantizers * subdimension != vectors.shape[1]:
        raise ValueError("PQ centroids do not match the embedding dimension")
    residual_blocks = (vectors - coarse_centroids).reshape(
        len(vectors), subquantizers, subdimension
    )
    distances = torch.sum(
        (
            residual_blocks[:, :, None, :]
            - pq_centroids[None, :, :, :]
        ).square(),
        dim=-1,
    )
    codes = torch.argmin(distances, dim=2)
    subspace = torch.arange(subquantizers, device=vectors.device)[None, :]
    chosen = pq_centroids[subspace, codes]
    hard = coarse_centroids + chosen.reshape_as(vectors)
    straight_through = vectors + (hard - vectors).detach()
    return straight_through, hard, codes


def build_pool_boundary_pairs(
    teacher_scores: np.ndarray,
    pq_scores: np.ndarray,
    relevance: np.ndarray,
    valid: np.ndarray,
    *,
    pool_k: int,
    negative_window: int = 16,
    negatives_per_positive: int = 4,
    margin_temperature: float = 0.05,
    damage_scale: float = 8.0,
    flip_bonus: float = 2.0,
) -> PairBatch:
    """Mine relevant-vs-unjudged pairs around the deployed pool cutoff.

    A zero relevance entry is called a *mined hard negative* only.  It is not
    interpreted as an explicit non-relevance judgment.  Weights increase for
    small teacher margins, positive PQ margin damage, and PQ-induced flips.
    """

    teacher = np.asarray(teacher_scores, dtype=np.float32)
    pq = np.asarray(pq_scores, dtype=np.float32)
    labels = np.asarray(relevance, dtype=np.uint8)
    mask = np.asarray(valid, dtype=bool)
    if not (teacher.shape == pq.shape == labels.shape == mask.shape):
        raise ValueError("score, label, and valid matrices must match")
    if teacher.ndim != 2 or not 0 < pool_k < teacher.shape[1]:
        raise ValueError("Require 0 < pool_k < candidate count")
    if negative_window <= 0 or negatives_per_positive <= 0:
        raise ValueError("Pair mining counts must be positive")
    if margin_temperature <= 0 or damage_scale < 0 or flip_bonus < 0:
        raise ValueError("Pair weighting parameters are invalid")

    query_rows: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    weights: list[float] = []
    teacher_margins: list[float] = []
    pq_margins: list[float] = []
    flips: list[bool] = []
    order = stable_descending_order(pq, mask)

    for query_index in range(len(teacher)):
        ranked = order[query_index]
        ranked = ranked[mask[query_index, ranked]]
        if len(ranked) <= pool_k:
            continue
        lo = max(0, pool_k - negative_window)
        hi = min(len(ranked), pool_k + negative_window)
        boundary = ranked[lo:hi]
        negative_pool = boundary[labels[query_index, boundary] == 0]
        positive_pool = np.flatnonzero(mask[query_index] & (labels[query_index] > 0))
        if not len(negative_pool) or not len(positive_pool):
            continue
        for positive in positive_pool:
            t_margin = teacher[query_index, positive] - teacher[
                query_index, negative_pool
            ]
            q_margin = pq[query_index, positive] - pq[query_index, negative_pool]
            eligible = np.isfinite(t_margin) & np.isfinite(q_margin) & (t_margin > 0)
            if not np.any(eligible):
                continue
            candidates = negative_pool[eligible]
            tm = t_margin[eligible]
            qm = q_margin[eligible]
            damage = np.maximum(0.0, tm - qm)
            is_flip = qm <= 0
            boundary_weight = np.exp(-np.maximum(tm, 0.0) / margin_temperature)
            pair_weight = (
                1.0
                + boundary_weight
                + damage_scale * damage
                + flip_bonus * is_flip.astype(np.float32)
            )
            selection = np.lexsort((candidates, tm, -pair_weight))
            for selected in selection[:negatives_per_positive]:
                query_rows.append(query_index)
                positives.append(int(positive))
                negatives.append(int(candidates[selected]))
                weights.append(float(pair_weight[selected]))
                teacher_margins.append(float(tm[selected]))
                pq_margins.append(float(qm[selected]))
                flips.append(bool(is_flip[selected]))

    if not query_rows:
        raise ValueError("No valid pool-boundary training pairs were found")
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array /= float(np.mean(weight_array))
    return PairBatch(
        query=np.asarray(query_rows, dtype=np.int64),
        positive=np.asarray(positives, dtype=np.int64),
        negative=np.asarray(negatives, dtype=np.int64),
        weight=weight_array,
        teacher_margin=np.asarray(teacher_margins, dtype=np.float32),
        pq_margin=np.asarray(pq_margins, dtype=np.float32),
        pq_flip=np.asarray(flips, dtype=bool),
    )


def recall_at_k_per_query(
    scores: np.ndarray,
    relevance: np.ndarray,
    relevant_counts: np.ndarray,
    valid: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(relevance, dtype=np.uint8)
    counts = np.asarray(relevant_counts, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)
    if not (values.shape == labels.shape == mask.shape):
        raise ValueError("score, relevance, and valid matrices must match")
    if counts.shape != (len(values),) or np.any(counts <= 0):
        raise ValueError("Every query must have a positive relevant denominator")
    if not 0 < k <= values.shape[1]:
        raise ValueError("k is outside the candidate matrix")
    order = stable_descending_order(values, mask)[:, :k]
    hits = np.take_along_axis(labels, order, axis=1).sum(axis=1)
    return hits.astype(np.float64) / counts


def known_positive_recall_at_k(
    retrieved_rows: np.ndarray,
    positive_rows: np.ndarray,
    positive_valid: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Recall against a padded per-query set of known-positive local rows.

    ``retrieved_rows`` must already be in retrieval order.  This helper is used
    for the end-to-end 100K pilot, where a positive absent from the returned
    IVF-PQ list must remain a miss rather than being appended to an evaluation
    candidate set.
    """

    retrieved = np.asarray(retrieved_rows, dtype=np.int64)
    positives = np.asarray(positive_rows, dtype=np.int64)
    valid = np.asarray(positive_valid, dtype=bool)
    if retrieved.ndim != 2 or positives.ndim != 2:
        raise ValueError("retrieved and positive rows must be matrices")
    if positives.shape != valid.shape or len(retrieved) != len(positives):
        raise ValueError("positive rows and masks must match retrieval queries")
    if not 0 < k <= retrieved.shape[1]:
        raise ValueError("k is outside the retrieved matrix")
    counts = valid.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("Every query must have at least one known positive")
    if np.any(positives[valid] < 0):
        raise ValueError("Valid positive rows must be non-negative")
    top = retrieved[:, :k]
    hits = np.any(
        (top[:, :, None] == positives[:, None, :])
        & valid[:, None, :],
        axis=2,
    ).sum(axis=1)
    return hits.astype(np.float64) / counts


def paired_bootstrap_mean_difference(
    treatment: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    left = np.asarray(treatment, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("Paired metric vectors must be matching and non-empty")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    delta = left - right
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    batch = 2048
    for start in range(0, replicates, batch):
        end = min(replicates, start + batch)
        indices = rng.integers(0, len(delta), size=(end - start, len(delta)))
        means[start:end] = delta[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(delta.mean()),
        "lower": float(np.quantile(means, tail, method="linear")),
        "upper": float(np.quantile(means, 1.0 - tail, method="linear")),
        "confidence": float(confidence),
        "replicates": int(replicates),
    }


def recovery_fraction(*, base: float, treatment: float, teacher: float) -> float:
    gap = float(teacher) - float(base)
    if gap <= 0:
        return 0.0
    return float((float(treatment) - float(base)) / gap)
