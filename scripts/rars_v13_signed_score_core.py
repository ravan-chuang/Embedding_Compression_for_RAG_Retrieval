#!/usr/bin/env python3
"""Numerical core for V13 fixed-assignment signed-score RPQ.

V12 moved each RPQ centroid toward residual vectors with a positive scalar
weight.  That objective cannot distinguish an over-estimated retrieval score
from an under-estimated one.  V13 instead distils the signed block score
``q_block @ residual_block`` into the centroid selected by the frozen
unsupervised RPQ assignment.  Each update is a four-dimensional anchored
ridge solve and never changes a stored code.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v13_signed_score_distilled_rpq_v1"


def _matrix(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def deterministic_query_priority(query_id: str) -> bytes:
    return hashlib.sha256(
        b"rars_v13_fresh_train_selection_v1\0"
        + str(query_id).encode("utf-8")
    ).digest()


def deterministic_fold_ids(
    query_ids: list[str], *, fold_count: int = 5
) -> np.ndarray:
    if fold_count <= 1 or not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("Fold assignment needs unique query ids and >=2 folds")
    output = np.empty(len(query_ids), dtype=np.int64)
    for index, query_id in enumerate(query_ids):
        digest = hashlib.sha256(
            b"rars_v13_fresh_fold_v1\0" + str(query_id).encode("utf-8")
        ).digest()
        output[index] = int.from_bytes(digest[:8], "big") % fold_count
    return output


def assign_product_codes(
    coefficients: Any,
    codebooks: Any,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Assign coefficient blocks to the nearest initial RPQ centroid."""

    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    books = np.asarray(codebooks, dtype=np.float32)
    if books.ndim != 3 or books.shape[1] != 256:
        raise ValueError("Codebooks must have shape (M, 256, block_dimension)")
    if values.shape[1] != books.shape[0] * books.shape[2] or batch_size <= 0:
        raise ValueError("Coefficient rank and product codebooks disagree")
    if not np.all(np.isfinite(books)):
        raise ValueError("Codebooks contain non-finite values")
    codes = np.empty((len(values), books.shape[0]), dtype=np.uint8)
    block_dimension = books.shape[2]
    for block in range(books.shape[0]):
        local_centroids = books[block]
        centroid_norm = np.sum(local_centroids * local_centroids, axis=1)
        left = block * block_dimension
        right = left + block_dimension
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            local = values[start:end, left:right]
            distances = (
                np.sum(local * local, axis=1)[:, None]
                - 2.0 * (local @ local_centroids.T)
                + centroid_norm[None, :]
            )
            codes[start:end, block] = np.argmin(distances, axis=1).astype(
                np.uint8
            )
    return codes


def _stable_top(scores: np.ndarray, rows: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or count > len(scores):
        raise ValueError("Stable selection count is outside the candidate pool")
    return np.lexsort((rows, -scores))[:count]


def build_signed_score_statistics(
    queries: Any,
    basis: Any,
    coefficients: Any,
    initial_codes: Any,
    candidate_rows: Any,
    residual_lookup: Any,
    base_scores: Any,
    relevance: Any,
    *,
    top_b: int,
    final_k: int,
    cutoff_boost: float,
    margin_temperature: float,
    known_positive_multiplier: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Accumulate sufficient statistics for signed block-score regression.

    Every Base Top-B candidate contributes sixteen observations.  For block
    ``m`` the regressor input is the projected query block and the target is
    its signed dot product with the exact residual coefficient block.  The
    stored code chooses which of the 256 independent regressors receives the
    observation.
    """

    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    codes = np.asarray(initial_codes, dtype=np.uint8)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    scores = _matrix(base_scores, name="base_scores", dtype=np.float32)
    labels = _matrix(relevance, name="relevance", dtype=np.uint8)
    if not (rows.shape == lookup.shape == scores.shape == labels.shape):
        raise ValueError("Candidate arrays must share a shape")
    if len(query_matrix) != len(rows) or query_matrix.shape[1] != projection.shape[0]:
        raise ValueError("Query and candidate dimensions disagree")
    if codes.ndim != 2 or len(codes) != len(values):
        raise ValueError("Initial codes differ from coefficient rows")
    if projection.shape[1] != values.shape[1] or values.shape[1] % codes.shape[1]:
        raise ValueError("Basis, coefficients, and product blocks disagree")
    if (
        np.any(lookup < 0)
        or np.any(lookup >= len(values))
        or not 0 < final_k <= top_b <= rows.shape[1]
        or cutoff_boost < 0
        or margin_temperature <= 0
        or known_positive_multiplier < 1
    ):
        raise ValueError("Invalid lookup or signed-score controls")

    subquantizers = codes.shape[1]
    block_dimension = values.shape[1] // subquantizers
    projected = (query_matrix @ projection).reshape(
        len(query_matrix), subquantizers, block_dimension
    )
    normal = np.zeros(
        (subquantizers, 256, block_dimension, block_dimension),
        dtype=np.float64,
    )
    rhs = np.zeros((subquantizers, 256, block_dimension), dtype=np.float64)
    target_square = np.zeros((subquantizers, 256), dtype=np.float64)
    mass = np.zeros((subquantizers, 256), dtype=np.float64)
    positive_observations = 0
    minimum_weight = np.inf
    maximum_weight = 0.0
    weight_sum = 0.0

    for query_index in range(len(rows)):
        ordering = _stable_top(
            scores[query_index], rows[query_index], top_b
        )
        threshold = float(scores[query_index, ordering[final_k - 1]])
        local_scores = scores[query_index, ordering].astype(np.float64)
        weights = 1.0 + cutoff_boost * np.exp(
            -np.abs(local_scores - threshold) / margin_temperature
        )
        local_positive = labels[query_index, ordering] == 1
        weights[local_positive] *= known_positive_multiplier
        positive_observations += int(local_positive.sum())
        minimum_weight = min(minimum_weight, float(weights.min()))
        maximum_weight = max(maximum_weight, float(weights.max()))
        weight_sum += float(weights.sum())
        selected_rows = lookup[query_index, ordering]
        local_codes = codes[selected_rows]
        for block in range(subquantizers):
            query_block = projected[query_index, block].astype(np.float64)
            left = block * block_dimension
            right = left + block_dimension
            targets = (
                values[selected_rows, left:right].astype(np.float64)
                @ query_block
            )
            centroid_ids = local_codes[:, block].astype(np.int64)
            weighted_outer = np.outer(query_block, query_block)
            for position, centroid_id in enumerate(centroid_ids):
                weight = float(weights[position])
                target = float(targets[position])
                normal[block, centroid_id] += weight * weighted_outer
                rhs[block, centroid_id] += weight * target * query_block
                target_square[block, centroid_id] += weight * target * target
                mass[block, centroid_id] += weight

    observation_count = len(rows) * top_b
    return {
        "normal": normal,
        "rhs": rhs,
        "target_square": target_square,
        "mass": mass,
    }, {
        "query_count": int(len(rows)),
        "candidate_observations": int(observation_count),
        "block_observations": int(observation_count * subquantizers),
        "known_positive_observations": int(positive_observations),
        "active_centroid_cells": int(np.sum(mass > 0)),
        "minimum_observation_weight": float(minimum_weight),
        "maximum_observation_weight": float(maximum_weight),
        "mean_observation_weight": float(weight_sum / observation_count),
    }


def _quadratic_objective(
    centroid: np.ndarray,
    normal: np.ndarray,
    rhs: np.ndarray,
    target_square: float,
    anchor: np.ndarray,
    anchor_matrix: np.ndarray,
) -> float:
    data = (
        float(centroid @ normal @ centroid)
        - 2.0 * float(rhs @ centroid)
        + float(target_square)
    )
    drift = centroid - anchor
    return data + float(drift @ anchor_matrix @ drift)


def fit_signed_score_codebooks(
    coefficients: Any,
    initial_codebooks: Any,
    statistics: dict[str, Any],
    *,
    anchor_ratio: float,
    maximum_drift_fraction: float,
    jitter: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve anchored signed-score regression with frozen assignments."""

    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    books = np.asarray(initial_codebooks, dtype=np.float32)
    if books.ndim != 3 or books.shape[1] != 256:
        raise ValueError("Initial codebooks must be (M, 256, block_dimension)")
    if values.shape[1] != books.shape[0] * books.shape[2]:
        raise ValueError("Coefficient rank differs from the codebooks")
    if anchor_ratio < 0 or not 0 < maximum_drift_fraction <= 1 or jitter <= 0:
        raise ValueError("Anchor, drift, or jitter controls are invalid")
    normal = np.asarray(statistics["normal"], dtype=np.float64)
    rhs = np.asarray(statistics["rhs"], dtype=np.float64)
    target_square = np.asarray(statistics["target_square"], dtype=np.float64)
    mass = np.asarray(statistics["mass"], dtype=np.float64)
    expected_normal = (books.shape[0], 256, books.shape[2], books.shape[2])
    if (
        normal.shape != expected_normal
        or rhs.shape != books.shape
        or target_square.shape != books.shape[:2]
        or mass.shape != books.shape[:2]
        or np.any(mass < 0)
        or not all(
            np.all(np.isfinite(array))
            for array in (normal, rhs, target_square, mass)
        )
    ):
        raise ValueError("Signed-score sufficient statistics are invalid")

    output = books.copy()
    identity = np.eye(books.shape[2], dtype=np.float64)
    before = 0.0
    after = 0.0
    active = 0
    clipped = 0
    maximum_observed_fraction = 0.0
    objective_increases = 0
    block_summaries: list[dict[str, Any]] = []
    for block in range(books.shape[0]):
        left = block * books.shape[2]
        right = left + books.shape[2]
        local_values = values[:, left:right].astype(np.float64)
        block_rms = float(np.sqrt(np.mean(local_values * local_values)))
        radius = maximum_drift_fraction * max(
            block_rms, float(np.finfo(np.float32).tiny)
        )
        total_mass = float(mass[block].sum())
        covariance = normal[block].sum(axis=0) / max(
            total_mass, np.finfo(np.float64).tiny
        )
        block_active = 0
        block_before = 0.0
        block_after = 0.0
        for centroid_id in range(256):
            if mass[block, centroid_id] <= 0:
                continue
            active += 1
            block_active += 1
            anchor = books[block, centroid_id].astype(np.float64)
            anchor_matrix = (
                anchor_ratio * mass[block, centroid_id] * covariance
            )
            system = normal[block, centroid_id] + anchor_matrix + jitter * identity
            target = rhs[block, centroid_id] + anchor_matrix @ anchor
            proposal = np.linalg.solve(system, target)
            drift = proposal - anchor
            norm = float(np.linalg.norm(drift))
            if norm > radius:
                drift *= radius / norm
                proposal = anchor + drift
                clipped += 1
                norm = radius
            local_before = _quadratic_objective(
                anchor,
                normal[block, centroid_id],
                rhs[block, centroid_id],
                target_square[block, centroid_id],
                anchor,
                anchor_matrix,
            )
            local_after = _quadratic_objective(
                proposal,
                normal[block, centroid_id],
                rhs[block, centroid_id],
                target_square[block, centroid_id],
                anchor,
                anchor_matrix,
            )
            tolerance = 1e-8 * max(1.0, abs(local_before))
            if local_after > local_before + tolerance:
                objective_increases += 1
                raise ValueError("Signed-score update increased its objective")
            output[block, centroid_id] = proposal.astype(np.float32)
            before += local_before
            after += local_after
            block_before += local_before
            block_after += local_after
            maximum_observed_fraction = max(
                maximum_observed_fraction,
                norm / max(block_rms, np.finfo(np.float64).tiny),
            )
        block_summaries.append(
            {
                "block": block,
                "active_centroids": block_active,
                "observation_mass": total_mass,
                "training_block_rms": block_rms,
                "drift_radius": radius,
                "objective_before": block_before,
                "objective_after": block_after,
            }
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Signed-score codebooks contain non-finite values")
    return output, {
        "active_centroids": active,
        "inactive_centroids": int(books.shape[0] * 256 - active),
        "clipped_centroids": clipped,
        "maximum_centroid_drift_fraction": maximum_observed_fraction,
        "signed_score_objective_before": before,
        "signed_score_objective_after": after,
        "objective_nonincreasing": bool(after <= before + 1e-8),
        "objective_increases": objective_increases,
        "assignment_changes": 0,
        "fixed_assignments": True,
        "anchor_ratio": float(anchor_ratio),
        "maximum_drift_fraction": float(maximum_drift_fraction),
        "jitter": float(jitter),
        "block_summaries": block_summaries,
    }


def signed_score_decision(
    *,
    primary_vs_unsupervised: dict[str, Any],
    primary_vs_pca16: dict[str, Any],
    primary_vs_base: dict[str, Any],
    seed_gains: list[float],
    fold_gains: list[float],
    candidate_gap_recovery: float,
    unsupervised_mrr: float,
    challenger_mrr: float,
    unsupervised_ndcg: float,
    challenger_ndcg: float,
    payload_bytes_per_document: int,
    full_corpus_codes_materialized: bool,
    all_objectives_nonincreasing: bool,
    maximum_centroid_drift_fraction: float,
    assignment_changes: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    if len(seed_gains) != 3 or len(fold_gains) != 5:
        raise ValueError("V13 requires three seed gains and five fold gains")
    support = int(primary_vs_unsupervised["improved_queries"])
    net_support = support - int(primary_vs_unsupervised["harmed_queries"])
    gates = {
        "minimum_gain_over_unsupervised": float(primary_vs_unsupervised["mean_difference"])
        >= float(thresholds["minimum_recall_gain_over_unsupervised"]),
        "minimum_gain_over_pca16": float(primary_vs_pca16["mean_difference"])
        >= float(thresholds["minimum_recall_gain_over_pca16"]),
        "minimum_gain_over_base": float(primary_vs_base["mean_difference"])
        >= float(thresholds["minimum_recall_gain_over_base"]),
        "bootstrap_lower_above_zero": float(primary_vs_unsupervised["lower"])
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "randomization_p_value": float(
            primary_vs_unsupervised["randomization_p_value_one_sided"]
        )
        <= float(thresholds["maximum_randomization_p_value"]),
        "improved_query_support": support
        >= int(thresholds["minimum_improved_queries"]),
        "net_improved_query_support": net_support
        >= int(thresholds["minimum_net_improved_queries"]),
        "all_seed_gains_nonnegative": min(seed_gains)
        >= float(thresholds["minimum_each_seed_gain"]),
        "median_seed_gain": float(np.median(seed_gains))
        >= float(thresholds["minimum_median_seed_gain"]),
        "worst_fold_gain": min(fold_gains)
        >= float(thresholds["minimum_worst_fold_gain"]),
        "candidate_gap_recovery": candidate_gap_recovery
        >= float(thresholds["minimum_candidate_gap_recovery_fraction"]),
        "mrr_guardrail": challenger_mrr - unsupervised_mrr
        >= float(thresholds["minimum_mrr_change"]),
        "ndcg_guardrail": challenger_ndcg - unsupervised_ndcg
        >= float(thresholds["minimum_ndcg_change"]),
        "payload_exactly_16_bytes": payload_bytes_per_document == 16,
        "full_corpus_codes_materialized": bool(full_corpus_codes_materialized),
        "objectives_nonincreasing": bool(all_objectives_nonincreasing),
        "centroid_drift_guardrail": maximum_centroid_drift_fraction
        <= float(thresholds["maximum_centroid_drift_fraction"]) + 1e-7,
        "assignments_unchanged": assignment_changes
        <= int(thresholds["maximum_assignment_changes"]),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    passed = all(gates.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "decision": thresholds["go_decision"] if passed else thresholds["stop_decision"],
        "all_gates_passed": passed,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "improved_queries": support,
        "harmed_queries": int(primary_vs_unsupervised["harmed_queries"]),
        "net_improved_queries": net_support,
    }
