#!/usr/bin/env python3
"""Numerical core for RARS-v12 anchored cutoff-aware residual PQ.

The V11 result established a strong 16-byte baseline: rank-64 PCA
coefficients encoded by 16 independent 8-bit product-quantizer blocks.  V12
does not relearn the residual basis and does not backpropagate through hard
assignments.  Instead it performs one closed-form, cutoff-weighted centroid
update inside every *frozen initial Voronoi cell*::

    c* = (sum_i w_i z_i + lambda c_0) / (sum_i w_i + lambda)

The update is clipped to a preregistered radius around the unsupervised
centroid.  This makes the learned component small, deterministic, auditable,
and exactly storage matched to the V11 RPQ baseline.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v12_anchored_cutoff_rpq_v1"


def _matrix(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def deterministic_fold_ids(
    query_ids: list[str], *, fold_count: int = 5
) -> np.ndarray:
    """Assign stable folds without depending on source order."""

    if fold_count <= 1 or not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("Fold assignment needs unique query ids and >=2 folds")
    output = np.empty(len(query_ids), dtype=np.int64)
    for index, query_id in enumerate(query_ids):
        digest = hashlib.sha256(
            b"rars_v12_fresh_fold_v1\0" + str(query_id).encode("utf-8")
        ).digest()
        output[index] = int.from_bytes(digest[:8], "big") % fold_count
    return output


def deterministic_query_priority(query_id: str) -> bytes:
    """Return the frozen selection key for fresh MS MARCO train queries."""

    return hashlib.sha256(
        b"rars_v12_fresh_train_selection_v1\0"
        + str(query_id).encode("utf-8")
    ).digest()


def assign_product_codes(
    coefficients: Any,
    codebooks: Any,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Assign every coefficient block to its nearest product centroid."""

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
        start_dimension = block * block_dimension
        end_dimension = start_dimension + block_dimension
        centroids = books[block]
        centroid_norm = np.sum(centroids * centroids, axis=1)
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            local = values[start:end, start_dimension:end_dimension]
            distances = (
                np.sum(local * local, axis=1)[:, None]
                - 2.0 * (local @ centroids.T)
                + centroid_norm[None, :]
            )
            codes[start:end, block] = np.argmin(distances, axis=1).astype(
                np.uint8
            )
    return codes


def build_cutoff_block_weights(
    queries: Any,
    basis: Any,
    pairs: Any,
    *,
    residual_count: int,
    subquantizers: int = 16,
    cutoff_boost: float = 8.0,
    protection_multiplier: float = 2.0,
    maximum_weight: float = 25.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build document/block weights from query-balanced cutoff pairs.

    A pair contributes to both documents.  Its mass is distributed across
    blocks according to the squared norm of the projected query in that
    block.  Each block is normalized independently so a high-energy direction
    cannot silently dominate merely because of scale.
    """

    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    if query_matrix.shape[1] != projection.shape[0]:
        raise ValueError("Query and basis dimensions disagree")
    if projection.shape[1] % subquantizers or residual_count <= 0:
        raise ValueError("Basis rank or residual count is incompatible")
    if cutoff_boost < 0 or protection_multiplier < 1 or maximum_weight <= 1:
        raise ValueError("Cutoff-weight controls are invalid")
    if len(pairs) == 0:
        return np.ones((residual_count, subquantizers), dtype=np.float32), {
            "pair_count": 0,
            "active_residual_rows": 0,
            "minimum_weight": 1.0,
            "maximum_weight": 1.0,
            "mean_weight": 1.0,
        }
    query_rows = np.asarray(pairs.query, dtype=np.int64)
    positive_rows = np.asarray(pairs.positive_residual_row, dtype=np.int64)
    challenger_rows = np.asarray(pairs.challenger_residual_row, dtype=np.int64)
    balanced = np.asarray(pairs.balanced_weight, dtype=np.float64)
    kinds = np.asarray(pairs.kind, dtype=np.uint8)
    if (
        np.any(query_rows < 0)
        or np.any(query_rows >= len(query_matrix))
        or np.any(positive_rows < 0)
        or np.any(challenger_rows < 0)
        or max(int(positive_rows.max()), int(challenger_rows.max()))
        >= residual_count
    ):
        raise ValueError("Cutoff pair indices are outside the supplied arrays")
    if np.any(balanced <= 0) or not np.all(np.isfinite(balanced)):
        raise ValueError("Cutoff-pair weights must be positive and finite")

    block_dimension = projection.shape[1] // subquantizers
    projected = (query_matrix[query_rows] @ projection).reshape(
        len(query_rows), subquantizers, block_dimension
    )
    sensitivity = np.sum(projected * projected, axis=2, dtype=np.float64)
    denominator = np.maximum(
        sensitivity.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny
    )
    sensitivity /= denominator
    # PROTECTION is encoded as 1 by the frozen V8 pair miner.
    role_scale = np.where(kinds == 1, protection_multiplier, 1.0)
    contribution = sensitivity * (balanced * role_scale)[:, None]
    raw = np.zeros((residual_count, subquantizers), dtype=np.float64)
    for block in range(subquantizers):
        np.add.at(raw[:, block], positive_rows, contribution[:, block])
        np.add.at(raw[:, block], challenger_rows, contribution[:, block])

    weights = np.ones_like(raw)
    for block in range(subquantizers):
        active = raw[:, block] > 0
        if np.any(active):
            local_mean = float(np.mean(raw[active, block]))
            weights[active, block] += cutoff_boost * raw[active, block] / local_mean
    weights = np.minimum(weights, maximum_weight).astype(np.float32)
    active_rows = np.any(raw > 0, axis=1)
    return weights, {
        "pair_count": int(len(pairs)),
        "active_residual_rows": int(active_rows.sum()),
        "active_fraction": float(active_rows.mean()),
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "mean_weight": float(weights.mean()),
        "cutoff_boost": float(cutoff_boost),
        "protection_multiplier": float(protection_multiplier),
    }


def _fixed_assignment_objective(
    values: np.ndarray,
    codes: np.ndarray,
    books: np.ndarray,
    weights: np.ndarray,
    anchor: np.ndarray,
    anchor_pseudocount: float,
) -> float:
    total = 0.0
    block_dimension = books.shape[2]
    for block in range(books.shape[0]):
        local = values[
            :, block * block_dimension : (block + 1) * block_dimension
        ].astype(np.float64)
        reconstruction = books[block, codes[:, block]].astype(np.float64)
        error = local - reconstruction
        total += float(
            np.sum(weights[:, block, None].astype(np.float64) * error * error)
        )
        drift = books[block].astype(np.float64) - anchor[block].astype(np.float64)
        total += float(anchor_pseudocount * np.sum(drift * drift))
    return total


def fit_anchored_cutoff_codebooks(
    coefficients: Any,
    initial_codes: Any,
    initial_codebooks: Any,
    block_weights: Any,
    *,
    anchor_pseudocount: float = 32.0,
    maximum_drift_fraction: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one closed-form weighted centroid update with frozen cells."""

    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    codes = np.asarray(initial_codes, dtype=np.uint8)
    books = np.asarray(initial_codebooks, dtype=np.float32)
    weights = np.asarray(block_weights, dtype=np.float32)
    if books.ndim != 3 or books.shape[1] != 256:
        raise ValueError("Initial codebooks must be 256-entry product books")
    if codes.shape != (len(values), books.shape[0]):
        raise ValueError("Initial code shape differs from the codebooks")
    if weights.shape != (len(values), books.shape[0]):
        raise ValueError("Block weights differ from the product-code shape")
    if values.shape[1] != books.shape[0] * books.shape[2]:
        raise ValueError("Coefficient rank differs from the product codebooks")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Block weights must be positive and finite")
    if anchor_pseudocount < 0 or not 0 < maximum_drift_fraction <= 1:
        raise ValueError("Anchor and drift controls are invalid")

    output = books.copy()
    block_dimension = books.shape[2]
    maximum_observed_fraction = 0.0
    clipped_centroids = 0
    for block in range(books.shape[0]):
        local = values[
            :, block * block_dimension : (block + 1) * block_dimension
        ].astype(np.float64)
        local_weights = weights[:, block].astype(np.float64)
        local_codes = codes[:, block].astype(np.int64)
        numerator = np.zeros((256, block_dimension), dtype=np.float64)
        np.add.at(numerator, local_codes, local * local_weights[:, None])
        denominator = np.bincount(
            local_codes, weights=local_weights, minlength=256
        ).astype(np.float64)
        anchor = books[block].astype(np.float64)
        proposal = (
            numerator + anchor_pseudocount * anchor
        ) / np.maximum(denominator + anchor_pseudocount, np.finfo(np.float64).tiny)[:, None]
        # Empty cells remain byte-identical to their unsupervised anchors.
        proposal[denominator == 0] = anchor[denominator == 0]
        block_rms = float(np.sqrt(np.mean(local * local)))
        radius = maximum_drift_fraction * max(
            block_rms, float(np.finfo(np.float32).tiny)
        )
        drift = proposal - anchor
        norms = np.linalg.norm(drift, axis=1)
        clipped = norms > radius
        if np.any(clipped):
            drift[clipped] *= (radius / norms[clipped])[:, None]
            clipped_centroids += int(clipped.sum())
        output[block] = (anchor + drift).astype(np.float32)
        maximum_observed_fraction = max(
            maximum_observed_fraction,
            float(np.max(np.linalg.norm(drift, axis=1)) / max(block_rms, 1e-30)),
        )

    before = _fixed_assignment_objective(
        values, codes, books, weights, books, anchor_pseudocount
    )
    after = _fixed_assignment_objective(
        values, codes, output, weights, books, anchor_pseudocount
    )
    if after > before + max(1e-8, abs(before) * 1e-10):
        raise AssertionError("Anchored closed-form update increased its objective")
    return output, {
        "fixed_assignment_objective_before": float(before),
        "fixed_assignment_objective_after": float(after),
        "relative_objective_change": float((after - before) / before)
        if before > 0
        else 0.0,
        "maximum_centroid_drift_fraction": float(maximum_observed_fraction),
        "drift_limit_fraction": float(maximum_drift_fraction),
        "clipped_centroids": int(clipped_centroids),
        "anchor_pseudocount": float(anchor_pseudocount),
        "payload_bytes_per_document": int(books.shape[0]),
        "codebook_bytes": int(output.nbytes),
        "frozen_assignment_update": True,
    }


def ca_rpq_decision(
    *,
    primary_vs_unsupervised: dict[str, Any],
    primary_vs_base: dict[str, Any],
    seed_gains: Any,
    fold_gains: Any,
    candidate_gap_recovery: float,
    unsupervised_mrr: float,
    ca_mrr: float,
    unsupervised_ndcg: float,
    ca_ndcg: float,
    payload_bytes_per_document: int,
    full_corpus_codes_materialized: bool,
    all_objectives_nonincreasing: bool,
    maximum_centroid_drift_fraction: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered development gate without method selection."""

    seeds = np.asarray(seed_gains, dtype=np.float64)
    folds = np.asarray(fold_gains, dtype=np.float64)
    if seeds.ndim != 1 or len(seeds) < 3 or np.any(~np.isfinite(seeds)):
        raise ValueError("At least three finite seed gains are required")
    if folds.ndim != 1 or len(folds) != 5 or np.any(~np.isfinite(folds)):
        raise ValueError("Exactly five finite fold gains are required")
    primary_gain = float(primary_vs_unsupervised["mean_difference"])
    gates = {
        "minimum_primary_gain": primary_gain
        >= float(thresholds["minimum_recall_gain_over_unsupervised"]),
        "bootstrap_lower_positive": primary_vs_unsupervised["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "randomization_significant": primary_vs_unsupervised[
            "randomization_p_value_one_sided"
        ] <= float(thresholds["maximum_randomization_p_value"]),
        "minimum_query_support": primary_vs_unsupervised["improved_queries"]
        >= int(thresholds["minimum_improved_queries"]),
        "minimum_net_support": (
            primary_vs_unsupervised["improved_queries"]
            - primary_vs_unsupervised["harmed_queries"]
        ) >= int(thresholds["minimum_net_improved_queries"]),
        "minimum_gain_over_base": primary_vs_base["mean_difference"]
        >= float(thresholds["minimum_recall_gain_over_base"]),
        "all_seed_gains_nonnegative": float(np.min(seeds))
        >= float(thresholds["minimum_each_seed_gain"]),
        "minimum_median_seed_gain": float(np.median(seeds))
        >= float(thresholds["minimum_median_seed_gain"]),
        "worst_fold_nonnegative": float(np.min(folds))
        >= float(thresholds["minimum_worst_fold_gain"]),
        "minimum_gap_recovery": candidate_gap_recovery
        >= float(thresholds["minimum_candidate_gap_recovery_fraction"]),
        "mrr_guardrail": ca_mrr - unsupervised_mrr
        >= float(thresholds["minimum_mrr_change"]),
        "ndcg_guardrail": ca_ndcg - unsupervised_ndcg
        >= float(thresholds["minimum_ndcg_change"]),
        "payload_exactly_sixteen_bytes": payload_bytes_per_document == 16,
        "full_corpus_codes_materialized": bool(full_corpus_codes_materialized),
        "closed_form_objectives_nonincreasing": bool(
            all_objectives_nonincreasing
        ),
        "centroid_drift_within_limit": maximum_centroid_drift_fraction
        <= float(thresholds["maximum_centroid_drift_fraction"]) + 1e-7,
    }
    decision = (
        thresholds["go_decision"]
        if all(gates.values())
        else thresholds["stop_decision"]
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "decision": decision,
        "gates": {name: bool(value) for name, value in gates.items()},
        "failed_gates": [name for name, value in gates.items() if not value],
        "primary_gain": primary_gain,
        "seed_gains": seeds.tolist(),
        "median_seed_gain": float(np.median(seeds)),
        "fold_gains": folds.tolist(),
        "fresh_confirmation_access_authorized": False,
        "go_authorizes_only_protocol_writing": True,
    }
