#!/usr/bin/env python3
"""Pure numerical core for the RARS-v17 million-scale setting diagnostic.

The module intentionally contains no dataframe, filesystem, model-training,
or plotting dependencies.  Its inputs and outputs are NumPy arrays and small
JSON-compatible dictionaries so the protocol can be tested independently of
the experiment driver.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np


PROTOCOL_ID = "rars_v17_million_scale_setting_transfer_v1"

DEFAULT_SETTING_THRESHOLDS: dict[str, float | int] = {
    "min_headroom": 0.01,
    "material_headroom": 0.02,
    "min_capacity_gain": 0.005,
    "min_coding_gap": 0.005,
    "min_objective_gain": 0.005,
    "min_domain_interaction": 0.005,
    "min_pooled_recovery": 0.5,
    "pooled_noninferiority": -0.002,
    "min_improved_queries": 20,
    "min_net_share": 0.005,
    "min_gap_recovery": 0.15,
    "worst_domain_floor": -0.002,
}

# Backward-compatible name for callers created before the V17 protocol stopped
# describing the two different IVF recipes as a pure domain-causal contrast.
DEFAULT_CAUSAL_THRESHOLDS = DEFAULT_SETTING_THRESHOLDS


def _matrix(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _vector(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def deterministic_query_priority(
    query_id: str, *, namespace: str = PROTOCOL_ID
) -> bytes:
    """Return an order-independent deterministic selection key."""

    return hashlib.sha256(
        str(namespace).encode("utf-8")
        + b"\0query-priority\0"
        + str(query_id).encode("utf-8")
    ).digest()


def deterministic_fold_ids(
    query_ids: list[str] | tuple[str, ...] | np.ndarray,
    *,
    fold_count: int = 5,
    namespace: str = PROTOCOL_ID,
) -> np.ndarray:
    """Assign unique query ids to stable folds, independent of input order."""

    ids = [str(query_id) for query_id in query_ids]
    if fold_count <= 1 or not ids or len(ids) != len(set(ids)):
        raise ValueError("Fold assignment needs unique query ids and >=2 folds")
    output = np.empty(len(ids), dtype=np.int64)
    for index, query_id in enumerate(ids):
        digest = hashlib.sha256(
            str(namespace).encode("utf-8")
            + b"\0fold\0"
            + query_id.encode("utf-8")
        ).digest()
        output[index] = int.from_bytes(digest[:8], "big") % int(fold_count)
    return output


def _stable_top_b(scores: np.ndarray, rows: np.ndarray, top_b: int) -> np.ndarray:
    """Select by descending score, breaking ties by ascending document row."""

    score_vector = np.asarray(scores)
    row_vector = np.asarray(rows)
    if score_vector.ndim != 1 or row_vector.shape != score_vector.shape:
        raise ValueError("score and row vectors must match")
    if not 0 < top_b <= len(score_vector):
        raise ValueError("Top-B is outside the candidate width")
    valid = row_vector >= 0
    safe_rows = np.where(valid, row_vector, np.iinfo(np.int64).max)
    safe_scores = np.where(valid, score_vector, -np.inf)
    return np.lexsort((safe_rows, -safe_scores))[:top_b]


def score_fp32_sidecar_candidates(
    queries: Any,
    candidate_rows: Any,
    residual_lookup: Any,
    base_scores: Any,
    residuals: Any,
    basis: Any,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    """Apply an FP32 residual-basis correction to the frozen Base Top-B only."""

    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _matrix(base_scores, name="base_scores", dtype=np.float32)
    residual_matrix = _matrix(residuals, name="residuals", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    if not (rows.shape == lookup.shape == base.shape):
        raise ValueError("Candidate rows, lookup, and scores must match")
    if (
        len(query_matrix) != len(rows)
        or query_matrix.shape[1] != projection.shape[0]
        or residual_matrix.shape[1] != projection.shape[0]
    ):
        raise ValueError("Query, candidate, residual, and basis dimensions disagree")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Alpha must be finite and non-negative")

    projected_queries = query_matrix @ projection
    projected_residuals = residual_matrix @ projection
    output = base.copy()
    for query_index in range(len(rows)):
        selected = _stable_top_b(base[query_index], rows[query_index], top_b)
        selected_lookup = lookup[query_index, selected]
        valid = selected_lookup >= 0
        if not np.any(valid):
            continue
        if np.max(selected_lookup[valid]) >= len(projected_residuals):
            raise ValueError("Residual lookup is outside the FP32 matrix")
        output[query_index, selected[valid]] += np.float32(alpha) * (
            projected_residuals[selected_lookup[valid]]
            @ projected_queries[query_index]
        )
    return output


# Preserve the established V11 spelling for experiment drivers.
score_float_sidecar_candidates = score_fp32_sidecar_candidates


def aggregate_score_error_weights(
    residual_rows: Any,
    score_errors: Any,
    *,
    residual_count: int,
) -> np.ndarray:
    """Aggregate absolute score error by residual row with ``np.bincount``."""

    rows = np.asarray(residual_rows, dtype=np.int64)
    errors = np.asarray(score_errors, dtype=np.float64)
    if rows.shape != errors.shape or rows.ndim not in (1, 2):
        raise ValueError("Residual rows and score errors must have matching shapes")
    if residual_count <= 0 or not np.all(np.isfinite(errors)):
        raise ValueError("Residual count and score errors must be valid")
    flat_rows = rows.reshape(-1)
    flat_errors = np.abs(errors.reshape(-1))
    valid = flat_rows >= 0
    if np.any(flat_rows[valid] >= residual_count):
        raise ValueError("Residual row is outside the residual matrix")
    return np.bincount(
        flat_rows[valid],
        weights=flat_errors[valid],
        minlength=residual_count,
    ).astype(np.float64, copy=False)


def _orient_columns(basis: np.ndarray) -> np.ndarray:
    output = np.asarray(basis, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


def fit_score_error_weighted_basis(
    residuals: Any,
    score_errors: Any,
    *,
    rank: int,
    residual_rows: Any | None = None,
    minimum_weight: float = 0.0,
) -> np.ndarray:
    """Fit a deterministic uncentred basis weighted by absolute score error.

    ``score_errors`` can contain one value per residual.  When candidate-level
    errors are supplied, ``residual_rows`` maps them to residual rows and
    duplicate contributions are aggregated without pandas.
    """

    values = _matrix(residuals, name="residuals", dtype=np.float64)
    if not 0 < rank <= min(values.shape) or minimum_weight < 0:
        raise ValueError("Basis rank or minimum weight is invalid")
    if residual_rows is None:
        weights = np.asarray(score_errors, dtype=np.float64)
        if weights.ndim != 1 or weights.shape != (len(values),):
            raise ValueError("Score errors must have one value per residual")
        if not np.all(np.isfinite(weights)):
            raise ValueError("Score errors contain non-finite values")
        weights = np.abs(weights)
    else:
        weights = aggregate_score_error_weights(
            residual_rows,
            score_errors,
            residual_count=len(values),
        )
    weights = np.maximum(weights, float(minimum_weight))
    if not np.any(weights > 0):
        raise ValueError("At least one positive score-error weight is required")
    weights /= float(np.mean(weights[weights > 0]))
    weighted = values * np.sqrt(weights[:, None])
    covariance = weighted.T @ weighted
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")[:rank]
    return _orient_columns(eigenvectors[:, order]).astype(np.float32)


def subspace_alignment_metrics(first_basis: Any, second_basis: Any) -> dict[str, Any]:
    """Measure basis-invariant principal-angle alignment between subspaces."""

    first = _matrix(first_basis, name="first_basis", dtype=np.float64)
    second = _matrix(second_basis, name="second_basis", dtype=np.float64)
    if first.shape[0] != second.shape[0] or not first.shape[1] or not second.shape[1]:
        raise ValueError("Subspaces must have the same ambient dimension")
    q_first, _ = np.linalg.qr(first, mode="reduced")
    q_second, _ = np.linalg.qr(second, mode="reduced")
    cosines = np.clip(
        np.linalg.svd(q_first.T @ q_second, compute_uv=False), 0.0, 1.0
    )
    angles = np.arccos(cosines)
    overlap = float(np.sum(cosines * cosines))
    return {
        "ambient_dimension": int(first.shape[0]),
        "first_rank": int(q_first.shape[1]),
        "second_rank": int(q_second.shape[1]),
        "principal_cosines": [float(value) for value in cosines],
        "principal_angles_radians": [float(value) for value in angles],
        "mean_squared_canonical_correlation": float(
            np.mean(cosines * cosines)
        ),
        "minimum_cosine": float(np.min(cosines)),
        "maximum_principal_angle_radians": float(np.max(angles)),
        "projection_frobenius_distance": float(
            math.sqrt(max(0.0, q_first.shape[1] + q_second.shape[1] - 2 * overlap))
        ),
    }


def paired_query_inference(
    treatment: Any,
    baseline: Any,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    randomization_replicates: int,
    randomization_seed: int,
    confidence: float = 0.95,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Deterministic paired query bootstrap and sign-flip randomization."""

    treated = _vector(treatment, name="treatment")
    control = _vector(baseline, name="baseline")
    if treated.shape != control.shape:
        raise ValueError("Treatment and baseline vectors must match")
    if (
        bootstrap_replicates <= 0
        or randomization_replicates <= 0
        or not 0 < confidence < 1
        or alternative not in {"greater", "less", "two-sided"}
    ):
        raise ValueError("Inference controls are invalid")
    differences = treated - control
    count = len(differences)
    observed = float(np.mean(differences))

    bootstrap_rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_means = np.empty(bootstrap_replicates, dtype=np.float64)
    for start in range(0, bootstrap_replicates, 2048):
        end = min(start + 2048, bootstrap_replicates)
        indices = bootstrap_rng.integers(0, count, size=(end - start, count))
        bootstrap_means[start:end] = np.mean(differences[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])

    randomization_rng = np.random.default_rng(int(randomization_seed))
    extreme = 0
    for start in range(0, randomization_replicates, 4096):
        end = min(start + 4096, randomization_replicates)
        signs = randomization_rng.integers(
            0, 2, size=(end - start, count), dtype=np.int8
        )
        permuted = np.mean(differences * (2 * signs - 1), axis=1)
        if alternative == "greater":
            extreme += int(np.count_nonzero(permuted >= observed))
        elif alternative == "less":
            extreme += int(np.count_nonzero(permuted <= observed))
        else:
            extreme += int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
    p_value = float((extreme + 1) / (randomization_replicates + 1))
    support = contrast_support(treated, control)
    return {
        "query_count": int(count),
        "mean_difference": observed,
        "lower": float(lower),
        "upper": float(upper),
        "confidence": float(confidence),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "randomization_replicates": int(randomization_replicates),
        "randomization_seed": int(randomization_seed),
        "alternative": alternative,
        "randomization_p_value": p_value,
        "randomization_p_value_one_sided": p_value
        if alternative != "two-sided"
        else None,
        **support,
    }


paired_inference = paired_query_inference


def contrast_support(
    treatment: Any,
    baseline: Any | None = None,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Count query-level support and report its signed net share."""

    treated = _vector(treatment, name="treatment")
    control = (
        np.zeros_like(treated)
        if baseline is None
        else _vector(baseline, name="baseline")
    )
    if treated.shape != control.shape or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Contrast vectors or tolerance are invalid")
    difference = treated - control
    improved = int(np.count_nonzero(difference > tolerance))
    harmed = int(np.count_nonzero(difference < -tolerance))
    unchanged = int(len(difference) - improved - harmed)
    return {
        "improved_queries": improved,
        "harmed_queries": harmed,
        "unchanged_queries": unchanged,
        "improved_share": float(improved / len(difference)),
        "harmed_share": float(harmed / len(difference)),
        "net_improved_queries": int(improved - harmed),
        "net_share": float((improved - harmed) / len(difference)),
    }


def candidate_gap_decomposition(
    method: Any,
    baseline: Any,
    ceiling: Any,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Decompose candidate-set headroom into recovery, harm, and remainder."""

    method_values = _vector(method, name="method")
    baseline_values = _vector(baseline, name="baseline")
    ceiling_values = _vector(ceiling, name="ceiling")
    if not (
        method_values.shape == baseline_values.shape == ceiling_values.shape
    ):
        raise ValueError("Method, baseline, and ceiling vectors must match")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Tolerance must be finite and non-negative")

    headroom = ceiling_values - baseline_values
    gain = method_values - baseline_values
    recoverable = np.maximum(headroom, 0.0)
    recovered = np.minimum(np.maximum(gain, 0.0), recoverable)
    harm = np.maximum(-gain, 0.0)
    overshoot = np.maximum(gain - recoverable, 0.0)
    total_headroom = float(np.mean(headroom))
    method_gain = float(np.mean(gain))
    recovery_fraction = (
        method_gain / total_headroom if total_headroom > tolerance else 0.0
    )
    recoverable_total = float(np.sum(recoverable))
    clipped_recovery_fraction = (
        float(np.sum(recovered) / recoverable_total)
        if recoverable_total > tolerance
        else 0.0
    )
    headroom_mask = headroom > tolerance
    return {
        "query_count": int(len(method_values)),
        "baseline_mean": float(np.mean(baseline_values)),
        "method_mean": float(np.mean(method_values)),
        "ceiling_mean": float(np.mean(ceiling_values)),
        "candidate_headroom": total_headroom,
        "method_gain": method_gain,
        "remaining_gap": float(np.mean(ceiling_values - method_values)),
        "gap_recovery_fraction": float(recovery_fraction),
        "clipped_gap_recovery_fraction": clipped_recovery_fraction,
        "mean_recovered_headroom": float(np.mean(recovered)),
        "mean_harm": float(np.mean(harm)),
        "mean_overshoot": float(np.mean(overshoot)),
        "headroom_queries": int(np.count_nonzero(headroom_mask)),
        "recovered_queries": int(
            np.count_nonzero(headroom_mask & (gain > tolerance))
        ),
        "fully_recovered_queries": int(
            np.count_nonzero(headroom_mask & (gain >= headroom - tolerance))
        ),
        "harmed_queries": int(np.count_nonzero(gain < -tolerance)),
    }


def _observed_number(
    observed: Mapping[str, Any], canonical: str, *aliases: str
) -> float:
    for key in (canonical, *aliases):
        if key in observed:
            value = float(observed[key])
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            return value
    raise ValueError(f"Missing observed value: {canonical}")


def setting_transfer_decision(
    observed: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered V17 million-scale setting decision tree."""

    limits = dict(DEFAULT_SETTING_THRESHOLDS)
    limits.update(thresholds)
    for key in DEFAULT_SETTING_THRESHOLDS:
        if key not in limits or not np.isfinite(float(limits[key])):
            raise ValueError(f"Invalid causal threshold: {key}")

    n_queries = int(
        _observed_number(observed, "n_queries", "query_count", "queries")
    )
    if n_queries <= 0:
        raise ValueError("n_queries must be positive")
    headroom = _observed_number(
        observed, "headroom", "candidate_headroom", "candidate_gap"
    )
    capacity_gain = _observed_number(
        observed, "capacity_gain", "fp32_capacity_gain"
    )
    coding_gap = _observed_number(
        observed, "coding_gap", "fp32_minus_coded_gain"
    )
    objective_gain = _observed_number(
        observed, "objective_gain", "learned_objective_gain"
    )
    setting_interaction = _observed_number(
        observed,
        "setting_interaction",
        "fit_setting_interaction",
        "domain_interaction",
        "domain_interaction_gain",
    )
    pooled_recovery = _observed_number(
        observed, "pooled_recovery", "pooled_recovery_fraction"
    )
    pooled_gain = _observed_number(
        observed, "pooled_gain", "pooled_difference", "pooled_delta"
    )
    improved_queries = int(
        _observed_number(observed, "improved_queries", "queries_improved")
    )
    harmed_queries = int(
        _observed_number(observed, "harmed_queries", "queries_harmed")
    )
    net_share = (
        float(observed["net_share"])
        if "net_share" in observed
        else (improved_queries - harmed_queries) / n_queries
    )
    if not np.isfinite(net_share):
        raise ValueError("net_share must be finite")
    gap_recovery = _observed_number(
        observed, "gap_recovery", "gap_recovery_fraction"
    )
    worst_domain_gain = _observed_number(
        observed, "worst_domain_gain", "minimum_domain_gain"
    )

    required_improved = max(
        int(limits["min_improved_queries"]),
        int(math.ceil(0.01 * n_queries)),
    )
    gates = {
        "has_minimum_headroom": headroom >= float(limits["min_headroom"]),
        "has_material_headroom": headroom >= float(limits["material_headroom"]),
        "capacity_gain_passes": capacity_gain >= float(limits["min_capacity_gain"]),
        "coding_gap_passes": coding_gap >= float(limits["min_coding_gap"]),
        "objective_gain_passes": objective_gain
        >= float(limits["min_objective_gain"]),
        "setting_interaction_passes": abs(setting_interaction)
        >= float(limits["min_domain_interaction"]),
        "pooled_recovery_passes": pooled_recovery
        >= float(limits["min_pooled_recovery"]),
        "pooled_noninferiority_passes": pooled_gain
        >= float(limits["pooled_noninferiority"]),
        "improved_query_support_passes": improved_queries >= required_improved,
        "net_share_passes": net_share >= float(limits["min_net_share"]),
        "gap_recovery_passes": gap_recovery
        >= float(limits["min_gap_recovery"]),
        "worst_domain_floor_passes": worst_domain_gain
        >= float(limits["worst_domain_floor"]),
    }
    common_support = (
        gates["improved_query_support_passes"]
        and gates["net_share_passes"]
        and gates["gap_recovery_passes"]
    )
    mechanism = {
        "capacity_bottleneck_supported": (
            gates["has_material_headroom"]
            and gates["capacity_gain_passes"]
            and common_support
        ),
        "coding_bottleneck_supported": (
            gates["has_material_headroom"]
            and gates["coding_gap_passes"]
            and common_support
        ),
        "setting_interaction_supported": (
            gates["objective_gain_passes"]
            and gates["setting_interaction_passes"]
            and common_support
            and not gates["worst_domain_floor_passes"]
        ),
        "objective_repair_supported": (
            gates["objective_gain_passes"]
            and gates["pooled_recovery_passes"]
            and gates["pooled_noninferiority_passes"]
            and gates["worst_domain_floor_passes"]
            and common_support
        ),
    }

    if not gates["has_minimum_headroom"]:
        decision = "STOP_FROZEN_CANDIDATE_METHOD"
    elif mechanism["objective_repair_supported"]:
        decision = "OBJECTIVE_REPAIR_SUPPORTED"
    elif mechanism["setting_interaction_supported"]:
        decision = "SETTING_INTERACTION_SUPPORTED"
    elif mechanism["coding_bottleneck_supported"]:
        decision = "CODING_BOTTLENECK_SUPPORTED"
    elif mechanism["capacity_bottleneck_supported"]:
        decision = "CAPACITY_BOTTLENECK_SUPPORTED"
    elif (
        gates["pooled_recovery_passes"]
        and gates["pooled_noninferiority_passes"]
        and not gates["objective_gain_passes"]
    ):
        decision = "STOP_LEARNING_CLAIM_KEEP_UNIFORM_RPQ"
    else:
        decision = "STOP_RARS_METHOD_EXPANSION"

    return {
        "decision": decision,
        **gates,
        **mechanism,
        "required_improved_queries": required_improved,
        "net_share": float(net_share),
        "thresholds": {
            key: int(value) if key == "min_improved_queries" else float(value)
            for key, value in limits.items()
        },
    }


# Compatibility alias for early, never-executed V17 drafts.  The returned
# decision vocabulary is nevertheless the frozen setting-transfer vocabulary.
causal_decision = setting_transfer_decision
