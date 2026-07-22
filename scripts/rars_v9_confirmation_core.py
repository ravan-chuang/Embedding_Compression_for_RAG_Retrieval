#!/usr/bin/env python3
"""Numerical core for locked within-program RARS-v8 confirmation.

The 803-query role is prospective relative to V8 method development, but its
queries came from a pool used by earlier RARS versions.  Nothing in this
module labels that role an independent test set.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v9_locked_confirmation_v1"


def _matrix(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return array


def stable_score_order(scores: Any, rows: Any) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    identifiers = np.asarray(rows, dtype=np.int64)
    if values.ndim != 1 or identifiers.shape != values.shape:
        raise ValueError("Score and row vectors must match")
    if np.any(~np.isfinite(values)):
        raise ValueError("Scores contain non-finite values")
    return np.lexsort((identifiers, -values))


def candidate_labels(
    candidate_rows: Any, positive_rows: Any, positive_valid: Any
) -> np.ndarray:
    candidates = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    positives = _matrix(positive_rows, name="positive_rows", dtype=np.int64)
    valid = _matrix(positive_valid, name="positive_valid", dtype=bool)
    if positives.shape != valid.shape or positives.shape[0] != candidates.shape[0]:
        raise ValueError("Candidate and positive-row populations disagree")
    if np.any(valid.sum(axis=1) <= 0):
        raise ValueError("Every query needs at least one mapped positive")
    labels = np.zeros(candidates.shape, dtype=np.uint8)
    for query_index in range(len(candidates)):
        selected = positives[query_index, valid[query_index]]
        labels[query_index] = np.isin(candidates[query_index], selected).astype(
            np.uint8
        )
    return labels


def per_query_metrics(
    scores: Any,
    candidate_rows: Any,
    positive_rows: Any,
    positive_valid: Any,
    *,
    k: int,
    tie_identifiers: Any | None = None,
) -> dict[str, np.ndarray]:
    values = _matrix(scores, name="scores", dtype=np.float64)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    if values.shape != rows.shape or not 0 < k <= values.shape[1]:
        raise ValueError("Score/row shapes or metric cutoff are invalid")
    tie_values = (
        rows
        if tie_identifiers is None
        else _matrix(
            tie_identifiers, name="tie_identifiers", dtype=np.int64
        )
    )
    if tie_values.shape != rows.shape:
        raise ValueError("Tie identifiers must match candidate rows")
    labels = candidate_labels(rows, positive_rows, positive_valid)
    denominators = np.asarray(positive_valid, dtype=bool).sum(axis=1)
    recall = np.empty(len(values), dtype=np.float64)
    success = np.empty(len(values), dtype=np.float64)
    reciprocal_rank = np.empty(len(values), dtype=np.float64)
    ndcg = np.empty(len(values), dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for query_index in range(len(values)):
        order = stable_score_order(
            values[query_index], tie_values[query_index]
        )[:k]
        gains = labels[query_index, order].astype(np.float64)
        hits = np.flatnonzero(gains)
        recall[query_index] = float(gains.sum() / denominators[query_index])
        success[query_index] = float(bool(len(hits)))
        reciprocal_rank[query_index] = (
            1.0 / (int(hits[0]) + 1) if len(hits) else 0.0
        )
        ideal_count = min(int(denominators[query_index]), k)
        ideal = float(np.sum(discounts[:ideal_count]))
        ndcg[query_index] = float(np.sum(gains * discounts) / ideal)
    return {
        "recall": recall,
        "success": success,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
    }


def paired_bootstrap(
    treatment: Any,
    baseline: Any,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    treatment_values = np.asarray(treatment, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if (
        treatment_values.ndim != 1
        or treatment_values.shape != baseline_values.shape
        or not len(treatment_values)
        or np.any(~np.isfinite(treatment_values))
        or np.any(~np.isfinite(baseline_values))
    ):
        raise ValueError("Paired vectors must match and be non-empty")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("Bootstrap controls are invalid")
    difference = treatment_values - baseline_values
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    block_size = 2048
    for start in range(0, replicates, block_size):
        end = min(replicates, start + block_size)
        draws = rng.integers(
            0, len(difference), size=(end - start, len(difference))
        )
        means[start:end] = difference[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(difference.mean()),
        "lower": float(np.quantile(means, tail, method="linear")),
        "upper": float(np.quantile(means, 1.0 - tail, method="linear")),
        "confidence": float(confidence),
        "replicates": int(replicates),
        "improved_queries": int(np.sum(difference > 0)),
        "harmed_queries": int(np.sum(difference < 0)),
        "unchanged_queries": int(np.sum(difference == 0)),
    }


def paired_randomization_p_value(
    treatment: Any,
    baseline: Any,
    *,
    replicates: int,
    seed: int,
) -> float:
    """One-sided paired sign-randomization p-value for a positive mean effect."""

    treatment_values = np.asarray(treatment, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if (
        treatment_values.ndim != 1
        or treatment_values.shape != baseline_values.shape
        or not len(treatment_values)
        or replicates <= 0
        or np.any(~np.isfinite(treatment_values))
        or np.any(~np.isfinite(baseline_values))
    ):
        raise ValueError("Randomization inputs are invalid")
    difference = treatment_values - baseline_values
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    block_size = 4096
    for start in range(0, replicates, block_size):
        end = min(replicates, start + block_size)
        signs = rng.integers(
            0, 2, size=(end - start, len(difference)), dtype=np.int8
        )
        signed = signs.astype(np.float64) * 2.0 - 1.0
        means = (signed * difference[None, :]).mean(axis=1)
        exceedances += int(np.sum(means >= observed))
    return float((exceedances + 1) / (replicates + 1))


def comparison(
    treatment: Any,
    baseline: Any,
    *,
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> dict[str, Any]:
    payload = paired_bootstrap(
        treatment,
        baseline,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    payload["randomization_p_value_one_sided"] = paired_randomization_p_value(
        treatment,
        baseline,
        replicates=randomization_replicates,
        seed=seed + 1,
    )
    return payload


def candidate_gap_recovery(
    treatment: Any, baseline: Any, same_candidate_exact: Any
) -> float:
    treatment_values = np.asarray(treatment, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    exact_values = np.asarray(same_candidate_exact, dtype=np.float64)
    if (
        treatment_values.ndim != 1
        or treatment_values.shape != baseline_values.shape
        or treatment_values.shape != exact_values.shape
        or not len(treatment_values)
        or np.any(~np.isfinite(treatment_values))
        or np.any(~np.isfinite(baseline_values))
        or np.any(~np.isfinite(exact_values))
    ):
        raise ValueError("Gap-recovery vectors must be matching and finite")
    treatment_mean = float(np.mean(treatment_values))
    baseline_mean = float(np.mean(baseline_values))
    exact_mean = float(np.mean(exact_values))
    gap = exact_mean - baseline_mean
    return float((treatment_mean - baseline_mean) / gap) if gap > 0 else 0.0


def confirmation_decision(
    *,
    rars_vs_base: dict[str, Any],
    pca_vs_base: dict[str, Any],
    rars_vs_pca: dict[str, Any],
    gap_recovery: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "minimum_rars_gain_over_base": rars_vs_base["mean_difference"]
        >= float(thresholds["minimum_rars_recall_at_10_gain_over_base"]),
        "rars_base_bootstrap_lower_positive": rars_vs_base["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "minimum_candidate_gap_recovery": gap_recovery
        >= float(thresholds["minimum_candidate_gap_recovery_fraction"]),
        "minimum_rars_improved_query_support": rars_vs_base[
            "improved_queries"
        ]
        >= int(thresholds["minimum_improved_queries_over_base"]),
        "positive_rars_net_query_support": (
            rars_vs_base["improved_queries"] - rars_vs_base["harmed_queries"]
        )
        >= int(thresholds["minimum_net_improved_queries_over_base"]),
    }
    algorithm = {
        "minimum_rars_gain_over_pca": rars_vs_pca["mean_difference"]
        >= float(thresholds["minimum_rars_recall_at_10_gain_over_pca"]),
        "rars_pca_bootstrap_lower_positive": rars_vs_pca["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "rars_pca_randomization_significant": rars_vs_pca[
            "randomization_p_value_one_sided"
        ]
        <= float(thresholds["maximum_primary_randomization_p_value"]),
        "minimum_rars_over_pca_improved_queries": rars_vs_pca[
            "improved_queries"
        ]
        >= int(thresholds["minimum_improved_queries_over_pca"]),
        "positive_rars_over_pca_net_support": (
            rars_vs_pca["improved_queries"] - rars_vs_pca["harmed_queries"]
        )
        >= int(thresholds["minimum_net_improved_queries_over_pca"]),
    }
    generic = {
        "pca_gain_over_base": pca_vs_base["mean_difference"]
        >= float(thresholds["minimum_generic_sidecar_gain_over_base"]),
        "pca_base_bootstrap_lower_positive": pca_vs_base["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
    }
    algorithm_path = {**common, **algorithm}
    generic_path = {**common, **generic}
    if all(algorithm_path.values()):
        decision = thresholds["algorithm_confirmation_decision"]
        required_path = "algorithm"
        required_gates = algorithm_path
    elif all(generic_path.values()):
        decision = thresholds["generic_sidecar_confirmation_decision"]
        required_path = "generic_sidecar"
        required_gates = generic_path
    else:
        decision = thresholds["stop_decision"]
        required_path = "stop"
        required_gates = algorithm_path
    return {
        "protocol_id": PROTOCOL_ID,
        "decision": decision,
        "selected_path": required_path,
        "gates": {
            "common": common,
            "algorithm": algorithm,
            "generic_sidecar": generic,
        },
        "failed_gates_for_selected_or_primary_path": [
            name for name, passed in required_gates.items() if not passed
        ],
        "method_or_threshold_tuning_authorized": False,
    }


def summarize_metric(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError("Metric vector is invalid")
    return float(math.fsum(array.tolist()) / len(array))
