#!/usr/bin/env python3
"""Numerical core for the V15 cross-fitted selective RPQ correction gate."""

from __future__ import annotations

from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v15_cross_fitted_selective_rpq_gate_v1"
FEATURE_NAMES = (
    "base_cutoff_margin",
    "sidecar_cutoff_margin",
    "base_top10_span",
    "sidecar_top10_span",
    "mean_abs_correction_top10_base",
    "mean_abs_correction_top40_base",
    "max_abs_correction_top40_base",
    "std_correction_top40_base",
    "mean_signed_correction_top10_base",
    "top10_disagreement_fraction",
    "correction_to_cutoff_margin",
    "top10_score_std_change",
)


def _rank(scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return deterministic descending-score order with row-id tie breaking."""

    return np.lexsort((rows, -scores))


def query_gate_features(
    base_scores: np.ndarray,
    sidecar_scores: np.ndarray,
    rows: np.ndarray,
    *,
    final_k: int = 10,
    top_b: int = 40,
    ratio_clip: float = 50.0,
) -> np.ndarray:
    """Build label-free query features from Base and always-on sidecar scores."""

    base = np.asarray(base_scores, dtype=np.float64)
    sidecar = np.asarray(sidecar_scores, dtype=np.float64)
    candidate_rows = np.asarray(rows, dtype=np.int64)
    if base.ndim != 2 or sidecar.shape != base.shape or candidate_rows.shape != base.shape:
        raise ValueError("Base, sidecar, and row arrays must share a two-dimensional shape")
    if final_k <= 0 or top_b <= final_k or top_b > base.shape[1]:
        raise ValueError("Require 0 < final_k < top_b <= candidate count")
    if ratio_clip <= 0 or np.any(~np.isfinite(base)) or np.any(~np.isfinite(sidecar)):
        raise ValueError("Scores and feature constants must be finite")

    output = np.empty((len(base), len(FEATURE_NAMES)), dtype=np.float64)
    epsilon = 1e-8
    for index in range(len(base)):
        base_order = _rank(base[index], candidate_rows[index])
        sidecar_order = _rank(sidecar[index], candidate_rows[index])
        base_top = base_order[:top_b]
        base_head = base_order[:final_k]
        sidecar_head = sidecar_order[:final_k]
        correction = sidecar[index] - base[index]
        top_correction = correction[base_top]
        head_correction = correction[base_head]
        base_sorted = base[index, base_order]
        sidecar_sorted = sidecar[index, sidecar_order]
        base_margin = max(
            float(base_sorted[final_k - 1] - base_sorted[final_k]), 0.0
        )
        sidecar_margin = max(
            float(sidecar_sorted[final_k - 1] - sidecar_sorted[final_k]), 0.0
        )
        overlap = len(
            set(candidate_rows[index, base_head].tolist())
            & set(candidate_rows[index, sidecar_head].tolist())
        )
        mean_abs_top = float(np.mean(np.abs(top_correction)))
        output[index] = (
            base_margin,
            sidecar_margin,
            float(base_sorted[0] - base_sorted[final_k - 1]),
            float(sidecar_sorted[0] - sidecar_sorted[final_k - 1]),
            float(np.mean(np.abs(head_correction))),
            mean_abs_top,
            float(np.max(np.abs(top_correction))),
            float(np.std(top_correction)),
            float(np.mean(head_correction)),
            1.0 - overlap / final_k,
            min(mean_abs_top / max(base_margin, epsilon), ratio_clip),
            float(np.std(sidecar[index, sidecar_head]) - np.std(base[index, base_head])),
        )
    if np.any(~np.isfinite(output)):
        raise ValueError("Gate features contain a non-finite value")
    return output


def fit_weighted_ridge_gate(
    features: np.ndarray,
    recall_gain: np.ndarray,
    *,
    ridge: float,
    neutral_weight: float,
    harm_weight: float,
    minimum_scale: float = 1e-6,
) -> dict[str, Any]:
    """Fit one deterministic weighted ridge utility model."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(recall_gain, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0:
        raise ValueError("Invalid gate training arrays")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Gate training arrays must be finite")
    if ridge <= 0 or neutral_weight <= 0 or harm_weight < 1 or minimum_scale <= 0:
        raise ValueError("Invalid gate fitting constants")
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), minimum_scale)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    weights = np.where(y < 0, harm_weight, np.where(y > 0, 1.0, neutral_weight))
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = ridge * 1e-6
    coefficients = np.linalg.solve(
        gram + penalty,
        design.T @ (weights * y),
    )
    predictions = design @ coefficients
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "training_rows": int(len(x)),
        "positive_rows": int(np.sum(y > 0)),
        "harm_rows": int(np.sum(y < 0)),
        "neutral_rows": int(np.sum(y == 0)),
        "weighted_mse": float(np.average((predictions - y) ** 2, weights=weights)),
        "condition_number": float(np.linalg.cond(gram + penalty)),
    }


def gate_utility_scores(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1:] != mean.shape or scale.shape != mean.shape:
        raise ValueError("Gate feature/model shape mismatch")
    if coefficients.shape != (x.shape[1] + 1,) or np.any(scale <= 0):
        raise ValueError("Gate model coefficients or scales are invalid")
    return coefficients[0] + ((x - mean) / scale) @ coefficients[1:]


def select_calibrated_threshold(
    utility_scores: np.ndarray,
    base_metrics: dict[str, np.ndarray],
    sidecar_metrics: dict[str, np.ndarray],
    *,
    quantile_grid: list[float],
    minimum_coverage: float,
    maximum_coverage: float,
    minimum_mrr_change: float,
    minimum_ndcg_change: float,
) -> dict[str, Any]:
    """Select a threshold on a disjoint calibration fold, else use always-on."""

    scores = np.asarray(utility_scores, dtype=np.float64)
    required = ("recall", "mrr", "ndcg")
    if scores.ndim != 1 or len(scores) == 0 or np.any(~np.isfinite(scores)):
        raise ValueError("Calibration utility scores are invalid")
    for metric in required:
        if np.asarray(base_metrics[metric]).shape != scores.shape:
            raise ValueError("Base calibration metric shape changed")
        if np.asarray(sidecar_metrics[metric]).shape != scores.shape:
            raise ValueError("Sidecar calibration metric shape changed")
    if not 0 <= minimum_coverage <= maximum_coverage <= 1:
        raise ValueError("Coverage bounds are invalid")
    if sorted(set(quantile_grid)) != quantile_grid or any(
        value < 0 or value > 1 for value in quantile_grid
    ):
        raise ValueError("Quantile grid must be sorted, unique, and within [0,1]")

    candidates: list[dict[str, Any]] = []
    for quantile in quantile_grid:
        threshold = float(np.quantile(scores, quantile, method="higher"))
        apply = scores >= threshold
        coverage = float(np.mean(apply))
        if coverage < minimum_coverage or coverage > maximum_coverage:
            continue
        changes = {}
        for metric in required:
            selected = np.where(
                apply,
                np.asarray(sidecar_metrics[metric], dtype=np.float64),
                np.asarray(base_metrics[metric], dtype=np.float64),
            )
            changes[metric] = float(
                np.mean(selected - np.asarray(sidecar_metrics[metric], dtype=np.float64))
            )
        if changes["mrr"] < minimum_mrr_change or changes["ndcg"] < minimum_ndcg_change:
            continue
        candidates.append(
            {
                "threshold": threshold,
                "quantile": float(quantile),
                "coverage": coverage,
                "changes_vs_always_on": changes,
                "fallback_always_on": False,
            }
        )
    beneficial = [row for row in candidates if row["changes_vs_always_on"]["recall"] > 0]
    if not beneficial:
        return {
            "threshold": float(np.nextafter(np.min(scores), -np.inf)),
            "quantile": None,
            "coverage": 1.0,
            "changes_vs_always_on": {name: 0.0 for name in required},
            "fallback_always_on": True,
            "feasible_candidate_count": len(candidates),
        }
    best = max(
        beneficial,
        key=lambda row: (
            row["changes_vs_always_on"]["recall"],
            row["changes_vs_always_on"]["mrr"],
            row["changes_vs_always_on"]["ndcg"],
            row["coverage"],
        ),
    )
    return {**best, "feasible_candidate_count": len(candidates)}


def apply_query_gate(utility_scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(utility_scores, dtype=np.float64)
    if scores.ndim != 1 or np.any(~np.isfinite(scores)) or np.isnan(threshold):
        raise ValueError("Invalid gate scores or threshold")
    return scores >= threshold


def selective_gate_decision(
    *,
    primary_vs_uniform: dict[str, Any],
    primary_vs_base: dict[str, Any],
    seed_gains: list[float],
    fold_gains: list[float],
    uniform_mrr: float,
    selective_mrr: float,
    uniform_ndcg: float,
    selective_ndcg: float,
    applied_coverages: list[float],
    improved_queries: int,
    harmed_queries: int,
    parent_payload_bytes_per_document: int,
    extra_document_bytes: int,
    global_model_bytes: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply every preregistered V15 development gate without rescue logic."""

    median_seed = float(np.median(np.asarray(seed_gains, dtype=np.float64)))
    gates = {
        "minimum_gain_over_uniform_rpq": primary_vs_uniform["mean_difference"]
        >= float(thresholds["minimum_recall_gain_over_uniform_rpq"]),
        "bootstrap_lower_above_zero": primary_vs_uniform["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "randomization_p_value": primary_vs_uniform["randomization_p_value_one_sided"]
        <= float(thresholds["maximum_randomization_p_value"]),
        "minimum_gain_over_base": primary_vs_base["mean_difference"]
        >= float(thresholds["minimum_recall_gain_over_base"]),
        "improved_query_support": improved_queries
        >= int(thresholds["minimum_improved_queries"]),
        "harm_query_guardrail": harmed_queries
        <= int(thresholds["maximum_harmed_queries"]),
        "net_improved_query_support": improved_queries - harmed_queries
        >= int(thresholds["minimum_net_improved_queries"]),
        "all_seed_gains_nonnegative": min(seed_gains)
        >= float(thresholds["minimum_each_seed_gain"]),
        "median_seed_gain": median_seed
        >= float(thresholds["minimum_median_seed_gain"]),
        "worst_fold_gain": min(fold_gains)
        >= float(thresholds["minimum_worst_fold_gain"]),
        "mrr_guardrail": selective_mrr - uniform_mrr
        >= float(thresholds["minimum_mrr_change"]),
        "ndcg_guardrail": selective_ndcg - uniform_ndcg
        >= float(thresholds["minimum_ndcg_change"]),
        "nontrivial_primary_coverage": float(thresholds["minimum_primary_coverage"])
        <= applied_coverages[0]
        <= float(thresholds["maximum_primary_coverage"]),
        "parent_payload_exactly_16_bytes": parent_payload_bytes_per_document == 16,
        "zero_extra_document_bytes": extra_document_bytes == 0,
        "global_gate_size": global_model_bytes
        <= int(thresholds["maximum_global_gate_bytes"]),
    }
    passed = all(gates.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "decision": thresholds["go_decision"] if passed else thresholds["stop_decision"],
        "all_gates_passed": passed,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "improved_queries": int(improved_queries),
        "harmed_queries": int(harmed_queries),
        "net_improved_queries": int(improved_queries - harmed_queries),
        "median_seed_gain": median_seed,
    }
