#!/usr/bin/env python3
"""Numerical core for PCA-anchored harm-constrained residual correction.

V10 is post-confirmation method development.  It never reads V9 outcomes and
does not modify the frozen V8 implementation.  The deployed representation is
still one orthonormal rank-16 basis with one int8 coefficient vector per
document.  Stability comes from a PCA trust region, query-level tail-harm
penalty, and monotone Riemannian line search rather than extra storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v10_pca_anchored_harm_constrained_v1"


def _matrix(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _orient_columns(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


def retract_qr(value: Any) -> np.ndarray:
    matrix = _matrix(value, name="retraction input")
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q *= signs[None, :]
    return _orient_columns(q)


def validate_basis(value: Any, *, dimension: int, rank: int) -> np.ndarray:
    basis = _matrix(value, name="basis")
    if basis.shape != (dimension, rank):
        raise ValueError(f"basis must have shape {(dimension, rank)}")
    if not np.allclose(basis.T @ basis, np.eye(rank), rtol=0.0, atol=2e-6):
        raise ValueError("basis is not orthonormal")
    return _orient_columns(basis)


def tangent_projection(basis: Any, gradient: Any) -> np.ndarray:
    b = _matrix(basis, name="basis")
    g = _matrix(gradient, name="gradient")
    if b.shape != g.shape:
        raise ValueError("basis and gradient shapes differ")
    symmetric = 0.5 * (b.T @ g + g.T @ b)
    return g - b @ symmetric


def maximum_principal_angle_degrees(anchor: Any, candidate: Any) -> float:
    p = _matrix(anchor, name="anchor")
    b = _matrix(candidate, name="candidate")
    if p.shape != b.shape:
        raise ValueError("anchor and candidate shapes differ")
    singular = np.linalg.svd(p.T @ b, compute_uv=False)
    minimum = float(np.clip(np.min(singular), 0.0, 1.0))
    return float(np.degrees(np.arccos(minimum)))


def score_float_sidecar_candidates(
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
    """Score a non-deployable FP32 coefficient ceiling for one fixed basis.

    Comparing this ceiling with the identical-basis int8 scorer isolates scalar
    coefficient-quantization headroom.  It does not measure rank truncation,
    routing, or candidate-pool loss.
    """

    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _matrix(base_scores, name="base_scores", dtype=np.float32)
    residual_matrix = _matrix(residuals, name="residuals", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    if not (rows.shape == lookup.shape == base.shape):
        raise ValueError("candidate rows, residual lookup, and scores must match")
    if len(query_matrix) != len(rows) or query_matrix.shape[1] != projection.shape[0]:
        raise ValueError("query/candidate/basis dimensions disagree")
    if residual_matrix.shape[1] != projection.shape[0]:
        raise ValueError("residual and basis dimensions disagree")
    if not np.isfinite(alpha) or alpha < 0 or not 0 < top_b <= rows.shape[1]:
        raise ValueError("invalid FP32 sidecar alpha or Top-B")
    output = base.copy()
    q_projected = query_matrix @ projection
    residual_projected = residual_matrix @ projection
    for query_index in range(len(rows)):
        valid_rows = np.where(
            rows[query_index] >= 0,
            rows[query_index],
            np.iinfo(np.int64).max,
        )
        valid_scores = np.where(
            rows[query_index] >= 0, base[query_index], -np.inf
        )
        order = np.lexsort((valid_rows, -valid_scores))
        selected = order[:top_b]
        selected_lookup = lookup[query_index, selected]
        valid = selected_lookup >= 0
        if not np.any(valid):
            continue
        if np.max(selected_lookup[valid]) >= len(residual_projected):
            raise ValueError("residual lookup is outside FP32 coefficient matrix")
        output[query_index, selected[valid]] += alpha * (
            residual_projected[selected_lookup[valid]] @ q_projected[query_index]
        )
    return output


def query_equal_weights(query: Any, raw_weight: Any) -> np.ndarray:
    queries = np.asarray(query, dtype=np.int64)
    raw = np.asarray(raw_weight, dtype=np.float64)
    if queries.ndim != 1 or queries.shape != raw.shape or not len(raw):
        raise ValueError("query and raw-weight vectors must match and be non-empty")
    if np.any(queries < 0) or np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("query equalisation inputs are invalid")
    represented = np.unique(queries)
    output = np.zeros(len(raw), dtype=np.float64)
    for query_index in represented:
        members = np.flatnonzero(queries == query_index)
        local = raw[members]
        output[members] = local / local.sum()
    for query_index in represented:
        if not np.isclose(output[queries == query_index].sum(), 1.0, atol=1e-12):
            raise AssertionError("query-local harm weights do not sum to one")
    return output


@dataclass(frozen=True)
class ObjectiveBatch:
    queries: np.ndarray
    residual_differences: np.ndarray
    exact_residual_margin: np.ndarray
    base_margin: np.ndarray
    balanced_weight: np.ndarray
    harm_weight: np.ndarray
    query_index: np.ndarray
    pca_prediction: np.ndarray

    def __post_init__(self) -> None:
        q = np.asarray(self.queries)
        dr = np.asarray(self.residual_differences)
        vectors = [
            np.asarray(self.exact_residual_margin),
            np.asarray(self.base_margin),
            np.asarray(self.balanced_weight),
            np.asarray(self.harm_weight),
            np.asarray(self.query_index),
            np.asarray(self.pca_prediction),
        ]
        if q.ndim != 2 or dr.shape != q.shape:
            raise ValueError("objective query and residual-difference matrices differ")
        if any(value.ndim != 1 or len(value) != len(q) for value in vectors):
            raise ValueError("objective vectors do not match pair count")
        if not len(q) or any(
            np.any(~np.isfinite(value)) for value in [q, dr, *vectors[:-1], vectors[-1]]
        ):
            raise ValueError("objective batch is empty or non-finite")
        if np.any(self.balanced_weight <= 0) or not np.isclose(
            np.sum(self.balanced_weight), 1.0, atol=1e-7
        ):
            raise ValueError("balanced pair weights must be positive and sum to one")
        if np.any(self.harm_weight <= 0) or np.any(self.query_index < 0):
            raise ValueError("harm weights or query indices are invalid")


def build_objective_batch(
    queries: Any,
    residuals: Any,
    pairs: Any,
    pca_basis: Any,
) -> ObjectiveBatch:
    query_matrix = _matrix(queries, name="queries")
    residual_matrix = _matrix(residuals, name="residuals")
    basis = validate_basis(
        pca_basis,
        dimension=query_matrix.shape[1],
        rank=np.asarray(pca_basis).shape[1],
    )
    query_index = np.asarray(pairs.query, dtype=np.int64)
    positive = np.asarray(pairs.positive_residual_row, dtype=np.int64)
    challenger = np.asarray(pairs.challenger_residual_row, dtype=np.int64)
    if not len(query_index):
        raise ValueError("V10 requires at least one cutoff pair")
    if np.max(query_index) >= len(query_matrix) or max(
        int(np.max(positive)), int(np.max(challenger))
    ) >= len(residual_matrix):
        raise ValueError("pair indices are outside objective inputs")
    q = query_matrix[query_index]
    delta_r = residual_matrix[positive] - residual_matrix[challenger]
    exact = np.asarray(pairs.target_residual_margin, dtype=np.float64)
    direct = np.einsum("pd,pd->p", q, delta_r)
    if not np.allclose(exact, direct, rtol=1e-4, atol=2e-5):
        raise ValueError("pair target does not equal exact residual margin")
    raw = np.asarray(pairs.raw_weight, dtype=np.float64)
    return ObjectiveBatch(
        queries=q,
        residual_differences=delta_r,
        exact_residual_margin=exact,
        base_margin=np.asarray(pairs.base_margin, dtype=np.float64),
        balanced_weight=np.asarray(pairs.balanced_weight, dtype=np.float64),
        harm_weight=query_equal_weights(query_index, raw),
        query_index=query_index,
        pca_prediction=np.einsum("pr,pr->p", q @ basis, delta_r @ basis),
    )


def _huber(error: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(error)
    loss = np.where(
        absolute <= delta,
        0.5 * error * error / delta,
        absolute - 0.5 * delta,
    )
    derivative = np.where(absolute <= delta, error / delta, np.sign(error))
    return loss, derivative


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def objective_and_gradient(
    basis: Any,
    batch: ObjectiveBatch,
    anchor_basis: Any,
    *,
    alpha: float,
    distillation_weight: float,
    cutoff_weight: float,
    harm_weight: float,
    anchor_weight: float,
    huber_delta: float,
    cutoff_temperature: float,
    margin_floor: float,
    harm_scale: float,
    cvar_fraction: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    b = _matrix(basis, name="basis")
    anchor = _matrix(anchor_basis, name="anchor_basis")
    if b.shape != anchor.shape or b.shape[0] != batch.queries.shape[1]:
        raise ValueError("basis, anchor, and pair dimensions disagree")
    controls = [
        alpha,
        distillation_weight,
        cutoff_weight,
        harm_weight,
        anchor_weight,
        huber_delta,
        cutoff_temperature,
        harm_scale,
        cvar_fraction,
    ]
    if any(not np.isfinite(value) for value in controls):
        raise ValueError("objective controls must be finite")
    if alpha <= 0 or min(distillation_weight, cutoff_weight, harm_weight, anchor_weight) < 0:
        raise ValueError("objective weights are invalid")
    if huber_delta <= 0 or cutoff_temperature <= 0 or harm_scale <= 0:
        raise ValueError("objective scales must be positive")
    if not 0 < cvar_fraction <= 1:
        raise ValueError("CVaR fraction must be within (0, 1]")

    q_projected = batch.queries @ b
    residual_projected = batch.residual_differences @ b
    prediction = np.einsum("pr,pr->p", q_projected, residual_projected)
    error = prediction - batch.exact_residual_margin
    distill_pair, distill_derivative = _huber(error, huber_delta)
    distill_loss = float(np.sum(batch.balanced_weight * distill_pair))
    coefficient = (
        distillation_weight * batch.balanced_weight * distill_derivative
    )

    corrected_margin = batch.base_margin + alpha * prediction
    cutoff_argument = (margin_floor - corrected_margin) / cutoff_temperature
    cutoff_pair = cutoff_temperature * np.logaddexp(0.0, cutoff_argument)
    cutoff_loss = float(np.sum(batch.balanced_weight * cutoff_pair))
    coefficient += (
        cutoff_weight
        * batch.balanced_weight
        * (-alpha * _sigmoid(cutoff_argument))
    )

    pca_margin = batch.base_margin + alpha * batch.pca_prediction
    shortfall = np.maximum(pca_margin - corrected_margin, 0.0)
    harm_pair = 0.5 * shortfall * shortfall / harm_scale
    represented = np.unique(batch.query_index)
    query_harm = np.empty(len(represented), dtype=np.float64)
    for offset, query_index in enumerate(represented):
        members = batch.query_index == query_index
        query_harm[offset] = float(
            np.sum(batch.harm_weight[members] * harm_pair[members])
        )
    tail_count = max(1, int(math.ceil(cvar_fraction * len(represented))))
    order = np.lexsort((represented, -query_harm))
    tail_queries = represented[order[:tail_count]]
    cvar_loss = float(np.mean(query_harm[order[:tail_count]]))
    tail_mask = np.isin(batch.query_index, tail_queries) & (shortfall > 0)
    coefficient[tail_mask] += (
        harm_weight
        * batch.harm_weight[tail_mask]
        / tail_count
        * (-alpha * shortfall[tail_mask] / harm_scale)
    )

    overlap = anchor.T @ b
    anchor_loss = anchor_weight * (
        1.0 - float(np.sum(overlap * overlap)) / b.shape[1]
    )
    gradient = batch.queries.T @ (
        coefficient[:, None] * residual_projected
    )
    gradient += batch.residual_differences.T @ (
        coefficient[:, None] * q_projected
    )
    gradient += -2.0 * anchor_weight / b.shape[1] * (
        anchor @ (anchor.T @ b)
    )
    total = (
        distillation_weight * distill_loss
        + cutoff_weight * cutoff_loss
        + harm_weight * cvar_loss
        + anchor_loss
    )
    diagnostics = {
        "loss": float(total),
        "distillation_loss": distill_loss,
        "cutoff_loss": cutoff_loss,
        "tail_harm_loss": cvar_loss,
        "anchor_loss": float(anchor_loss),
        "tail_query_count": int(tail_count),
        "maximum_query_harm": float(np.max(query_harm)),
        "mean_corrected_margin": float(np.mean(corrected_margin)),
        "minimum_corrected_margin": float(np.min(corrected_margin)),
    }
    return float(total), gradient, diagnostics


def gradient_direction_audit(
    basis: Any,
    batch: ObjectiveBatch,
    anchor_basis: Any,
    objective_kwargs: dict[str, Any],
    *,
    epsilon: float = 1e-5,
    maximum_relative_error: float = 2e-4,
) -> dict[str, Any]:
    b = _matrix(basis, name="basis")
    if epsilon <= 0 or maximum_relative_error <= 0:
        raise ValueError("gradient-audit controls must be positive")
    loss, euclidean, _ = objective_and_gradient(
        b, batch, anchor_basis, **objective_kwargs
    )
    riemannian = tangent_projection(b, euclidean)
    raw = np.sin(np.arange(b.size, dtype=np.float64) + 1.0).reshape(b.shape)
    direction = tangent_projection(b, raw)
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise AssertionError("deterministic audit direction is zero")
    direction /= norm
    plus = retract_qr(b + epsilon * direction)
    minus = retract_qr(b - epsilon * direction)
    plus_loss = objective_and_gradient(
        plus, batch, anchor_basis, **objective_kwargs
    )[0]
    minus_loss = objective_and_gradient(
        minus, batch, anchor_basis, **objective_kwargs
    )[0]
    numerical = float((plus_loss - minus_loss) / (2.0 * epsilon))
    analytical = float(np.sum(riemannian * direction))
    denominator = max(1e-10, abs(numerical), abs(analytical))
    relative_error = float(abs(numerical - analytical) / denominator)
    return {
        "status": "PASS" if relative_error <= maximum_relative_error else "FAIL",
        "loss": loss,
        "analytical_directional_derivative": analytical,
        "numerical_directional_derivative": numerical,
        "relative_error": relative_error,
        "epsilon": float(epsilon),
        "maximum_relative_error": float(maximum_relative_error),
    }


def fit_stable_basis(
    batch: ObjectiveBatch,
    pca_basis: Any,
    objective_kwargs: dict[str, Any],
    *,
    maximum_steps: int,
    initial_step_size: float,
    backtracking_factor: float,
    armijo_constant: float,
    maximum_backtracks: int,
    maximum_principal_angle: float,
    gradient_tolerance: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    anchor = validate_basis(
        pca_basis,
        dimension=batch.queries.shape[1],
        rank=np.asarray(pca_basis).shape[1],
    )
    if maximum_steps <= 0 or initial_step_size <= 0 or maximum_backtracks <= 0:
        raise ValueError("optimizer iteration controls are invalid")
    if not 0 < backtracking_factor < 1 or not 0 < armijo_constant < 1:
        raise ValueError("line-search controls are invalid")
    if maximum_principal_angle <= 0 or gradient_tolerance < 0:
        raise ValueError("trust-region controls are invalid")
    basis = anchor.copy()
    current_loss, euclidean, current_diag = objective_and_gradient(
        basis, batch, anchor, **objective_kwargs
    )
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "accepted": True,
            "step_size": 0.0,
            "backtracks": 0,
            "pre_loss": current_loss,
            "proposal_loss": current_loss,
            "post_retraction_loss": current_loss,
            "full_step_loss_change": 0.0,
            "gradient_norm": float(
                np.linalg.norm(tangent_projection(basis, euclidean))
            ),
            "maximum_principal_angle_degrees": 0.0,
            **current_diag,
        }
    ]
    for step in range(1, maximum_steps + 1):
        tangent = tangent_projection(basis, euclidean)
        gradient_norm = float(np.linalg.norm(tangent))
        if gradient_norm <= gradient_tolerance:
            break
        direction = -tangent
        step_size = initial_step_size
        accepted = False
        selected: tuple[np.ndarray, float, np.ndarray, dict[str, Any], float, float] | None = None
        for backtracks in range(maximum_backtracks + 1):
            proposal = basis + step_size * direction
            proposal_loss = objective_and_gradient(
                proposal, batch, anchor, **objective_kwargs
            )[0]
            candidate = retract_qr(proposal)
            angle = maximum_principal_angle_degrees(anchor, candidate)
            candidate_loss, candidate_gradient, candidate_diag = objective_and_gradient(
                candidate, batch, anchor, **objective_kwargs
            )
            sufficient = candidate_loss <= (
                current_loss
                - armijo_constant * step_size * gradient_norm * gradient_norm
                + 1e-14
            )
            if sufficient and angle <= maximum_principal_angle:
                selected = (
                    candidate,
                    candidate_loss,
                    candidate_gradient,
                    candidate_diag,
                    proposal_loss,
                    angle,
                )
                accepted = True
                break
            step_size *= backtracking_factor
        if not accepted or selected is None:
            history.append(
                {
                    "step": step,
                    "accepted": False,
                    "step_size": float(step_size),
                    "backtracks": int(maximum_backtracks + 1),
                    "pre_loss": current_loss,
                    "proposal_loss": float(proposal_loss),
                    "post_retraction_loss": float(candidate_loss),
                    "full_step_loss_change": float(candidate_loss - current_loss),
                    "gradient_norm": gradient_norm,
                    "maximum_principal_angle_degrees": float(angle),
                    **current_diag,
                }
            )
            break
        candidate, candidate_loss, candidate_gradient, candidate_diag, proposal_loss, angle = selected
        previous_loss = current_loss
        basis = candidate
        current_loss = candidate_loss
        euclidean = candidate_gradient
        current_diag = candidate_diag
        history.append(
            {
                "step": step,
                "accepted": True,
                "step_size": float(step_size),
                "backtracks": int(backtracks),
                "pre_loss": float(previous_loss),
                "proposal_loss": float(proposal_loss),
                "post_retraction_loss": float(candidate_loss),
                "full_step_loss_change": float(candidate_loss - previous_loss),
                "gradient_norm": gradient_norm,
                "maximum_principal_angle_degrees": float(angle),
                **candidate_diag,
            }
        )
    accepted_losses = [
        float(item["post_retraction_loss"])
        for item in history
        if item["accepted"]
    ]
    if any(
        later > earlier + 1e-12
        for earlier, later in zip(accepted_losses, accepted_losses[1:])
    ):
        raise AssertionError("accepted V10 objective increased")
    return validate_basis(
        basis, dimension=basis.shape[0], rank=basis.shape[1]
    ).astype(np.float32), history


def paired_inference(
    treatment: Any,
    baseline: Any,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    randomization_replicates: int,
    randomization_seed: int,
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
        raise ValueError("paired inference inputs are invalid")
    if bootstrap_replicates <= 0 or randomization_replicates <= 0:
        raise ValueError("paired inference replicates must be positive")
    difference = treatment_values - baseline_values
    rng = np.random.default_rng(bootstrap_seed)
    means = np.empty(bootstrap_replicates, dtype=np.float64)
    for start in range(0, bootstrap_replicates, 2048):
        end = min(bootstrap_replicates, start + 2048)
        draws = rng.integers(
            0, len(difference), size=(end - start, len(difference))
        )
        means[start:end] = difference[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    observed = float(np.mean(difference))
    rng = np.random.default_rng(randomization_seed)
    exceedances = 0
    for start in range(0, randomization_replicates, 4096):
        end = min(randomization_replicates, start + 4096)
        signs = rng.integers(
            0, 2, size=(end - start, len(difference)), dtype=np.int8
        )
        randomized = (signs.astype(np.float64) * 2.0 - 1.0) * difference
        exceedances += int(np.sum(randomized.mean(axis=1) >= observed))
    return {
        "mean_difference": observed,
        "lower": float(np.quantile(means, tail, method="linear")),
        "upper": float(np.quantile(means, 1.0 - tail, method="linear")),
        "confidence": float(confidence),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "randomization_replicates": int(randomization_replicates),
        "randomization_seed": int(randomization_seed),
        "randomization_p_value_one_sided": float(
            (exceedances + 1) / (randomization_replicates + 1)
        ),
        "improved_queries": int(np.sum(difference > 0)),
        "harmed_queries": int(np.sum(difference < 0)),
        "unchanged_queries": int(np.sum(difference == 0)),
    }


def scalar_quantization_headroom_decision(
    comparison: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    gates = {
        "minimum_recall_gain": comparison["mean_difference"]
        >= float(thresholds["minimum_recall_at_10_gain"]),
        "bootstrap_lower_positive": comparison["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "randomization_significant": comparison[
            "randomization_p_value_one_sided"
        ] <= float(thresholds["maximum_randomization_p_value"]),
        "minimum_improved_queries": comparison["improved_queries"]
        >= int(thresholds["minimum_improved_queries"]),
        "minimum_net_improved_queries": (
            comparison["improved_queries"] - comparison["harmed_queries"]
        ) >= int(thresholds["minimum_net_improved_queries"]),
    }
    passed = all(gates.values())
    return {
        "decision": (
            thresholds["go_decision"] if passed else thresholds["stop_decision"]
        ),
        "gates": {name: bool(value) for name, value in gates.items()},
        "failed_gates": [name for name, value in gates.items() if not value],
        "diagnostic_only": True,
        "codebook_training_performed": False,
    }


def stable_development_decision(
    *,
    v10_vs_base: dict[str, Any],
    v10_vs_pca: dict[str, Any],
    fold_gains_over_pca: Any,
    gap_recovery: float,
    pca_mrr: float,
    v10_mrr: float,
    pca_ndcg: float,
    v10_ndcg: float,
    optimizer_audits_pass: bool,
    accepted_losses_monotone: bool,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    folds = np.asarray(fold_gains_over_pca, dtype=np.float64)
    if folds.ndim != 1 or not len(folds) or np.any(~np.isfinite(folds)):
        raise ValueError("fold gains must be a finite vector")
    gates = {
        "minimum_gain_over_base": v10_vs_base["mean_difference"]
        >= float(thresholds["minimum_recall_at_10_gain_over_base"]),
        "base_bootstrap_lower_positive": v10_vs_base["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "minimum_gain_over_pca": v10_vs_pca["mean_difference"]
        >= float(thresholds["minimum_recall_at_10_gain_over_pca"]),
        "pca_bootstrap_lower_positive": v10_vs_pca["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "pca_randomization_significant": v10_vs_pca[
            "randomization_p_value_one_sided"
        ] <= float(thresholds["maximum_randomization_p_value"]),
        "minimum_improved_queries_over_pca": v10_vs_pca["improved_queries"]
        >= int(thresholds["minimum_improved_queries_over_pca"]),
        "minimum_net_improved_queries_over_pca": (
            v10_vs_pca["improved_queries"] - v10_vs_pca["harmed_queries"]
        ) >= int(thresholds["minimum_net_improved_queries_over_pca"]),
        "worst_fold_nonnegative": float(np.min(folds))
        >= float(thresholds["minimum_worst_fold_gain_over_pca"]),
        "minimum_gap_recovery": gap_recovery
        >= float(thresholds["minimum_candidate_gap_recovery_fraction"]),
        "mrr_guardrail": v10_mrr - pca_mrr
        >= float(thresholds["minimum_mrr_change_vs_pca"]),
        "ndcg_guardrail": v10_ndcg - pca_ndcg
        >= float(thresholds["minimum_ndcg_change_vs_pca"]),
        "gradient_audits_pass": bool(optimizer_audits_pass),
        "accepted_losses_monotone": bool(accepted_losses_monotone),
    }
    passed = all(gates.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "decision": (
            thresholds["go_decision"] if passed else thresholds["stop_decision"]
        ),
        "gates": {name: bool(value) for name, value in gates.items()},
        "failed_gates": [name for name, value in gates.items() if not value],
        "fresh_external_access_authorized": False,
        "v9_reuse_authorized": False,
    }
