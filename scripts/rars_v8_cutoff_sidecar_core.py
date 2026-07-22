#!/usr/bin/env python3
"""Deterministic core for the RARS-v8 frozen-index residual sidecar.

V8 deliberately removes the unconstrained query/document projections used by
earlier development versions.  It learns one orthonormal residual subspace,
initialised from uncentred PCA, and keeps the deployed score symmetric::

    score(q, x) = score_pq(q, x) + alpha * (q B) dot dequant(code_B(r_x))

Only explicit positive qrels are treated as relevant.  Every other candidate
is an *unjudged challenger*, never an explicit negative.  Training pairs are
query-balanced and split equally between promotion and protection whenever
both roles are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "rars_v8_cutoff_sidecar_v1"
PROMOTION = np.uint8(0)
PROTECTION = np.uint8(1)


def _as_matrix(value: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return array


def _orient_columns(basis: np.ndarray) -> np.ndarray:
    """Give every orthonormal column a deterministic sign."""

    output = np.asarray(basis, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


def validate_orthonormal_basis(
    basis: Any, *, dimension: int, rank: int, atol: float = 2e-4
) -> np.ndarray:
    value = _as_matrix(basis, name="basis", dtype=np.float64)
    if value.shape != (dimension, rank):
        raise ValueError(
            f"basis must have shape {(dimension, rank)}, got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("basis contains non-finite values")
    if not np.allclose(
        value.T @ value, np.eye(rank), rtol=0.0, atol=atol
    ):
        raise ValueError("basis is not orthonormal")
    return _orient_columns(value).astype(np.float32)


@dataclass(frozen=True)
class CutoffPairBatch:
    """Positive/challenger pairs tied to a query and a cutoff role."""

    query: np.ndarray
    positive_position: np.ndarray
    challenger_position: np.ndarray
    positive_residual_row: np.ndarray
    challenger_residual_row: np.ndarray
    base_margin: np.ndarray
    teacher_margin: np.ndarray
    target_residual_margin: np.ndarray
    raw_weight: np.ndarray
    balanced_weight: np.ndarray
    kind: np.ndarray

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(getattr(self, field))
            for field in self.__dataclass_fields__
        ]
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("Cutoff-pair fields have different lengths")
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("Cutoff-pair fields must be vectors")
        if len(self.query):
            for name in (
                "query",
                "positive_position",
                "challenger_position",
                "positive_residual_row",
                "challenger_residual_row",
            ):
                if np.any(np.asarray(getattr(self, name)) < 0):
                    raise ValueError(f"{name} must be non-negative")
            for name in (
                "base_margin",
                "teacher_margin",
                "target_residual_margin",
                "raw_weight",
                "balanced_weight",
            ):
                if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                    raise ValueError(f"{name} contains non-finite values")
            if np.any(np.asarray(self.teacher_margin) <= 0):
                raise ValueError("Teacher margins must be positive")
            if np.any(np.asarray(self.raw_weight) <= 0) or np.any(
                np.asarray(self.balanced_weight) <= 0
            ):
                raise ValueError("Pair weights must be positive")
            if not set(np.unique(self.kind)).issubset(
                {int(PROMOTION), int(PROTECTION)}
            ):
                raise ValueError("Unknown cutoff-pair role")

    def __len__(self) -> int:
        return int(len(self.query))


def _empty_pairs() -> CutoffPairBatch:
    integer = np.empty(0, dtype=np.int64)
    floating = np.empty(0, dtype=np.float32)
    return CutoffPairBatch(
        query=integer,
        positive_position=integer.copy(),
        challenger_position=integer.copy(),
        positive_residual_row=integer.copy(),
        challenger_residual_row=integer.copy(),
        base_margin=floating,
        teacher_margin=floating.copy(),
        target_residual_margin=floating.copy(),
        raw_weight=floating.copy(),
        balanced_weight=floating.copy(),
        kind=np.empty(0, dtype=np.uint8),
    )


def query_role_balanced_weights(
    query: Any,
    kind: Any,
    raw_weight: Any,
    *,
    promotion_mass: float = 0.5,
) -> np.ndarray:
    """Allocate equal mass to each represented query within each role.

    Unlike the v7 implementation, a role with more represented queries cannot
    silently receive more total loss mass.  If only one role is present its
    requested mass is renormalised to one.
    """

    queries = np.asarray(query, dtype=np.int64)
    kinds = np.asarray(kind, dtype=np.uint8)
    raw = np.asarray(raw_weight, dtype=np.float64)
    if not (queries.shape == kinds.shape == raw.shape) or raw.ndim != 1:
        raise ValueError("Weight inputs must be matching vectors")
    if len(raw) == 0:
        return np.empty(0, dtype=np.float32)
    if not 0.0 <= promotion_mass <= 1.0:
        raise ValueError("promotion_mass must be within [0, 1]")
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("Raw weights must be finite and positive")
    requested = {
        int(PROMOTION): float(promotion_mass),
        int(PROTECTION): float(1.0 - promotion_mass),
    }
    active = [role for role in requested if np.any(kinds == role)]
    active_mass = sum(requested[role] for role in active)
    if active_mass <= 0:
        raise ValueError("Active pair roles have zero requested mass")
    output = np.zeros(len(raw), dtype=np.float64)
    for role in active:
        role_mask = kinds == role
        represented = np.unique(queries[role_mask])
        role_mass = requested[role] / active_mass
        query_mass = role_mass / len(represented)
        for query_index in represented:
            members = np.flatnonzero(role_mask & (queries == query_index))
            local = raw[members]
            output[members] = query_mass * local / local.sum()
    if not np.isclose(output.sum(), 1.0, rtol=0.0, atol=1e-7):
        raise AssertionError("Internal query-role weight normalisation failed")
    return output.astype(np.float32)


def _stable_score_order(scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
    valid_rows = np.where(rows >= 0, rows, np.iinfo(np.int64).max)
    valid_scores = np.where(rows >= 0, scores, -np.inf)
    return np.lexsort((valid_rows, -valid_scores))


def mine_cutoff_pairs(
    candidate_rows: Any,
    residual_lookup: Any,
    base_scores: Any,
    teacher_scores: Any,
    positive_labels: Any,
    *,
    final_k: int = 10,
    top_b: int = 40,
    protection_window: int = 16,
    max_challengers_per_positive: int = 4,
    margin_temperature: float = 0.02,
    damage_scale: float = 4.0,
    promotion_mass: float = 0.5,
) -> CutoffPairBatch:
    """Mine static, quantisation-induced Top-k promotion/protection pairs.

    Promotion requires a known positive outside Base Top-k but inside the
    correctable Top-B whose same-route FP32 score exceeds an unjudged Base
    Top-k challenger.  Protection retains a known positive already in Base
    Top-k against unjudged ranks k+1..k+window when the FP32 teacher agrees.
    """

    rows = _as_matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _as_matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _as_matrix(base_scores, name="base_scores", dtype=np.float32)
    teacher = _as_matrix(teacher_scores, name="teacher_scores", dtype=np.float32)
    labels = _as_matrix(positive_labels, name="positive_labels")
    if not (rows.shape == lookup.shape == base.shape == teacher.shape == labels.shape):
        raise ValueError("Candidate, lookup, score, and label matrices must match")
    if not 0 < final_k < top_b <= rows.shape[1]:
        raise ValueError("Require 0 < final_k < top_b <= candidate count")
    if protection_window <= 0 or max_challengers_per_positive <= 0:
        raise ValueError("Pair mining counts must be positive")
    if margin_temperature <= 0 or damage_scale < 0:
        raise ValueError("Pair weighting parameters are invalid")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("positive_labels must be binary")

    records: list[tuple[int, int, int, int, int, float, float, float, float, int]] = []
    for query_index in range(len(rows)):
        order = _stable_score_order(base[query_index], rows[query_index])
        correctable = order[:top_b]
        top = order[:final_k]
        protection_band = order[final_k : min(top_b, final_k + protection_window)]

        promotion_positives = [
            int(position)
            for position in correctable[final_k:]
            if labels[query_index, position] > 0 and lookup[query_index, position] >= 0
        ]
        promotion_challengers = [
            int(position)
            for position in top
            if labels[query_index, position] == 0 and lookup[query_index, position] >= 0
        ]
        protection_positives = [
            int(position)
            for position in top
            if labels[query_index, position] > 0 and lookup[query_index, position] >= 0
        ]
        protection_challengers = [
            int(position)
            for position in protection_band
            if labels[query_index, position] == 0 and lookup[query_index, position] >= 0
        ]

        def append_pairs(
            positives: Iterable[int], challengers: Iterable[int], role: np.uint8
        ) -> None:
            for positive in positives:
                local: list[tuple[float, int, float, float, float]] = []
                for challenger in challengers:
                    teacher_margin = float(
                        teacher[query_index, positive]
                        - teacher[query_index, challenger]
                    )
                    if teacher_margin <= 0:
                        continue
                    base_margin = float(
                        base[query_index, positive] - base[query_index, challenger]
                    )
                    residual_margin = teacher_margin - base_margin
                    damage = max(0.0, teacher_margin - base_margin)
                    boundary = np.exp(-abs(base_margin) / margin_temperature)
                    raw_weight = 1.0 + float(boundary) + damage_scale * damage
                    local.append(
                        (damage, challenger, base_margin, teacher_margin, raw_weight)
                    )
                local.sort(key=lambda item: (-item[0], int(rows[query_index, item[1]])))
                for _, challenger, base_margin, teacher_margin, raw_weight in local[
                    :max_challengers_per_positive
                ]:
                    records.append(
                        (
                            query_index,
                            positive,
                            challenger,
                            int(lookup[query_index, positive]),
                            int(lookup[query_index, challenger]),
                            base_margin,
                            teacher_margin,
                            teacher_margin - base_margin,
                            raw_weight,
                            int(role),
                        )
                    )

        append_pairs(promotion_positives, promotion_challengers, PROMOTION)
        append_pairs(protection_positives, protection_challengers, PROTECTION)

    if not records:
        return _empty_pairs()
    columns = tuple(zip(*records))
    query = np.asarray(columns[0], dtype=np.int64)
    kinds = np.asarray(columns[9], dtype=np.uint8)
    raw = np.asarray(columns[8], dtype=np.float32)
    return CutoffPairBatch(
        query=query,
        positive_position=np.asarray(columns[1], dtype=np.int64),
        challenger_position=np.asarray(columns[2], dtype=np.int64),
        positive_residual_row=np.asarray(columns[3], dtype=np.int64),
        challenger_residual_row=np.asarray(columns[4], dtype=np.int64),
        base_margin=np.asarray(columns[5], dtype=np.float32),
        teacher_margin=np.asarray(columns[6], dtype=np.float32),
        target_residual_margin=np.asarray(columns[7], dtype=np.float32),
        raw_weight=raw,
        balanced_weight=query_role_balanced_weights(
            query, kinds, raw, promotion_mass=promotion_mass
        ),
        kind=kinds,
    )


def subset_pairs(batch: CutoffPairBatch, query_indices: Any) -> CutoffPairBatch:
    allowed = np.asarray(query_indices, dtype=np.int64)
    selected = np.flatnonzero(np.isin(batch.query, allowed))
    values = {
        field: np.asarray(getattr(batch, field))[selected]
        for field in batch.__dataclass_fields__
    }
    if len(selected):
        values["balanced_weight"] = query_role_balanced_weights(
            values["query"], values["kind"], values["raw_weight"]
        )
    return CutoffPairBatch(**values)


def summarize_pairs(batch: CutoffPairBatch) -> dict[str, Any]:
    output: dict[str, Any] = {"total_pairs": len(batch)}
    for label, role in (("promotion", PROMOTION), ("protection", PROTECTION)):
        members = np.flatnonzero(batch.kind == role)
        output[label] = {
            "pairs": int(len(members)),
            "queries": int(np.unique(batch.query[members]).size) if len(members) else 0,
            "positive_residual_rows": int(
                np.unique(batch.positive_residual_row[members]).size
            )
            if len(members)
            else 0,
            "challenger_residual_rows": int(
                np.unique(batch.challenger_residual_row[members]).size
            )
            if len(members)
            else 0,
            "balanced_weight_sum": float(batch.balanced_weight[members].sum())
            if len(members)
            else 0.0,
        }
    return output


def fit_uncentered_pca_basis(
    residuals: Any, *, rank: int, batch_size: int = 8192
) -> np.ndarray:
    """Fit deterministic uncentred PCA through a streamed second moment."""

    values = _as_matrix(residuals, name="residuals")
    if not 0 < rank <= values.shape[1] or batch_size <= 0:
        raise ValueError("Invalid PCA rank or batch size")
    covariance = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    count = 0
    for start in range(0, len(values), batch_size):
        block = np.asarray(values[start : start + batch_size], dtype=np.float64)
        if not np.all(np.isfinite(block)):
            raise ValueError("Residuals contain non-finite values")
        covariance += block.T @ block
        count += len(block)
    if count == 0:
        raise ValueError("Cannot fit PCA from an empty residual matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance / count)
    order = np.argsort(-eigenvalues, kind="stable")[:rank]
    basis = _orient_columns(eigenvectors[:, order])
    return validate_orthonormal_basis(
        basis, dimension=values.shape[1], rank=rank
    )


def fit_cutoff_aware_basis(
    queries: Any,
    residuals: Any,
    pairs: CutoffPairBatch,
    anchor_basis: Any,
    *,
    steps: int = 160,
    learning_rate: float = 0.02,
    anchor_weight: float = 0.10,
    huber_delta: float = 0.02,
    gradient_clip: float = 5.0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Fit a symmetric orthonormal subspace by projected deterministic Adam.

    The target is the exact pairwise residual score correction, not a global
    embedding reconstruction loss.  QR retraction after every update prevents
    arbitrary scale growth and guarantees a deployable single-basis scorer.
    """

    query_matrix = _as_matrix(queries, name="queries", dtype=np.float64)
    residual_matrix = _as_matrix(residuals, name="residuals", dtype=np.float64)
    if query_matrix.shape[1] != residual_matrix.shape[1]:
        raise ValueError("Query and residual dimensions disagree")
    anchor = validate_orthonormal_basis(
        anchor_basis,
        dimension=query_matrix.shape[1],
        rank=np.asarray(anchor_basis).shape[1],
    ).astype(np.float64)
    if len(pairs) == 0:
        return anchor.astype(np.float32), [{"step": 0.0, "loss": 0.0}]
    if np.max(pairs.query) >= len(query_matrix):
        raise ValueError("Pair query index is outside the query matrix")
    if max(
        int(np.max(pairs.positive_residual_row)),
        int(np.max(pairs.challenger_residual_row)),
    ) >= len(residual_matrix):
        raise ValueError("Pair residual row is outside the residual matrix")
    if steps <= 0 or learning_rate <= 0 or anchor_weight < 0:
        raise ValueError("Invalid optimisation controls")
    if huber_delta <= 0 or gradient_clip <= 0:
        raise ValueError("Huber delta and gradient clip must be positive")

    q = query_matrix[pairs.query]
    delta_r = (
        residual_matrix[pairs.positive_residual_row]
        - residual_matrix[pairs.challenger_residual_row]
    )
    target = np.asarray(pairs.target_residual_margin, dtype=np.float64)
    direct_target = np.einsum("pd,pd->p", q, delta_r)
    if not np.allclose(target, direct_target, rtol=1e-4, atol=2e-5):
        raise ValueError("Registered residual-margin targets do not match residuals")
    weights = np.asarray(pairs.balanced_weight, dtype=np.float64)
    weights /= weights.sum()

    basis = anchor.copy()
    first_moment = np.zeros_like(basis)
    second_moment = np.zeros_like(basis)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    history: list[dict[str, float]] = []
    anchor_projector = anchor @ anchor.T

    def objective(candidate: np.ndarray) -> tuple[float, float, float]:
        candidate_q = q @ candidate
        candidate_residual = delta_r @ candidate
        candidate_prediction = np.einsum(
            "pr,pr->p", candidate_q, candidate_residual
        )
        candidate_error = candidate_prediction - target
        candidate_absolute = np.abs(candidate_error)
        candidate_huber = np.where(
            candidate_absolute <= huber_delta,
            0.5 * candidate_error * candidate_error / huber_delta,
            candidate_absolute - 0.5 * huber_delta,
        )
        candidate_overlap = anchor.T @ candidate
        candidate_anchor = anchor_weight * (
            1.0
            - float(np.sum(candidate_overlap * candidate_overlap))
            / candidate.shape[1]
        )
        candidate_pair = float(np.sum(weights * candidate_huber))
        return candidate_pair, float(candidate_anchor), float(
            candidate_pair + candidate_anchor
        )

    for step in range(1, steps + 1):
        q_projected = q @ basis
        residual_projected = delta_r @ basis
        predicted = np.einsum("pr,pr->p", q_projected, residual_projected)
        error = predicted - target
        absolute = np.abs(error)
        huber = np.where(
            absolute <= huber_delta,
            0.5 * error * error / huber_delta,
            absolute - 0.5 * huber_delta,
        )
        derivative = np.where(
            absolute <= huber_delta, error / huber_delta, np.sign(error)
        )
        coefficient = weights * derivative
        gradient = q.T @ (coefficient[:, None] * residual_projected)
        gradient += delta_r.T @ (coefficient[:, None] * q_projected)

        overlap = anchor.T @ basis
        anchor_loss = anchor_weight * (
            1.0 - float(np.sum(overlap * overlap)) / basis.shape[1]
        )
        gradient += (
            -2.0
            * anchor_weight
            / basis.shape[1]
            * (anchor_projector @ basis)
        )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > gradient_clip:
            gradient *= gradient_clip / gradient_norm

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        corrected_first = first_moment / (1.0 - beta1**step)
        corrected_second = second_moment / (1.0 - beta2**step)
        proposal = basis - learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        proposal_pair, proposal_anchor, proposal_total = objective(proposal)
        retracted, _ = np.linalg.qr(proposal, mode="reduced")
        basis = _orient_columns(retracted)
        retracted_pair, retracted_anchor, retracted_total = objective(basis)
        pre_pair = float(np.sum(weights * huber))
        pre_total = float(pre_pair + anchor_loss)
        history.append(
            {
                "step": float(step),
                # Backward-compatible fields record the pre-update objective,
                # matching every frozen V8 development packet.
                "pair_huber_loss": pre_pair,
                "anchor_loss": float(anchor_loss),
                "loss": pre_total,
                "gradient_norm_before_clip": gradient_norm,
                # These audit-only fields do not alter the update.  They make
                # the Adam-versus-QR loss direction observable in later
                # development diagnostics without rewriting frozen V8.
                "proposal_pair_huber_loss": proposal_pair,
                "proposal_anchor_loss": proposal_anchor,
                "proposal_loss": proposal_total,
                "retracted_pair_huber_loss": retracted_pair,
                "retracted_anchor_loss": retracted_anchor,
                "retracted_loss": retracted_total,
                "proposal_loss_change": float(proposal_total - pre_total),
                "retraction_loss_change": float(
                    retracted_total - proposal_total
                ),
                "full_step_loss_change": float(
                    retracted_total - pre_total
                ),
            }
        )
    return validate_orthonormal_basis(
        basis, dimension=query_matrix.shape[1], rank=basis.shape[1]
    ), history


def fit_int8_scales(
    residuals: Any, basis: Any, *, batch_size: int = 8192
) -> np.ndarray:
    values = _as_matrix(residuals, name="residuals")
    projection = _as_matrix(basis, name="basis", dtype=np.float32)
    if values.shape[1] != projection.shape[0] or batch_size <= 0:
        raise ValueError("Residual/basis dimensions or batch size are invalid")
    maximum = np.zeros(projection.shape[1], dtype=np.float32)
    for start in range(0, len(values), batch_size):
        coefficients = (
            np.asarray(values[start : start + batch_size], dtype=np.float32)
            @ projection
        )
        maximum = np.maximum(maximum, np.max(np.abs(coefficients), axis=0))
    if not len(values):
        raise ValueError("Cannot calibrate scales from no residuals")
    return np.maximum(maximum / 127.0, np.finfo(np.float32).tiny).astype(np.float32)


def encode_residuals_int8(
    residuals: Any, basis: Any, scales: Any
) -> tuple[np.ndarray, dict[str, float]]:
    values = _as_matrix(residuals, name="residuals", dtype=np.float32)
    projection = _as_matrix(basis, name="basis", dtype=np.float32)
    scale = np.asarray(scales, dtype=np.float32)
    if values.shape[1] != projection.shape[0] or scale.shape != (
        projection.shape[1],
    ):
        raise ValueError("Residual, basis, and scale shapes disagree")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Scales must be finite and positive")
    raw = values @ projection / scale[None, :]
    rounded = np.rint(raw)
    saturated = np.abs(rounded) > 127
    codes = np.clip(rounded, -127, 127).astype(np.int8)
    return codes, {
        "coefficient_count": int(raw.size),
        "saturated_coefficients": int(saturated.sum()),
        "saturation_fraction": float(saturated.mean()) if raw.size else 0.0,
    }


def score_sidecar_candidates(
    queries: Any,
    candidate_rows: Any,
    residual_lookup: Any,
    base_scores: Any,
    basis: Any,
    codes: Any,
    scales: Any,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    query_matrix = _as_matrix(queries, name="queries", dtype=np.float32)
    rows = _as_matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _as_matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _as_matrix(base_scores, name="base_scores", dtype=np.float32)
    projection = _as_matrix(basis, name="basis", dtype=np.float32)
    code_matrix = _as_matrix(codes, name="codes")
    scale = np.asarray(scales, dtype=np.float32)
    if not (rows.shape == lookup.shape == base.shape):
        raise ValueError("Candidate rows, lookup, and scores must match")
    if len(query_matrix) != len(rows) or query_matrix.shape[1] != projection.shape[0]:
        raise ValueError("Query/candidate/basis shapes disagree")
    if code_matrix.shape[1] != projection.shape[1] or scale.shape != (
        projection.shape[1],
    ):
        raise ValueError("Code or scale rank differs from basis")
    if not np.isfinite(alpha) or alpha < 0 or not 0 < top_b <= rows.shape[1]:
        raise ValueError("Invalid alpha or Top-B")
    output = base.copy()
    q_projected = query_matrix @ projection
    for query_index in range(len(rows)):
        order = _stable_score_order(base[query_index], rows[query_index])
        selected = order[:top_b]
        selected_lookup = lookup[query_index, selected]
        valid = selected_lookup >= 0
        if not np.any(valid):
            continue
        if np.max(selected_lookup[valid]) >= len(code_matrix):
            raise ValueError("Residual lookup is outside the code matrix")
        coefficients = (
            np.asarray(code_matrix[selected_lookup[valid]], dtype=np.float32)
            * scale[None, :]
        )
        output[query_index, selected[valid]] += alpha * (
            coefficients @ q_projected[query_index]
        )
    return output


def per_query_metrics(
    scores: Any,
    candidate_rows: Any,
    positive_labels: Any,
    relevant_counts: Any,
    *,
    k: int,
) -> dict[str, np.ndarray]:
    values = _as_matrix(scores, name="scores", dtype=np.float64)
    rows = _as_matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    labels = _as_matrix(positive_labels, name="positive_labels")
    denominators = np.asarray(relevant_counts, dtype=np.int64)
    if not (values.shape == rows.shape == labels.shape):
        raise ValueError("Metric score, row, and label matrices must match")
    if denominators.shape != (len(values),) or np.any(denominators <= 0):
        raise ValueError("Every query needs a positive relevance denominator")
    if not 0 < k <= values.shape[1]:
        raise ValueError("Metric cutoff is outside candidate count")
    recall = np.empty(len(values), dtype=np.float64)
    reciprocal_rank = np.empty(len(values), dtype=np.float64)
    ndcg = np.empty(len(values), dtype=np.float64)
    for query_index in range(len(values)):
        order = _stable_score_order(values[query_index], rows[query_index])[:k]
        gains = (labels[query_index, order] > 0).astype(np.float64)
        recall[query_index] = float(gains.sum() / denominators[query_index])
        hits = np.flatnonzero(gains)
        reciprocal_rank[query_index] = 1.0 / (int(hits[0]) + 1) if len(hits) else 0.0
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum(gains * discounts))
        ideal_count = min(int(denominators[query_index]), k)
        ideal = float(np.sum(discounts[:ideal_count]))
        ndcg[query_index] = dcg / ideal if ideal > 0 else 0.0
    return {"recall": recall, "mrr": reciprocal_rank, "ndcg": ndcg}


def paired_bootstrap(
    treatment: Any,
    baseline: Any,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    treatment_values = np.asarray(treatment, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if (
        treatment_values.shape != baseline_values.shape
        or treatment_values.ndim != 1
        or not len(treatment_values)
    ):
        raise ValueError("Paired metric vectors must match and be non-empty")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap controls")
    difference = treatment_values - baseline_values
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    block = 2048
    for start in range(0, replicates, block):
        end = min(replicates, start + block)
        draw = rng.integers(0, len(difference), size=(end - start, len(difference)))
        means[start:end] = difference[draw].mean(axis=1)
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


def candidate_gap_recovery(
    treatment: Any, baseline: Any, teacher: Any
) -> float:
    treatment_mean = float(np.mean(np.asarray(treatment, dtype=np.float64)))
    baseline_mean = float(np.mean(np.asarray(baseline, dtype=np.float64)))
    teacher_mean = float(np.mean(np.asarray(teacher, dtype=np.float64)))
    gap = teacher_mean - baseline_mean
    return float((treatment_mean - baseline_mean) / gap) if gap > 0 else 0.0


def development_decision(
    *,
    rars_vs_base: dict[str, float],
    pca_vs_base: dict[str, float],
    rars_vs_pca: dict[str, float],
    gap_recovery: float,
    pair_support: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Return an evidence-tier decision without changing the claim post hoc."""

    common = {
        "minimum_rars_gain_over_base": rars_vs_base["mean_difference"]
        >= float(thresholds["minimum_rars_recall_at_10_gain_over_base"]),
        "rars_base_bootstrap_lower_positive": rars_vs_base["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "minimum_gap_recovery": gap_recovery
        >= float(thresholds["minimum_candidate_gap_recovery_fraction"]),
        "minimum_improved_query_support": rars_vs_base["improved_queries"]
        >= int(thresholds["minimum_improved_queries"]),
        "positive_net_query_support": rars_vs_base["improved_queries"]
        - rars_vs_base["harmed_queries"]
        >= int(thresholds["minimum_net_improved_queries"]),
        "promotion_query_support": pair_support["promotion"]["queries"]
        >= int(thresholds["minimum_promotion_queries"]),
        "protection_query_support": pair_support["protection"]["queries"]
        >= int(thresholds["minimum_protection_queries"]),
    }
    algorithm = {
        "minimum_rars_gain_over_pca": rars_vs_pca["mean_difference"]
        >= float(thresholds["minimum_rars_recall_at_10_gain_over_pca"]),
        "rars_pca_bootstrap_lower_positive": rars_vs_pca["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
    }
    generic = {
        "pca_or_rars_improves_base": max(
            pca_vs_base["mean_difference"], rars_vs_base["mean_difference"]
        )
        >= float(thresholds["minimum_generic_sidecar_gain_over_base"])
    }
    if all(common.values()) and all(algorithm.values()):
        decision = thresholds["algorithm_go_decision"]
    elif all(common.values()) and all(generic.values()):
        decision = thresholds["generic_sidecar_go_decision"]
    else:
        decision = thresholds["stop_decision"]
    gates = {**common, **algorithm, **generic}
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V8_DEVELOPMENT_COMPLETE",
        "decision": decision,
        "gates": {name: bool(value) for name, value in gates.items()},
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "future_access_authorized": False,
        "interpretation": (
            "Development-only tiering. A separate frozen evaluator is required "
            "before opening any prospective method holdout."
        ),
    }
