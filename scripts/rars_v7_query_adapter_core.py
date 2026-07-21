#!/usr/bin/env python3
"""Deterministic numerical core for the RARS-v7 query-adapter pilot.

The document index is immutable.  Training changes only a small query-side
map.  Positive labels are explicit qrels; all other candidates are described
as unjudged challengers rather than explicit negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "rars_v7_query_adapter_pilot_v1"
PROMOTION = 0
PROTECTION = 1


@dataclass(frozen=True)
class QuerySplit:
    training: np.ndarray
    selection: np.ndarray
    training_qids: tuple[str, ...]
    selection_qids: tuple[str, ...]

    def __post_init__(self) -> None:
        training = np.asarray(self.training)
        selection = np.asarray(self.selection)
        if training.ndim != 1 or selection.ndim != 1:
            raise ValueError("Query split indices must be vectors")
        if not np.issubdtype(training.dtype, np.integer) or not np.issubdtype(
            selection.dtype, np.integer
        ):
            raise ValueError("Query split indices must be integers")
        if len(training) != len(self.training_qids) or len(selection) != len(
            self.selection_qids
        ):
            raise ValueError("Query IDs must match split index counts")
        if len(np.intersect1d(training, selection)):
            raise ValueError("Training and selection queries overlap")


@dataclass(frozen=True)
class CutoffPairBatch:
    """Flattened positive/challenger pairs tied to a query and cutoff role."""

    query: np.ndarray
    positive_row: np.ndarray
    challenger_row: np.ndarray
    teacher_margin: np.ndarray
    base_pq_margin: np.ndarray
    raw_weight: np.ndarray
    balanced_weight: np.ndarray
    kind: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(getattr(self, name)) for name in self.__dataclass_fields__)
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("Cutoff-pair fields must have identical lengths")
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("Cutoff-pair fields must be vectors")
        if len(self.query):
            if np.any(np.asarray(self.query) < 0):
                raise ValueError("Query indices must be non-negative")
            if np.any(np.asarray(self.positive_row) < 0) or np.any(
                np.asarray(self.challenger_row) < 0
            ):
                raise ValueError("Document rows must be non-negative")
            if np.any(~np.isfinite(self.teacher_margin)) or np.any(
                np.asarray(self.teacher_margin) <= 0
            ):
                raise ValueError("Teacher margins must be finite and positive")
            if np.any(~np.isfinite(self.base_pq_margin)):
                raise ValueError("Base-PQ margins must be finite")
            if np.any(~np.isfinite(self.raw_weight)) or np.any(
                np.asarray(self.raw_weight) <= 0
            ):
                raise ValueError("Raw weights must be finite and positive")
            if np.any(~np.isfinite(self.balanced_weight)) or np.any(
                np.asarray(self.balanced_weight) <= 0
            ):
                raise ValueError("Balanced weights must be finite and positive")
            if not set(np.unique(self.kind)).issubset({PROMOTION, PROTECTION}):
                raise ValueError("Unknown cutoff-pair kind")

    def __len__(self) -> int:
        return len(self.query)


def _empty_pairs() -> CutoffPairBatch:
    integer = np.empty(0, dtype=np.int64)
    floating = np.empty(0, dtype=np.float32)
    return CutoffPairBatch(
        query=integer,
        positive_row=integer.copy(),
        challenger_row=integer.copy(),
        teacher_margin=floating,
        base_pq_margin=floating.copy(),
        raw_weight=floating.copy(),
        balanced_weight=floating.copy(),
        kind=np.empty(0, dtype=np.uint8),
    )


def deterministic_query_split(
    query_ids: Iterable[str], *, selection_count: int, salt: str
) -> QuerySplit:
    """Return a label-blind, exact-size split ordered by salted SHA-256."""

    qids = tuple(str(value) for value in query_ids)
    if not qids or len(qids) != len(set(qids)):
        raise ValueError("Query IDs must be non-empty and unique")
    if not salt or not 0 < selection_count < len(qids):
        raise ValueError("Invalid split salt or selection count")
    ranked = sorted(
        range(len(qids)),
        key=lambda index: (
            hashlib.sha256(f"{salt}\0{qids[index]}".encode("utf-8")).digest(),
            qids[index],
        ),
    )
    selection = np.asarray(sorted(ranked[:selection_count]), dtype=np.int64)
    training = np.asarray(sorted(ranked[selection_count:]), dtype=np.int64)
    return QuerySplit(
        training=training,
        selection=selection,
        training_qids=tuple(qids[index] for index in training),
        selection_qids=tuple(qids[index] for index in selection),
    )


def newline_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _take_pairs(batch: CutoffPairBatch, indices: np.ndarray) -> CutoffPairBatch:
    chosen = np.asarray(indices, dtype=np.int64)
    return CutoffPairBatch(
        **{
            name: np.asarray(getattr(batch, name))[chosen]
            for name in batch.__dataclass_fields__
        }
    )


def subset_pairs(batch: CutoffPairBatch, query_indices: np.ndarray) -> CutoffPairBatch:
    allowed = np.asarray(query_indices, dtype=np.int64)
    return _take_pairs(batch, np.flatnonzero(np.isin(batch.query, allowed)))


def query_balanced_weights(
    query: np.ndarray, kind: np.ndarray, raw_weight: np.ndarray
) -> np.ndarray:
    """Normalize each represented query to equal total weight per pair type."""

    query_values = np.asarray(query, dtype=np.int64)
    kind_values = np.asarray(kind, dtype=np.uint8)
    raw = np.asarray(raw_weight, dtype=np.float64)
    if not (query_values.shape == kind_values.shape == raw.shape) or raw.ndim != 1:
        raise ValueError("Weight inputs must be matching vectors")
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("Raw pair weights must be finite and positive")
    balanced = np.empty_like(raw)
    for pair_kind in np.unique(kind_values):
        kind_mask = kind_values == pair_kind
        for query_index in np.unique(query_values[kind_mask]):
            members = np.flatnonzero(kind_mask & (query_values == query_index))
            balanced[members] = raw[members] / raw[members].sum()
    return balanced.astype(np.float32)


def make_pair_batch(
    records: list[tuple[int, int, int, float, float, float, int]],
) -> CutoffPairBatch:
    if not records:
        return _empty_pairs()
    columns = tuple(zip(*records))
    query = np.asarray(columns[0], dtype=np.int64)
    positive = np.asarray(columns[1], dtype=np.int64)
    challenger = np.asarray(columns[2], dtype=np.int64)
    teacher = np.asarray(columns[3], dtype=np.float32)
    pq = np.asarray(columns[4], dtype=np.float32)
    raw = np.asarray(columns[5], dtype=np.float32)
    kind = np.asarray(columns[6], dtype=np.uint8)
    balanced = query_balanced_weights(query, kind, raw)
    return CutoffPairBatch(
        query, positive, challenger, teacher, pq, raw, balanced, kind
    )


def promotion_pairs_from_v6(batch: Any) -> CutoffPairBatch:
    """Convert the frozen v6 capped flip structure into v7 promotion pairs."""

    records = [
        (
            int(batch.query[index]),
            int(batch.positive_row[index]),
            int(batch.unjudged_row[index]),
            float(batch.exact_margin[index]),
            float(batch.pq_margin[index]),
            float(batch.weight[index]),
            PROMOTION,
        )
        for index in range(len(batch.query))
    ]
    return make_pair_batch(records)


def mine_top10_protection_pairs(
    base_rows: np.ndarray,
    candidate_rows: np.ndarray,
    exact_scores: np.ndarray,
    pq_scores: np.ndarray,
    positive_rows: np.ndarray,
    positive_valid: np.ndarray,
    *,
    negative_window: int = 16,
    max_challengers_per_positive: int = 4,
    margin_temperature: float = 0.05,
    damage_scale: float = 8.0,
) -> CutoffPairBatch:
    """Protect known positives already in Base-PQ Top-10.

    Challengers come from Base-PQ ranks 11--(10 + ``negative_window``).
    A pair is admitted only when the FP32 same-route teacher ranks the positive
    above the unjudged challenger.  Selection is deterministic and capped per
    query-positive.
    """

    base = np.asarray(base_rows, dtype=np.int64)
    candidates = np.asarray(candidate_rows, dtype=np.int64)
    exact = np.asarray(exact_scores, dtype=np.float32)
    pq = np.asarray(pq_scores, dtype=np.float32)
    positives = np.asarray(positive_rows, dtype=np.int64)
    positive_mask = np.asarray(positive_valid, dtype=bool)
    if base.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Base and candidate rows must be matrices")
    if not (candidates.shape == exact.shape == pq.shape):
        raise ValueError("Candidate rows and score matrices must match")
    if len(base) != len(candidates) or len(positives) != len(candidates):
        raise ValueError("Query counts must match")
    if positives.shape != positive_mask.shape:
        raise ValueError("Positive rows and masks must match")
    if negative_window <= 0 or max_challengers_per_positive <= 0:
        raise ValueError("Protection mining counts must be positive")
    if margin_temperature <= 0 or damage_scale < 0:
        raise ValueError("Protection weighting parameters are invalid")

    records: list[tuple[int, int, int, float, float, float, int]] = []
    for query_index in range(len(base)):
        known = set(int(row) for row in positives[query_index, positive_mask[query_index]])
        row_to_position = {
            int(row): position
            for position, row in enumerate(candidates[query_index])
            if row >= 0
        }
        top10_positives = [int(row) for row in base[query_index, :10] if int(row) in known]
        challenger_rows = [
            int(row)
            for row in base[query_index, 10 : 10 + negative_window]
            if row >= 0 and int(row) not in known
        ]
        for positive_row in sorted(set(top10_positives)):
            if positive_row not in row_to_position:
                raise ValueError("Candidate union omitted a protected positive")
            positive_position = row_to_position[positive_row]
            local: list[tuple[float, int, float, float, float]] = []
            for challenger_row in challenger_rows:
                challenger_position = row_to_position.get(challenger_row)
                if challenger_position is None:
                    raise ValueError("Candidate union omitted a Top-10 challenger")
                teacher_margin = float(
                    exact[query_index, positive_position]
                    - exact[query_index, challenger_position]
                )
                if teacher_margin <= 0.0:
                    continue
                pq_margin = float(
                    pq[query_index, positive_position]
                    - pq[query_index, challenger_position]
                )
                damage = max(0.0, teacher_margin - pq_margin)
                raw_weight = (
                    1.0
                    + math.exp(-teacher_margin / margin_temperature)
                    + damage_scale * damage
                )
                local.append(
                    (damage, challenger_row, teacher_margin, pq_margin, raw_weight)
                )
            local.sort(key=lambda value: (-value[0], value[1]))
            for _, challenger_row, teacher_margin, pq_margin, raw_weight in local[
                :max_challengers_per_positive
            ]:
                records.append(
                    (
                        query_index,
                        positive_row,
                        challenger_row,
                        teacher_margin,
                        pq_margin,
                        raw_weight,
                        PROTECTION,
                    )
                )
    return make_pair_batch(records)


def concatenate_pairs(*batches: CutoffPairBatch) -> CutoffPairBatch:
    nonempty = [batch for batch in batches if len(batch)]
    if not nonempty:
        return _empty_pairs()
    combined = CutoffPairBatch(
        **{
            name: np.concatenate([np.asarray(getattr(batch, name)) for batch in nonempty])
            for name in CutoffPairBatch.__dataclass_fields__
        }
    )
    return replace(
        combined,
        balanced_weight=query_balanced_weights(
            combined.query, combined.kind, combined.raw_weight
        ),
    )


def summarize_pairs(batch: CutoffPairBatch) -> dict[str, Any]:
    output: dict[str, Any] = {"total_pairs": int(len(batch))}
    for label, code in (("promotion", PROMOTION), ("protection", PROTECTION)):
        members = np.flatnonzero(batch.kind == code)
        weights = np.asarray(batch.balanced_weight[members], dtype=np.float64)
        output[label] = {
            "pairs": int(len(members)),
            "queries": int(np.unique(batch.query[members]).size) if len(members) else 0,
            "positive_documents": int(np.unique(batch.positive_row[members]).size)
            if len(members)
            else 0,
            "challenger_documents": int(np.unique(batch.challenger_row[members]).size)
            if len(members)
            else 0,
            "balanced_weight_sum": float(weights.sum()) if len(weights) else 0.0,
        }
    return output


def paired_bootstrap_mean_difference(
    treatment: np.ndarray,
    baseline: np.ndarray,
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
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration")
    delta = treatment_values - baseline_values
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    block = 2048
    for start in range(0, replicates, block):
        end = min(replicates, start + block)
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


def eligible_checkpoint(
    record: dict[str, Any], *, base_r10: float, teacher_r100: float,
    maximum_r10_drop: float, maximum_teacher_drop: float
) -> bool:
    return (
        float(record["hard_pq_recall_at_10"]) >= base_r10 - maximum_r10_drop
        and float(record["adapted_same_ivf_fp32_recall_at_100"])
        >= teacher_r100 - maximum_teacher_drop
    )


def select_checkpoint(
    history: list[dict[str, Any]],
    *,
    base_r10: float,
    teacher_r100: float,
    maximum_r10_drop: float,
    maximum_teacher_drop: float,
) -> int:
    if not history or [int(row["epoch"]) for row in history] != list(
        range(len(history))
    ):
        raise ValueError("History must start at identity epoch zero and be contiguous")
    eligible = [
        row
        for row in history
        if eligible_checkpoint(
            row,
            base_r10=base_r10,
            teacher_r100=teacher_r100,
            maximum_r10_drop=maximum_r10_drop,
            maximum_teacher_drop=maximum_teacher_drop,
        )
    ]
    if not eligible:
        raise ValueError("Identity epoch must make checkpoint selection non-empty")
    best = max(
        eligible,
        key=lambda row: (
            float(row["hard_pq_recall_at_100"]),
            float(row["hard_pq_recall_at_10"]),
            float(row["adapted_same_ivf_fp32_recall_at_100"]),
            float(row["mean_query_cosine"]),
            -int(row["epoch"]),
        ),
    )
    return int(best["epoch"])


def pilot_gate_decision(
    *,
    selected_epoch: int,
    base_r10: float,
    adapted_r10: float,
    base_r100: float,
    adapted_r100: float,
    teacher_r100: float,
    adapted_teacher_r100: float,
    bootstrap_lower: float,
    improved_queries: int,
    harmed_queries: int,
    mean_query_cosine: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    gain = adapted_r100 - base_r100
    gap = teacher_r100 - base_r100
    recovery = gain / gap if gap > 0.0 else 0.0
    gates = {
        "selected_epoch_nonzero": selected_epoch > 0,
        "minimum_recall_at_100_gain": gain
        >= float(thresholds["minimum_hard_pq_recall_at_100_gain"]),
        "bootstrap_lower_above_zero": bootstrap_lower
        > float(thresholds["paired_bootstrap_95_lower_must_exceed"]),
        "minimum_gap_recovery": recovery
        >= float(thresholds["minimum_same_route_teacher_gap_recovery_fraction"]),
        "minimum_improved_query_support": improved_queries
        >= int(thresholds["minimum_improved_selection_queries"]),
        "minimum_net_improved_query_support": improved_queries - harmed_queries
        >= int(thresholds["minimum_net_improved_selection_queries"]),
        "recall_at_10_guardrail": adapted_r10
        >= base_r10 - float(thresholds["maximum_hard_pq_recall_at_10_drop"]),
        "same_ivf_fp32_guardrail": adapted_teacher_r100
        >= teacher_r100
        - float(thresholds["maximum_adapted_same_route_fp32_recall_at_100_drop"]),
        "query_cosine_guardrail": mean_query_cosine
        >= float(thresholds["minimum_mean_query_cosine"]),
    }
    passed = all(gates.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "V7_QUERY_ADAPTER_PILOT_COMPLETE",
        "decision": thresholds["go_decision"] if passed else thresholds["stop_decision"],
        "gates": {name: bool(value) for name, value in gates.items()},
        "failed_gates": [name for name, value in gates.items() if not value],
        "recall_at_100_gain": float(gain),
        "same_route_teacher_gap_recovery_fraction": float(recovery),
        "development_only": True,
        "future_access_authorized": False,
        "rars_combination_authorized": False,
    }
