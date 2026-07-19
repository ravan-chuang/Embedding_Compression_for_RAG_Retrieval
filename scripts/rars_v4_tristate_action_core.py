#!/usr/bin/env python3
"""Pure NumPy contracts for the RARS-v4 tri-state Phase-0 gate.

The module intentionally contains no trainer.  It distinguishes explicit
non-relevance from missing judgments, measures query-level boundary support,
and solves a non-deployable post-PQ action oracle with a lexicographic
positive-first, explicit-negative-second objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rars_v3_oracle_core import (
    gain_diagnostics,
    paired_bootstrap_mean_delta,
    recall_at_k_per_query,
    stable_topk,
    topk_membership,
)


PROTOCOL_ID = "rars_v4_tristate_action_feasibility_v1"
DESIGN_ROLE_ID = "v4_design_observed"
AUDIT_ROLE_ID = "v4_diagnostic_audit"
POSITIVE = np.int8(1)
EXPLICIT_NEGATIVE = np.int8(-1)
UNJUDGED = np.int8(0)
FOLD_SALT = b"rars_v4_tristate_fold_v1\0"


@dataclass(frozen=True)
class LabelSupportResult:
    """Query-level masks and the JSON-safe label-support summary."""

    base_topk_membership: np.ndarray
    positive_in: np.ndarray
    positive_out_actionable: np.ndarray
    explicit_negative_in: np.ndarray
    explicit_negative_out_actionable: np.ndarray
    label_swap_gain: np.ndarray
    promotion_pair_queries: np.ndarray
    protection_pair_queries: np.ndarray
    summary: dict[str, Any]


@dataclass(frozen=True)
class ActionReachabilityResult:
    """Frozen-action reachability without assuming a learned allocator."""

    downward_feasible_queries: np.ndarray
    upward_feasible_queries: np.ndarray
    joint_swap_reachable_queries: np.ndarray
    summary: dict[str, Any]


@dataclass(frozen=True)
class TriageOracleResult:
    """Exact per-query lexicographic optimum under an action-cost budget."""

    positive_hits_at_k: np.ndarray
    explicit_negative_hits_at_k: np.ndarray
    recall_at_k: np.ndarray
    action_cost: np.ndarray
    rate_assignments: np.ndarray
    topk_membership: np.ndarray


def validate_tristate_labels(
    labels: np.ndarray,
    relevant_counts: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fail closed unless labels preserve P/N/U as int8 values -1/0/+1."""

    states = np.asarray(labels)
    counts = np.asarray(relevant_counts)
    if states.dtype != np.dtype(np.int8):
        raise ValueError("Tri-state candidate judgments must use int8")
    if states.ndim != 2 or (expected_shape is not None and states.shape != expected_shape):
        raise ValueError("Tri-state candidate-judgment shape changed")
    if counts.dtype != np.dtype(np.int32) or counts.shape != (states.shape[0],):
        raise ValueError("Relevant counts must use int32 with one value per query")
    if np.any(counts <= 0):
        raise ValueError("Every Phase-0 query must have at least one judged positive")
    if np.any((states != POSITIVE) & (states != EXPLICIT_NEGATIVE) & (states != UNJUDGED)):
        raise ValueError("Tri-state judgments contain a value outside {-1, 0, +1}")
    if np.any(np.sum(states == POSITIVE, axis=1) > counts):
        raise ValueError("Candidate positives exceed corpus-level relevant counts")
    return states, counts


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float | int]:
    """Return a deterministic two-sided Wilson score interval."""

    if isinstance(successes, (bool, np.bool_)) or isinstance(total, (bool, np.bool_)):
        raise ValueError("Wilson counts must be integers")
    if not isinstance(successes, (int, np.integer)) or not isinstance(total, (int, np.integer)):
        raise ValueError("Wilson counts must be integers")
    if total <= 0 or successes < 0 or successes > total or not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid Wilson interval inputs")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "count": int(successes),
        "total": int(total),
        "fraction": float(proportion),
        "lower": float(max(0.0, center - half)),
        "upper": float(min(1.0, center + half)),
    }


def _coverage(mask: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(mask)
    if values.dtype != np.bool_ or values.ndim != 1 or not len(values):
        raise ValueError("Coverage requires a non-empty boolean query mask")
    return wilson_interval(int(np.sum(values)), len(values))


def _state_counts(labels: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    selected = labels[mask]
    return {
        "positive": int(np.sum(selected == POSITIVE)),
        "explicit_negative": int(np.sum(selected == EXPLICIT_NEGATIVE)),
        "unjudged": int(np.sum(selected == UNJUDGED)),
        "total": int(selected.size),
    }


def label_support_diagnostics(
    base_scores: np.ndarray,
    document_ids: np.ndarray,
    labels: np.ndarray,
    relevant_counts: np.ndarray,
    *,
    final_k: int,
    correction_depth: int,
) -> LabelSupportResult:
    """Measure protect/promote/penalize support on every eligible query.

    Frozen candidate columns 1..``correction_depth`` are the only actionable
    candidates.  The Base Top-k itself is recomputed with the registered stable
    tie-break, so tied scores cannot silently alter the role definitions.
    """

    scores = np.asarray(base_scores)
    docids = np.asarray(document_ids)
    states, counts = validate_tristate_labels(
        labels, relevant_counts, expected_shape=scores.shape
    )
    if scores.ndim != 2 or docids.shape != scores.shape:
        raise ValueError("Base scores and document IDs must be matching matrices")
    if not np.all(np.isfinite(scores)) or not np.issubdtype(docids.dtype, np.integer):
        raise ValueError("Base ranking inputs must be finite scores and integer IDs")
    if not 0 < final_k < correction_depth <= scores.shape[1]:
        raise ValueError("Invalid Top-k or correction depth")
    base_topk = topk_membership(scores, docids, k=final_k)
    actionable = np.zeros(scores.shape, dtype=bool)
    actionable[:, :correction_depth] = True
    outside_topk = ~base_topk
    tail = np.zeros(scores.shape, dtype=bool)
    tail[:, correction_depth:] = True

    positive = states == POSITIVE
    negative = states == EXPLICIT_NEGATIVE
    positive_in = positive & base_topk
    positive_out_actionable = positive & outside_topk & actionable
    positive_out_tail = positive & outside_topk & tail
    negative_in = negative & base_topk
    negative_out_actionable = negative & outside_topk & actionable

    positive_in_count = np.sum(positive_in, axis=1)
    positive_out_actionable_count = np.sum(positive_out_actionable, axis=1)
    negative_in_count = np.sum(negative_in, axis=1)
    negative_out_actionable_count = np.sum(negative_out_actionable, axis=1)
    negative_100_count = np.sum(negative, axis=1)
    promotion_pair = (positive_out_actionable_count > 0) & (negative_in_count > 0)
    protection_pair = (positive_in_count > 0) & (negative_out_actionable_count > 0)
    label_swap_count = np.minimum(positive_out_actionable_count, negative_in_count)
    label_swap_gain = label_swap_count.astype(np.float64) / counts.astype(np.float64)

    top_band = base_topk
    middle_band = actionable & outside_topk
    summary = {
        "query_count": int(len(scores)),
        "candidate_count": int(scores.shape[1]),
        "final_k": int(final_k),
        "correction_depth": int(correction_depth),
        "candidate_judged_fraction": float(np.mean(states != UNJUDGED)),
        "state_counts": {
            "base_topk": _state_counts(states, top_band),
            "base_ranks_11_to_correction_depth": _state_counts(states, middle_band),
            "base_ranks_after_correction_depth": _state_counts(states, tail),
            "candidate_100": _state_counts(states, np.ones(states.shape, dtype=bool)),
        },
        "coverage": {
            "explicit_negative_top100": _coverage(negative_100_count > 0),
            "penalty_topk_explicit_negative": _coverage(negative_in_count > 0),
            "promotion_pair": _coverage(promotion_pair),
            "protection_pair": _coverage(protection_pair),
            "outside_positive_actionable": _coverage(
                positive_out_actionable_count > 0
            ),
            "outside_positive_tail": _coverage(np.sum(positive_out_tail, axis=1) > 0),
            "nonredundant_unary_penalty": _coverage(promotion_pair),
        },
        "per_query_count_distribution": {
            "positive_in_topk_mean": float(np.mean(positive_in_count)),
            "positive_out_actionable_mean": float(
                np.mean(positive_out_actionable_count)
            ),
            "explicit_negative_in_topk_mean": float(np.mean(negative_in_count)),
            "explicit_negative_out_actionable_mean": float(
                np.mean(negative_out_actionable_count)
            ),
            "explicit_negative_top100_mean": float(np.mean(negative_100_count)),
            "label_swap_count_mean": float(np.mean(label_swap_count)),
        },
        "label_swap_ceiling": {
            "mean_recall_gain": float(np.mean(label_swap_gain)),
            "positive_query_count": int(np.sum(label_swap_gain > 0)),
            "positive_query_fraction": float(np.mean(label_swap_gain > 0)),
            **gain_diagnostics(label_swap_gain),
        },
    }
    return LabelSupportResult(
        base_topk_membership=base_topk,
        positive_in=positive_in,
        positive_out_actionable=positive_out_actionable,
        explicit_negative_in=negative_in,
        explicit_negative_out_actionable=negative_out_actionable,
        label_swap_gain=label_swap_gain,
        promotion_pair_queries=promotion_pair,
        protection_pair_queries=protection_pair,
        summary=summary,
    )


def _strictly_ranks_above(
    left_score: float,
    left_docid: int,
    right_score: float,
    right_docid: int,
) -> bool:
    return left_score > right_score or (
        left_score == right_score and left_docid < right_docid
    )


def action_reachability_diagnostics(
    tier_scores: np.ndarray,
    document_ids: np.ndarray,
    support: LabelSupportResult,
    *,
    correction_depth: int,
) -> ActionReachabilityResult:
    """Report whether frozen actions can move supported boundary pairs."""

    scores = np.asarray(tier_scores)
    docids = np.asarray(document_ids)
    if scores.ndim != 3 or docids.shape != (scores.shape[0], scores.shape[2]):
        raise ValueError("Tier scores must be [Q,T,C] with matching document IDs")
    if not np.all(np.isfinite(scores)) or not 0 < correction_depth <= scores.shape[2]:
        raise ValueError("Invalid action-reachability inputs")
    if support.base_topk_membership.shape != docids.shape:
        raise ValueError("Label support and action scores use different candidates")

    query_count = scores.shape[0]
    downward = np.zeros(query_count, dtype=bool)
    upward = np.zeros(query_count, dtype=bool)
    joint = np.zeros(query_count, dtype=bool)
    for query_index in range(query_count):
        positives = np.flatnonzero(
            support.positive_out_actionable[query_index, :correction_depth]
        )
        negatives = np.flatnonzero(
            support.explicit_negative_in[query_index, :correction_depth]
        )
        if len(positives):
            upward[query_index] = any(
                float(np.max(scores[query_index, :, candidate]))
                > float(scores[query_index, 0, candidate])
                for candidate in positives
            )
        if len(negatives):
            downward[query_index] = any(
                float(np.min(scores[query_index, :, candidate]))
                < float(scores[query_index, 0, candidate])
                for candidate in negatives
            )
        for positive_index in positives:
            best_positive = float(np.max(scores[query_index, :, positive_index]))
            for negative_index in negatives:
                best_negative = float(np.min(scores[query_index, :, negative_index]))
                if _strictly_ranks_above(
                    best_positive,
                    int(docids[query_index, positive_index]),
                    best_negative,
                    int(docids[query_index, negative_index]),
                ):
                    joint[query_index] = True
                    break
            if joint[query_index]:
                break
    summary = {
        "query_count": int(query_count),
        "downward_feasible": _coverage(downward),
        "upward_feasible": _coverage(upward),
        "joint_swap_reachable": _coverage(joint),
        "joint_recovery_of_label_supported_queries": (
            0.0
            if not np.any(support.promotion_pair_queries)
            else float(
                np.sum(joint & support.promotion_pair_queries)
                / np.sum(support.promotion_pair_queries)
            )
        ),
    }
    return ActionReachabilityResult(downward, upward, joint, summary)


def _validate_tier_costs(
    tier_costs: Iterable[int], *, expected_count: int
) -> np.ndarray:
    values = np.asarray(tuple(tier_costs))
    if values.shape != (expected_count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Action tiers require one integer cost per score tier")
    values = values.astype(np.int64)
    if values[0] != 0 or np.any(values < 0) or np.any(np.diff(values) <= 0):
        raise ValueError("Action tier costs must start at zero and strictly increase")
    return values


def _selected_at_threshold(
    score: float,
    document_id: int,
    threshold_score: float,
    threshold_document_id: int,
) -> int:
    return int(
        score > threshold_score
        or (score == threshold_score and document_id <= threshold_document_id)
    )


def _label_utility(state: int, *, final_k: int) -> int:
    if state == int(POSITIVE):
        return final_k + 1
    if state == int(EXPLICIT_NEGATIVE):
        return -1
    return 0


def _exact_query_triage_oracle(
    tier_scores: np.ndarray,
    tier_costs: np.ndarray,
    labels: np.ndarray,
    document_ids: np.ndarray,
    *,
    final_k: int,
    correction_depth: int,
    budget: int,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Exact threshold DP for the lexicographic P-first/N-second objective."""

    scores = np.asarray(tier_scores, dtype=np.float64)
    states = np.asarray(labels)
    docids = np.asarray(document_ids)
    if scores.ndim != 2:
        raise ValueError("Per-query tier scores must have shape [T,C]")
    tier_count, candidate_count = scores.shape
    costs = _validate_tier_costs(tier_costs, expected_count=tier_count)
    if states.shape != (candidate_count,) or docids.shape != (candidate_count,):
        raise ValueError("Per-query action-oracle arrays disagree")
    if not np.all(np.isfinite(scores)) or not np.issubdtype(docids.dtype, np.integer):
        raise ValueError("Action-oracle scores and IDs must be valid")
    if not 0 < final_k < correction_depth <= candidate_count:
        raise ValueError("Invalid action-oracle Top-k or correction depth")
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("Invalid action-oracle budget")
    positive_costs = costs[costs > 0]
    quantum = int(np.gcd.reduce(positive_costs)) if len(positive_costs) else 1
    if budget % quantum or np.any(costs % quantum):
        raise ValueError("Budget and actions must share an integer cost quantum")
    cost_units = costs // quantum
    budget_units = int(budget // quantum)

    threshold_keys = {
        (float(scores[tier, candidate]), int(docids[candidate]))
        for candidate in range(correction_depth)
        for tier in range(tier_count)
    }
    threshold_keys.update(
        (float(scores[0, candidate]), int(docids[candidate]))
        for candidate in range(correction_depth, candidate_count)
    )
    negative_inf = np.int16(-30_000)
    best_utility = int(negative_inf)
    best_cost_units = budget_units + 1
    best_threshold: tuple[float, int] | None = None
    tail_scores = scores[0, correction_depth:]
    tail_docids = docids[correction_depth:]
    tail_states = states[correction_depth:]
    variable_order = np.argsort(docids[:correction_depth], kind="stable")

    for threshold_score, threshold_docid in sorted(
        threshold_keys, key=lambda value: (-value[0], value[1])
    ):
        fixed_selected = (tail_scores > threshold_score) | (
            (tail_scores == threshold_score) & (tail_docids <= threshold_docid)
        )
        fixed_count = int(np.sum(fixed_selected))
        if fixed_count > final_k:
            continue
        fixed_utility = int(
            sum(
                _label_utility(int(value), final_k=final_k)
                for value in tail_states[fixed_selected]
            )
        )
        dp = np.full((final_k + 1, budget_units + 1), negative_inf, dtype=np.int16)
        dp[fixed_count, 0] = np.int16(fixed_utility)
        for candidate in variable_order:
            updated = np.full_like(dp, negative_inf)
            for tier in range(tier_count):
                selected = _selected_at_threshold(
                    float(scores[tier, candidate]),
                    int(docids[candidate]),
                    threshold_score,
                    threshold_docid,
                )
                added = selected * _label_utility(
                    int(states[candidate]), final_k=final_k
                )
                cost = int(cost_units[tier])
                row_limit = final_k + 1 - selected
                col_limit = budget_units + 1 - cost
                if row_limit <= 0 or col_limit <= 0:
                    continue
                source = dp[:row_limit, :col_limit]
                destination = updated[
                    selected : selected + row_limit,
                    cost : cost + col_limit,
                ]
                candidate_values = np.where(
                    source > negative_inf,
                    source + np.int16(added),
                    negative_inf,
                )
                np.maximum(destination, candidate_values, out=destination)
            dp = updated
        row = dp[final_k]
        threshold_utility = int(np.max(row))
        if threshold_utility <= int(negative_inf):
            continue
        threshold_cost = int(np.flatnonzero(row == threshold_utility)[0])
        if threshold_utility > best_utility or (
            threshold_utility == best_utility and threshold_cost < best_cost_units
        ):
            best_utility = threshold_utility
            best_cost_units = threshold_cost
            best_threshold = (threshold_score, threshold_docid)
    if best_threshold is None:
        raise RuntimeError("Exact triage oracle found no feasible Top-k")

    threshold_score, threshold_docid = best_threshold
    fixed_selected = (tail_scores > threshold_score) | (
        (tail_scores == threshold_score) & (tail_docids <= threshold_docid)
    )
    fixed_count = int(np.sum(fixed_selected))
    fixed_utility = int(
        sum(
            _label_utility(int(value), final_k=final_k)
            for value in tail_states[fixed_selected]
        )
    )
    dp = np.full((final_k + 1, budget_units + 1), negative_inf, dtype=np.int16)
    dp[fixed_count, 0] = np.int16(fixed_utility)
    step_count = len(variable_order)
    previous_count = np.full(
        (step_count, final_k + 1, budget_units + 1), -1, dtype=np.int16
    )
    previous_cost = np.full_like(previous_count, -1)
    chosen_tier = np.full_like(previous_count, -1)
    for step, candidate in enumerate(variable_order):
        updated = np.full_like(dp, negative_inf)
        for tier in range(tier_count):
            selected = _selected_at_threshold(
                float(scores[tier, candidate]),
                int(docids[candidate]),
                threshold_score,
                threshold_docid,
            )
            added = selected * _label_utility(
                int(states[candidate]), final_k=final_k
            )
            cost = int(cost_units[tier])
            for old_count in range(final_k + 1 - selected):
                for old_cost in range(budget_units + 1 - cost):
                    old_value = int(dp[old_count, old_cost])
                    if old_value <= int(negative_inf):
                        continue
                    new_count = old_count + selected
                    new_cost = old_cost + cost
                    new_value = old_value + added
                    if new_value > int(updated[new_count, new_cost]):
                        updated[new_count, new_cost] = np.int16(new_value)
                        previous_count[step, new_count, new_cost] = old_count
                        previous_cost[step, new_count, new_cost] = old_cost
                        chosen_tier[step, new_count, new_cost] = tier
        dp = updated
    if int(dp[final_k, best_cost_units]) != best_utility:
        raise AssertionError("Triage-oracle backpointer pass changed the optimum")
    rates = np.zeros(correction_depth, dtype=np.int16)
    current_count = final_k
    current_cost = best_cost_units
    for step in range(step_count - 1, -1, -1):
        tier = int(chosen_tier[step, current_count, current_cost])
        if tier < 0:
            raise AssertionError("Triage-oracle backpointer is incomplete")
        candidate = int(variable_order[step])
        rates[candidate] = np.int16(costs[tier])
        current_count, current_cost = (
            int(previous_count[step, current_count, current_cost]),
            int(previous_cost[step, current_count, current_cost]),
        )
    if current_count != fixed_count or current_cost != 0:
        raise AssertionError("Triage-oracle backpointer did not reach its initial state")

    tier_by_cost = {int(value): index for index, value in enumerate(costs)}
    selected_scores = scores[0].copy()
    for candidate in range(correction_depth):
        selected_scores[candidate] = scores[
            tier_by_cost[int(rates[candidate])], candidate
        ]
    order = np.lexsort((docids, -selected_scores))
    membership = np.zeros(candidate_count, dtype=bool)
    membership[order[:final_k]] = True
    positive_hits = int(np.sum(states[membership] == POSITIVE))
    negative_hits = int(np.sum(states[membership] == EXPLICIT_NEGATIVE))
    reconstructed_utility = positive_hits * (final_k + 1) - negative_hits
    if reconstructed_utility != best_utility:
        raise AssertionError("Reconstructed triage assignment does not attain optimum")
    return (
        positive_hits,
        negative_hits,
        best_cost_units * quantum,
        rates,
        membership,
    )


def exact_triage_action_oracle(
    tier_scores: np.ndarray,
    tier_costs: Iterable[int],
    labels: np.ndarray,
    document_ids: np.ndarray,
    relevant_counts: np.ndarray,
    *,
    final_k: int,
    correction_depth: int,
    budget: int,
) -> TriageOracleResult:
    """Solve the frozen action space exactly for every query."""

    scores = np.asarray(tier_scores)
    if scores.ndim != 3 or not np.all(np.isfinite(scores)):
        raise ValueError("Tier scores must be a finite [Q,T,C] tensor")
    states, counts = validate_tristate_labels(
        labels, relevant_counts, expected_shape=(scores.shape[0], scores.shape[2])
    )
    docids = np.asarray(document_ids)
    if docids.shape != states.shape or not np.issubdtype(docids.dtype, np.integer):
        raise ValueError("Document IDs do not match tri-state labels")
    costs = _validate_tier_costs(tier_costs, expected_count=scores.shape[1])
    query_count = scores.shape[0]
    positives = np.empty(query_count, dtype=np.int32)
    negatives = np.empty(query_count, dtype=np.int32)
    used = np.empty(query_count, dtype=np.int32)
    rates = np.empty((query_count, correction_depth), dtype=np.int16)
    membership = np.zeros(states.shape, dtype=bool)
    for query_index in range(query_count):
        (
            positives[query_index],
            negatives[query_index],
            used[query_index],
            rates[query_index],
            membership[query_index],
        ) = _exact_query_triage_oracle(
            scores[query_index],
            costs,
            states[query_index],
            docids[query_index],
            final_k=final_k,
            correction_depth=correction_depth,
            budget=budget,
        )
    if np.any(used > budget):
        raise AssertionError("Internal triage-oracle budget violation")
    recall = positives.astype(np.float64) / counts.astype(np.float64)
    return TriageOracleResult(positives, negatives, recall, used, rates, membership)


def design_fold_ids(qids: Iterable[str]) -> np.ndarray:
    values = [str(value) for value in qids]
    if not values or len(values) != len(set(values)):
        raise ValueError("Fold query IDs must be non-empty and unique")
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(FOLD_SALT + value.encode("utf-8")).digest()[:8],
                "big",
            )
            % 5
            for value in values
        ],
        dtype=np.uint8,
    )


def label_swap_bootstrap(
    label_swap_gain: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int]:
    values = np.asarray(label_swap_gain, dtype=np.float64)
    return paired_bootstrap_mean_delta(
        values,
        np.zeros_like(values),
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )


def oracle_diagnostics(
    oracle: TriageOracleResult,
    comparator_scores: np.ndarray,
    document_ids: np.ndarray,
    labels: np.ndarray,
    relevant_counts: np.ndarray,
    label_swap_gain: np.ndarray,
    joint_swap_reachable: np.ndarray,
    *,
    qids: Iterable[str],
    final_k: int,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Summarize the exact oracle against one frozen comparator."""

    states, counts = validate_tristate_labels(labels, relevant_counts)
    scores = np.asarray(comparator_scores)
    docids = np.asarray(document_ids)
    if scores.shape != states.shape or docids.shape != states.shape:
        raise ValueError("Comparator arrays do not match the triage role")
    comparator_recall = recall_at_k_per_query(
        scores,
        docids,
        (states == POSITIVE).astype(np.uint8),
        counts,
        k=final_k,
    )
    delta = np.asarray(oracle.recall_at_k, dtype=np.float64) - comparator_recall
    bootstrap = paired_bootstrap_mean_delta(
        oracle.recall_at_k,
        comparator_recall,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    comparator_membership = topk_membership(scores, docids, k=final_k)
    entered_positive = np.any(
        oracle.topk_membership & ~comparator_membership & (states == POSITIVE),
        axis=1,
    )
    exited_negative = np.any(
        comparator_membership
        & ~oracle.topk_membership
        & (states == EXPLICIT_NEGATIVE),
        axis=1,
    )
    explicit_swap_event = entered_positive & exited_negative
    positive_gain = np.maximum(delta, 0.0)
    positive_mass = float(np.sum(positive_gain))
    label_values = np.asarray(label_swap_gain, dtype=np.float64)
    label_mass = float(np.sum(label_values))
    folds = design_fold_ids(qids)
    fold_gains = [
        float(np.mean(delta[folds == fold])) if np.any(folds == fold) else None
        for fold in range(5)
    ]
    if any(value is None for value in fold_gains):
        raise ValueError("Every registered design fold must be non-empty")
    return {
        "mean_comparator_recall_at_10": float(np.mean(comparator_recall)),
        "mean_oracle_recall_at_10": float(np.mean(oracle.recall_at_k)),
        "mean_oracle_gain": float(np.mean(delta)),
        "gain_diagnostics": gain_diagnostics(delta),
        "bootstrap": bootstrap,
        "label_ceiling_recovery": (
            0.0
            if label_mass <= 0
            else float(np.sum(np.minimum(positive_gain, label_values)) / label_mass)
        ),
        "explicit_swap_attribution": (
            0.0
            if positive_mass <= 0
            else float(np.sum(positive_gain[explicit_swap_event]) / positive_mass)
        ),
        "explicit_swap_event_queries": int(np.sum(explicit_swap_event)),
        "joint_swap_reachable": _coverage(np.asarray(joint_swap_reachable, dtype=bool)),
        "action_cost": {
            "mean": float(np.mean(oracle.action_cost)),
            "p95": float(np.quantile(oracle.action_cost, 0.95, method="linear")),
            "maximum": int(np.max(oracle.action_cost)),
        },
        "design_fold_gains": [float(value) for value in fold_gains],
    }


def pre_action_decision(
    *,
    role_id: str,
    explicit_negative_semantics_preserved: bool,
    support_summary: dict[str, Any],
    label_bootstrap: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply schema/support gates before running an expensive action oracle."""

    if role_id not in {DESIGN_ROLE_ID, AUDIT_ROLE_ID}:
        raise ValueError("Unsupported v4 Phase-0 role")
    query_count = int(support_summary["query_count"])
    minimum_queries = int(
        thresholds[
            "minimum_design_queries"
            if role_id == DESIGN_ROLE_ID
            else "minimum_audit_queries"
        ]
    )
    if query_count < minimum_queries:
        return {
            "decision": "STOP_UNDERPOWERED",
            "action_oracle_authorized": False,
            "checks": {"minimum_query_count": False},
        }
    if not explicit_negative_semantics_preserved:
        return {
            "decision": "STOP_NO_EXPLICIT_NEGATIVE_SEMANTICS",
            "action_oracle_authorized": False,
            "checks": {
                "minimum_query_count": True,
                "explicit_negative_semantics_preserved": False,
            },
        }
    coverage = support_summary["coverage"]
    negative_checks = {
        "explicit_negative_top100_fraction": coverage["explicit_negative_top100"][
            "fraction"
        ]
        >= float(thresholds["minimum_explicit_negative_top100_coverage"]),
        "explicit_negative_top100_wilson": coverage["explicit_negative_top100"][
            "lower"
        ]
        >= float(thresholds["minimum_explicit_negative_top100_wilson_lower"]),
        "penalty_fraction": coverage["penalty_topk_explicit_negative"]["fraction"]
        >= float(thresholds["minimum_penalty_query_coverage"]),
        "penalty_wilson": coverage["penalty_topk_explicit_negative"]["lower"]
        >= float(thresholds["minimum_penalty_query_wilson_lower"]),
    }
    if role_id == AUDIT_ROLE_ID:
        negative_checks.update(
            {
                "audit_explicit_negative_count": coverage[
                    "explicit_negative_top100"
                ]["count"]
                >= int(thresholds["audit_minimum_explicit_negative_top100_queries"]),
                "audit_penalty_count": coverage["penalty_topk_explicit_negative"][
                    "count"
                ]
                >= int(thresholds["audit_minimum_penalty_queries"]),
            }
        )
    if not all(negative_checks.values()):
        return {
            "decision": "STOP_NEGATIVE_SUPPORT",
            "action_oracle_authorized": False,
            "checks": {"minimum_query_count": True, **negative_checks},
        }

    nonredundant_minimum = max(
        int(thresholds["minimum_nonredundant_actionable_queries"]),
        int(
            math.ceil(
                float(thresholds["minimum_nonredundant_actionable_fraction"])
                * query_count
            )
        ),
    )
    nonredundant_count = int(
        coverage["nonredundant_unary_penalty"]["count"]
    )
    if nonredundant_count < nonredundant_minimum:
        return {
            "decision": "STOP_NO_NOVEL_ACTION_SUPPORT",
            "action_oracle_authorized": False,
            "checks": {
                "minimum_query_count": True,
                **negative_checks,
                "nonredundant_actionable_support": False,
            },
            "minimum_nonredundant_actionable_queries_effective": nonredundant_minimum,
        }

    swap_checks = {
        "promotion_fraction": coverage["promotion_pair"]["fraction"]
        >= float(thresholds["minimum_promotion_pair_coverage"]),
        "promotion_wilson": coverage["promotion_pair"]["lower"]
        >= float(thresholds["minimum_promotion_pair_wilson_lower"]),
        "protection_fraction": coverage["protection_pair"]["fraction"]
        >= float(thresholds["minimum_protection_pair_coverage"]),
        "protection_wilson": coverage["protection_pair"]["lower"]
        >= float(thresholds["minimum_protection_pair_wilson_lower"]),
        "label_swap_mean": support_summary["label_swap_ceiling"]["mean_recall_gain"]
        >= float(thresholds["minimum_mean_label_swap_ceiling"]),
        "label_swap_bootstrap": float(label_bootstrap["lower"])
        > float(thresholds["minimum_label_swap_bootstrap_lower"]),
        "label_swap_support": support_summary["label_swap_ceiling"][
            "positive_query_fraction"
        ]
        >= float(thresholds["minimum_positive_support_fraction"]),
        "label_swap_effective_support": support_summary["label_swap_ceiling"][
            "effective_positive_support"
        ]
        >= float(thresholds["minimum_effective_positive_support"]),
        "label_swap_concentration": support_summary["label_swap_ceiling"][
            "top_1pct_positive_gain_concentration"
        ]
        <= float(thresholds["maximum_top_1pct_positive_mass_concentration"]),
    }
    if role_id == AUDIT_ROLE_ID:
        swap_checks.update(
            {
                "audit_promotion_count": coverage["promotion_pair"]["count"]
                >= int(thresholds["audit_minimum_promotion_queries"]),
                "audit_protection_count": coverage["protection_pair"]["count"]
                >= int(thresholds["audit_minimum_protection_queries"]),
            }
        )
    decision = "ACTION_ORACLE_AUTHORIZED" if all(swap_checks.values()) else "STOP_SWAP_SUPPORT"
    return {
        "decision": decision,
        "action_oracle_authorized": all(swap_checks.values()),
        "checks": {
            "minimum_query_count": True,
            **negative_checks,
            "nonredundant_actionable_support": True,
            **swap_checks,
        },
        "minimum_nonredundant_actionable_queries_effective": nonredundant_minimum,
    }


def final_action_decision(
    *,
    role_id: str,
    progressive_diagnostics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply the registered compression-consistent action-space gate."""

    gains = progressive_diagnostics["gain_diagnostics"]
    minimum_support = max(
        int(thresholds["minimum_positive_support_queries"]),
        int(
            math.ceil(
                float(thresholds["minimum_positive_support_fraction"])
                * int(gains["improved"] + gains["harmed"] + gains["unchanged"])
            )
        ),
    )
    folds = np.asarray(progressive_diagnostics["design_fold_gains"], dtype=np.float64)
    checks = {
        "progressive_gain": float(progressive_diagnostics["mean_oracle_gain"])
        >= float(thresholds["minimum_progressive_oracle_gain"]),
        "progressive_bootstrap": float(
            progressive_diagnostics["bootstrap"]["lower"]
        )
        > float(thresholds["minimum_progressive_bootstrap_lower"]),
        "positive_support": int(gains["improved"]) >= minimum_support,
        "effective_support": float(gains["effective_positive_support"])
        >= float(thresholds["minimum_effective_positive_support"]),
        "gain_concentration": float(gains["top_1pct_positive_gain_concentration"])
        <= float(thresholds["maximum_top_1pct_positive_mass_concentration"]),
        "label_ceiling_recovery": float(
            progressive_diagnostics["label_ceiling_recovery"]
        )
        >= float(thresholds["minimum_label_ceiling_recovery"]),
        "explicit_swap_attribution": float(
            progressive_diagnostics["explicit_swap_attribution"]
        )
        >= float(thresholds["minimum_explicit_swap_attribution"]),
        "joint_swap_reachability": float(
            progressive_diagnostics["joint_swap_reachable"]["fraction"]
        )
        >= float(thresholds["minimum_joint_swap_reachable_coverage"]),
        "maximum_accessed_bytes": int(
            progressive_diagnostics["action_cost"]["maximum"]
        )
        <= int(thresholds["maximum_accessed_bytes_per_query"]),
        "positive_design_folds": int(np.sum(folds > 0))
        >= int(thresholds["minimum_positive_design_folds"]),
        "worst_design_fold": float(np.min(folds))
        >= float(thresholds["minimum_worst_design_fold_gain"]),
    }
    if role_id == AUDIT_ROLE_ID:
        checks["audit_joint_reachable_count"] = int(
            progressive_diagnostics["joint_swap_reachable"]["count"]
        ) >= int(thresholds["audit_minimum_joint_reachable_queries"])
    if all(checks.values()):
        decision = (
            "DESIGN_GO_TO_DIAGNOSTIC_AUDIT"
            if role_id == DESIGN_ROLE_ID
            else "GO_FREEZE_FP32_DEVELOPMENT_PROTOCOL"
        )
    else:
        decision = "STOP_NO_COMPRESSION_CONSISTENT_HEADROOM"
    return {
        "decision": decision,
        "all_required_checks_passed": all(checks.values()),
        "checks": checks,
        "minimum_positive_support_queries_effective": minimum_support,
    }
