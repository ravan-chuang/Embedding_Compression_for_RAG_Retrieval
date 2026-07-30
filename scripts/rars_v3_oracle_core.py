#!/usr/bin/env python3
"""Deterministic NumPy contracts for the RARS-v3 oracle-first gate.

The module deliberately contains no training loop, Torch dependency, or
external-test access.  It owns the development split, progressive residual
tiers, exact candidate rescoring diagnostics, the exact matched-access
query-label oracle, and the preregistered go/no-go statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "rars_v3_oracle_first_feasibility_v1"
DESIGN_ROLE_ID = "oracle_design"
AUDIT_ROLE_ID = "oracle_audit"
FUTURE_ROLE_ID = "future_method_holdout"
SPLIT_SALT = b"rars_v3_split_v1\0"
FOLD_SALT = b"rars_v3_fold_v1\0"
ALLOWED_TIERS = (0, 8, 16, 32)
FORBIDDEN_ROLE_MARKERS = (
    "inner_validation",
    "outer",
    "clean_test",
    "evaluation",
    "posthoc",
    "sealed",
    "nq",
    "trec",
)


@dataclass(frozen=True)
class AccessOracleResult:
    """Exact per-query optimum under a document-code byte budget."""

    hits_at_k: np.ndarray
    recall_at_k: np.ndarray
    accessed_bytes: np.ndarray
    rate_assignments: np.ndarray
    topk_membership: np.ndarray


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _split_bucket(qid: str) -> int:
    digest = hashlib.sha256(SPLIT_SALT + str(qid).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % 10


def split_development_qids(
    qids: Iterable[str],
) -> dict[str, np.ndarray]:
    """Freeze the 60/20/20 v3 split inside the old 3,961-query train role."""

    values = [str(value) for value in qids]
    if not values or len(values) != len(set(values)):
        raise ValueError("Parent development qids must be non-empty and unique")
    buckets = np.asarray([_split_bucket(value) for value in values], dtype=np.uint8)
    roles = {
        DESIGN_ROLE_ID: np.flatnonzero(buckets <= 5).astype(np.int64),
        AUDIT_ROLE_ID: np.flatnonzero((buckets >= 6) & (buckets <= 7)).astype(
            np.int64
        ),
        FUTURE_ROLE_ID: np.flatnonzero(buckets >= 8).astype(np.int64),
    }
    if any(not len(indices) for indices in roles.values()):
        raise ValueError("A v3 development role is empty")
    combined = np.sort(np.concatenate(list(roles.values())))
    if not np.array_equal(combined, np.arange(len(values), dtype=np.int64)):
        raise AssertionError("Internal v3 split coverage failure")
    return roles


def design_fold_ids(qids: Iterable[str]) -> np.ndarray:
    """Return fixed five-fold diagnostic IDs without changing primary roles."""

    values = [str(value) for value in qids]
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(FOLD_SALT + value.encode("utf-8")).digest()[:8],
                "big",
                signed=False,
            )
            % 5
            for value in values
        ],
        dtype=np.uint8,
    )


def validate_bundle_manifest(
    manifest: dict[str, Any], *, expected_role_id: str
) -> None:
    """Fail closed unless a bundle is one of the two permitted v3 roles."""

    if expected_role_id not in {DESIGN_ROLE_ID, AUDIT_ROLE_ID}:
        raise ValueError(f"Unsupported v3 role request: {expected_role_id!r}")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Bundle does not use the frozen v3 protocol")
    role_id = str(manifest.get("role_id", "")).casefold()
    if role_id != expected_role_id:
        raise ValueError(
            f"Expected role_id={expected_role_id!r}, found {role_id!r}"
        )
    marker = next(
        (value for value in FORBIDDEN_ROLE_MARKERS if value in role_id), None
    )
    if marker is not None:
        raise ValueError(f"V3 oracle evaluator forbids role marker {marker!r}")
    if manifest.get("evidence_status") != "DEVELOPMENT_ONLY":
        raise ValueError("V3 bundle must remain DEVELOPMENT_ONLY")
    source_commit = str(manifest.get("source_commit", ""))
    if len(source_commit) != 40 or any(
        value not in "0123456789abcdef" for value in source_commit
    ):
        raise ValueError("Bundle manifest lacks an exact source_commit")
    for field in (
        "query_ids_sha256",
        "query_rows_sha256",
        "split_audit_sha256",
        "builder_sha256",
        "protocol_sha256",
        "parent_v2_2_manifest_sha256",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Bundle manifest lacks a valid {field}")
    access = manifest.get("data_access")
    if not isinstance(access, dict):
        raise ValueError("Bundle manifest lacks explicit data_access")
    required_false = (
        "v2_2_inner_validation_values_used",
        "outer_relevance_values_used",
        "clean_test_relevance_values_used",
        "nq_relevance_values_used",
        "trec_relevance_values_used",
        "future_method_holdout_relevance_values_used",
    )
    for field in required_false:
        if access.get(field) is not False:
            raise ValueError(f"Forbidden data-access flag is not false: {field}")


def _validate_document_ids(
    document_ids: np.ndarray, *, expected_shape: tuple[int, ...]
) -> np.ndarray:
    """Return integer document IDs after enforcing per-query uniqueness."""

    docids = np.asarray(document_ids)
    if docids.shape != expected_shape:
        raise ValueError("document_ids do not match the score shape")
    if not np.issubdtype(docids.dtype, np.integer):
        raise ValueError("document_ids must use an integer dtype")
    rows = docids.reshape((-1, docids.shape[-1]))
    if any(np.unique(row).size != row.size for row in rows):
        raise ValueError("document_ids must be unique within every query")
    return docids


def _validate_tier_costs(
    tier_bytes: Iterable[int], *, expected_count: int | None = None
) -> np.ndarray:
    """Return the only admissible progressive cost schedule: 0 < ... ."""

    raw = tuple(tier_bytes)
    if not raw or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError("Tier byte costs must be a non-empty integer sequence")
    costs = np.asarray(raw, dtype=np.int64)
    if expected_count is not None and costs.shape != (expected_count,):
        raise ValueError("tier_bytes do not match tier score count")
    if costs[0] != 0 or np.any(np.diff(costs) <= 0):
        raise ValueError(
            "Tier byte costs must be unique, strictly increasing, and start at zero"
        )
    if costs[-1] > np.iinfo(np.int16).max:
        raise ValueError("Tier byte costs exceed the auditable assignment dtype")
    return costs


def stable_topk(
    scores: np.ndarray, document_ids: np.ndarray, k: int
) -> np.ndarray:
    """Top-k by score descending and document ID ascending."""

    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError("scores and document_ids must be matching [Q, C] arrays")
    docids = _validate_document_ids(document_ids, expected_shape=values.shape)
    if not 0 < k < values.shape[1]:
        raise ValueError("Require 0 < k < candidate count")
    if not np.all(np.isfinite(values)):
        raise ValueError("Scores must be finite")
    order = np.empty(values.shape, dtype=np.int64)
    for query_index in range(len(values)):
        order[query_index] = np.lexsort(
            (docids[query_index], -values[query_index])
        )
    return order[:, :k]


def recall_at_k_per_query(
    scores: np.ndarray,
    document_ids: np.ndarray,
    labels: np.ndarray,
    relevant_counts: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    relevance = np.asarray(labels)
    counts = np.asarray(relevant_counts)
    if relevance.shape != np.asarray(scores).shape:
        raise ValueError("labels and scores must have matching shapes")
    if counts.shape != (len(relevance),) or np.any(counts <= 0):
        raise ValueError("relevant_counts must be positive with one value per query")
    top = stable_topk(scores, document_ids, k)
    hits = np.take_along_axis(relevance, top, axis=1).sum(axis=1)
    return hits.astype(np.float64) / counts.astype(np.float64)


def topk_membership(
    scores: np.ndarray, document_ids: np.ndarray, *, k: int
) -> np.ndarray:
    top = stable_topk(scores, document_ids, k)
    membership = np.zeros(np.asarray(scores).shape, dtype=bool)
    np.put_along_axis(membership, top, True, axis=1)
    return membership


def orient_basis_deterministically(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float32).copy()
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            result[:, column_index] *= -1.0
    return result


def fit_progressive_pca(
    residuals: np.ndarray,
    *,
    rank: int,
    max_samples: int,
    seed: int,
    scale_batch_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit an uncentered nested SVD basis, scales, rows, and full spectrum."""

    values = residuals
    if values.ndim != 2 or not 0 < rank <= values.shape[1]:
        raise ValueError("Invalid residual matrix or progressive rank")
    if max_samples <= 0 or scale_batch_size <= 0:
        raise ValueError("Sample and batch sizes must be positive")
    rng = np.random.default_rng(seed)
    take = min(max_samples, len(values))
    sample_rows = np.sort(
        rng.choice(len(values), size=take, replace=False).astype(np.int64)
    )
    sample = np.asarray(values[sample_rows], dtype=np.float32)
    if not np.all(np.isfinite(sample)):
        raise ValueError("PCA sample contains non-finite values")
    _, singular_values, vh = np.linalg.svd(sample, full_matrices=False)
    basis = orient_basis_deterministically(vh[:rank].T)
    max_abs = np.zeros(rank, dtype=np.float32)
    for start in range(0, len(values), scale_batch_size):
        end = min(start + scale_batch_size, len(values))
        coefficients = np.asarray(values[start:end], dtype=np.float32) @ basis
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Residual coefficients contain non-finite values")
        max_abs = np.maximum(max_abs, np.max(np.abs(coefficients), axis=0))
    scales = np.maximum(max_abs, np.float32(1e-12)) / np.float32(127.0)
    return (
        basis.astype(np.float32),
        scales.astype(np.float32),
        sample_rows,
        singular_values.astype(np.float64),
    )


def progressive_tier_scores(
    queries: np.ndarray,
    ann_scores: np.ndarray,
    residual_lookup: np.ndarray,
    residuals: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    *,
    tiers: tuple[int, ...] = ALLOWED_TIERS,
    alpha: float,
    top_b: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Return [Q, T, C] scores from prefixes of one nested int8 code."""

    query_values = np.asarray(queries)
    base = np.asarray(ann_scores)
    lookup = np.asarray(residual_lookup)
    residual_values = np.asarray(residuals)
    basis_values = np.asarray(basis, dtype=np.float32)
    scale_values = np.asarray(scales, dtype=np.float32)
    if query_values.ndim != 2 or base.ndim != 2 or lookup.shape != base.shape:
        raise ValueError("Invalid query, score, or lookup shapes")
    if base.shape[0] != query_values.shape[0]:
        raise ValueError("Query and candidate counts disagree")
    if (
        residual_values.ndim != 2
        or residual_values.shape[1] != query_values.shape[1]
    ):
        raise ValueError("Residuals do not match the query dimension")
    if basis_values.shape != (query_values.shape[1], len(scale_values)):
        raise ValueError("Basis/scales do not match query dimension")
    if (
        not tiers
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in tiers
        )
        or tuple(sorted(set(tiers))) != tuple(tiers)
        or tiers[0] != 0
    ):
        raise ValueError("Tiers must be unique, increasing, and start at zero")
    if any(value < 0 or value > basis_values.shape[1] for value in tiers):
        raise ValueError("A tier exceeds the progressive basis rank")
    if not 0 < top_b <= base.shape[1] or batch_size <= 0:
        raise ValueError("Invalid correction depth or batch size")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    selected_lookup = lookup[:, :top_b]
    if not np.issubdtype(selected_lookup.dtype, np.integer):
        raise ValueError("Residual lookup must use an integer dtype")
    if np.any(selected_lookup < 0) or np.any(
        selected_lookup >= len(residual_values)
    ):
        raise ValueError("Correctable candidates have out-of-bounds residual rows")
    if not (
        np.all(np.isfinite(query_values))
        and np.all(np.isfinite(base))
        and np.all(np.isfinite(residual_values[selected_lookup]))
        and np.all(np.isfinite(basis_values))
        and np.all(np.isfinite(scale_values))
        and np.all(scale_values > 0)
    ):
        raise ValueError("Progressive score inputs must be finite with positive scales")

    result = np.repeat(
        np.asarray(base, dtype=np.float32)[:, None, :], len(tiers), axis=1
    )
    for start in range(0, len(base), batch_size):
        end = min(start + batch_size, len(base))
        q_projection = (
            np.asarray(query_values[start:end], dtype=np.float32) @ basis_values
        )
        selected = np.asarray(lookup[start:end, :top_b], dtype=np.int64)
        coefficients = np.einsum(
            "qcd,dr->qcr",
            np.asarray(residual_values[selected], dtype=np.float32),
            basis_values,
        )
        codes = np.clip(
            np.rint(coefficients / scale_values[None, None, :]), -127, 127
        ).astype(np.int8)
        dequantized = codes.astype(np.float32) * scale_values[None, None, :]
        component_scores = q_projection[:, None, :] * dequantized
        prefix = np.cumsum(component_scores, axis=2, dtype=np.float32)
        for tier_index, tier in enumerate(tiers):
            if tier:
                result[start:end, tier_index, :top_b] += (
                    np.float32(alpha) * prefix[:, :, tier - 1]
                )
    return result


def exact_residual_scores(
    queries: np.ndarray,
    ann_scores: np.ndarray,
    residual_lookup: np.ndarray,
    residuals: np.ndarray,
    *,
    top_b: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Replace compressed candidate scores with full residual correction."""

    query_values = np.asarray(queries)
    result = np.asarray(ann_scores, dtype=np.float32).copy()
    lookup = np.asarray(residual_lookup)
    residual_values = np.asarray(residuals)
    if query_values.ndim != 2 or result.ndim != 2 or lookup.shape != result.shape:
        raise ValueError("Invalid exact-rescore inputs")
    if result.shape[0] != query_values.shape[0]:
        raise ValueError("Query and candidate counts disagree")
    if (
        residual_values.ndim != 2
        or residual_values.shape[1] != query_values.shape[1]
    ):
        raise ValueError("Residuals do not match the query dimension")
    if not 0 < top_b <= result.shape[1]:
        raise ValueError("Invalid exact correction depth")
    selected_lookup = lookup[:, :top_b]
    if not np.issubdtype(selected_lookup.dtype, np.integer):
        raise ValueError("Residual lookup must use an integer dtype")
    if np.any(selected_lookup < 0) or np.any(
        selected_lookup >= len(residual_values)
    ):
        raise ValueError("Exact-correctable candidates have out-of-bounds residual rows")
    if not (
        np.all(np.isfinite(query_values))
        and np.all(np.isfinite(result))
        and np.all(np.isfinite(residual_values[selected_lookup]))
    ):
        raise ValueError("Exact-rescore inputs must be finite")
    for start in range(0, len(result), batch_size):
        end = min(start + batch_size, len(result))
        selected = np.asarray(lookup[start:end, :top_b], dtype=np.int64)
        correction = np.einsum(
            "qd,qcd->qc",
            np.asarray(query_values[start:end], dtype=np.float32),
            np.asarray(residual_values[selected], dtype=np.float32),
        )
        result[start:end, :top_b] += correction.astype(np.float32)
    return result


def _key_at_least(
    score: float,
    document_id: int,
    threshold_score: float,
    threshold_document_id: int,
) -> int:
    return int(
        score > threshold_score
        or (score == threshold_score and document_id <= threshold_document_id)
    )


def _exact_query_access_oracle(
    tier_scores: np.ndarray,
    tier_bytes: np.ndarray,
    labels: np.ndarray,
    document_ids: np.ndarray,
    *,
    final_k: int,
    top_b: int,
    budget_bytes: int,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Exact DP over stable score thresholds, Top-k count, and byte state."""

    scores = np.asarray(tier_scores, dtype=np.float64)
    relevance = np.asarray(labels)
    if scores.ndim != 2:
        raise ValueError("Per-query tier_scores must have shape [T, C]")
    tier_count, candidate_count = scores.shape
    costs = _validate_tier_costs(tier_bytes, expected_count=tier_count)
    docids = _validate_document_ids(
        document_ids, expected_shape=(candidate_count,)
    )
    if relevance.shape != (candidate_count,):
        raise ValueError("Per-query oracle arrays disagree")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(relevance)):
        raise ValueError("Per-query oracle scores and labels must be finite")
    if not 0 < final_k < candidate_count or not final_k < top_b <= candidate_count:
        raise ValueError("Invalid Top-k or correction depth")
    if (
        isinstance(budget_bytes, (bool, np.bool_))
        or not isinstance(budget_bytes, (int, np.integer))
        or budget_bytes < 0
    ):
        raise ValueError("Invalid accessed-byte budget")
    positive_costs = costs[costs > 0]
    quantum = int(np.gcd.reduce(positive_costs)) if len(positive_costs) else 1
    if budget_bytes % quantum or np.any(costs % quantum):
        raise ValueError("Budget and tiers must share an integer byte quantum")
    cost_units = costs // quantum
    budget_units = budget_bytes // quantum

    threshold_keys = {
        (float(scores[tier, candidate]), int(docids[candidate]))
        for candidate in range(top_b)
        for tier in range(tier_count)
    }
    threshold_keys.update(
        (float(scores[0, candidate]), int(docids[candidate]))
        for candidate in range(top_b, candidate_count)
    )
    best_hits = -1
    best_cost_units = budget_units + 1
    best_threshold: tuple[float, int] | None = None
    negative = np.int16(-30_000)
    tail_scores = scores[0, top_b:]
    tail_docids = docids[top_b:]
    tail_labels = relevance[top_b:]

    variable_order = np.argsort(docids[:top_b], kind="stable")
    for threshold_score, threshold_docid in sorted(
        threshold_keys, key=lambda value: (-value[0], value[1])
    ):
        fixed_selected = (
            (tail_scores > threshold_score)
            | ((tail_scores == threshold_score) & (tail_docids <= threshold_docid))
        )
        fixed_count = int(np.sum(fixed_selected))
        if fixed_count > final_k:
            continue
        fixed_hits = int(np.sum(tail_labels[fixed_selected] > 0))
        dp = np.full(
            (final_k + 1, budget_units + 1), negative, dtype=np.int16
        )
        dp[fixed_count, 0] = np.int16(fixed_hits)
        for candidate in variable_order:
            updated = np.full_like(dp, negative)
            for tier in range(tier_count):
                selected = _key_at_least(
                    float(scores[tier, candidate]),
                    int(docids[candidate]),
                    threshold_score,
                    threshold_docid,
                )
                add_hit = int(selected and relevance[candidate] > 0)
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
                valid = source > negative
                candidate_values = np.where(
                    valid, source + np.int16(add_hit), negative
                )
                np.maximum(destination, candidate_values, out=destination)
            dp = updated
        row = dp[final_k]
        threshold_best = int(np.max(row))
        if threshold_best < 0:
            continue
        threshold_cost = int(np.flatnonzero(row == threshold_best)[0])
        if threshold_best > best_hits or (
            threshold_best == best_hits and threshold_cost < best_cost_units
        ):
            best_hits = threshold_best
            best_cost_units = threshold_cost
            best_threshold = (threshold_score, threshold_docid)
    if best_hits < 0:
        raise RuntimeError("Exact accessed-byte oracle found no feasible Top-k")
    if best_threshold is None:
        raise AssertionError("Internal oracle threshold selection failure")

    # Re-run the chosen threshold with backpointers so the exact rate vector and
    # Top-k membership can be audited.  Tiers and candidates are visited in
    # ascending order, making ties deterministic without changing the optimum.
    threshold_score, threshold_docid = best_threshold
    fixed_selected = (
        (tail_scores > threshold_score)
        | ((tail_scores == threshold_score) & (tail_docids <= threshold_docid))
    )
    fixed_count = int(np.sum(fixed_selected))
    fixed_hits = int(np.sum(tail_labels[fixed_selected] > 0))
    dp = np.full((final_k + 1, budget_units + 1), negative, dtype=np.int16)
    dp[fixed_count, 0] = np.int16(fixed_hits)
    steps = len(variable_order)
    previous_count = np.full(
        (steps, final_k + 1, budget_units + 1), -1, dtype=np.int16
    )
    previous_cost = np.full_like(previous_count, -1)
    chosen_tier = np.full_like(previous_count, -1)
    for step, candidate in enumerate(variable_order):
        updated = np.full_like(dp, negative)
        for tier in range(tier_count):
            selected = _key_at_least(
                float(scores[tier, candidate]),
                int(docids[candidate]),
                threshold_score,
                threshold_docid,
            )
            add_hit = int(selected and relevance[candidate] > 0)
            cost = int(cost_units[tier])
            for old_count in range(final_k + 1 - selected):
                for old_cost in range(budget_units + 1 - cost):
                    old_value = int(dp[old_count, old_cost])
                    if old_value <= int(negative):
                        continue
                    new_count = old_count + selected
                    new_cost = old_cost + cost
                    new_value = old_value + add_hit
                    if new_value > int(updated[new_count, new_cost]):
                        updated[new_count, new_cost] = np.int16(new_value)
                        previous_count[step, new_count, new_cost] = old_count
                        previous_cost[step, new_count, new_cost] = old_cost
                        chosen_tier[step, new_count, new_cost] = tier
        dp = updated
    if int(dp[final_k, best_cost_units]) != best_hits:
        raise AssertionError("Oracle backpointer pass changed the optimum")
    rates = np.zeros(top_b, dtype=np.int16)
    current_count = final_k
    current_cost = best_cost_units
    for step in range(steps - 1, -1, -1):
        tier = int(chosen_tier[step, current_count, current_cost])
        if tier < 0:
            raise AssertionError("Oracle backpointer is incomplete")
        candidate = int(variable_order[step])
        rates[candidate] = np.int16(costs[tier])
        old_count = int(previous_count[step, current_count, current_cost])
        old_cost = int(previous_cost[step, current_count, current_cost])
        current_count, current_cost = old_count, old_cost
    if current_count != fixed_count or current_cost != 0:
        raise AssertionError("Oracle backpointer did not return to its initial state")
    tier_by_bytes = {int(value): index for index, value in enumerate(costs)}
    selected_scores = scores[0].copy()
    for candidate in range(top_b):
        selected_scores[candidate] = scores[tier_by_bytes[int(rates[candidate])], candidate]
    order = np.lexsort((docids, -selected_scores))
    membership = np.zeros(candidate_count, dtype=bool)
    membership[order[:final_k]] = True
    reconstructed_hits = int(np.sum(relevance[membership] > 0))
    if reconstructed_hits != best_hits:
        raise AssertionError("Reconstructed oracle assignment does not attain optimum")
    return best_hits, best_cost_units * quantum, rates, membership


def exact_accessed_byte_oracle(
    tier_scores: np.ndarray,
    tier_bytes: Iterable[int],
    labels: np.ndarray,
    document_ids: np.ndarray,
    relevant_counts: np.ndarray,
    *,
    final_k: int,
    top_b: int,
    budget_bytes: int,
) -> AccessOracleResult:
    """Solve every query exactly; the oracle is label-aware and non-deployable."""

    scores = np.asarray(tier_scores)
    relevance = np.asarray(labels)
    counts = np.asarray(relevant_counts)
    if scores.ndim != 3:
        raise ValueError("tier_scores must have shape [Q, T, C]")
    if not np.all(np.isfinite(scores)):
        raise ValueError("tier_scores must be finite")
    if relevance.shape != (scores.shape[0], scores.shape[2]):
        raise ValueError("labels do not match tier scores")
    if not np.all(np.isfinite(relevance)):
        raise ValueError("labels must be finite")
    docids = _validate_document_ids(document_ids, expected_shape=relevance.shape)
    if (
        counts.shape != (scores.shape[0],)
        or not np.all(np.isfinite(counts))
        or np.any(counts <= 0)
    ):
        raise ValueError("Invalid relevant_counts")
    costs = _validate_tier_costs(tier_bytes, expected_count=scores.shape[1])
    if not 0 < final_k < scores.shape[2] or not final_k < top_b <= scores.shape[2]:
        raise ValueError("Invalid Top-k or correction depth")
    hits = np.empty(scores.shape[0], dtype=np.int32)
    used = np.empty(scores.shape[0], dtype=np.int32)
    assignments = np.empty((scores.shape[0], top_b), dtype=np.int16)
    membership = np.zeros((scores.shape[0], scores.shape[2]), dtype=bool)
    for query_index in range(scores.shape[0]):
        (
            hits[query_index],
            used[query_index],
            assignments[query_index],
            membership[query_index],
        ) = _exact_query_access_oracle(
            scores[query_index],
            costs,
            relevance[query_index],
            docids[query_index],
            final_k=final_k,
            top_b=top_b,
            budget_bytes=budget_bytes,
        )
    if np.any(used > budget_bytes):
        raise AssertionError("Internal oracle budget violation")
    recall = hits.astype(np.float64) / counts.astype(np.float64)
    return AccessOracleResult(hits, recall, used, assignments, membership)


def candidate_relevance_ceiling(
    labels: np.ndarray, relevant_counts: np.ndarray, *, k: int, depth: int
) -> np.ndarray:
    relevance = np.asarray(labels)
    counts = np.asarray(relevant_counts)
    if relevance.ndim != 2 or counts.shape != (len(relevance),):
        raise ValueError("Invalid relevance ceiling arrays")
    if not 0 < k <= depth <= relevance.shape[1] or np.any(counts <= 0):
        raise ValueError("Invalid ceiling depth or relevant counts")
    hits = np.minimum(k, np.sum(relevance[:, :depth] > 0, axis=1))
    return hits.astype(np.float64) / counts.astype(np.float64)


def paired_bootstrap_mean_delta(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
    chunk_size: int = 256,
) -> dict[str, float | int]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("Paired bootstrap inputs must have identical shapes")
    if (
        left_values.ndim != 1
        or not len(left_values)
        or not np.all(np.isfinite(left_values))
        or not np.all(np.isfinite(right_values))
    ):
        raise ValueError("Bootstrap inputs must be finite non-empty vectors")
    delta = left_values - right_values
    if replicates <= 0 or not 0 < confidence < 1 or chunk_size <= 0:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        indices = rng.integers(0, len(delta), size=(end - start, len(delta)))
        means[start:end] = np.mean(delta[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "point_estimate": float(np.mean(delta)),
        "quantile_method": "linear",
        "lower": float(np.quantile(means, tail, method="linear")),
        "upper": float(np.quantile(means, 1.0 - tail, method="linear")),
    }


def gain_diagnostics(delta: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Gain diagnostics require one finite value per query")
    positive = np.maximum(values, 0.0)
    negative = np.maximum(-values, 0.0)
    positive_mass = float(np.sum(positive))
    negative_mass = float(np.sum(negative))
    top_count = max(1, int(math.ceil(0.01 * len(values))))
    top_mass = float(np.sum(np.sort(positive)[-top_count:]))
    squared = float(np.sum(positive * positive))
    return {
        "improved": int(np.sum(values > 0)),
        "harmed": int(np.sum(values < 0)),
        "unchanged": int(np.sum(values == 0)),
        "positive_support_fraction": float(np.mean(values > 0)),
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "harm_to_positive_mass_ratio": (
            None if positive_mass == 0 else negative_mass / positive_mass
        ),
        "top_1pct_positive_gain_concentration": (
            1.0 if positive_mass == 0 else top_mass / positive_mass
        ),
        "effective_positive_support": (
            0.0 if squared == 0 else positive_mass * positive_mass / squared
        ),
    }


def compression_recovery_diagnostics(
    reference_scores: np.ndarray,
    exact_scores: np.ndarray,
    oracle_recall: np.ndarray,
    reference_recall: np.ndarray,
    exact_recall: np.ndarray,
    oracle_topk_membership: np.ndarray | None,
    document_ids: np.ndarray,
    *,
    k: int,
    reference_name: str,
) -> dict[str, float | str | None]:
    """Measure oracle recovery relative to one explicitly named reference.

    Call this once with the uncorrected base and once with the design-frozen
    comparator.  Only the comparator-relative result is eligible for hard
    recovery gates; the base-relative result remains a useful diagnostic.
    """

    if not isinstance(reference_name, str) or not reference_name.strip():
        raise ValueError("Compression recovery requires an explicit reference_name")
    reference_score_values = np.asarray(reference_scores)
    exact_score_values = np.asarray(exact_scores)
    if (
        reference_score_values.ndim != 2
        or exact_score_values.shape != reference_score_values.shape
    ):
        raise ValueError("Compression recovery score arrays disagree")
    if not (
        np.all(np.isfinite(reference_score_values))
        and np.all(np.isfinite(exact_score_values))
    ):
        raise ValueError("Compression recovery scores must be finite")
    docids = _validate_document_ids(
        document_ids, expected_shape=reference_score_values.shape
    )
    if not 0 < k < reference_score_values.shape[1]:
        raise ValueError("Invalid compression-recovery Top-k")

    reference_values = np.asarray(reference_recall, dtype=np.float64)
    exact_values = np.asarray(exact_recall, dtype=np.float64)
    oracle_values = np.asarray(oracle_recall, dtype=np.float64)
    if not (
        reference_values.shape == exact_values.shape == oracle_values.shape
        and reference_values.shape == (reference_score_values.shape[0],)
    ):
        raise ValueError("Compression recovery recall arrays disagree")
    if not (
        np.all(np.isfinite(reference_values))
        and np.all(np.isfinite(exact_values))
        and np.all(np.isfinite(oracle_values))
    ):
        raise ValueError("Compression recovery recall arrays must be finite")
    compression_positive = np.maximum(exact_values - reference_values, 0.0)
    oracle_positive = np.maximum(oracle_values - reference_values, 0.0)
    recoverable = np.minimum(oracle_positive, compression_positive)
    headroom = float(np.sum(compression_positive))
    if oracle_topk_membership is None:
        recovered = float(np.sum(recoverable))
        return {
            "reference_name": reference_name,
            "compression_positive_mass": headroom,
            "compression_consistent_recovered_mass": recovered,
            "counterfactual_recovery_fraction": (
                0.0 if headroom <= 0 else recovered / headroom
            ),
            "positive_gain_mass_with_exact_distance_reduction_fraction": None,
        }
    reference_membership = topk_membership(reference_score_values, docids, k=k)
    exact_membership = topk_membership(exact_score_values, docids, k=k)
    oracle_membership = np.asarray(oracle_topk_membership)
    if oracle_membership.shape != reference_membership.shape:
        raise ValueError("Oracle membership shape does not match candidate arrays")
    if oracle_membership.dtype != np.bool_:
        raise ValueError("Oracle membership must use a boolean dtype")
    if np.any(np.sum(oracle_membership, axis=1) != k):
        raise ValueError("Oracle membership must select exactly k documents per query")
    reference_distance = np.sum(reference_membership != exact_membership, axis=1)
    oracle_distance = np.sum(oracle_membership != exact_membership, axis=1)
    aligned = oracle_distance < reference_distance
    recovered = float(np.sum(recoverable[aligned]))
    aligned_mass = float(np.sum(oracle_positive[aligned]))
    total_oracle_mass = float(np.sum(oracle_positive))
    return {
        "reference_name": reference_name,
        "compression_positive_mass": headroom,
        "compression_consistent_recovered_mass": recovered,
        "counterfactual_recovery_fraction": (
            0.0 if headroom <= 0 else recovered / headroom
        ),
        "positive_gain_mass_with_exact_distance_reduction_fraction": (
            0.0 if total_oracle_mass == 0 else aligned_mass / total_oracle_mass
        ),
    }


def decide_oracle_gate(
    *,
    oracle_recall: np.ndarray,
    comparator_recall: np.ndarray,
    exact40_recall: np.ndarray,
    base_recall: np.ndarray,
    base_relative_cfr8: float,
    base_relative_cfr16: float,
    base_relative_alignment16: float,
    comparator_relative_cfr8: float,
    comparator_relative_cfr16: float,
    comparator_relative_alignment16: float,
    design_fold_gains: Iterable[float],
    bootstrap: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen gate, using comparator-relative recovery as hard evidence.

    Base-relative recovery is retained and returned for continuity, but cannot
    satisfy a hard gate when the selected comparator is stronger than Base.
    """

    oracle = np.asarray(oracle_recall, dtype=np.float64)
    comparator = np.asarray(comparator_recall, dtype=np.float64)
    exact40 = np.asarray(exact40_recall, dtype=np.float64)
    base = np.asarray(base_recall, dtype=np.float64)
    if not (
        oracle.shape == comparator.shape == exact40.shape == base.shape
        and oracle.ndim == 1
        and len(oracle) > 0
    ):
        raise ValueError("Gate recall arrays disagree")
    if not all(
        np.all(np.isfinite(value))
        for value in (oracle, comparator, exact40, base)
    ):
        raise ValueError("Gate recall arrays must be finite")

    recovery_values = {
        "base_relative_cfr8": base_relative_cfr8,
        "base_relative_cfr16": base_relative_cfr16,
        "base_relative_alignment16": base_relative_alignment16,
        "comparator_relative_cfr8": comparator_relative_cfr8,
        "comparator_relative_cfr16": comparator_relative_cfr16,
        "comparator_relative_alignment16": comparator_relative_alignment16,
    }
    normalized_recovery: dict[str, float] = {}
    for name, value in recovery_values.items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite fraction")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite fraction") from error
        if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"{name} must be a finite fraction in [0, 1]")
        normalized_recovery[name] = normalized

    if not isinstance(bootstrap, dict):
        raise ValueError("Bootstrap evidence must be a dictionary")
    normalized_bootstrap: dict[str, Any] = dict(bootstrap)
    for name in ("point_estimate", "lower", "upper", "confidence"):
        try:
            value = float(bootstrap[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Bootstrap requires finite {name}") from error
        if not np.isfinite(value):
            raise ValueError(f"Bootstrap requires finite {name}")
        normalized_bootstrap[name] = value
    if not 0.0 < normalized_bootstrap["confidence"] < 1.0:
        raise ValueError("Bootstrap confidence must be in (0, 1)")
    for name in ("replicates", "seed"):
        value = bootstrap.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"Bootstrap requires finite integer {name}")
        if name == "replicates" and value <= 0:
            raise ValueError("Bootstrap replicates must be positive")
        normalized_bootstrap[name] = int(value)

    delta = oracle - comparator
    diagnostics = gain_diagnostics(delta)
    folds = np.asarray(tuple(design_fold_gains), dtype=np.float64)
    if folds.shape != (5,) or not np.all(np.isfinite(folds)):
        raise ValueError("Exactly five finite design-fold gains are required")
    minimum_support = max(
        int(thresholds["minimum_positive_support_queries"]),
        int(math.ceil(float(thresholds["minimum_positive_support_fraction"]) * len(delta))),
    )
    checks = {
        "exact40_headroom": float(np.mean(exact40 - comparator))
        >= float(thresholds["minimum_exact40_gain_over_comparator"]),
        "oracle_gain": float(np.mean(delta))
        >= float(thresholds["minimum_oracle_gain_over_comparator"]),
        "bootstrap_lower_bound": normalized_bootstrap["lower"]
        > float(thresholds["minimum_bootstrap_lower_bound"]),
        "positive_support": int(diagnostics["improved"]) >= minimum_support,
        "harm_mass": diagnostics["harm_to_positive_mass_ratio"] is not None
        and float(diagnostics["harm_to_positive_mass_ratio"])
        <= float(thresholds["maximum_harm_to_positive_mass_ratio"]),
        "gain_concentration": float(
            diagnostics["top_1pct_positive_gain_concentration"]
        )
        <= float(thresholds["maximum_top_1pct_positive_gain_concentration"]),
        "effective_support": float(diagnostics["effective_positive_support"])
        >= float(thresholds["minimum_effective_positive_support"]),
        "oracle8_comparator_relative_cfr": normalized_recovery[
            "comparator_relative_cfr8"
        ]
        >= float(
            thresholds[
                "minimum_comparator_counterfactual_recovery_fraction_8b"
            ]
        ),
        "oracle16_comparator_relative_cfr": normalized_recovery[
            "comparator_relative_cfr16"
        ]
        >= float(
            thresholds[
                "minimum_comparator_counterfactual_recovery_fraction_16b"
            ]
        ),
        "oracle16_comparator_exact_membership_alignment": normalized_recovery[
            "comparator_relative_alignment16"
        ]
        >= float(
            thresholds[
                "minimum_comparator_positive_gain_mass_with_exact_distance_reduction_fraction"
            ]
        ),
        "design_fold_direction": int(np.sum(folds > 0))
        >= int(thresholds["minimum_positive_design_folds"]),
        "design_worst_fold": float(np.min(folds))
        >= float(thresholds["minimum_worst_design_fold_gain"]),
        "comparator_compression_positive_mass": float(
            np.sum(np.maximum(exact40 - comparator, 0.0))
        )
        > 0.0,
    }
    if not checks["exact40_headroom"] or not checks[
        "comparator_compression_positive_mass"
    ]:
        decision = "KILL_NO_SCORE_HEADROOM"
    elif all(checks.values()):
        decision = "GO_TO_STATIC_STORAGE_ORACLE"
    else:
        decision = "STOP_NO_HEADROOM"
    return {
        "decision": decision,
        "all_required_checks_passed": all(checks.values()),
        "checks": checks,
        "minimum_positive_support_queries_effective": minimum_support,
        "oracle_gain_over_comparator": float(np.mean(delta)),
        "exact40_gain_over_comparator": float(np.mean(exact40 - comparator)),
        "compression_recovery": {
            "base_relative": {
                "oracle8_cfr": normalized_recovery["base_relative_cfr8"],
                "oracle16_cfr": normalized_recovery["base_relative_cfr16"],
                "oracle16_exact_membership_alignment": normalized_recovery[
                    "base_relative_alignment16"
                ],
                "exact40_positive_mass": float(
                    np.sum(np.maximum(exact40 - base, 0.0))
                ),
            },
            "comparator_relative": {
                "oracle8_cfr": normalized_recovery["comparator_relative_cfr8"],
                "oracle16_cfr": normalized_recovery[
                    "comparator_relative_cfr16"
                ],
                "oracle16_exact_membership_alignment": normalized_recovery[
                    "comparator_relative_alignment16"
                ],
                "exact40_positive_mass": float(
                    np.sum(np.maximum(exact40 - comparator, 0.0))
                ),
            },
        },
        "design_fold_gains": folds.tolist(),
        "gain_diagnostics": diagnostics,
        "bootstrap": normalized_bootstrap,
    }


def build_run_fingerprint(payload: dict[str, Any]) -> str:
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Run fingerprint must use the v3 protocol")
    return canonical_sha256(payload)
