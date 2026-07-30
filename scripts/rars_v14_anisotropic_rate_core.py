#!/usr/bin/env python3
"""Numerical core for V14 query-whitened anisotropic rate-RPQ."""

from __future__ import annotations

from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v14_query_whitened_anisotropic_rate_rpq_diagnostic_v1"


def _matrix(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix")
    return array


def _stable_top(scores: np.ndarray, rows: np.ndarray, count: int) -> np.ndarray:
    if scores.ndim != 1 or rows.shape != scores.shape or not 0 < count <= len(scores):
        raise ValueError("Stable top selection received an invalid vector")
    return np.lexsort((rows, -scores))[:count]


def cutoff_weights(
    scores: Any,
    rows: Any,
    *,
    top_b: int,
    final_k: int,
    cutoff_boost: float,
    margin_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Base Top-B positions and label-free cutoff proximity weights."""

    values = np.asarray(scores, dtype=np.float64)
    identifiers = np.asarray(rows, dtype=np.int64)
    if (
        values.ndim != 1
        or identifiers.shape != values.shape
        or not 0 < final_k <= top_b <= len(values)
        or cutoff_boost < 0
        or margin_temperature <= 0
    ):
        raise ValueError("Invalid cutoff-weight inputs")
    ordering = _stable_top(values, identifiers, top_b)
    threshold = float(values[ordering[final_k - 1]])
    weights = 1.0 + cutoff_boost * np.exp(
        -np.abs(values[ordering] - threshold) / margin_temperature
    )
    return ordering, weights.astype(np.float64)


def fit_query_metric_transforms(
    queries: Any,
    basis: Any,
    candidate_rows: Any,
    base_scores: Any,
    *,
    top_b: int,
    final_k: int,
    cutoff_boost: float,
    margin_temperature: float,
    ridge_fraction: float,
    block_dimension: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one trace-normalized score metric per PCA coefficient block.

    No relevance labels are accepted by this API.  Repeating each query by
    its weighted Base Top-B mass makes the metric reflect online cutoff use.
    """

    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    scores = _matrix(base_scores, name="base_scores", dtype=np.float32)
    if rows.shape != scores.shape or len(rows) != len(query_matrix):
        raise ValueError("Query and candidate arrays disagree")
    if query_matrix.shape[1] != projection.shape[0]:
        raise ValueError("Query dimension differs from the PCA basis")
    if projection.shape[1] % block_dimension or ridge_fraction <= 0:
        raise ValueError("Invalid block dimension or ridge fraction")
    subquantizers = projection.shape[1] // block_dimension
    projected = (query_matrix @ projection).reshape(
        len(query_matrix), subquantizers, block_dimension
    )
    covariance = np.zeros(
        (subquantizers, block_dimension, block_dimension), dtype=np.float64
    )
    mass = 0.0
    for index in range(len(query_matrix)):
        _, weights = cutoff_weights(
            scores[index],
            rows[index],
            top_b=top_b,
            final_k=final_k,
            cutoff_boost=cutoff_boost,
            margin_temperature=margin_temperature,
        )
        local_mass = float(weights.sum())
        mass += local_mass
        for block in range(subquantizers):
            q = projected[index, block].astype(np.float64)
            covariance[block] += local_mass * np.outer(q, q)
    if mass <= 0:
        raise ValueError("Query metric has zero mass")
    transforms = np.empty_like(covariance)
    eigenvalue_ranges: list[list[float]] = []
    conditions: list[float] = []
    for block in range(subquantizers):
        metric = covariance[block] / mass
        ridge = ridge_fraction * max(
            float(np.trace(metric)) / block_dimension,
            np.finfo(np.float64).tiny,
        )
        metric += ridge * np.eye(block_dimension, dtype=np.float64)
        metric *= block_dimension / float(np.trace(metric))
        eigenvalues = np.linalg.eigvalsh(metric)
        if np.any(eigenvalues <= 0):
            raise ValueError("Regularized query metric is not positive definite")
        transforms[block] = np.linalg.cholesky(metric)
        eigenvalue_ranges.append([float(eigenvalues[0]), float(eigenvalues[-1])])
        conditions.append(float(eigenvalues[-1] / eigenvalues[0]))
    return transforms.astype(np.float32), {
        "query_count": int(len(query_matrix)),
        "weighted_candidate_mass": mass,
        "subquantizers": subquantizers,
        "block_dimension": block_dimension,
        "ridge_fraction": float(ridge_fraction),
        "trace_normalization_target": float(block_dimension),
        "eigenvalue_ranges": eigenvalue_ranges,
        "maximum_condition_number": float(max(conditions)),
        "labels_used": False,
    }


def block_rate_sensitivity(
    coefficients: Any,
    transforms: Any,
    residual_lookup: Any,
    candidate_rows: Any,
    base_scores: Any,
    *,
    top_b: int,
    final_k: int,
    cutoff_boost: float,
    margin_temperature: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate label-free transformed residual energy at the cutoff."""

    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    matrices = np.asarray(transforms, dtype=np.float32)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    scores = _matrix(base_scores, name="base_scores", dtype=np.float32)
    if rows.shape != scores.shape or lookup.shape != rows.shape:
        raise ValueError("Candidate arrays disagree")
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Transforms must be square block matrices")
    blocks, block_dimension, _ = matrices.shape
    if values.shape[1] != blocks * block_dimension:
        raise ValueError("Coefficient rank differs from transforms")
    numerator = np.zeros(blocks, dtype=np.float64)
    denominator = 0.0
    for query_index in range(len(rows)):
        ordering, weights = cutoff_weights(
            scores[query_index],
            rows[query_index],
            top_b=top_b,
            final_k=final_k,
            cutoff_boost=cutoff_boost,
            margin_temperature=margin_temperature,
        )
        selected = lookup[query_index, ordering]
        if np.any(selected < 0) or np.any(selected >= len(values)):
            raise ValueError("Residual lookup is outside coefficient rows")
        denominator += float(weights.sum())
        for block in range(blocks):
            left = block * block_dimension
            right = left + block_dimension
            transformed = values[selected, left:right] @ matrices[block]
            energy = np.sum(transformed.astype(np.float64) ** 2, axis=1)
            numerator[block] += float(weights @ energy)
    if denominator <= 0:
        raise ValueError("Rate sensitivity has zero mass")
    sensitivity = np.maximum(
        numerator / denominator, np.finfo(np.float64).tiny
    )
    return sensitivity, {
        "candidate_observations": int(len(rows) * top_b),
        "weighted_candidate_mass": denominator,
        "minimum_sensitivity": float(sensitivity.min()),
        "maximum_sensitivity": float(sensitivity.max()),
        "anisotropy_ratio": float(sensitivity.max() / sensitivity.min()),
        "labels_used": False,
    }


def allocate_bits_dynamic_programming(
    sensitivity: Any,
    *,
    total_bits: int,
    minimum_bits: int,
    maximum_bits: int,
    block_dimension: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the exact bounded integer high-rate allocation problem."""

    values = np.asarray(sensitivity, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or np.any(~np.isfinite(values))
        or np.any(values <= 0)
        or minimum_bits <= 0
        or maximum_bits < minimum_bits
        or block_dimension <= 0
        or not len(values) * minimum_bits <= total_bits <= len(values) * maximum_bits
    ):
        raise ValueError("Invalid bit-allocation problem")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for value in values:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (cost, prefix) in states.items():
            for bits in range(minimum_bits, maximum_bits + 1):
                updated = used + bits
                if updated > total_bits:
                    continue
                candidate = (
                    cost + float(value) * 2.0 ** (-2.0 * bits / block_dimension),
                    prefix + (bits,),
                )
                current = next_states.get(updated)
                if current is None or candidate[0] < current[0] - 1e-18 or (
                    abs(candidate[0] - current[0]) <= 1e-18
                    and candidate[1] < current[1]
                ):
                    next_states[updated] = candidate
        states = next_states
    if total_bits not in states:
        raise ValueError("No feasible bit allocation")
    objective, allocation = states[total_bits]
    output = np.asarray(allocation, dtype=np.int64)
    uniform_bits = total_bits // len(values)
    uniform_objective = float(
        np.sum(values * 2.0 ** (-2.0 * uniform_bits / block_dimension))
    )
    return output, {
        "total_bits": int(output.sum()),
        "minimum_allocated_bits": int(output.min()),
        "maximum_allocated_bits": int(output.max()),
        "nonuniform": bool(np.any(output != output[0])),
        "objective": float(objective),
        "uniform_objective": uniform_objective,
        "proxy_reduction_fraction": float(
            (uniform_objective - objective) / uniform_objective
        ),
    }


def pack_variable_codes(codes: Any, bit_allocation: Any) -> np.ndarray:
    matrix = np.asarray(codes)
    bits = np.asarray(bit_allocation, dtype=np.int64)
    if matrix.ndim != 2 or bits.shape != (matrix.shape[1],) or np.any(bits <= 0):
        raise ValueError("Codes and bit allocation disagree")
    total_bits = int(bits.sum())
    if total_bits % 8:
        raise ValueError("Total code width must be byte aligned")
    if np.any(matrix < 0):
        raise ValueError("Product codes must be nonnegative")
    output_bits = np.empty((len(matrix), total_bits), dtype=np.uint8)
    offset = 0
    for block, width in enumerate(bits):
        values = matrix[:, block].astype(np.uint64, copy=False)
        if np.any(values >= (1 << int(width))):
            raise ValueError("A product code exceeds its allocated width")
        for shift in range(int(width)):
            output_bits[:, offset + shift] = ((values >> shift) & 1).astype(np.uint8)
        offset += int(width)
    return np.packbits(output_bits, axis=1, bitorder="little")


def unpack_variable_codes(packed: Any, bit_allocation: Any) -> np.ndarray:
    payload = np.asarray(packed, dtype=np.uint8)
    bits = np.asarray(bit_allocation, dtype=np.int64)
    if payload.ndim != 2 or bits.ndim != 1 or np.any(bits <= 0):
        raise ValueError("Packed codes or bit allocation are invalid")
    total_bits = int(bits.sum())
    if payload.shape[1] * 8 != total_bits:
        raise ValueError("Packed payload width differs from the bit allocation")
    expanded = np.unpackbits(payload, axis=1, bitorder="little")
    codes = np.zeros((len(payload), len(bits)), dtype=np.uint16)
    offset = 0
    for block, width in enumerate(bits):
        value = np.zeros(len(payload), dtype=np.uint16)
        for shift in range(int(width)):
            value |= expanded[:, offset + shift].astype(np.uint16) << shift
        codes[:, block] = value
        offset += int(width)
    return codes


def concatenate_codebooks(codebooks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not codebooks:
        raise ValueError("At least one codebook is required")
    dimension = np.asarray(codebooks[0]).shape[-1]
    normalized: list[np.ndarray] = []
    offsets = [0]
    for value in codebooks:
        book = _matrix(value, name="codebook", dtype=np.float32)
        if book.shape[1] != dimension:
            raise ValueError("Variable codebooks have inconsistent dimensions")
        normalized.append(book)
        offsets.append(offsets[-1] + len(book))
    return np.concatenate(normalized).astype(np.float32), np.asarray(offsets, dtype=np.int64)


def split_codebooks(values: Any, offsets: Any) -> list[np.ndarray]:
    books = _matrix(values, name="concatenated_codebooks", dtype=np.float32)
    boundaries = np.asarray(offsets, dtype=np.int64)
    if (
        boundaries.ndim != 1
        or len(boundaries) < 2
        or boundaries[0] != 0
        or boundaries[-1] != len(books)
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("Codebook offsets are invalid")
    return [books[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]


def fit_variable_block_quantizers(
    coefficients: Any,
    transforms: Any,
    bit_allocation: Any,
    faiss_module: Any,
    *,
    iterations: int,
    seed: int,
    max_points_per_centroid: int,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Fit independent transformed-space k-means codebooks."""

    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    matrices = np.asarray(transforms, dtype=np.float32)
    bits = np.asarray(bit_allocation, dtype=np.int64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Transforms must be square")
    blocks, block_dimension, _ = matrices.shape
    if bits.shape != (blocks,) or values.shape[1] != blocks * block_dimension:
        raise ValueError("Coefficient, transform, and bit shapes disagree")
    if iterations <= 0 or max_points_per_centroid <= 0:
        raise ValueError("Invalid k-means controls")
    faiss_module.omp_set_num_threads(1)
    codes = np.empty((len(values), blocks), dtype=np.uint16)
    decoded_books: list[np.ndarray] = []
    block_summaries: list[dict[str, Any]] = []
    for block in range(blocks):
        left = block * block_dimension
        right = left + block_dimension
        transformed = np.ascontiguousarray(
            values[:, left:right] @ matrices[block], dtype=np.float32
        )
        centroid_count = 1 << int(bits[block])
        clustering = faiss_module.Clustering(block_dimension, centroid_count)
        clustering.niter = int(iterations)
        clustering.nredo = 1
        clustering.seed = int(seed + block)
        clustering.verbose = False
        clustering.min_points_per_centroid = 1
        clustering.max_points_per_centroid = int(max_points_per_centroid)
        index = faiss_module.IndexFlatL2(block_dimension)
        clustering.train(transformed, index)
        transformed_book = np.asarray(
            faiss_module.vector_to_array(clustering.centroids), dtype=np.float32
        ).reshape(centroid_count, block_dimension)
        assignment_index = faiss_module.IndexFlatL2(block_dimension)
        assignment_index.add(transformed_book)
        _, assignment = assignment_index.search(transformed, 1)
        local_codes = assignment[:, 0].astype(np.uint16)
        codes[:, block] = local_codes
        decoded = transformed_book @ np.linalg.inv(matrices[block])
        decoded_books.append(decoded.astype(np.float32))
        error = transformed - transformed_book[local_codes]
        block_summaries.append(
            {
                "block": block,
                "bits": int(bits[block]),
                "centroids": centroid_count,
                "occupied_centroids": int(np.unique(local_codes).size),
                "transformed_mse": float(np.mean(error * error)),
            }
        )
    packed = pack_variable_codes(codes, bits)
    return packed, decoded_books, {
        "training_rows": int(len(values)),
        "iterations": int(iterations),
        "seed": int(seed),
        "total_bits": int(bits.sum()),
        "payload_bytes_per_document": int(packed.shape[1]),
        "block_summaries": block_summaries,
    }


def assign_variable_codes(
    coefficients: Any,
    transforms: Any,
    decoded_codebooks: list[np.ndarray],
    bit_allocation: Any,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    matrices = np.asarray(transforms, dtype=np.float32)
    bits = np.asarray(bit_allocation, dtype=np.int64)
    blocks, block_dimension, _ = matrices.shape
    if len(decoded_codebooks) != blocks or bits.shape != (blocks,) or batch_size <= 0:
        raise ValueError("Variable assignment contract changed")
    codes = np.empty((len(values), blocks), dtype=np.uint16)
    for block, decoded in enumerate(decoded_codebooks):
        book = _matrix(decoded, name="decoded_codebook", dtype=np.float32)
        if book.shape != (1 << int(bits[block]), block_dimension):
            raise ValueError("Decoded codebook size differs from bit allocation")
        transformed_book = book @ matrices[block]
        book_norm = np.sum(transformed_book * transformed_book, axis=1)
        left = block * block_dimension
        right = left + block_dimension
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            local = values[start:end, left:right] @ matrices[block]
            distance = (
                np.sum(local * local, axis=1)[:, None]
                - 2.0 * local @ transformed_book.T
                + book_norm[None, :]
            )
            codes[start:end, block] = np.argmin(distance, axis=1).astype(np.uint16)
    return pack_variable_codes(codes, bits)


def score_variable_sidecar_candidates(
    queries: Any,
    candidate_rows: Any,
    residual_lookup: Any,
    base_scores: Any,
    basis: Any,
    packed_codes: Any,
    bit_allocation: Any,
    codebooks: list[np.ndarray],
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _matrix(base_scores, name="base_scores", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    bits = np.asarray(bit_allocation, dtype=np.int64)
    if not (rows.shape == lookup.shape == base.shape) or len(rows) != len(query_matrix):
        raise ValueError("Candidate arrays disagree")
    if len(codebooks) != len(bits) or projection.shape[1] % len(bits):
        raise ValueError("Codebooks differ from basis blocks")
    codes = unpack_variable_codes(packed_codes, bits)
    block_dimension = projection.shape[1] // len(bits)
    projected_queries = query_matrix @ projection
    output = base.copy()
    for query_index in range(len(rows)):
        selected = _stable_top(base[query_index], rows[query_index], top_b)
        selected_lookup = lookup[query_index, selected]
        if np.any(selected_lookup < 0) or np.any(selected_lookup >= len(codes)):
            raise ValueError("Residual lookup is outside variable codes")
        local_codes = codes[selected_lookup]
        query_blocks = projected_queries[query_index].reshape(len(bits), block_dimension)
        correction = np.zeros(len(selected), dtype=np.float32)
        for block, book in enumerate(codebooks):
            lookup_table = np.asarray(book, dtype=np.float32) @ query_blocks[block]
            correction += lookup_table[local_codes[:, block]]
        output[query_index, selected] += float(alpha) * correction
    return output


def multi_seed_consensus(
    challenger_recall: Any, baseline_recall: Any
) -> dict[str, int]:
    challenger = np.asarray(challenger_recall, dtype=np.float64)
    baseline = np.asarray(baseline_recall, dtype=np.float64)
    if challenger.ndim != 2 or challenger.shape != baseline.shape or len(challenger) != 3:
        raise ValueError("Consensus requires three matched seed-by-query matrices")
    differences = challenger - baseline
    improved = np.sum(differences > 0, axis=0)
    harmed = np.sum(differences < 0, axis=0)
    return {
        "improved_in_at_least_two_seeds": int(np.sum(improved >= 2)),
        "harmed_in_at_least_two_seeds": int(np.sum(harmed >= 2)),
        "improved_in_all_three_seeds": int(np.sum(improved == 3)),
        "harmed_in_all_three_seeds": int(np.sum(harmed == 3)),
    }


def anisotropic_rate_decision(
    *,
    primary_vs_uniform_rpq: dict[str, Any],
    primary_vs_uniform_whitened: dict[str, Any],
    primary_vs_pca16: dict[str, Any],
    primary_vs_base: dict[str, Any],
    seed_gains: list[float],
    fold_gains: list[float],
    candidate_gap_recovery: float,
    uniform_rpq_mrr: float,
    challenger_mrr: float,
    uniform_rpq_ndcg: float,
    challenger_ndcg: float,
    consensus: dict[str, int],
    allocations: list[list[int]],
    payload_bytes_per_document: int,
    full_corpus_codes_materialized: bool,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    if len(seed_gains) != 3 or len(fold_gains) != 5 or not allocations:
        raise ValueError("V14 requires three seeds, five folds, and allocations")
    support = int(primary_vs_uniform_rpq["improved_queries"])
    harmed = int(primary_vs_uniform_rpq["harmed_queries"])
    total_bits = int(thresholds["require_exact_total_bits"])
    exact_allocations = all(sum(allocation) == total_bits for allocation in allocations)
    nonuniform = all(len(set(allocation)) > 1 for allocation in allocations)
    gates = {
        "minimum_gain_over_v13_uniform_rpq": float(primary_vs_uniform_rpq["mean_difference"])
        >= float(thresholds["minimum_recall_gain_over_v13_uniform_rpq"]),
        "minimum_gain_over_uniform_whitened": float(primary_vs_uniform_whitened["mean_difference"])
        >= float(thresholds["minimum_recall_gain_over_uniform_whitened"]),
        "minimum_gain_over_pca16": float(primary_vs_pca16["mean_difference"])
        >= float(thresholds["minimum_recall_gain_over_pca16"]),
        "minimum_gain_over_base": float(primary_vs_base["mean_difference"])
        >= float(thresholds["minimum_recall_gain_over_base"]),
        "bootstrap_lower_above_zero": float(primary_vs_uniform_rpq["lower"])
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "randomization_p_value": float(primary_vs_uniform_rpq["randomization_p_value_one_sided"])
        <= float(thresholds["maximum_randomization_p_value"]),
        "improved_query_support": support >= int(thresholds["minimum_improved_queries"]),
        "net_improved_query_support": support - harmed
        >= int(thresholds["minimum_net_improved_queries"]),
        "all_seed_gains_nonnegative": min(seed_gains)
        >= float(thresholds["minimum_each_seed_gain"]),
        "median_seed_gain": float(np.median(seed_gains))
        >= float(thresholds["minimum_median_seed_gain"]),
        "worst_fold_gain": min(fold_gains)
        >= float(thresholds["minimum_worst_fold_gain"]),
        "candidate_gap_recovery": candidate_gap_recovery
        >= float(thresholds["minimum_candidate_gap_recovery_fraction"]),
        "mrr_guardrail": challenger_mrr - uniform_rpq_mrr
        >= float(thresholds["minimum_mrr_change"]),
        "ndcg_guardrail": challenger_ndcg - uniform_rpq_ndcg
        >= float(thresholds["minimum_ndcg_change"]),
        "multi_seed_improvement_support": consensus["improved_in_at_least_two_seeds"]
        >= int(thresholds["minimum_queries_improved_in_at_least_two_seeds"]),
        "multi_seed_harm_guardrail": consensus["harmed_in_at_least_two_seeds"]
        <= int(thresholds["maximum_queries_harmed_in_at_least_two_seeds"]),
        "exact_128_bit_allocations": exact_allocations,
        "nonuniform_allocations": nonuniform if thresholds["require_nonuniform_allocation"] else True,
        "payload_exactly_16_bytes": payload_bytes_per_document == 16,
        "full_corpus_codes_materialized": bool(full_corpus_codes_materialized),
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
        "harmed_queries": harmed,
        "net_improved_queries": support - harmed,
    }
