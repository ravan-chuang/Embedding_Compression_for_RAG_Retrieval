#!/usr/bin/env python3
"""Numerical core for the RARS-v11 rank--rate capacity diagnostic.

V11 does not learn a cutoff-aware method.  It asks a narrower engineering
question forced by the V10 result: can a wider residual subspace retain useful
ranking information when the per-document payload remains exactly 16 bytes?

The deployable candidates are packed rank-32 int4 coefficients and 16-byte
residual product codes.  FP32 rank-32/rank-64 scorers are diagnostic ceilings.
"""

from __future__ import annotations

from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v11_rank_rate_diagnostic_v1"


def _matrix(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _stable_top_b(scores: np.ndarray, rows: np.ndarray, top_b: int) -> np.ndarray:
    if scores.ndim != 1 or rows.shape != scores.shape:
        raise ValueError("score and row vectors must match")
    if not 0 < top_b <= len(scores):
        raise ValueError("Top-B is outside the candidate width")
    valid = rows >= 0
    safe_rows = np.where(valid, rows, np.iinfo(np.int64).max)
    safe_scores = np.where(valid, scores, -np.inf)
    return np.lexsort((safe_rows, -safe_scores))[:top_b]


def fit_int4_scales(
    residuals: Any, basis: Any, *, batch_size: int = 8192
) -> np.ndarray:
    values = _matrix(residuals, name="residuals", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    if values.shape[1] != projection.shape[0] or batch_size <= 0:
        raise ValueError("Residual/basis dimensions or batch size are invalid")
    maximum = np.zeros(projection.shape[1], dtype=np.float32)
    count = 0
    for start in range(0, len(values), batch_size):
        block = np.asarray(values[start : start + batch_size], dtype=np.float32)
        coefficients = block @ projection
        maximum = np.maximum(maximum, np.max(np.abs(coefficients), axis=0))
        count += len(block)
    if not count:
        raise ValueError("Cannot calibrate int4 scales from no residuals")
    return np.maximum(maximum / 7.0, np.finfo(np.float32).tiny).astype(np.float32)


def pack_signed_int4(codes: Any) -> np.ndarray:
    values = np.asarray(codes)
    if values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("Signed int4 codes must be an even-width matrix")
    if np.any(values < -8) or np.any(values > 7):
        raise ValueError("Signed int4 values must lie in [-8, 7]")
    signed = values.astype(np.int8, copy=False)
    low = signed[:, 0::2].astype(np.uint8) & np.uint8(0x0F)
    high = signed[:, 1::2].astype(np.uint8) & np.uint8(0x0F)
    return (low | (high << np.uint8(4))).astype(np.uint8)


def unpack_signed_int4(packed: Any) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError("Packed int4 values must be a matrix")
    low = (values & np.uint8(0x0F)).astype(np.int8)
    high = ((values >> np.uint8(4)) & np.uint8(0x0F)).astype(np.int8)
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)
    output = np.empty((len(values), values.shape[1] * 2), dtype=np.int8)
    output[:, 0::2] = low
    output[:, 1::2] = high
    return output


def encode_residuals_int4(
    residuals: Any, basis: Any, scales: Any
) -> tuple[np.ndarray, dict[str, float]]:
    values = _matrix(residuals, name="residuals", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    scale = np.asarray(scales, dtype=np.float32)
    if values.shape[1] != projection.shape[0] or projection.shape[1] % 2:
        raise ValueError("Residual/int4 basis dimensions are invalid")
    if scale.shape != (projection.shape[1],) or np.any(scale <= 0):
        raise ValueError("Int4 scales differ from the basis rank")
    raw = values @ projection / scale[None, :]
    rounded = np.rint(raw)
    saturated = np.abs(rounded) > 7
    signed = np.clip(rounded, -7, 7).astype(np.int8)
    packed = pack_signed_int4(signed)
    return packed, {
        "coefficient_count": int(raw.size),
        "saturated_coefficients": int(saturated.sum()),
        "saturation_fraction": float(saturated.mean()) if raw.size else 0.0,
        "payload_bytes_per_document": int(packed.shape[1]),
    }


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
    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _matrix(base_scores, name="base_scores", dtype=np.float32)
    residual_matrix = _matrix(residuals, name="residuals", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    if not (rows.shape == lookup.shape == base.shape):
        raise ValueError("Candidate rows, lookup, and scores must match")
    if len(query_matrix) != len(rows) or query_matrix.shape[1] != projection.shape[0]:
        raise ValueError("Query/candidate/basis dimensions disagree")
    if residual_matrix.shape[1] != projection.shape[0]:
        raise ValueError("Residual and basis dimensions disagree")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Alpha must be finite and non-negative")
    projected_queries = query_matrix @ projection
    projected_residuals = residual_matrix @ projection
    output = base.copy()
    for query_index in range(len(rows)):
        selected = _stable_top_b(base[query_index], rows[query_index], top_b)
        selected_lookup = lookup[query_index, selected]
        valid = selected_lookup >= 0
        if np.any(valid):
            if np.max(selected_lookup[valid]) >= len(projected_residuals):
                raise ValueError("Residual lookup is outside the FP32 matrix")
            output[query_index, selected[valid]] += alpha * (
                projected_residuals[selected_lookup[valid]]
                @ projected_queries[query_index]
            )
    return output


def score_int4_sidecar_candidates(
    queries: Any,
    candidate_rows: Any,
    residual_lookup: Any,
    base_scores: Any,
    basis: Any,
    packed_codes: Any,
    scales: Any,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _matrix(base_scores, name="base_scores", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    packed = np.asarray(packed_codes, dtype=np.uint8)
    scale = np.asarray(scales, dtype=np.float32)
    rank = projection.shape[1]
    if not (rows.shape == lookup.shape == base.shape):
        raise ValueError("Candidate rows, lookup, and scores must match")
    if packed.ndim != 2 or packed.shape[1] * 2 != rank:
        raise ValueError("Packed int4 payload does not match the basis rank")
    if scale.shape != (rank,) or len(query_matrix) != len(rows):
        raise ValueError("Int4 query/scale dimensions disagree")
    decoded = unpack_signed_int4(packed).astype(np.float32) * scale[None, :]
    projected_queries = query_matrix @ projection
    output = base.copy()
    for query_index in range(len(rows)):
        selected = _stable_top_b(base[query_index], rows[query_index], top_b)
        selected_lookup = lookup[query_index, selected]
        valid = selected_lookup >= 0
        if np.any(valid):
            if np.max(selected_lookup[valid]) >= len(decoded):
                raise ValueError("Residual lookup is outside the int4 code matrix")
            output[query_index, selected[valid]] += alpha * (
                decoded[selected_lookup[valid]] @ projected_queries[query_index]
            )
    return output


def fit_faiss_product_quantizer(
    coefficients: Any,
    faiss_module: Any,
    *,
    subquantizers: int,
    bits: int,
    iterations: int,
    seed: int,
    max_points_per_centroid: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = _matrix(coefficients, name="RPQ coefficients", dtype=np.float32)
    if (
        subquantizers <= 0
        or values.shape[1] % subquantizers
        or bits != 8
        or iterations <= 0
        or max_points_per_centroid <= 0
    ):
        raise ValueError("Invalid product-quantizer contract")
    faiss_module.omp_set_num_threads(1)
    quantizer = faiss_module.ProductQuantizer(
        values.shape[1], subquantizers, bits
    )
    quantizer.cp.niter = int(iterations)
    quantizer.cp.nredo = 1
    quantizer.cp.seed = int(seed)
    quantizer.cp.verbose = False
    quantizer.cp.min_points_per_centroid = 1
    quantizer.cp.max_points_per_centroid = int(max_points_per_centroid)
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    quantizer.train(contiguous)
    if not quantizer.is_trained:
        raise ValueError("Faiss product quantizer did not finish training")
    codes = np.asarray(quantizer.compute_codes(contiguous), dtype=np.uint8)
    if codes.shape != (len(values), subquantizers):
        raise ValueError(f"Unexpected RPQ code shape: {codes.shape}")
    codebook_size = 1 << bits
    block_dimension = values.shape[1] // subquantizers
    codebooks = np.asarray(
        faiss_module.vector_to_array(quantizer.centroids), dtype=np.float32
    ).reshape(subquantizers, codebook_size, block_dimension)
    reconstructed = decode_product_codes(codes, codebooks)
    error = values - reconstructed
    return codes, codebooks, {
        "rank": int(values.shape[1]),
        "subquantizers": int(subquantizers),
        "bits_per_subquantizer": int(bits),
        "payload_bytes_per_document": int(codes.shape[1]),
        "codebook_bytes": int(codebooks.nbytes),
        "training_rows": int(len(values)),
        "iterations": int(iterations),
        "seed": int(seed),
        "omp_threads": 1,
        "coefficient_mse": float(np.mean(error * error)),
    }


def decode_product_codes(codes: Any, codebooks: Any) -> np.ndarray:
    code_matrix = np.asarray(codes, dtype=np.uint8)
    books = np.asarray(codebooks, dtype=np.float32)
    if code_matrix.ndim != 2 or books.ndim != 3:
        raise ValueError("RPQ codes/codebooks have invalid rank")
    if code_matrix.shape[1] != books.shape[0] or books.shape[1] != 256:
        raise ValueError("RPQ code width or codebook size changed")
    blocks = [books[index, code_matrix[:, index]] for index in range(books.shape[0])]
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def score_product_sidecar_candidates(
    queries: Any,
    candidate_rows: Any,
    residual_lookup: Any,
    base_scores: Any,
    basis: Any,
    codes: Any,
    codebooks: Any,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    rows = _matrix(candidate_rows, name="candidate_rows", dtype=np.int64)
    lookup = _matrix(residual_lookup, name="residual_lookup", dtype=np.int64)
    base = _matrix(base_scores, name="base_scores", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    code_matrix = np.asarray(codes, dtype=np.uint8)
    books = np.asarray(codebooks, dtype=np.float32)
    if not (rows.shape == lookup.shape == base.shape):
        raise ValueError("Candidate rows, lookup, and scores must match")
    if code_matrix.ndim != 2 or books.ndim != 3:
        raise ValueError("RPQ codes/codebooks have invalid rank")
    if code_matrix.shape[1] != books.shape[0] or books.shape[1] != 256:
        raise ValueError("RPQ code width or codebook size changed")
    rank = books.shape[0] * books.shape[2]
    if projection.shape[1] != rank or len(query_matrix) != len(rows):
        raise ValueError("RPQ basis/query dimensions disagree")
    projected_queries = query_matrix @ projection
    output = base.copy()
    subquantizers = books.shape[0]
    block_dimension = books.shape[2]
    for query_index in range(len(rows)):
        selected = _stable_top_b(base[query_index], rows[query_index], top_b)
        selected_lookup = lookup[query_index, selected]
        valid = selected_lookup >= 0
        if not np.any(valid):
            continue
        if np.max(selected_lookup[valid]) >= len(code_matrix):
            raise ValueError("Residual lookup is outside the RPQ code matrix")
        local_codes = code_matrix[selected_lookup[valid]]
        query_blocks = projected_queries[query_index].reshape(
            subquantizers, block_dimension
        )
        lookup_table = np.einsum("mb,mkb->mk", query_blocks, books)
        correction = np.zeros(len(local_codes), dtype=np.float32)
        for block in range(subquantizers):
            correction += lookup_table[block, local_codes[:, block]]
        output[query_index, selected[valid]] += alpha * correction
    return output


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
    first = np.asarray(treatment, dtype=np.float64)
    second = np.asarray(baseline, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or not len(first):
        raise ValueError("Paired metric vectors must match and be non-empty")
    if not 0 < confidence < 1 or bootstrap_replicates <= 0:
        raise ValueError("Invalid paired-inference controls")
    difference = first - second
    rng = np.random.default_rng(bootstrap_seed)
    means = np.empty(bootstrap_replicates, dtype=np.float64)
    for start in range(0, bootstrap_replicates, 2048):
        end = min(start + 2048, bootstrap_replicates)
        draws = rng.integers(0, len(difference), size=(end - start, len(difference)))
        means[start:end] = difference[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    observed = float(np.mean(difference))
    rng = np.random.default_rng(randomization_seed)
    exceedances = 0
    for start in range(0, randomization_replicates, 4096):
        end = min(start + 4096, randomization_replicates)
        signs = rng.integers(0, 2, size=(end - start, len(difference)), dtype=np.int8)
        randomized = (2.0 * signs.astype(np.float64) - 1.0) * difference
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


def rank_rate_decision(
    *,
    rank64_fp32_vs_pca: dict[str, Any],
    rank64_rpq_vs_pca: dict[str, Any],
    rank64_rpq_vs_base: dict[str, Any],
    fold_gains_over_pca: Any,
    gap_recovery: float,
    pca_mrr: float,
    rpq_mrr: float,
    pca_ndcg: float,
    rpq_ndcg: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    folds = np.asarray(fold_gains_over_pca, dtype=np.float64)
    if folds.ndim != 1 or not len(folds) or np.any(~np.isfinite(folds)):
        raise ValueError("Fold gains must be a finite vector")
    fp32_gain = float(rank64_fp32_vs_pca["mean_difference"])
    rpq_gain = float(rank64_rpq_vs_pca["mean_difference"])
    retention = rpq_gain / fp32_gain if fp32_gain > 0 else 0.0
    capacity_gates = {
        "minimum_rank64_fp32_gain": fp32_gain
        >= float(thresholds["minimum_rank64_fp32_gain_over_pca"]),
        "rank64_fp32_bootstrap_lower_positive": rank64_fp32_vs_pca["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "rank64_fp32_randomization_significant": rank64_fp32_vs_pca[
            "randomization_p_value_one_sided"
        ] <= float(thresholds["maximum_randomization_p_value"]),
        "rank64_fp32_query_support": rank64_fp32_vs_pca["improved_queries"]
        >= int(thresholds["minimum_improved_queries"]),
        "rank64_fp32_net_support": (
            rank64_fp32_vs_pca["improved_queries"]
            - rank64_fp32_vs_pca["harmed_queries"]
        ) >= int(thresholds["minimum_net_improved_queries"]),
    }
    encoding_gates = {
        "minimum_rank64_rpq_gain": rpq_gain
        >= float(thresholds["minimum_rank64_rpq_gain_over_pca"]),
        "rank64_rpq_bootstrap_lower_positive": rank64_rpq_vs_pca["lower"]
        > float(thresholds["bootstrap_lower_must_exceed"]),
        "rank64_rpq_randomization_significant": rank64_rpq_vs_pca[
            "randomization_p_value_one_sided"
        ] <= float(thresholds["maximum_randomization_p_value"]),
        "rank64_rpq_query_support": rank64_rpq_vs_pca["improved_queries"]
        >= int(thresholds["minimum_improved_queries"]),
        "rank64_rpq_net_support": (
            rank64_rpq_vs_pca["improved_queries"]
            - rank64_rpq_vs_pca["harmed_queries"]
        ) >= int(thresholds["minimum_net_improved_queries"]),
        "minimum_headroom_retention": retention
        >= float(thresholds["minimum_rank64_headroom_retention_fraction"]),
        "minimum_gain_over_base": rank64_rpq_vs_base["mean_difference"]
        >= float(thresholds["minimum_rank64_rpq_gain_over_base"]),
        "worst_fold_nonnegative": float(np.min(folds))
        >= float(thresholds["minimum_worst_fold_gain_over_pca"]),
        "minimum_gap_recovery": gap_recovery
        >= float(thresholds["minimum_candidate_gap_recovery_fraction"]),
        "mrr_guardrail": rpq_mrr - pca_mrr
        >= float(thresholds["minimum_mrr_change_vs_pca"]),
        "ndcg_guardrail": rpq_ndcg - pca_ndcg
        >= float(thresholds["minimum_ndcg_change_vs_pca"]),
    }
    capacity_pass = all(capacity_gates.values())
    encoding_pass = all(encoding_gates.values())
    if not capacity_pass:
        decision = thresholds["stop_no_rank_headroom_decision"]
    elif not encoding_pass:
        decision = thresholds["stop_rpq_encoding_decision"]
    else:
        decision = thresholds["go_decision"]
    all_gates = {**capacity_gates, **encoding_gates}
    return {
        "protocol_id": PROTOCOL_ID,
        "decision": decision,
        "capacity_gates": {name: bool(value) for name, value in capacity_gates.items()},
        "encoding_gates": {name: bool(value) for name, value in encoding_gates.items()},
        "failed_gates": [name for name, value in all_gates.items() if not value],
        "rank64_headroom_retention_fraction": float(retention),
        "cutoff_training_performed": False,
        "old_holdout_reuse_authorized": False,
        "fresh_confirmation_access_authorized": False,
    }
