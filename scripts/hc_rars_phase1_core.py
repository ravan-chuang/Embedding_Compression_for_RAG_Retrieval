#!/usr/bin/env python3
"""Numerical core for HC-RARS Phase 1: rank-64, 16-byte residual PQ.

The module is intentionally filesystem-free. Training and evaluation drivers
may import it, while protocol tests can validate the codec independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL_ID = "hc_rars_phase1_rpq64_16b_v1"


@dataclass(frozen=True)
class RPQCodec:
    """A block product codec over projected residual coefficients."""

    basis: np.ndarray          # [ambient_dim, rank], float32
    codebooks: np.ndarray      # [blocks, centroids, block_dim], float32

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def block_count(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def centroids_per_block(self) -> int:
        return int(self.codebooks.shape[1])

    @property
    def block_dimension(self) -> int:
        return int(self.codebooks.shape[2])

    @property
    def payload_bytes_per_document(self) -> int:
        if self.centroids_per_block > 256:
            raise ValueError("uint8 codes support at most 256 centroids per block")
        return self.block_count


def _matrix(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _orient_columns(basis: np.ndarray) -> np.ndarray:
    output = np.asarray(basis, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


def fit_uncentred_pca_basis(residuals: Any, *, rank: int) -> np.ndarray:
    """Fit a deterministic uncentred PCA basis on training residuals only."""
    values = _matrix(residuals, name="residuals", dtype=np.float64)
    if not 0 < rank <= min(values.shape):
        raise ValueError("rank is outside the residual matrix dimensions")
    covariance = values.T @ values
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")[:rank]
    return _orient_columns(eigenvectors[:, order]).astype(np.float32)


def project_residuals(residuals: Any, basis: Any) -> np.ndarray:
    values = _matrix(residuals, name="residuals", dtype=np.float32)
    projection = _matrix(basis, name="basis", dtype=np.float32)
    if values.shape[1] != projection.shape[0]:
        raise ValueError("residual and basis dimensions disagree")
    return values @ projection


def _squared_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # (x-c)^2 = x^2 + c^2 - 2xc; clamp roundoff below zero.
    distances = (
        np.sum(values * values, axis=1, keepdims=True)
        + np.sum(centroids * centroids, axis=1)[None, :]
        - 2.0 * (values @ centroids.T)
    )
    return np.maximum(distances, 0.0)


def _deterministic_kmeans_plus_plus(
    values: np.ndarray, *, centroid_count: int, rng: np.random.Generator
) -> np.ndarray:
    n_rows = len(values)
    if centroid_count > n_rows:
        raise ValueError("centroid_count cannot exceed the number of training rows")
    first = int(rng.integers(0, n_rows))
    indices = [first]
    closest = _squared_distances(values, values[[first]])[:, 0]
    for _ in range(1, centroid_count):
        total = float(np.sum(closest))
        if total <= 0.0:
            unused = np.setdiff1d(np.arange(n_rows), np.asarray(indices), assume_unique=False)
            indices.append(int(unused[0]))
        else:
            threshold = float(rng.random()) * total
            next_index = int(np.searchsorted(np.cumsum(closest), threshold, side="right"))
            next_index = min(next_index, n_rows - 1)
            if next_index in indices:
                unused = np.setdiff1d(np.arange(n_rows), np.asarray(indices), assume_unique=False)
                next_index = int(unused[0])
            indices.append(next_index)
        candidate = _squared_distances(values, values[[indices[-1]]])[:, 0]
        closest = np.minimum(closest, candidate)
    return values[np.asarray(indices)].copy()


def fit_kmeans_codebook(
    values: Any,
    *,
    centroid_count: int = 256,
    seed: int = 20260731,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Fit deterministic Lloyd k-means for one RPQ block."""
    matrix = _matrix(values, name="values", dtype=np.float32)
    if centroid_count <= 1 or centroid_count > 256:
        raise ValueError("centroid_count must be in [2, 256]")
    if max_iterations <= 0 or tolerance < 0:
        raise ValueError("invalid k-means controls")
    rng = np.random.default_rng(int(seed))
    centroids = _deterministic_kmeans_plus_plus(
        matrix, centroid_count=centroid_count, rng=rng
    ).astype(np.float32)
    previous_inertia = np.inf
    for _ in range(max_iterations):
        distances = _squared_distances(matrix, centroids)
        assignments = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(len(matrix)), assignments]))
        updated = centroids.copy()
        nearest_error = distances[np.arange(len(matrix)), assignments]
        rescue_order = np.argsort(-nearest_error, kind="stable")
        rescue_cursor = 0
        for centroid_index in range(centroid_count):
            members = matrix[assignments == centroid_index]
            if len(members):
                updated[centroid_index] = np.mean(members, axis=0)
            else:
                updated[centroid_index] = matrix[rescue_order[rescue_cursor]]
                rescue_cursor += 1
        centroids = updated.astype(np.float32)
        relative = abs(previous_inertia - inertia) / max(1.0, abs(previous_inertia))
        if relative <= tolerance:
            break
        previous_inertia = inertia
    return centroids


def fit_rpq_codec(
    residuals: Any,
    *,
    rank: int = 64,
    block_count: int = 16,
    centroid_count: int = 256,
    seed: int = 20260731,
    max_iterations: int = 50,
) -> RPQCodec:
    """Fit PCA then one independent 8-bit codebook per coefficient block."""
    values = _matrix(residuals, name="residuals", dtype=np.float32)
    if rank % block_count:
        raise ValueError("rank must be divisible by block_count")
    block_dimension = rank // block_count
    basis = fit_uncentred_pca_basis(values, rank=rank)
    coefficients = project_residuals(values, basis)
    codebooks = np.empty(
        (block_count, centroid_count, block_dimension), dtype=np.float32
    )
    for block in range(block_count):
        start = block * block_dimension
        stop = start + block_dimension
        codebooks[block] = fit_kmeans_codebook(
            coefficients[:, start:stop],
            centroid_count=centroid_count,
            seed=int(seed) + block,
            max_iterations=max_iterations,
        )
    codec = RPQCodec(basis=basis, codebooks=codebooks)
    validate_codec(codec, expected_payload_bytes=block_count)
    return codec


def validate_codec(codec: RPQCodec, *, expected_payload_bytes: int = 16) -> None:
    basis = _matrix(codec.basis, name="basis", dtype=np.float32)
    codebooks = np.asarray(codec.codebooks, dtype=np.float32)
    if codebooks.ndim != 3 or not np.all(np.isfinite(codebooks)):
        raise ValueError("codebooks must be a finite 3D tensor")
    if basis.shape[1] != codebooks.shape[0] * codebooks.shape[2]:
        raise ValueError("basis rank and codebook block dimensions disagree")
    if codebooks.shape[1] > 256:
        raise ValueError("uint8 RPQ supports at most 256 centroids")
    if codec.payload_bytes_per_document != expected_payload_bytes:
        raise ValueError("codec violates the exact payload budget")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), atol=2e-4, rtol=2e-4):
        raise ValueError("basis columns are not orthonormal")


def encode_projected(coefficients: Any, codebooks: Any) -> np.ndarray:
    values = _matrix(coefficients, name="coefficients", dtype=np.float32)
    books = np.asarray(codebooks, dtype=np.float32)
    if books.ndim != 3 or not np.all(np.isfinite(books)):
        raise ValueError("codebooks must be a finite 3D tensor")
    block_count, centroid_count, block_dimension = books.shape
    if values.shape[1] != block_count * block_dimension:
        raise ValueError("coefficient rank and codebooks disagree")
    if centroid_count > 256:
        raise ValueError("centroid count exceeds uint8 capacity")
    codes = np.empty((len(values), block_count), dtype=np.uint8)
    for block in range(block_count):
        start = block * block_dimension
        stop = start + block_dimension
        distances = _squared_distances(values[:, start:stop], books[block])
        codes[:, block] = np.argmin(distances, axis=1).astype(np.uint8)
    return codes


def encode_residuals(residuals: Any, codec: RPQCodec) -> np.ndarray:
    validate_codec(codec, expected_payload_bytes=codec.block_count)
    return encode_projected(project_residuals(residuals, codec.basis), codec.codebooks)


def decode_codes(codes: Any, codebooks: Any) -> np.ndarray:
    encoded = np.asarray(codes)
    books = np.asarray(codebooks, dtype=np.float32)
    if encoded.ndim != 2 or encoded.dtype != np.uint8:
        raise ValueError("codes must be a uint8 matrix")
    if books.ndim != 3 or encoded.shape[1] != books.shape[0]:
        raise ValueError("codes and codebooks disagree")
    output = np.empty(
        (len(encoded), books.shape[0] * books.shape[2]), dtype=np.float32
    )
    for block in range(books.shape[0]):
        start = block * books.shape[2]
        stop = start + books.shape[2]
        output[:, start:stop] = books[block, encoded[:, block]]
    return output


def build_query_lut(projected_query: Any, codebooks: Any) -> np.ndarray:
    query = np.asarray(projected_query, dtype=np.float32)
    books = np.asarray(codebooks, dtype=np.float32)
    if query.ndim != 1 or books.ndim != 3:
        raise ValueError("query must be a vector and codebooks a 3D tensor")
    block_count, _, block_dimension = books.shape
    if query.shape != (block_count * block_dimension,):
        raise ValueError("projected query and codebooks disagree")
    lut = np.empty((block_count, books.shape[1]), dtype=np.float32)
    for block in range(block_count):
        start = block * block_dimension
        stop = start + block_dimension
        lut[block] = books[block] @ query[start:stop]
    return lut


def score_codes_with_lut(codes: Any, lut: Any) -> np.ndarray:
    encoded = np.asarray(codes)
    table = np.asarray(lut, dtype=np.float32)
    if encoded.ndim != 2 or encoded.dtype != np.uint8:
        raise ValueError("codes must be a uint8 matrix")
    if table.ndim != 2 or encoded.shape[1] != table.shape[0]:
        raise ValueError("codes and LUT disagree")
    rows = np.arange(encoded.shape[1])[:, None]
    return np.sum(table[rows, encoded.T], axis=0, dtype=np.float32)


def score_rpq_candidates(
    queries: Any,
    candidate_rows: Any,
    base_scores: Any,
    document_codes: Any,
    codec: RPQCodec,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    """Correct only the frozen Base Top-B candidates using RPQ lookup tables."""
    query_matrix = _matrix(queries, name="queries", dtype=np.float32)
    rows = np.asarray(candidate_rows, dtype=np.int64)
    base = _matrix(base_scores, name="base_scores", dtype=np.float32)
    codes = np.asarray(document_codes)
    validate_codec(codec, expected_payload_bytes=codec.block_count)
    if rows.ndim != 2 or rows.shape != base.shape or len(rows) != len(query_matrix):
        raise ValueError("candidate rows and base scores must match queries")
    if codes.ndim != 2 or codes.dtype != np.uint8 or codes.shape[1] != codec.block_count:
        raise ValueError("document_codes violate the RPQ contract")
    if query_matrix.shape[1] != codec.basis.shape[0]:
        raise ValueError("query and basis dimensions disagree")
    if not 0 <= top_b <= rows.shape[1] or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("invalid top_b or alpha")
    output = base.copy()
    projected_queries = query_matrix @ codec.basis
    for query_index in range(len(rows)):
        valid_positions = np.flatnonzero(rows[query_index] >= 0)
        if not len(valid_positions) or top_b == 0:
            continue
        # Stable descending base score, then ascending document row.
        order = np.lexsort((rows[query_index, valid_positions], -base[query_index, valid_positions]))
        selected = valid_positions[order[:top_b]]
        document_rows = rows[query_index, selected]
        if np.max(document_rows) >= len(codes):
            raise ValueError("candidate row is outside document_codes")
        lut = build_query_lut(projected_queries[query_index], codec.codebooks)
        correction = score_codes_with_lut(codes[document_rows], lut)
        output[query_index, selected] += np.float32(alpha) * correction
    return output
