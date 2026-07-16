#!/usr/bin/env python3
"""Pure NumPy contracts for the RARS-v2.2 development experiment.

This module intentionally has no Torch or Faiss dependency.  It owns the
versioned bundle contract, PCA warm start, FP32 scorer, dynamic boundary miner,
and canonical run identity used by the GPU trainer and its unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v2_2_boundary_loss_development_v1"
FIT_ROLE_ID = "inner_train"
SELECTION_ROLE_ID = "inner_validation"
FORBIDDEN_ROLE_MARKERS = ("outer", "test", "evaluation", "posthoc", "sealed")
PROMOTION = np.uint8(0)
PROTECTION = np.uint8(1)


@dataclass(frozen=True)
class BoundaryPairBatch:
    """Dynamically mined relevance pairs and macro-query weights."""

    query: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    kind: np.ndarray
    weight: np.ndarray
    target_margin: np.ndarray

    def __len__(self) -> int:
        return int(len(self.query))

    @property
    def promotion_count(self) -> int:
        return int(np.sum(self.kind == PROMOTION))

    @property
    def protection_count(self) -> int:
        return int(np.sum(self.kind == PROTECTION))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_bundle_manifest(
    manifest: dict[str, Any], *, expected_role_id: str
) -> None:
    """Fail closed unless a bundle is an explicitly frozen inner role."""

    if expected_role_id not in {FIT_ROLE_ID, SELECTION_ROLE_ID}:
        raise ValueError(f"Unsupported v2.2 role request: {expected_role_id!r}")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Bundle does not use the frozen v2.2 protocol")
    role_id = str(manifest.get("role_id", "")).casefold()
    if role_id != expected_role_id:
        raise ValueError(
            f"Expected role_id={expected_role_id!r}, found {role_id!r}"
        )
    marker = next(
        (value for value in FORBIDDEN_ROLE_MARKERS if value in role_id), None
    )
    if marker is not None:
        raise ValueError(f"Development trainer forbids role marker {marker!r}")
    expected_split_role = "train" if role_id == FIT_ROLE_ID else "validation"
    if manifest.get("split_role") != expected_split_role:
        raise ValueError("role_id and split_role disagree")
    if manifest.get("evidence_status") != "DEVELOPMENT_ONLY":
        raise ValueError("Inner bundle must be labeled DEVELOPMENT_ONLY")
    for field in (
        "query_ids_sha256",
        "query_rows_sha256",
        "split_audit_sha256",
        "source_bundle_manifest_sha256",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Bundle manifest lacks a valid {field}")
    access = manifest.get("data_access")
    if not isinstance(access, dict):
        raise ValueError("Bundle manifest lacks explicit data_access")
    if access.get("outer_outcomes_used") is not False:
        raise ValueError("Outer outcomes must not enter v2.2 development")
    if access.get("closed_test_relevance_values_used") is not False:
        raise ValueError("Closed-test relevance values must not enter development")


def load_pca_warm_start(
    basis_path: Path,
    config_path: Path,
    *,
    dimension: int,
    rank: int,
    top_b: int,
    orthogonality_atol: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return Wq, Wd whose unbounded score equals the PCA correction."""

    basis = np.asarray(np.load(basis_path), dtype=np.float32)
    config = read_json(config_path)
    if basis.shape != (dimension, rank):
        raise ValueError(
            f"PCA basis must have shape {(dimension, rank)}, got {basis.shape}"
        )
    if not np.all(np.isfinite(basis)):
        raise ValueError("PCA basis contains non-finite values")
    if int(config.get("rank", -1)) != rank:
        raise ValueError("PCA config rank does not match the v2.2 rank")
    if int(config.get("top_b", -1)) != top_b:
        raise ValueError("PCA config Top-B does not match v2.2")
    alpha = float(config.get("alpha", float("nan")))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("PCA config alpha must be finite and positive")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(rank), atol=orthogonality_atol, rtol=0.0):
        raise ValueError("PCA basis is not orthonormal within tolerance")
    query_projection = (alpha * basis).astype(np.float32)
    document_projection = basis.astype(np.float32, copy=True)
    return query_projection, document_projection, alpha


def _validate_score_arrays(arrays: dict[str, np.ndarray]) -> tuple[int, int, int]:
    queries = np.asarray(arrays["queries"])
    scores = np.asarray(arrays["ann_scores"])
    lookup = np.asarray(arrays["residual_lookup"])
    residuals = np.asarray(arrays["residuals"])
    if queries.ndim != 2 or scores.ndim != 2 or lookup.shape != scores.shape:
        raise ValueError("Invalid query, score, or residual-lookup shape")
    if residuals.ndim != 2 or residuals.shape[1] != queries.shape[1]:
        raise ValueError("Residual dimension does not match query dimension")
    if scores.shape[0] != queries.shape[0]:
        raise ValueError("Query and candidate counts disagree")
    return scores.shape[0], scores.shape[1], queries.shape[1]


def score_candidates_fp32(
    arrays: dict[str, np.ndarray],
    query_projection: np.ndarray,
    document_projection: np.ndarray,
    *,
    top_b: int,
    max_correction: float | None,
    batch_size: int = 256,
) -> np.ndarray:
    """Apply the no-gate v2.2 scorer only to frozen ANN positions < Top-B."""

    query_count, candidate_count, dimension = _validate_score_arrays(arrays)
    query_matrix = np.asarray(query_projection, dtype=np.float32)
    document_matrix = np.asarray(document_projection, dtype=np.float32)
    if query_matrix.ndim != 2 or document_matrix.shape != query_matrix.shape:
        raise ValueError("Query and document projections must have matching shape")
    if query_matrix.shape[0] != dimension:
        raise ValueError("Projection dimension does not match the bundle")
    if not 0 < top_b <= candidate_count or batch_size <= 0:
        raise ValueError("Invalid Top-B or batch size")
    if max_correction is not None and max_correction <= 0:
        raise ValueError("max_correction must be positive or None")
    result = np.asarray(arrays["ann_scores"], dtype=np.float32).copy()
    lookup = np.asarray(arrays["residual_lookup"], dtype=np.int64)
    if np.any(lookup[:, :top_b] < 0):
        raise ValueError("Correctable candidates have invalid residual rows")
    residuals = arrays["residuals"]
    for start in range(0, query_count, batch_size):
        end = min(start + batch_size, query_count)
        q_projected = (
            np.asarray(arrays["queries"][start:end], dtype=np.float32)
            @ query_matrix
        )
        selected = np.asarray(lookup[start:end, :top_b], dtype=np.int64)
        residual_projected = np.einsum(
            "qcd,dr->qcr",
            np.asarray(residuals[selected], dtype=np.float32),
            document_matrix,
        )
        raw = np.einsum("qr,qcr->qc", q_projected, residual_projected)
        if max_correction is not None:
            raw = max_correction * np.tanh(raw / max_correction)
        result[start:end, :top_b] += raw.astype(np.float32)
    return result


def pca_fp32_scores(
    arrays: dict[str, np.ndarray],
    basis: np.ndarray,
    *,
    alpha: float,
    top_b: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute the storage-matched PCA comparator from FP32 residuals."""

    basis = np.asarray(basis, dtype=np.float32)
    return score_candidates_fp32(
        arrays,
        alpha * basis,
        basis,
        top_b=top_b,
        max_correction=None,
        batch_size=batch_size,
    )


def _macro_query_weights(
    query: np.ndarray,
    kind: np.ndarray,
    *,
    promotion_mix: float,
) -> np.ndarray:
    if not 0 < promotion_mix <= 1:
        raise ValueError("promotion_mix must be in (0, 1]")
    query = np.asarray(query, dtype=np.int64)
    kind = np.asarray(kind, dtype=np.uint8)
    if query.shape != kind.shape:
        raise ValueError("Pair query and kind arrays disagree")
    weights = np.zeros(len(query), dtype=np.float64)
    requested = {int(PROMOTION): promotion_mix, int(PROTECTION): 1 - promotion_mix}
    active = [value for value in requested if np.any(kind == value)]
    active_mass = sum(requested[value] for value in active)
    if active_mass <= 0:
        raise ValueError("No active pair type has positive loss mass")
    for kind_value in active:
        mask = kind == kind_value
        kind_queries, counts = np.unique(query[mask], return_counts=True)
        mass = requested[kind_value] / active_mass
        for query_value, count in zip(kind_queries, counts, strict=True):
            pair_mask = mask & (query == query_value)
            weights[pair_mask] = mass / (len(kind_queries) * int(count))
    weights *= len(weights)
    if not np.isclose(np.mean(weights), 1.0):
        raise AssertionError("Internal pair-weight normalization failed")
    return weights.astype(np.float32)


def mine_dynamic_boundary_pairs(
    labels: np.ndarray,
    current_scores: np.ndarray,
    *,
    final_k: int = 10,
    top_b: int = 40,
    max_negatives_per_positive: int = 8,
    promotion_mix: float = 0.8,
    minimum_margin: float = 1e-4,
    margin_multiplier: float = 1.0,
) -> BoundaryPairBatch:
    """Mine deterministic promotion/protection pairs from current rankings."""

    labels = np.asarray(labels)
    scores = np.asarray(current_scores, dtype=np.float32)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("labels and scores must be matching [Q, C] arrays")
    if not 0 < final_k < top_b <= scores.shape[1]:
        raise ValueError("Require 0 < final_k < top_b <= candidate count")
    if max_negatives_per_positive <= 0:
        raise ValueError("max_negatives_per_positive must be positive")
    if minimum_margin < 0 or margin_multiplier < 0:
        raise ValueError("Margin controls must be non-negative")

    records: list[tuple[int, int, int, int, float]] = []
    correctable = np.arange(scores.shape[1]) < top_b
    for query_index in range(len(labels)):
        order = np.argsort(-scores[query_index], kind="stable")
        top = order[:final_k]
        outside = order[final_k:]
        boundary_gap = abs(
            float(scores[query_index, order[final_k - 1]])
            - float(scores[query_index, order[final_k]])
        )
        target = max(minimum_margin, margin_multiplier * boundary_gap)

        promotion_positive = [
            int(value)
            for value in outside
            if correctable[value] and labels[query_index, value] > 0
        ]
        promotion_negative = [
            int(value) for value in top if labels[query_index, value] <= 0
        ]
        for positive in promotion_positive:
            ranked_negatives = sorted(
                promotion_negative,
                key=lambda negative: (
                    -float(
                        scores[query_index, negative]
                        - scores[query_index, positive]
                    ),
                    negative,
                ),
            )[:max_negatives_per_positive]
            records.extend(
                (query_index, positive, negative, int(PROMOTION), target)
                for negative in ranked_negatives
            )

        protection_positive = [
            int(value) for value in top if labels[query_index, value] > 0
        ]
        protection_negative = [
            int(value)
            for value in outside
            if correctable[value] and labels[query_index, value] <= 0
        ]
        for positive in protection_positive:
            ranked_negatives = sorted(
                protection_negative,
                key=lambda negative: (
                    -float(
                        scores[query_index, negative]
                        - scores[query_index, positive]
                    ),
                    negative,
                ),
            )[:max_negatives_per_positive]
            records.extend(
                (query_index, positive, negative, int(PROTECTION), target)
                for negative in ranked_negatives
            )

    if not records:
        empty_i64 = np.empty(0, dtype=np.int64)
        return BoundaryPairBatch(
            query=empty_i64,
            positive=empty_i64.copy(),
            negative=empty_i64.copy(),
            kind=np.empty(0, dtype=np.uint8),
            weight=np.empty(0, dtype=np.float32),
            target_margin=np.empty(0, dtype=np.float32),
        )
    values = np.asarray(records, dtype=np.float64)
    query = values[:, 0].astype(np.int64)
    positive = values[:, 1].astype(np.int64)
    negative = values[:, 2].astype(np.int64)
    kind = values[:, 3].astype(np.uint8)
    target_margin = values[:, 4].astype(np.float32)
    weight = _macro_query_weights(query, kind, promotion_mix=promotion_mix)
    return BoundaryPairBatch(
        query=query,
        positive=positive,
        negative=negative,
        kind=kind,
        weight=weight,
        target_margin=target_margin,
    )


def build_run_fingerprint(payload: dict[str, Any]) -> str:
    """Return the only identity under which a completed run may be reused."""

    required = {
        "protocol_id",
        "source_commit",
        "trainer_sha256",
        "train_bundle_manifest_sha256",
        "selection_bundle_manifest_sha256",
        "pca_basis_sha256",
        "pca_config_sha256",
        "configuration",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Run fingerprint payload lacks: {', '.join(missing)}")
    if payload["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Run fingerprint uses the wrong protocol")
    source_commit = str(payload["source_commit"])
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be an exact lowercase 40-hex commit")
    return canonical_sha256(payload)
