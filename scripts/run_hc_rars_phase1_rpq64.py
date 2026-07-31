#!/usr/bin/env python3
"""Run the HC-RARS Phase-1 PCA64-RPQ-16B development baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.hc_rars_phase1_core import (
    build_query_lut,
    decode_codes,
    encode_residuals,
    fit_rpq_codec,
    project_residuals,
    score_codes_with_lut,
    validate_codec,
)


REQUIRED_ARRAYS = (
    "query_vectors.float32.npy",
    "fold_ids.int64.npy",
    "ann_rows.int64.npy",
    "ann_scores.float32.npy",
    "ann_residual_rows.int64.npy",
    "candidate_doc_rows.int64.npy",
    "candidate_residuals.float32.npy",
    "candidate_relevance.uint8.npy",
    "relevant_counts.int32.npy",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_bundle(root: Path) -> dict[str, np.ndarray]:
    missing = [name for name in REQUIRED_ARRAYS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Bundle is incomplete; missing: " + ", ".join(missing)
        )

    arrays = {
        name: np.load(root / name, mmap_mode="r", allow_pickle=False)
        for name in REQUIRED_ARRAYS
    }

    queries = arrays["query_vectors.float32.npy"]
    rows = arrays["ann_rows.int64.npy"]
    scores = arrays["ann_scores.float32.npy"]
    lookup = arrays["ann_residual_rows.int64.npy"]
    candidate_rows = arrays["candidate_doc_rows.int64.npy"]
    residuals = arrays["candidate_residuals.float32.npy"]
    labels = arrays["candidate_relevance.uint8.npy"]
    relevant_counts = arrays["relevant_counts.int32.npy"]
    folds = arrays["fold_ids.int64.npy"]

    query_count = len(queries)

    if queries.shape != (query_count, 384) or queries.dtype != np.float32:
        raise ValueError("query_vectors must be float32[Q, 384]")
    if rows.shape != scores.shape or rows.shape != lookup.shape:
        raise ValueError("candidate rows, scores and lookup shapes differ")
    if labels.shape != rows.shape:
        raise ValueError("candidate relevance shape differs from candidate rows")
    if residuals.ndim != 2 or residuals.shape[1] != 384:
        raise ValueError("candidate_residuals must have shape [N, 384]")
    if residuals.dtype != np.float32:
        raise ValueError("candidate_residuals must use float32")
    if candidate_rows.shape != (len(residuals),):
        raise ValueError("candidate_doc_rows length differs from residual count")
    if candidate_rows.dtype != np.int64:
        raise ValueError("candidate_doc_rows must use int64")
    if relevant_counts.shape != (query_count,):
        raise ValueError("relevant_counts must have shape [Q]")
    if folds.shape != (query_count,):
        raise ValueError("fold_ids must have shape [Q]")
    if np.any(lookup < 0) or np.any(lookup >= len(residuals)):
        raise ValueError("residual lookup contains out-of-range values")
    if not np.array_equal(np.asarray(candidate_rows)[lookup], np.asarray(rows)):
        raise ValueError("candidate rows are inconsistent with residual lookup")
    if np.any(~np.isfinite(queries)):
        raise ValueError("query vectors contain non-finite values")
    if np.any(~np.isfinite(scores)):
        raise ValueError("base scores contain non-finite values")
    if np.any(~np.isfinite(residuals)):
        raise ValueError("candidate residuals contain non-finite values")

    return arrays


def score_candidate_union_codes(
    *,
    queries: np.ndarray,
    residual_lookup: np.ndarray,
    base_scores: np.ndarray,
    candidate_codes: np.ndarray,
    codec: Any,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    """Score candidate-union codes using local residual lookup indices."""
    if residual_lookup.shape != base_scores.shape:
        raise ValueError("lookup and base score shapes differ")
    if len(queries) != len(base_scores):
        raise ValueError("query and score counts differ")
    if candidate_codes.dtype != np.uint8:
        raise ValueError("candidate codes must use uint8")
    if not 0 <= top_b <= base_scores.shape[1]:
        raise ValueError("top_b is outside candidate depth")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")

    output = np.asarray(base_scores, dtype=np.float32).copy()
    projected_queries = np.asarray(queries, dtype=np.float32) @ codec.basis

    for query_index in range(len(output)):
        valid_positions = np.flatnonzero(residual_lookup[query_index] >= 0)
        if len(valid_positions) == 0 or top_b == 0:
            continue

        order = np.lexsort(
            (
                residual_lookup[query_index, valid_positions],
                -base_scores[query_index, valid_positions],
            )
        )
        selected = valid_positions[order[:top_b]]
        local_rows = residual_lookup[query_index, selected]

        lut = build_query_lut(
            projected_queries[query_index],
            codec.codebooks,
        )
        correction = score_codes_with_lut(
            candidate_codes[local_rows],
            lut,
        )
        output[query_index, selected] += np.float32(alpha) * correction

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top-b", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_bundle(args.bundle_root)

    residuals = np.asarray(
        arrays["candidate_residuals.float32.npy"],
        dtype=np.float32,
    )
    queries = np.asarray(
        arrays["query_vectors.float32.npy"],
        dtype=np.float32,
    )
    base_scores = np.asarray(
        arrays["ann_scores.float32.npy"],
        dtype=np.float32,
    )
    lookup = np.asarray(
        arrays["ann_residual_rows.int64.npy"],
        dtype=np.int64,
    )

    codec = fit_rpq_codec(
        residuals,
        rank=64,
        block_count=16,
        centroid_count=256,
        seed=args.seed,
        max_iterations=args.iterations,
    )
    validate_codec(codec, expected_payload_bytes=16)

    coefficients = project_residuals(residuals, codec.basis)
    codes = encode_residuals(residuals, codec)

    if codes.dtype != np.uint8:
        raise AssertionError("RPQ code dtype is not uint8")
    if codes.shape != (len(residuals), 16):
        raise AssertionError("RPQ payload is not exactly 16 bytes per candidate")

    reconstructed = decode_codes(codes, codec.codebooks)

    candidate_scores = score_candidate_union_codes(
        queries=queries,
        residual_lookup=lookup,
        base_scores=base_scores,
        candidate_codes=codes,
        codec=codec,
        alpha=args.alpha,
        top_b=args.top_b,
    )

    artifacts = {
        "pca_basis_rank64.float32.npy": np.asarray(
            codec.basis,
            dtype=np.float32,
        ),
        "rpq_codebooks.float32.npy": np.asarray(
            codec.codebooks,
            dtype=np.float32,
        ),
        "candidate_codes.uint8.npy": codes,
        "candidate_scores.float32.npy": np.asarray(
            candidate_scores,
            dtype=np.float32,
        ),
    }

    for name, value in artifacts.items():
        np.save(args.output_dir / name, value)

    manifest = {
        "schema_version": 1,
        "status": "HC_RARS_PHASE1_RPQ64_16B_COMPLETE",
        "evaluation_role": "development_only",
        "bundle_root": str(args.bundle_root.resolve()),
        "query_count": int(len(queries)),
        "candidate_union_count": int(len(residuals)),
        "dimension": 384,
        "rank": 64,
        "block_count": 16,
        "subvector_dimension": 4,
        "centroid_count": 256,
        "payload_dtype": "uint8",
        "payload_bytes_per_candidate": 16,
        "seed": int(args.seed),
        "iterations": int(args.iterations),
        "alpha": float(args.alpha),
        "top_b": int(args.top_b),
        "coefficient_reconstruction_mse": float(
            np.mean(
                (
                    np.asarray(coefficients, dtype=np.float32)
                    - np.asarray(reconstructed, dtype=np.float32)
                )
                ** 2
            )
        ),
        "artifacts": {},
    }

    for name in artifacts:
        path = args.output_dir / name
        manifest["artifacts"][name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
