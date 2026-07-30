#!/usr/bin/env python3
"""Run the frozen fresh-query V13 signed score-distilled RPQ experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v11_rank_rate_core import (  # noqa: E402
    fit_faiss_product_quantizer,
    paired_inference,
    score_product_sidecar_candidates,
)
from rars_v13_signed_score_core import (  # noqa: E402
    PROTOCOL_ID,
    assign_product_codes,
    build_signed_score_statistics,
    deterministic_fold_ids,
    fit_signed_score_codebooks,
    signed_score_decision,
)
from rars_v8_cutoff_sidecar_core import (  # noqa: E402
    candidate_gap_recovery,
    encode_residuals_int8,
    fit_int8_scales,
    fit_uncentered_pca_basis,
    per_query_metrics,
    score_sidecar_candidates,
)
from train_rars_v8_cutoff_sidecar import (  # noqa: E402
    atomic_json,
    atomic_save,
    exact_candidate_scores,
    file_record,
    read_json,
    validate_runtime,
)


CANONICAL_PROTOCOL = Path(
    "protocols/rars_v13_signed_score_distilled_rpq_v1.json"
)
SOURCE_FILES = (
    CANONICAL_PROTOCOL,
    Path("scripts/rars_v13_signed_score_core.py"),
    Path("scripts/freeze_rars_v13_fresh_queries.py"),
    Path("scripts/build_rars_v13_fresh_bundle.py"),
    Path("scripts/train_rars_v13_signed_score_rpq.py"),
    Path("scripts/verify_rars_v13_signed_score_rpq_packet.py"),
    Path("scripts/rars_v11_rank_rate_core.py"),
    Path("scripts/rars_v8_cutoff_sidecar_core.py"),
    Path("scripts/train_rars_v8_cutoff_sidecar.py"),
)
REQUIRED_BUNDLE_ARRAYS = (
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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Registered {label} hash changed")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("--source-commit must be exact lowercase 40-hex")
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must use canonical path: {canonical}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or status:
        raise ValueError("V13 training requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != (
        "FROZEN_BEFORE_FIRST_V13_FRESH_DEVELOPMENT_RUN"
    ):
        raise ValueError("Unexpected V13 protocol identity or status")
    return protocol, {
        str(relative): file_record((repo_root / relative).resolve(strict=True))
        for relative in SOURCE_FILES
    }


def load_bundle(
    root: Path,
    protocol: dict[str, Any],
    repo_root: Path,
    source_commit: str,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = root / "fresh_bundle_manifest.json"
    complete_path = root / "fresh_bundle_complete.json"
    manifest = read_json(manifest_path)
    complete = read_json(complete_path)
    if manifest.get("status") != "RARS_V13_FRESH_DEVELOPMENT_BUNDLE_FROZEN":
        raise ValueError("V13 fresh development bundle is not frozen")
    if complete.get("status") != "RARS_V13_FRESH_BUNDLE_COMPLETE":
        raise ValueError("V13 fresh development bundle is incomplete")
    if manifest.get("source_commit") != source_commit or complete.get(
        "source_commit"
    ) != source_commit:
        raise ValueError("V13 fresh bundle source commit changed")
    if manifest.get("role_id") != "fresh_train_development":
        raise ValueError("V13 refuses any non-fresh development role")
    if manifest.get("metrics_computed") is not False:
        raise ValueError("V13 bundle was not frozen before metrics")
    _verify_record(manifest_path, complete["manifest"], "bundle manifest")
    for relative, record in manifest.get("source_blobs", {}).items():
        _verify_record(repo_root / relative, record, f"bundle source blob {relative}")
    for name in (*REQUIRED_BUNDLE_ARRAYS, "query_ids.utf8.txt"):
        record = manifest.get("files", {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Bundle manifest does not register {name}")
        _verify_record(root / name, record, f"bundle file {name}")
    qids = (root / "query_ids.utf8.txt").read_text(encoding="utf-8").splitlines()
    target = int(protocol["fresh_query_freeze"]["target_query_count"])
    if len(qids) != target or len(qids) != len(set(qids)):
        raise ValueError("Fresh V13 qids changed count or uniqueness")
    arrays = {
        name: np.load(root / name, mmap_mode="r", allow_pickle=False)
        for name in REQUIRED_BUNDLE_ARRAYS
    }
    folds = np.asarray(arrays["fold_ids.int64.npy"], dtype=np.int64)
    if not np.array_equal(folds, deterministic_fold_ids(qids)):
        raise ValueError("Fresh V13 fold ids changed")
    queries = arrays["query_vectors.float32.npy"]
    rows = arrays["ann_rows.int64.npy"]
    scores = arrays["ann_scores.float32.npy"]
    lookup = arrays["ann_residual_rows.int64.npy"]
    labels = arrays["candidate_relevance.uint8.npy"]
    candidate_rows = arrays["candidate_doc_rows.int64.npy"]
    residuals = arrays["candidate_residuals.float32.npy"]
    relevant_counts = arrays["relevant_counts.int32.npy"]
    expected_shape = (target, int(protocol["frozen_index_contract"]["candidate_pool"]))
    if queries.shape != (target, 384) or queries.dtype != np.float32:
        raise ValueError("Fresh query array contract changed")
    expected_dtypes = {
        "rows": np.dtype(np.int64),
        "scores": np.dtype(np.float32),
        "lookup": np.dtype(np.int64),
        "labels": np.dtype(np.uint8),
    }
    for name, value in (("rows", rows), ("scores", scores), ("lookup", lookup), ("labels", labels)):
        if value.shape != expected_shape:
            raise ValueError(f"Fresh candidate {name} shape changed")
        if value.dtype != expected_dtypes[name]:
            raise ValueError(f"Fresh candidate {name} dtype changed")
    if residuals.shape != (len(candidate_rows), 384) or residuals.dtype != np.float32:
        raise ValueError("Fresh candidate residual shape changed")
    if candidate_rows.dtype != np.int64 or not np.array_equal(
        candidate_rows, np.unique(candidate_rows)
    ):
        raise ValueError("Fresh candidate union is not sorted and unique")
    if relevant_counts.shape != (target,) or relevant_counts.dtype != np.int32 or np.any(relevant_counts <= 0):
        raise ValueError("Fresh relevant-count denominator changed")
    if np.any(~np.isfinite(queries)) or np.any(~np.isfinite(scores)) or np.any(~np.isfinite(residuals)):
        raise ValueError("Fresh bundle contains non-finite floating-point values")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("Fresh relevance labels must be binary")
    if np.any(lookup < 0) or np.any(lookup >= len(candidate_rows)):
        raise ValueError("Fresh residual lookup is outside the union")
    if not np.array_equal(np.asarray(candidate_rows)[lookup], np.asarray(rows)):
        raise ValueError("Fresh residual lookup no longer reproduces candidates")
    return qids, folds, arrays, {
        "fresh_bundle_manifest": file_record(manifest_path),
        "fresh_bundle_complete": file_record(complete_path),
        "registered_embeddings": manifest["inputs"]["embeddings"],
        "registered_index": manifest["inputs"]["index"],
    }


def _inference_kwargs(protocol: dict[str, Any], comparison: str) -> dict[str, Any]:
    inference = protocol["inference"]
    seeds = inference[comparison]
    return {
        "bootstrap_replicates": int(inference["bootstrap_replicates"]),
        "bootstrap_seed": int(seeds["bootstrap_seed"]),
        "randomization_replicates": int(inference["randomization_replicates"]),
        "randomization_seed": int(seeds["randomization_seed"]),
        "confidence": float(inference["confidence"]),
    }


def _metric_summary(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(array)) for name, array in values.items()}


def _save_metric_arrays(
    output_dir: Path, prefix: str, values: dict[str, np.ndarray]
) -> list[str]:
    names = []
    for metric, array in values.items():
        name = f"per_query_{prefix}_{metric}_at_10.float64.npy"
        atomic_save(output_dir / name, np.asarray(array, dtype=np.float64))
        names.append(name)
    return names


def _materialize_full_codes(
    *,
    embeddings_path: Path,
    index_path: Path,
    output_path: Path,
    basis: np.ndarray,
    codebooks: np.ndarray,
    n_docs: int,
    dimension: int,
    batch_size: int,
) -> dict[str, Any]:
    import faiss

    if output_path.exists():
        raise ValueError("Refusing to overwrite existing full-corpus V13 codes")
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    index = faiss.read_index(str(index_path))
    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    ivf.make_direct_map()
    embeddings = np.memmap(
        embeddings_path,
        dtype=np.float16,
        mode="r",
        shape=(n_docs, dimension),
    )
    codes = np.memmap(
        temporary,
        dtype=np.uint8,
        mode="w+",
        shape=(n_docs, codebooks.shape[0]),
    )
    histograms = np.zeros((codebooks.shape[0], 256), dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        rows = np.arange(start, end, dtype=np.int64)
        reconstructed = np.asarray(
            index.reconstruct_batch(rows), dtype=np.float32
        )
        residuals = np.asarray(embeddings[start:end], dtype=np.float32) - reconstructed
        coefficients = residuals @ basis
        local_codes = assign_product_codes(
            coefficients, codebooks, batch_size=min(batch_size, 8192)
        )
        codes[start:end] = local_codes
        for block in range(codebooks.shape[0]):
            histograms[block] += np.bincount(
                local_codes[:, block], minlength=256
            )
    codes.flush()
    del codes
    os.replace(temporary, output_path)
    expected_bytes = n_docs * codebooks.shape[0]
    if output_path.stat().st_size != expected_bytes:
        raise ValueError("Full-corpus V13 code payload byte count is wrong")
    occupied = np.sum(histograms > 0, axis=1)
    return {
        "document_count": n_docs,
        "code_shape": [n_docs, int(codebooks.shape[0])],
        "dtype": "uint8",
        "payload_bytes": expected_bytes,
        "payload_bytes_per_document": int(codebooks.shape[0]),
        "occupied_centroids_per_block": occupied.tolist(),
        "minimum_occupied_centroids_per_block": int(occupied.min()),
        "maximum_occupied_centroids_per_block": int(occupied.max()),
        "code_histograms": histograms.tolist(),
        "wall_seconds": float(time.perf_counter() - started),
        "record": file_record(output_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the preregistered V13 five-fold development experiment."""

    import faiss

    repo_root = Path(__file__).resolve().parents[1]
    forbidden = (
        "oracle_design",
        "oracle_audit",
        "future_method_holdout",
        "rars-v9",
        "rars-v10",
        "rars-v11",
        "rars-v12",
    )
    for path in (args.bundle_root, args.output_dir):
        if any(token in str(path).lower() for token in forbidden):
            raise ValueError(f"V13 refuses historical role/result path: {path}")
    protocol, source_blobs = validate_source(
        repo_root, args.protocol, args.source_commit
    )
    environment = validate_runtime(protocol)
    expected_faiss = protocol["execution_environment_contract"]["faiss_version"]
    if environment["faiss_version"] != expected_faiss:
        raise ValueError(
            f"faiss={environment['faiss_version']}; expected {expected_faiss}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty V13 development output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qids, folds, arrays, input_records = load_bundle(
        args.bundle_root, protocol, repo_root, args.source_commit
    )
    _verify_record(
        args.embeddings,
        input_records.pop("registered_embeddings"),
        "frozen embeddings",
    )
    _verify_record(
        args.index,
        input_records.pop("registered_index"),
        "frozen index",
    )
    started_path = args.output_dir / "development_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V13_FRESH_DEVELOPMENT_STARTED",
            "source_commit": args.source_commit,
            "environment": environment,
            "source_blobs": source_blobs,
            "inputs": {
                **input_records,
                "embeddings": file_record(args.embeddings),
                "index": file_record(args.index),
            },
            "query_count": len(qids),
            "opened_role": "fresh_train_development",
            "v9_packet_opened": False,
            "v10_packet_opened": False,
            "v11_packet_opened": False,
            "v12_packet_opened": False,
            "old_holdout_opened": False,
        },
    )

    queries = np.asarray(arrays["query_vectors.float32.npy"], dtype=np.float32)
    rows = np.asarray(arrays["ann_rows.int64.npy"], dtype=np.int64)
    base_scores = np.asarray(arrays["ann_scores.float32.npy"], dtype=np.float32)
    lookup = np.asarray(arrays["ann_residual_rows.int64.npy"], dtype=np.int64)
    residuals = np.asarray(
        arrays["candidate_residuals.float32.npy"], dtype=np.float32
    )
    labels = np.asarray(
        arrays["candidate_relevance.uint8.npy"], dtype=np.uint8
    )
    relevant_counts = np.asarray(
        arrays["relevant_counts.int32.npy"], dtype=np.int64
    )
    teacher_scores = exact_candidate_scores(
        queries, base_scores, lookup, residuals
    )
    final_k = int(protocol["frozen_index_contract"]["final_cutoff"])
    method = protocol["method"]
    pca_cfg = protocol["pca16_comparator"]
    score_cfg = protocol["score_distillation"]
    training_cfg = protocol["rpq_training"]
    seeds = [int(value) for value in training_cfg["seeds"]]
    primary_seed = int(training_cfg["primary_seed"])
    primary_index = seeds.index(primary_seed)
    fold_count = int(protocol["cross_validation"]["fold_count"])

    base_metrics = per_query_metrics(
        base_scores, rows, labels, relevant_counts, k=final_k
    )
    exact_metrics = per_query_metrics(
        teacher_scores, rows, labels, relevant_counts, k=final_k
    )
    metric_names = ("recall", "mrr", "ndcg")
    pca16 = {
        name: np.full(len(qids), np.nan, dtype=np.float64)
        for name in metric_names
    }
    unsupervised = {
        name: np.full((len(seeds), len(qids)), np.nan, dtype=np.float64)
        for name in metric_names
    }
    challenger = {
        name: np.full((len(seeds), len(qids)), np.nan, dtype=np.float64)
        for name in metric_names
    }
    fold_seed_diagnostics: list[dict[str, Any]] = []
    fold_pca_diagnostics: list[dict[str, Any]] = []
    maximum_drift = 0.0
    all_objectives_nonincreasing = True
    assignment_changes = 0

    for fold in range(fold_count):
        train_queries = np.flatnonzero(folds != fold)
        heldout_queries = np.flatnonzero(folds == fold)
        training_residual_rows = np.unique(lookup[train_queries].reshape(-1))

        pca_basis = fit_uncentered_pca_basis(
            residuals[training_residual_rows], rank=int(pca_cfg["rank"])
        )
        pca_scales = fit_int8_scales(
            residuals[training_residual_rows], pca_basis
        )
        pca_codes, pca_encoding = encode_residuals_int8(
            residuals, pca_basis, pca_scales
        )
        pca_scores = score_sidecar_candidates(
            queries[heldout_queries],
            rows[heldout_queries],
            lookup[heldout_queries],
            base_scores[heldout_queries],
            pca_basis,
            pca_codes,
            pca_scales,
            alpha=float(pca_cfg["alpha"]),
            top_b=int(pca_cfg["top_b"]),
        )
        pca_metrics = per_query_metrics(
            pca_scores,
            rows[heldout_queries],
            labels[heldout_queries],
            relevant_counts[heldout_queries],
            k=final_k,
        )
        for name in metric_names:
            pca16[name][heldout_queries] = pca_metrics[name]
        fold_pca_diagnostics.append(
            {
                "fold": fold,
                "training_query_count": int(len(train_queries)),
                "heldout_query_count": int(len(heldout_queries)),
                "training_residual_count": int(len(training_residual_rows)),
                "encoding": pca_encoding,
                "metrics": _metric_summary(pca_metrics),
            }
        )

        basis = fit_uncentered_pca_basis(
            residuals[training_residual_rows], rank=int(method["rank"])
        )
        coefficients = np.asarray(residuals @ basis, dtype=np.float32)
        for seed_index, seed in enumerate(seeds):
            _, initial_books, rpq_summary = fit_faiss_product_quantizer(
                coefficients[training_residual_rows],
                faiss,
                subquantizers=int(method["subquantizers"]),
                bits=int(method["bits_per_subquantizer"]),
                iterations=int(training_cfg["iterations"]),
                seed=seed,
                max_points_per_centroid=int(
                    training_cfg["maximum_points_per_centroid"]
                ),
            )
            initial_codes = assign_product_codes(coefficients, initial_books)
            statistics, statistics_summary = build_signed_score_statistics(
                queries[train_queries],
                basis,
                coefficients,
                initial_codes,
                rows[train_queries],
                lookup[train_queries],
                base_scores[train_queries],
                labels[train_queries],
                top_b=int(method["top_b"]),
                final_k=final_k,
                cutoff_boost=float(score_cfg["cutoff_boost"]),
                margin_temperature=float(score_cfg["margin_temperature"]),
                known_positive_multiplier=float(
                    score_cfg["known_positive_multiplier"]
                ),
            )
            updated_books, update_summary = fit_signed_score_codebooks(
                coefficients[training_residual_rows],
                initial_books,
                statistics,
                anchor_ratio=float(score_cfg["anchor_ratio"]),
                maximum_drift_fraction=float(
                    score_cfg[
                        "maximum_centroid_drift_fraction_of_training_block_rms"
                    ]
                ),
                jitter=float(score_cfg["linear_solve_jitter"]),
            )
            baseline_scores = score_product_sidecar_candidates(
                queries[heldout_queries],
                rows[heldout_queries],
                lookup[heldout_queries],
                base_scores[heldout_queries],
                basis,
                initial_codes,
                initial_books,
                alpha=float(method["alpha"]),
                top_b=int(method["top_b"]),
            )
            challenger_scores = score_product_sidecar_candidates(
                queries[heldout_queries],
                rows[heldout_queries],
                lookup[heldout_queries],
                base_scores[heldout_queries],
                basis,
                initial_codes,
                updated_books,
                alpha=float(method["alpha"]),
                top_b=int(method["top_b"]),
            )
            baseline_metrics = per_query_metrics(
                baseline_scores,
                rows[heldout_queries],
                labels[heldout_queries],
                relevant_counts[heldout_queries],
                k=final_k,
            )
            challenger_metrics = per_query_metrics(
                challenger_scores,
                rows[heldout_queries],
                labels[heldout_queries],
                relevant_counts[heldout_queries],
                k=final_k,
            )
            for name in metric_names:
                unsupervised[name][seed_index, heldout_queries] = (
                    baseline_metrics[name]
                )
                challenger[name][seed_index, heldout_queries] = (
                    challenger_metrics[name]
                )
            maximum_drift = max(
                maximum_drift,
                float(update_summary["maximum_centroid_drift_fraction"]),
            )
            all_objectives_nonincreasing &= bool(
                update_summary["objective_nonincreasing"]
            )
            assignment_changes += int(update_summary["assignment_changes"])
            fold_seed_diagnostics.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "training_query_count": int(len(train_queries)),
                    "heldout_query_count": int(len(heldout_queries)),
                    "training_residual_count": int(len(training_residual_rows)),
                    "statistics": statistics_summary,
                    "rpq_summary": rpq_summary,
                    "update_summary": update_summary,
                    "unsupervised_metrics": _metric_summary(baseline_metrics),
                    "challenger_metrics": _metric_summary(challenger_metrics),
                }
            )

    for collection in (pca16, unsupervised, challenger):
        if any(np.any(~np.isfinite(values)) for values in collection.values()):
            raise ValueError("OOF metric arrays are incomplete")

    # Export-only all-development fit. It cannot alter the OOF decision.
    final_pca16_basis = fit_uncentered_pca_basis(
        residuals, rank=int(pca_cfg["rank"])
    )
    final_pca16_scales = fit_int8_scales(residuals, final_pca16_basis)
    final_basis = fit_uncentered_pca_basis(residuals, rank=int(method["rank"]))
    final_coefficients = np.asarray(residuals @ final_basis, dtype=np.float32)
    _, final_initial_books, final_rpq_summary = fit_faiss_product_quantizer(
        final_coefficients,
        faiss,
        subquantizers=int(method["subquantizers"]),
        bits=int(method["bits_per_subquantizer"]),
        iterations=int(training_cfg["iterations"]),
        seed=primary_seed,
        max_points_per_centroid=int(
            training_cfg["maximum_points_per_centroid"]
        ),
    )
    final_codes = assign_product_codes(final_coefficients, final_initial_books)
    final_statistics, final_statistics_summary = build_signed_score_statistics(
        queries,
        final_basis,
        final_coefficients,
        final_codes,
        rows,
        lookup,
        base_scores,
        labels,
        top_b=int(method["top_b"]),
        final_k=final_k,
        cutoff_boost=float(score_cfg["cutoff_boost"]),
        margin_temperature=float(score_cfg["margin_temperature"]),
        known_positive_multiplier=float(score_cfg["known_positive_multiplier"]),
    )
    final_books, final_update_summary = fit_signed_score_codebooks(
        final_coefficients,
        final_initial_books,
        final_statistics,
        anchor_ratio=float(score_cfg["anchor_ratio"]),
        maximum_drift_fraction=float(
            score_cfg["maximum_centroid_drift_fraction_of_training_block_rms"]
        ),
        jitter=float(score_cfg["linear_solve_jitter"]),
    )
    maximum_drift = max(
        maximum_drift,
        float(final_update_summary["maximum_centroid_drift_fraction"]),
    )
    all_objectives_nonincreasing &= bool(
        final_update_summary["objective_nonincreasing"]
    )
    assignment_changes += int(final_update_summary["assignment_changes"])
    exports = {
        "final_pca16_basis.float32.npy": final_pca16_basis,
        "final_pca16_scales.float32.npy": final_pca16_scales,
        "final_pca_basis_rank64.float32.npy": final_basis,
        "final_unsupervised_codebooks.float32.npy": final_initial_books,
        "final_signed_score_codebooks.float32.npy": final_books,
    }
    for name, value in exports.items():
        atomic_save(args.output_dir / name, value)
    full_codes = _materialize_full_codes(
        embeddings_path=args.embeddings,
        index_path=args.index,
        output_path=(
            args.output_dir
            / "full_corpus_signed_score_assignments.uint8.memmap"
        ),
        basis=final_basis,
        codebooks=final_initial_books,
        n_docs=int(protocol["frozen_index_contract"]["document_count"]),
        dimension=int(protocol["frozen_index_contract"]["embedding_dimension"]),
        batch_size=args.full_corpus_batch_size,
    )
    if full_codes["minimum_occupied_centroids_per_block"] < int(
        protocol["full_corpus_sidecar"]["minimum_occupied_centroids_per_block"]
    ):
        raise ValueError("V13 full-corpus code occupancy is too small")

    primary_unsupervised = {
        name: values[primary_index] for name, values in unsupervised.items()
    }
    primary_challenger = {
        name: values[primary_index] for name, values in challenger.items()
    }
    versus_unsupervised = paired_inference(
        primary_challenger["recall"],
        primary_unsupervised["recall"],
        **_inference_kwargs(protocol, "primary_vs_unsupervised"),
    )
    versus_pca16 = paired_inference(
        primary_challenger["recall"],
        pca16["recall"],
        **_inference_kwargs(protocol, "primary_vs_pca16"),
    )
    versus_base = paired_inference(
        primary_challenger["recall"],
        base_metrics["recall"],
        **_inference_kwargs(protocol, "primary_vs_base"),
    )
    seed_gains = [
        float(
            np.mean(
                challenger["recall"][index]
                - unsupervised["recall"][index]
            )
        )
        for index in range(len(seeds))
    ]
    fold_gains = [
        float(
            np.mean(
                primary_challenger["recall"][folds == fold]
                - primary_unsupervised["recall"][folds == fold]
            )
        )
        for fold in range(fold_count)
    ]
    gap_recovery = candidate_gap_recovery(
        primary_challenger["recall"],
        base_metrics["recall"],
        exact_metrics["recall"],
    )
    decision = signed_score_decision(
        primary_vs_unsupervised=versus_unsupervised,
        primary_vs_pca16=versus_pca16,
        primary_vs_base=versus_base,
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        candidate_gap_recovery=gap_recovery,
        unsupervised_mrr=float(np.mean(primary_unsupervised["mrr"])),
        challenger_mrr=float(np.mean(primary_challenger["mrr"])),
        unsupervised_ndcg=float(np.mean(primary_unsupervised["ndcg"])),
        challenger_ndcg=float(np.mean(primary_challenger["ndcg"])),
        payload_bytes_per_document=int(
            full_codes["payload_bytes_per_document"]
        ),
        full_corpus_codes_materialized=True,
        all_objectives_nonincreasing=all_objectives_nonincreasing,
        maximum_centroid_drift_fraction=maximum_drift,
        assignment_changes=assignment_changes,
        thresholds=protocol["development_gate"],
    )

    output_names: list[str] = []
    qids_path = args.output_dir / "query_ids.utf8.txt"
    qids_path.write_text("\n".join(qids) + "\n", encoding="utf-8")
    output_names.append(qids_path.name)
    atomic_save(args.output_dir / "fold_ids.int64.npy", folds)
    output_names.append("fold_ids.int64.npy")
    output_names += _save_metric_arrays(args.output_dir, "base", base_metrics)
    output_names += _save_metric_arrays(
        args.output_dir, "same_candidate_exact", exact_metrics
    )
    output_names += _save_metric_arrays(args.output_dir, "pca16", pca16)
    for seed_index, seed in enumerate(seeds):
        output_names += _save_metric_arrays(
            args.output_dir,
            f"unsupervised_seed{seed}",
            {name: values[seed_index] for name, values in unsupervised.items()},
        )
        output_names += _save_metric_arrays(
            args.output_dir,
            f"signed_score_seed{seed}",
            {name: values[seed_index] for name, values in challenger.items()},
        )
    diagnostics_path = args.output_dir / "fold_seed_diagnostics.json"
    atomic_json(
        diagnostics_path,
        {
            "pca16": fold_pca_diagnostics,
            "signed_score": fold_seed_diagnostics,
        },
    )
    output_names.append(diagnostics_path.name)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V13_FRESH_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "evidence_tier": protocol["evidence_boundary"]["tier"],
        "query_count": len(qids),
        "candidate_residual_count": len(residuals),
        "primary_seed": primary_seed,
        "metrics": {
            "base": _metric_summary(base_metrics),
            "same_candidate_exact": _metric_summary(exact_metrics),
            "pca16": _metric_summary(pca16),
            "unsupervised_primary": _metric_summary(primary_unsupervised),
            "signed_score_primary": _metric_summary(primary_challenger),
            "unsupervised_by_seed": [
                _metric_summary(
                    {name: values[index] for name, values in unsupervised.items()}
                )
                for index in range(len(seeds))
            ],
            "signed_score_by_seed": [
                _metric_summary(
                    {name: values[index] for name, values in challenger.items()}
                )
                for index in range(len(seeds))
            ],
        },
        "comparisons": {
            "signed_score_vs_unsupervised": versus_unsupervised,
            "signed_score_vs_pca16": versus_pca16,
            "signed_score_vs_base": versus_base,
        },
        "seed_gains": seed_gains,
        "fold_gains": fold_gains,
        "candidate_gap_recovery_fraction": gap_recovery,
        "decision": decision,
        "maximum_centroid_drift_fraction": maximum_drift,
        "assignment_changes": assignment_changes,
        "all_objectives_nonincreasing": all_objectives_nonincreasing,
        "final_fit": {
            "statistics": final_statistics_summary,
            "rpq_summary": final_rpq_summary,
            "update_summary": final_update_summary,
            "full_corpus_codes": full_codes,
        },
        "v9_packet_opened": False,
        "v10_packet_opened": False,
        "v11_packet_opened": False,
        "v12_packet_opened": False,
        "old_holdout_opened": False,
        "fresh_confirmation_access_authorized": False,
        "interpretation": (
            "Fresh-query five-fold method development only. A GO permits "
            "only writing a new independent-confirmation protocol."
        ),
    }
    result_path = args.output_dir / "development_result.json"
    atomic_json(result_path, result)
    output_names.append(result_path.name)
    method_freeze_path = args.output_dir / "method_freeze.json"
    atomic_json(
        method_freeze_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V13_METHOD_AND_DECISION_FROZEN",
            "source_commit": args.source_commit,
            "formal_decision": decision["decision"],
            "method": protocol["method"],
            "pca16_comparator": protocol["pca16_comparator"],
            "score_distillation": protocol["score_distillation"],
            "rpq_training": protocol["rpq_training"],
            "development_gate": protocol["development_gate"],
            "development_result": file_record(result_path),
            "old_holdout_reuse_authorized": False,
            "fresh_confirmation_access_authorized": False,
        },
    )
    output_names.append(method_freeze_path.name)
    output_names.extend(exports)
    output_names.append("full_corpus_signed_score_assignments.uint8.memmap")
    complete_path = args.output_dir / "development_complete.json"
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V13_FRESH_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "started": file_record(started_path),
        "outputs": {
            name: file_record(args.output_dir / name)
            for name in sorted(set(output_names))
        },
        "old_holdout_opened": False,
        "fresh_confirmation_access_authorized": False,
    }
    atomic_json(complete_path, complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--full-corpus-batch-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
