#!/usr/bin/env python3
"""Run the frozen V14 query-whitened anisotropic rate-RPQ diagnostic."""

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

from rars_v11_rank_rate_core import paired_inference  # noqa: E402
from rars_v14_anisotropic_rate_core import (  # noqa: E402
    PROTOCOL_ID,
    allocate_bits_dynamic_programming,
    anisotropic_rate_decision,
    assign_variable_codes,
    block_rate_sensitivity,
    concatenate_codebooks,
    cutoff_weights,
    fit_query_metric_transforms,
    fit_variable_block_quantizers,
    multi_seed_consensus,
    score_variable_sidecar_candidates,
    unpack_variable_codes,
)
from rars_v8_cutoff_sidecar_core import (  # noqa: E402
    candidate_gap_recovery,
    fit_uncentered_pca_basis,
    per_query_metrics,
)
from train_rars_v13_signed_score_rpq import load_bundle  # noqa: E402
from train_rars_v8_cutoff_sidecar import (  # noqa: E402
    atomic_json,
    atomic_save,
    file_record,
    read_json,
    validate_runtime,
)
from verify_rars_v13_committed_closure import verify_closure  # noqa: E402
from verify_rars_v13_signed_score_rpq_packet import verify_packet as verify_v13_packet  # noqa: E402


CANONICAL_PROTOCOL = Path(
    "protocols/rars_v14_query_whitened_anisotropic_rate_rpq_diagnostic_v1.json"
)
V13_PROTOCOL = Path("protocols/rars_v13_signed_score_distilled_rpq_v1.json")
V13_SOURCE_COMMIT = "d8cb761c289fe17ea2c2bfb92059e8b5553cfd74"
SOURCE_FILES = (
    CANONICAL_PROTOCOL,
    Path("scripts/rars_v14_anisotropic_rate_core.py"),
    Path("scripts/evaluate_rars_v14_anisotropic_rate_rpq.py"),
    Path("scripts/verify_rars_v14_anisotropic_rate_rpq_packet.py"),
    Path("scripts/rars_v13_signed_score_core.py"),
    Path("scripts/train_rars_v13_signed_score_rpq.py"),
    Path("scripts/verify_rars_v13_signed_score_rpq_packet.py"),
    Path("scripts/verify_rars_v13_committed_closure.py"),
    Path("scripts/rars_v11_rank_rate_core.py"),
    Path("scripts/rars_v8_cutoff_sidecar_core.py"),
    Path("scripts/train_rars_v8_cutoff_sidecar.py"),
)
METRICS = ("recall", "mrr", "ndcg")


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
        raise ValueError(f"Registered {label} SHA-256 changed")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("--source-commit must be exact lowercase 40-hex")
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"V14 requires canonical protocol path: {canonical}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or status:
        raise ValueError("V14 diagnostic requires a clean exact checkout")
    protocol = read_json(canonical)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "FROZEN_BEFORE_FIRST_V14_DIAGNOSTIC_RUN"
    ):
        raise ValueError("Unexpected V14 protocol identity or status")
    return protocol, {
        str(relative): file_record((repo_root / relative).resolve(strict=True))
        for relative in SOURCE_FILES
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
    names: list[str] = []
    for metric, array in values.items():
        name = f"per_query_{prefix}_{metric}_at_10.float64.npy"
        atomic_save(output_dir / name, np.asarray(array, dtype=np.float64))
        names.append(name)
    return names


def _load_v13_metrics(
    packet_root: Path,
    qids: list[str],
    folds: np.ndarray,
    seeds: list[int],
) -> dict[str, Any]:
    packet_qids = (packet_root / "query_ids.utf8.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    packet_folds = np.load(packet_root / "fold_ids.int64.npy", allow_pickle=False)
    if packet_qids != qids or not np.array_equal(packet_folds, folds):
        raise ValueError("V13 packet and bundle query/fold identities differ")

    def load(prefix: str) -> dict[str, np.ndarray]:
        output = {
            metric: np.load(
                packet_root / f"per_query_{prefix}_{metric}_at_10.float64.npy",
                allow_pickle=False,
            )
            for metric in METRICS
        }
        if any(value.shape != (len(qids),) or value.dtype != np.float64 for value in output.values()):
            raise ValueError(f"V13 metric contract changed for {prefix}")
        return output

    return {
        "base": load("base"),
        "same_candidate_exact": load("same_candidate_exact"),
        "pca16": load("pca16"),
        "uniform": {seed: load(f"unsupervised_seed{seed}") for seed in seeds},
    }


def _training_residual_rows(
    query_indices: np.ndarray,
    rows: np.ndarray,
    scores: np.ndarray,
    lookup: np.ndarray,
    *,
    top_b: int,
    final_k: int,
    cutoff_boost: float,
    margin_temperature: float,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for query_index in query_indices:
        ordering, _ = cutoff_weights(
            scores[query_index],
            rows[query_index],
            top_b=top_b,
            final_k=final_k,
            cutoff_boost=cutoff_boost,
            margin_temperature=margin_temperature,
        )
        selected.append(lookup[query_index, ordering])
    output = np.unique(np.concatenate(selected)).astype(np.int64)
    if np.any(output < 0):
        raise ValueError("Training residual lookup contains an invalid row")
    return output


def _fit_representation(
    *,
    residuals: np.ndarray,
    queries: np.ndarray,
    rows: np.ndarray,
    base_scores: np.ndarray,
    lookup: np.ndarray,
    training_queries: np.ndarray,
    protocol: dict[str, Any],
    faiss_module: Any,
    seed: int,
    uniform: bool,
    geometry: dict[str, Any] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    method = protocol["method"]
    metric = protocol["query_metric"]
    training = protocol["quantizer_training"]
    final_k = int(protocol["frozen_index_contract"]["final_cutoff"])
    top_b = int(method["top_b"])
    if geometry is None:
        training_rows = _training_residual_rows(
            training_queries,
            rows,
            base_scores,
            lookup,
            top_b=top_b,
            final_k=final_k,
            cutoff_boost=float(metric["cutoff_boost"]),
            margin_temperature=float(metric["margin_temperature"]),
        )
        basis = fit_uncentered_pca_basis(
            residuals[training_rows], rank=int(method["rank"])
        )
        coefficients = np.asarray(residuals @ basis, dtype=np.float32)
        transforms, metric_summary = fit_query_metric_transforms(
            queries[training_queries],
            basis,
            rows[training_queries],
            base_scores[training_queries],
            top_b=top_b,
            final_k=final_k,
            cutoff_boost=float(metric["cutoff_boost"]),
            margin_temperature=float(metric["margin_temperature"]),
            ridge_fraction=float(metric["ridge_fraction_of_trace"]),
            block_dimension=int(method["block_dimension"]),
        )
        sensitivity, rate_summary = block_rate_sensitivity(
            coefficients,
            transforms,
            lookup[training_queries],
            rows[training_queries],
            base_scores[training_queries],
            top_b=top_b,
            final_k=final_k,
            cutoff_boost=float(metric["cutoff_boost"]),
            margin_temperature=float(metric["margin_temperature"]),
        )
        geometry = {
            "basis": basis,
            "coefficients": coefficients,
            "transforms": transforms,
            "sensitivity": sensitivity,
            "training_rows": training_rows,
            "metric_summary": metric_summary,
            "rate_summary": rate_summary,
        }
    else:
        basis = geometry["basis"]
        coefficients = geometry["coefficients"]
        transforms = geometry["transforms"]
        sensitivity = geometry["sensitivity"]
        training_rows = geometry["training_rows"]
        metric_summary = geometry["metric_summary"]
        rate_summary = geometry["rate_summary"]
    if uniform:
        allocation = np.full(
            int(method["subquantizers"]),
            int(method["total_bits_per_document"]) // int(method["subquantizers"]),
            dtype=np.int64,
        )
        allocation_summary = {
            "total_bits": int(allocation.sum()),
            "minimum_allocated_bits": int(allocation.min()),
            "maximum_allocated_bits": int(allocation.max()),
            "nonuniform": False,
            "ablation": "uniform_eight_bits",
        }
    else:
        allocation, allocation_summary = allocate_bits_dynamic_programming(
            sensitivity,
            total_bits=int(method["total_bits_per_document"]),
            minimum_bits=int(method["minimum_bits_per_block"]),
            maximum_bits=int(method["maximum_bits_per_block"]),
            block_dimension=int(method["block_dimension"]),
        )
    _, books, quantizer_summary = fit_variable_block_quantizers(
        coefficients[training_rows],
        transforms,
        allocation,
        faiss_module,
        iterations=int(training["iterations"]),
        seed=seed,
        max_points_per_centroid=int(training["maximum_points_per_centroid"]),
    )
    packed = assign_variable_codes(coefficients, transforms, books, allocation)
    return basis, transforms, allocation, books, {
        "training_query_count": int(len(training_queries)),
        "training_residual_count": int(len(training_rows)),
        "metric": metric_summary,
        "rate": rate_summary,
        "allocation": allocation.tolist(),
        "allocation_summary": allocation_summary,
        "quantizer": quantizer_summary,
        "candidate_codes": packed,
    }, geometry


def _materialize_full_codes(
    *,
    embeddings_path: Path,
    index_path: Path,
    output_path: Path,
    basis: np.ndarray,
    transforms: np.ndarray,
    allocation: np.ndarray,
    books: list[np.ndarray],
    n_docs: int,
    dimension: int,
    batch_size: int,
) -> dict[str, Any]:
    import faiss

    if output_path.exists() or batch_size <= 0:
        raise ValueError("Refusing invalid or existing V14 full payload")
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    index = faiss.read_index(str(index_path))
    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    ivf.make_direct_map()
    embeddings = np.memmap(
        embeddings_path, dtype=np.float16, mode="r", shape=(n_docs, dimension)
    )
    output = np.memmap(
        temporary, dtype=np.uint8, mode="w+", shape=(n_docs, 16)
    )
    histograms = [np.zeros(1 << int(bits), dtype=np.int64) for bits in allocation]
    started = time.perf_counter()
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        doc_rows = np.arange(start, end, dtype=np.int64)
        reconstructed = np.asarray(index.reconstruct_batch(doc_rows), dtype=np.float32)
        residuals = np.asarray(embeddings[start:end], dtype=np.float32) - reconstructed
        coefficients = np.asarray(residuals @ basis, dtype=np.float32)
        packed = assign_variable_codes(
            coefficients,
            transforms,
            books,
            allocation,
            batch_size=min(batch_size, 2048),
        )
        output[start:end] = packed
        codes = unpack_variable_codes(packed, allocation)
        for block in range(len(allocation)):
            histograms[block] += np.bincount(
                codes[:, block], minlength=len(histograms[block])
            )
    output.flush()
    del output
    os.replace(temporary, output_path)
    expected_bytes = n_docs * 16
    if output_path.stat().st_size != expected_bytes:
        raise ValueError("V14 full payload byte count is wrong")
    occupied = [int(np.sum(histogram > 0)) for histogram in histograms]
    return {
        "document_count": n_docs,
        "code_shape": [n_docs, 16],
        "dtype": "uint8",
        "bit_allocation": allocation.tolist(),
        "payload_bytes": expected_bytes,
        "payload_bytes_per_document": 16,
        "occupied_centroids_per_block": occupied,
        "code_histograms": [histogram.tolist() for histogram in histograms],
        "wall_seconds": float(time.perf_counter() - started),
        "record": file_record(output_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = validate_source(
        repo_root, args.protocol, args.source_commit
    )
    environment = validate_runtime(protocol)
    if environment["faiss_version"] != protocol["execution_environment_contract"]["faiss_version"]:
        raise ValueError("V14 Faiss version differs from the protocol")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty V14 output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    closure = verify_closure(
        repo_root / "results/rars_v13_signed_score_rpq", repo_root
    )
    if closure["formal_decision"] != protocol["parent_evidence"]["v13_formal_decision"]:
        raise ValueError("Committed V13 parent decision changed")
    v13_verification = verify_v13_packet(args.v13_packet_root, repo_root)
    if v13_verification["source_commit"] != V13_SOURCE_COMMIT:
        raise ValueError("V13 Drive packet source commit changed")
    v13_protocol = read_json(repo_root / V13_PROTOCOL)
    qids, folds, arrays, bundle_records = load_bundle(
        args.v13_bundle_root, v13_protocol, repo_root, V13_SOURCE_COMMIT
    )
    seeds = [int(value) for value in protocol["quantizer_training"]["seeds"]]
    primary_seed = int(protocol["quantizer_training"]["primary_seed"])
    baseline = _load_v13_metrics(args.v13_packet_root, qids, folds, seeds)
    _verify_record(args.embeddings, bundle_records["registered_embeddings"], "embeddings")
    _verify_record(args.index, bundle_records["registered_index"], "index")
    started_path = args.output_dir / "diagnostic_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V14_ANISOTROPIC_RATE_DIAGNOSTIC_STARTED",
            "source_commit": args.source_commit,
            "environment": environment,
            "source_blobs": source_blobs,
            "inputs": {
                **bundle_records,
                "v13_packet_result": file_record(args.v13_packet_root / "development_result.json"),
                "embeddings": file_record(args.embeddings),
                "index": file_record(args.index),
            },
            "v13_committed_closure": closure,
            "v13_packet_verification": v13_verification,
            "evidence_tier": protocol["evidence_boundary"]["tier"],
            "labels_used_for_representation_learning": False,
            "future_method_holdout_opened": False,
            "old_rars_holdout_opened": False,
        },
    )

    queries = np.asarray(arrays["query_vectors.float32.npy"], dtype=np.float32)
    rows = np.asarray(arrays["ann_rows.int64.npy"], dtype=np.int64)
    base_scores = np.asarray(arrays["ann_scores.float32.npy"], dtype=np.float32)
    lookup = np.asarray(arrays["ann_residual_rows.int64.npy"], dtype=np.int64)
    residuals = np.asarray(arrays["candidate_residuals.float32.npy"], dtype=np.float32)
    labels = np.asarray(arrays["candidate_relevance.uint8.npy"], dtype=np.uint8)
    relevant_counts = np.asarray(arrays["relevant_counts.int32.npy"], dtype=np.int64)
    final_k = int(protocol["frozen_index_contract"]["final_cutoff"])
    method = protocol["method"]
    fold_count = int(protocol["cross_validation"]["fold_count"])
    challenger = {
        metric: np.full((len(seeds), len(qids)), np.nan, dtype=np.float64)
        for metric in METRICS
    }
    uniform_whitened = {
        metric: np.full(len(qids), np.nan, dtype=np.float64) for metric in METRICS
    }
    diagnostics: list[dict[str, Any]] = []
    allocations: list[list[int]] = []
    for fold in range(fold_count):
        heldout = np.flatnonzero(folds == fold)
        training_queries = np.flatnonzero(folds != fold)
        geometry = None
        for seed_index, seed in enumerate(seeds):
            basis, transforms, allocation, books, summary, geometry = _fit_representation(
                residuals=residuals,
                queries=queries,
                rows=rows,
                base_scores=base_scores,
                lookup=lookup,
                training_queries=training_queries,
                protocol=protocol,
                faiss_module=faiss,
                seed=seed,
                uniform=False,
                geometry=geometry,
            )
            packed = summary.pop("candidate_codes")
            scores = score_variable_sidecar_candidates(
                queries[heldout],
                rows[heldout],
                lookup[heldout],
                base_scores[heldout],
                basis,
                packed,
                allocation,
                books,
                alpha=float(method["alpha"]),
                top_b=int(method["top_b"]),
            )
            metrics = per_query_metrics(
                scores,
                rows[heldout],
                labels[heldout],
                relevant_counts[heldout],
                k=final_k,
            )
            for name in METRICS:
                challenger[name][seed_index, heldout] = metrics[name]
            if seed == primary_seed:
                allocations.append(allocation.tolist())
            diagnostics.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "heldout_query_count": int(len(heldout)),
                    "representation": summary,
                    "metrics": _metric_summary(metrics),
                }
            )
        uniform_basis, uniform_transforms, uniform_bits, uniform_books, uniform_summary, _ = _fit_representation(
            residuals=residuals,
            queries=queries,
            rows=rows,
            base_scores=base_scores,
            lookup=lookup,
            training_queries=training_queries,
            protocol=protocol,
            faiss_module=faiss,
            seed=primary_seed,
            uniform=True,
            geometry=geometry,
        )
        uniform_codes = uniform_summary.pop("candidate_codes")
        uniform_scores = score_variable_sidecar_candidates(
            queries[heldout],
            rows[heldout],
            lookup[heldout],
            base_scores[heldout],
            uniform_basis,
            uniform_codes,
            uniform_bits,
            uniform_books,
            alpha=float(method["alpha"]),
            top_b=int(method["top_b"]),
        )
        uniform_metrics = per_query_metrics(
            uniform_scores,
            rows[heldout],
            labels[heldout],
            relevant_counts[heldout],
            k=final_k,
        )
        for name in METRICS:
            uniform_whitened[name][heldout] = uniform_metrics[name]
        diagnostics.append(
            {
                "fold": fold,
                "seed": primary_seed,
                "ablation": "uniform_whitened_8bit",
                "heldout_query_count": int(len(heldout)),
                "representation": uniform_summary,
                "metrics": _metric_summary(uniform_metrics),
            }
        )
    if any(np.any(~np.isfinite(value)) for value in challenger.values()) or any(
        np.any(~np.isfinite(value)) for value in uniform_whitened.values()
    ):
        raise ValueError("V14 OOF arrays are incomplete")

    all_queries = np.arange(len(qids), dtype=np.int64)
    final_basis, final_transforms, final_allocation, final_books, final_summary, _ = _fit_representation(
        residuals=residuals,
        queries=queries,
        rows=rows,
        base_scores=base_scores,
        lookup=lookup,
        training_queries=all_queries,
        protocol=protocol,
        faiss_module=faiss,
        seed=primary_seed,
        uniform=False,
    )
    final_summary.pop("candidate_codes")
    concatenated_books, offsets = concatenate_codebooks(final_books)
    exports = {
        "final_pca_basis_rank64.float32.npy": final_basis,
        "final_query_metric_transforms.float32.npy": final_transforms,
        "final_bit_allocation.int64.npy": final_allocation,
        "final_codebooks.float32.npy": concatenated_books,
        "final_codebook_offsets.int64.npy": offsets,
    }
    for name, value in exports.items():
        atomic_save(args.output_dir / name, value)
    full_codes = _materialize_full_codes(
        embeddings_path=args.embeddings,
        index_path=args.index,
        output_path=args.output_dir / "full_corpus_qw_ar_rpq_codes.uint8.memmap",
        basis=final_basis,
        transforms=final_transforms,
        allocation=final_allocation,
        books=final_books,
        n_docs=int(protocol["frozen_index_contract"]["document_count"]),
        dimension=int(protocol["frozen_index_contract"]["embedding_dimension"]),
        batch_size=args.full_corpus_batch_size,
    )
    allocations.append(final_allocation.tolist())

    primary_index = seeds.index(primary_seed)
    primary = {name: challenger[name][primary_index] for name in METRICS}
    primary_uniform = baseline["uniform"][primary_seed]
    comparisons = {
        "anisotropic_vs_v13_uniform_rpq": paired_inference(
            primary["recall"],
            primary_uniform["recall"],
            **_inference_kwargs(protocol, "primary_vs_v13_uniform_rpq"),
        ),
        "anisotropic_vs_uniform_whitened": paired_inference(
            primary["recall"],
            uniform_whitened["recall"],
            **_inference_kwargs(protocol, "primary_vs_uniform_whitened"),
        ),
        "anisotropic_vs_pca16": paired_inference(
            primary["recall"],
            baseline["pca16"]["recall"],
            **_inference_kwargs(protocol, "primary_vs_pca16"),
        ),
        "anisotropic_vs_base": paired_inference(
            primary["recall"],
            baseline["base"]["recall"],
            **_inference_kwargs(protocol, "primary_vs_base"),
        ),
    }
    seed_gains = [
        float(
            np.mean(
                challenger["recall"][index] - baseline["uniform"][seed]["recall"]
            )
        )
        for index, seed in enumerate(seeds)
    ]
    fold_gains = [
        float(
            np.mean(
                primary["recall"][folds == fold]
                - primary_uniform["recall"][folds == fold]
            )
        )
        for fold in range(fold_count)
    ]
    consensus = multi_seed_consensus(
        challenger["recall"],
        np.stack([baseline["uniform"][seed]["recall"] for seed in seeds]),
    )
    gap = candidate_gap_recovery(
        primary["recall"],
        baseline["base"]["recall"],
        baseline["same_candidate_exact"]["recall"],
    )
    decision = anisotropic_rate_decision(
        primary_vs_uniform_rpq=comparisons["anisotropic_vs_v13_uniform_rpq"],
        primary_vs_uniform_whitened=comparisons["anisotropic_vs_uniform_whitened"],
        primary_vs_pca16=comparisons["anisotropic_vs_pca16"],
        primary_vs_base=comparisons["anisotropic_vs_base"],
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        candidate_gap_recovery=gap,
        uniform_rpq_mrr=float(np.mean(primary_uniform["mrr"])),
        challenger_mrr=float(np.mean(primary["mrr"])),
        uniform_rpq_ndcg=float(np.mean(primary_uniform["ndcg"])),
        challenger_ndcg=float(np.mean(primary["ndcg"])),
        consensus=consensus,
        allocations=allocations,
        payload_bytes_per_document=int(full_codes["payload_bytes_per_document"]),
        full_corpus_codes_materialized=True,
        thresholds=protocol["diagnostic_gate"],
    )

    output_names: list[str] = []
    qids_path = args.output_dir / "query_ids.utf8.txt"
    qids_path.write_text("\n".join(qids) + "\n", encoding="utf-8")
    output_names.append(qids_path.name)
    atomic_save(args.output_dir / "fold_ids.int64.npy", folds)
    output_names.append("fold_ids.int64.npy")
    for prefix in ("base", "same_candidate_exact", "pca16"):
        output_names += _save_metric_arrays(args.output_dir, prefix, baseline[prefix])
    output_names += _save_metric_arrays(args.output_dir, "uniform_whitened_primary", uniform_whitened)
    for seed_index, seed in enumerate(seeds):
        output_names += _save_metric_arrays(
            args.output_dir, f"v13_uniform_seed{seed}", baseline["uniform"][seed]
        )
        output_names += _save_metric_arrays(
            args.output_dir,
            f"anisotropic_seed{seed}",
            {name: challenger[name][seed_index] for name in METRICS},
        )
    diagnostics_path = args.output_dir / "fold_seed_diagnostics.json"
    atomic_json(diagnostics_path, diagnostics)
    output_names.append(diagnostics_path.name)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V14_ANISOTROPIC_RATE_DIAGNOSTIC_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "evidence_tier": protocol["evidence_boundary"]["tier"],
        "query_count": len(qids),
        "metrics": {
            "base": _metric_summary(baseline["base"]),
            "same_candidate_exact": _metric_summary(baseline["same_candidate_exact"]),
            "pca16": _metric_summary(baseline["pca16"]),
            "v13_uniform_primary": _metric_summary(primary_uniform),
            "uniform_whitened_primary": _metric_summary(uniform_whitened),
            "anisotropic_primary": _metric_summary(primary),
        },
        "comparisons": comparisons,
        "seed_gains": seed_gains,
        "fold_gains": fold_gains,
        "multi_seed_consensus": consensus,
        "candidate_gap_recovery_fraction": gap,
        "allocations": allocations,
        "decision": decision,
        "final_fit": {
            "representation": final_summary,
            "codebook_offsets": offsets.tolist(),
            "codebook_bytes": int(concatenated_books.nbytes),
            "metric_bytes": int(final_transforms.nbytes),
            "basis_bytes": int(final_basis.nbytes),
            "full_corpus_codes": full_codes,
        },
        "labels_used_for_representation_learning": False,
        "future_method_holdout_opened": False,
        "old_rars_holdout_opened": False,
        "fresh_query_access_authorized": False,
        "interpretation": "Outcome-informed architecture diagnostic only. GO authorizes only writing a disjoint fresh-query protocol.",
    }
    result_path = args.output_dir / "diagnostic_result.json"
    atomic_json(result_path, result)
    output_names.append(result_path.name)
    freeze_path = args.output_dir / "diagnostic_freeze.json"
    atomic_json(
        freeze_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V14_METHOD_AND_DECISION_FROZEN",
            "source_commit": args.source_commit,
            "formal_decision": decision["decision"],
            "method": protocol["method"],
            "query_metric": protocol["query_metric"],
            "quantizer_training": protocol["quantizer_training"],
            "diagnostic_gate": protocol["diagnostic_gate"],
            "diagnostic_result": file_record(result_path),
            "fresh_query_access_authorized": False,
        },
    )
    output_names.append(freeze_path.name)
    output_names.extend(exports)
    output_names.append("full_corpus_qw_ar_rpq_codes.uint8.memmap")
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V14_ANISOTROPIC_RATE_DIAGNOSTIC_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "started": file_record(started_path),
        "outputs": {
            name: file_record(args.output_dir / name)
            for name in sorted(set(output_names))
        },
        "fresh_query_access_authorized": False,
    }
    atomic_json(args.output_dir / "diagnostic_complete.json", complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-bundle-root", type=Path, required=True)
    parser.add_argument("--v13-packet-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--full-corpus-batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
