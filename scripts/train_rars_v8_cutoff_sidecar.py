#!/usr/bin/env python3
"""Run RARS-v8 development without opening any future or audit role.

The script consumes the already frozen and labelled ``oracle_design`` bundle,
computes five-fold out-of-fold metrics for a storage-matched PCA sidecar and
the revised cutoff-aware RARS basis, and emits a method freeze.  It does not
encode the full corpus or authorize a holdout evaluation; that must happen in
a separate, commit-pinned stage after this development packet is closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v3_oracle_core import design_fold_ids  # noqa: E402
from rars_v8_cutoff_sidecar_core import (  # noqa: E402
    PROTOCOL_ID,
    development_decision,
    candidate_gap_recovery,
    encode_residuals_int8,
    fit_cutoff_aware_basis,
    fit_int8_scales,
    fit_uncentered_pca_basis,
    mine_cutoff_pairs,
    paired_bootstrap,
    per_query_metrics,
    score_sidecar_candidates,
    subset_pairs,
    summarize_pairs,
)
from verify_rars_v6_1m_headroom_packet import verify_packet as verify_v6_packet  # noqa: E402


CANONICAL_PROTOCOL = Path("protocols/rars_v8_cutoff_sidecar_v1.json")
REQUIRED_CANDIDATE_FILES = (
    "query_vectors.float32.npy",
    "ann_rows.int64.npy",
    "ann_scores.float32.npy",
    "candidate_doc_rows.int64.npy",
    "ann_residual_rows.int64.npy",
    "candidate_residuals.float32.npy",
)
REQUIRED_LABEL_FILES = (
    "candidate_relevance.uint8.npy",
    "relevant_counts.int32.npy",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing registered {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Registered {label} SHA-256 changed")


def _validate_exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_exact_commit(source_commit)
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must use the canonical path: {canonical}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if head != source_commit:
        raise ValueError(f"Git HEAD {head} does not match {source_commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if status:
        raise ValueError("V8 development requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected V8 protocol identity")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_V8_DEVELOPMENT_RUN":
        raise ValueError("V8 protocol is not frozen for its first development run")
    source_blobs = {
        "protocol": file_record(canonical),
        "trainer": file_record(Path(__file__).resolve()),
        "core": file_record(SCRIPT_DIR / "rars_v8_cutoff_sidecar_core.py"),
    }
    return protocol, source_blobs


def validate_runtime(protocol: dict[str, Any]) -> dict[str, Any]:
    import faiss
    import torch

    expected = protocol["execution_environment_contract"]
    observed = {
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    for key in ("python_version", "numpy_version", "torch_version", "torch_cuda_version"):
        if observed[key] != expected[key]:
            raise ValueError(f"Runtime {key}={observed[key]!r}; expected {expected[key]!r}")
    if not observed["cuda_available"] or expected["gpu_name_must_contain"] not in str(
        observed["gpu_name"]
    ):
        raise ValueError("V8 requires the registered T4 CUDA environment")
    if observed["cublas_workspace_config"] != expected["cublas_workspace_config"]:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from the V8 contract")
    if observed["cudnn_benchmark"] is not expected["cudnn_benchmark"]:
        raise ValueError("cuDNN benchmark state differs from the V8 contract")
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Deterministic Torch algorithms are not enabled")
    return observed


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("Refusing to reuse a non-empty V8 development output")
    path.mkdir(parents=True, exist_ok=True)


def load_design_bundle(
    role_dir: Path, protocol: dict[str, Any]
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    role_dir = role_dir.resolve()
    if role_dir.name != "oracle_design":
        raise ValueError("V8 accepts only the oracle_design role")
    candidate_manifest_path = role_dir / "v3_candidate_manifest.json"
    label_manifest_path = role_dir / "v3_role_labels_manifest.json"
    query_manifest_path = role_dir / "query_manifest.json"
    for path in (candidate_manifest_path, label_manifest_path, query_manifest_path):
        if not path.is_file():
            raise ValueError(f"Missing V8 design input: {path}")
    candidate_manifest = read_json(candidate_manifest_path)
    label_manifest = read_json(label_manifest_path)
    query_manifest = read_json(query_manifest_path)
    if candidate_manifest.get("role_id") != "oracle_design":
        raise ValueError("Candidate manifest is not oracle_design")
    if label_manifest.get("role_id") != "oracle_design" or label_manifest.get(
        "status"
    ) != "ROLE_LABELS_MATERIALIZED_FROM_FROZEN_PARENT":
        raise ValueError("Design labels are not the frozen oracle_design labels")
    if candidate_manifest.get("data_access", {}).get("qrels_opened_or_parsed") is not False:
        raise ValueError("Candidate builder unexpectedly opened qrels")
    qids = [str(value) for value in query_manifest.get("query_ids", [])]
    role_contract = protocol["data_policy"]["development_role"]
    if len(qids) != int(role_contract["query_count"]) or len(qids) != len(set(qids)):
        raise ValueError("Design query count or uniqueness changed")
    ordered_hash = hashlib.sha256(
        ("\n".join(qids) + "\n").encode("utf-8")
    ).hexdigest()
    numeric_hash = hashlib.sha256(
        ("\n".join(sorted(qids, key=int)) + "\n").encode("utf-8")
    ).hexdigest()
    if ordered_hash != role_contract["source_order_newline_qid_sha256"]:
        raise ValueError("Design source-order query hash changed")
    if numeric_hash != role_contract["numeric_sorted_newline_qid_sha256"]:
        raise ValueError("Design numeric-order query hash changed")

    for filename in REQUIRED_CANDIDATE_FILES:
        record = candidate_manifest.get("files", {}).get(filename)
        if not isinstance(record, dict):
            raise ValueError(f"Candidate manifest does not register {filename}")
        _verify_record(role_dir / filename, record, filename)
    for filename in REQUIRED_LABEL_FILES:
        record = label_manifest.get("files", {}).get(filename)
        if not isinstance(record, dict):
            raise ValueError(f"Label manifest does not register {filename}")
        _verify_record(role_dir / filename, record, filename)
    _verify_record(
        query_manifest_path, candidate_manifest["query_manifest"], "query manifest"
    )

    arrays = {
        filename: np.load(role_dir / filename, mmap_mode="r", allow_pickle=False)
        for filename in (*REQUIRED_CANDIDATE_FILES, *REQUIRED_LABEL_FILES)
    }
    query_count = len(qids)
    candidate_count = int(candidate_manifest["candidate_count"])
    matrix_shape = (query_count, candidate_count)
    queries = arrays["query_vectors.float32.npy"]
    if queries.dtype != np.float32 or queries.shape != (
        query_count,
        int(protocol["frozen_index_contract"]["embedding_dimension"]),
    ):
        raise ValueError("Design query vectors changed shape or dtype")
    for filename in (
        "ann_rows.int64.npy",
        "ann_scores.float32.npy",
        "ann_residual_rows.int64.npy",
        "candidate_relevance.uint8.npy",
    ):
        if arrays[filename].shape != matrix_shape:
            raise ValueError(f"Design matrix shape changed: {filename}")
    if arrays["relevant_counts.int32.npy"].shape != (query_count,):
        raise ValueError("Relevant-count denominator shape changed")
    candidate_rows = arrays["candidate_doc_rows.int64.npy"]
    residuals = arrays["candidate_residuals.float32.npy"]
    if residuals.shape != (
        len(candidate_rows),
        int(protocol["frozen_index_contract"]["embedding_dimension"]),
    ):
        raise ValueError("Candidate residual union shape changed")
    lookup = np.asarray(arrays["ann_residual_rows.int64.npy"], dtype=np.int64)
    if np.any(lookup < 0) or np.any(lookup >= len(candidate_rows)):
        raise ValueError("Residual lookup is out of range")
    if not np.array_equal(
        np.asarray(candidate_rows)[lookup], np.asarray(arrays["ann_rows.int64.npy"])
    ):
        raise ValueError("Residual lookup does not reproduce ANN rows")
    expected_folds = design_fold_ids(qids)
    registered_folds = np.asarray(query_manifest.get("diagnostic_fold_ids"), dtype=np.int64)
    if not np.array_equal(expected_folds, registered_folds):
        raise ValueError("Registered V3 diagnostic folds changed")
    counts = np.bincount(expected_folds, minlength=5).tolist()
    if counts != protocol["data_policy"]["cross_validation"]["fold_counts"]:
        raise ValueError("V8 fold counts differ from the frozen protocol")
    records = {
        "candidate_manifest": file_record(candidate_manifest_path),
        "label_manifest": file_record(label_manifest_path),
        "query_manifest": file_record(query_manifest_path),
    }
    return qids, expected_folds, arrays, records


def exact_candidate_scores(
    queries: np.ndarray,
    ann_scores: np.ndarray,
    residual_lookup: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    output = np.asarray(ann_scores, dtype=np.float32).copy()
    for start in range(0, len(queries), 128):
        end = min(start + 128, len(queries))
        selected = np.asarray(residual_lookup[start:end], dtype=np.int64)
        correction = np.einsum(
            "qd,qcd->qc",
            np.asarray(queries[start:end], dtype=np.float32),
            np.asarray(residuals[selected], dtype=np.float32),
        )
        output[start:end] += correction.astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("Exact candidate scores contain non-finite values")
    return output


def _metric_summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(value)) for name, value in metrics.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = validate_source(repo_root, args.protocol, args.source_commit)
    environment = validate_runtime(protocol)
    v6_verification = verify_v6_packet(args.v6_packet_root)
    prepare_output(args.output_dir)
    qids, folds, arrays, input_records = load_design_bundle(args.design_role_dir, protocol)

    started_path = args.output_dir / "development_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V8_DEVELOPMENT_STARTED",
            "source_commit": args.source_commit,
            "environment": environment,
            "source_blobs": source_blobs,
            "inputs": input_records,
            "v6_verification": v6_verification,
            "opened_roles": ["oracle_design"],
            "future_method_holdout_opened": False,
            "oracle_audit_opened": False,
        },
    )

    queries = np.asarray(arrays["query_vectors.float32.npy"], dtype=np.float32)
    rows = np.asarray(arrays["ann_rows.int64.npy"], dtype=np.int64)
    base_scores = np.asarray(arrays["ann_scores.float32.npy"], dtype=np.float32)
    lookup = np.asarray(arrays["ann_residual_rows.int64.npy"], dtype=np.int64)
    residuals = np.asarray(arrays["candidate_residuals.float32.npy"], dtype=np.float32)
    labels = np.asarray(arrays["candidate_relevance.uint8.npy"], dtype=np.uint8)
    relevant_counts = np.asarray(arrays["relevant_counts.int32.npy"], dtype=np.int64)
    teacher_scores = exact_candidate_scores(queries, base_scores, lookup, residuals)

    method = protocol["method"]
    mining = protocol["pair_mining"]
    optimisation = protocol["basis_optimization"]
    pairs = mine_cutoff_pairs(
        rows,
        lookup,
        base_scores,
        teacher_scores,
        labels,
        final_k=int(protocol["frozen_index_contract"]["final_cutoff"]),
        top_b=int(method["top_b"]),
        protection_window=int(mining["protection_window"]),
        max_challengers_per_positive=int(mining["maximum_challengers_per_positive"]),
        margin_temperature=float(mining["margin_temperature"]),
        damage_scale=float(mining["damage_scale"]),
        promotion_mass=float(mining["promotion_total_loss_mass"]),
    )
    support = summarize_pairs(pairs)
    support_path = args.output_dir / "pair_support.json"
    atomic_json(support_path, support)
    if not len(pairs):
        raise ValueError("V8 found no cutoff pairs")

    rank = int(method["rank"])
    pca_basis = fit_uncentered_pca_basis(residuals, rank=rank)
    pca_scales = fit_int8_scales(residuals, pca_basis)
    pca_codes, pca_quantization = encode_residuals_int8(
        residuals, pca_basis, pca_scales
    )
    pca_scores = score_sidecar_candidates(
        queries,
        rows,
        lookup,
        base_scores,
        pca_basis,
        pca_codes,
        pca_scales,
        alpha=float(method["alpha"]),
        top_b=int(method["top_b"]),
    )

    oof_rars_scores = np.full(base_scores.shape, np.nan, dtype=np.float32)
    fold_results: list[dict[str, Any]] = []
    for fold in range(int(protocol["data_policy"]["cross_validation"]["fold_count"])):
        train_queries = np.flatnonzero(folds != fold)
        validation_queries = np.flatnonzero(folds == fold)
        training_pairs = subset_pairs(pairs, train_queries)
        basis, history = fit_cutoff_aware_basis(
            queries,
            residuals,
            training_pairs,
            pca_basis,
            steps=int(optimisation["steps"]),
            learning_rate=float(optimisation["learning_rate"]),
            anchor_weight=float(optimisation["pca_anchor_weight"]),
            huber_delta=float(optimisation["huber_delta"]),
            gradient_clip=float(optimisation["gradient_norm_clip"]),
        )
        scales = fit_int8_scales(residuals, basis)
        codes, quantization = encode_residuals_int8(residuals, basis, scales)
        fold_scores = score_sidecar_candidates(
            queries[validation_queries],
            rows[validation_queries],
            lookup[validation_queries],
            base_scores[validation_queries],
            basis,
            codes,
            scales,
            alpha=float(method["alpha"]),
            top_b=int(method["top_b"]),
        )
        oof_rars_scores[validation_queries] = fold_scores
        fold_metrics = per_query_metrics(
            fold_scores,
            rows[validation_queries],
            labels[validation_queries],
            relevant_counts[validation_queries],
            k=int(protocol["frozen_index_contract"]["final_cutoff"]),
        )
        fold_results.append(
            {
                "fold": fold,
                "training_query_count": int(len(train_queries)),
                "validation_query_count": int(len(validation_queries)),
                "training_pair_support": summarize_pairs(training_pairs),
                "optimization_initial_loss": float(history[0]["loss"]),
                "optimization_final_loss": float(history[-1]["loss"]),
                "quantization": quantization,
                "validation_rars": _metric_summary(fold_metrics),
            }
        )
    if not np.all(np.isfinite(oof_rars_scores)):
        raise AssertionError("RARS OOF score matrix was not filled exactly once")

    final_basis, final_history = fit_cutoff_aware_basis(
        queries,
        residuals,
        pairs,
        pca_basis,
        steps=int(optimisation["steps"]),
        learning_rate=float(optimisation["learning_rate"]),
        anchor_weight=float(optimisation["pca_anchor_weight"]),
        huber_delta=float(optimisation["huber_delta"]),
        gradient_clip=float(optimisation["gradient_norm_clip"]),
    )
    final_scales = fit_int8_scales(residuals, final_basis)
    _, final_quantization = encode_residuals_int8(
        residuals, final_basis, final_scales
    )

    final_k = int(protocol["frozen_index_contract"]["final_cutoff"])
    base_metrics = per_query_metrics(
        base_scores, rows, labels, relevant_counts, k=final_k
    )
    pca_metrics = per_query_metrics(
        pca_scores, rows, labels, relevant_counts, k=final_k
    )
    rars_metrics = per_query_metrics(
        oof_rars_scores, rows, labels, relevant_counts, k=final_k
    )
    teacher_metrics = per_query_metrics(
        teacher_scores, rows, labels, relevant_counts, k=final_k
    )
    bootstrap_contract = protocol["development_metrics"]
    bootstrap_arguments = {
        "replicates": int(bootstrap_contract["bootstrap_replicates"]),
        "seed": int(bootstrap_contract["bootstrap_seed"]),
    }
    rars_vs_base = paired_bootstrap(
        rars_metrics["recall"], base_metrics["recall"], **bootstrap_arguments
    )
    pca_vs_base = paired_bootstrap(
        pca_metrics["recall"], base_metrics["recall"], **bootstrap_arguments
    )
    rars_vs_pca = paired_bootstrap(
        rars_metrics["recall"], pca_metrics["recall"], **bootstrap_arguments
    )
    recovery = candidate_gap_recovery(
        rars_metrics["recall"], base_metrics["recall"], teacher_metrics["recall"]
    )
    decision = development_decision(
        rars_vs_base=rars_vs_base,
        pca_vs_base=pca_vs_base,
        rars_vs_pca=rars_vs_pca,
        gap_recovery=recovery,
        pair_support=support,
        thresholds=protocol["development_gate"],
    )

    arrays_to_write = {
        "pca_basis_rank16.float32.npy": pca_basis.astype(np.float32),
        "rars_basis_rank16.float32.npy": final_basis.astype(np.float32),
        "pca_scales_rank16.float32.npy": pca_scales.astype(np.float32),
        "rars_scales_rank16.float32.npy": final_scales.astype(np.float32),
        "oof_base_recall_at_10.float64.npy": base_metrics["recall"],
        "oof_pca_recall_at_10.float64.npy": pca_metrics["recall"],
        "oof_rars_recall_at_10.float64.npy": rars_metrics["recall"],
        "oof_teacher_recall_at_10.float64.npy": teacher_metrics["recall"],
    }
    for filename, value in arrays_to_write.items():
        atomic_save(args.output_dir / filename, np.asarray(value))
    fold_path = args.output_dir / "fold_results.json"
    atomic_json(fold_path, fold_results)

    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V8_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "query_count": len(qids),
        "metrics": {
            "base": _metric_summary(base_metrics),
            "pca_oof": _metric_summary(pca_metrics),
            "rars_oof": _metric_summary(rars_metrics),
            "same_candidate_exact": _metric_summary(teacher_metrics),
        },
        "comparisons": {
            "rars_vs_base_recall_at_10": rars_vs_base,
            "pca_vs_base_recall_at_10": pca_vs_base,
            "rars_vs_pca_recall_at_10": rars_vs_pca,
        },
        "candidate_gap_recovery_fraction": recovery,
        "pair_support": support,
        "pca_quantization": pca_quantization,
        "final_rars_quantization_on_development_union": final_quantization,
        "final_optimization": {
            "initial_loss": float(final_history[0]["loss"]),
            "final_loss": float(final_history[-1]["loss"]),
            "steps": len(final_history),
        },
        "decision": decision,
        "opened_roles": ["oracle_design"],
        "future_method_holdout_opened": False,
        "oracle_audit_opened": False,
        "full_corpus_sidecar_encoded": False,
        "interpretation": (
            "Five-fold outcome-informed development evidence. These metrics are "
            "not an independent or official MS MARCO evaluation."
        ),
    }
    result_path = args.output_dir / "development_result.json"
    atomic_json(result_path, result)
    freeze = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V8_METHOD_FROZEN_AFTER_DEVELOPMENT",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "method": protocol["method"],
        "pair_mining": protocol["pair_mining"],
        "basis_optimization": protocol["basis_optimization"],
        "comparators": protocol["comparators"],
        "pca_basis": file_record(args.output_dir / "pca_basis_rank16.float32.npy"),
        "rars_basis": file_record(args.output_dir / "rars_basis_rank16.float32.npy"),
        "development_result": file_record(result_path),
        "future_access_authorized": False,
        "next_required_action": (
            "freeze a separate full-corpus artifact builder and prospective "
            "evaluation protocol; do not open future labels from this file"
        ),
    }
    freeze_path = args.output_dir / "method_freeze.json"
    atomic_json(freeze_path, freeze)
    output_records = {
        filename: file_record(args.output_dir / filename)
        for filename in (
            *arrays_to_write,
            "pair_support.json",
            "fold_results.json",
            "development_result.json",
            "method_freeze.json",
        )
    }
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V8_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "started": file_record(started_path),
        "outputs": output_records,
        "future_method_holdout_opened": False,
        "oracle_audit_opened": False,
        "full_corpus_sidecar_encoded": False,
    }
    complete_path = args.output_dir / "development_complete.json"
    atomic_json(complete_path, complete)
    missing = [
        filename
        for filename in protocol["required_development_outputs"]
        if not (args.output_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"Required V8 outputs were not written: {missing}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-role-dir", required=True, type=Path)
    parser.add_argument("--v6-packet-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
