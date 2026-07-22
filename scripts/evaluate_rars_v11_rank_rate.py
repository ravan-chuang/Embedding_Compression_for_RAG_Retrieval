#!/usr/bin/env python3
"""Execute the frozen V11 rank--rate architecture diagnostic once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v11_rank_rate_core import (  # noqa: E402
    PROTOCOL_ID,
    encode_residuals_int4,
    fit_faiss_product_quantizer,
    fit_int4_scales,
    paired_inference,
    rank_rate_decision,
    score_float_sidecar_candidates,
    score_int4_sidecar_candidates,
    score_product_sidecar_candidates,
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
    load_design_bundle,
    read_json,
    validate_runtime,
)
from verify_rars_v6_1m_headroom_packet import (  # noqa: E402
    verify_packet as verify_v6_packet,
)


CANONICAL_PROTOCOL = Path("protocols/rars_v11_rank_rate_diagnostic_v1.json")
SOURCE_FILES = (
    CANONICAL_PROTOCOL,
    Path("scripts/rars_v11_rank_rate_core.py"),
    Path("scripts/evaluate_rars_v11_rank_rate.py"),
    Path("scripts/rars_v8_cutoff_sidecar_core.py"),
    Path("scripts/train_rars_v8_cutoff_sidecar.py"),
    Path("scripts/rars_v3_oracle_core.py"),
    Path("scripts/verify_rars_v6_1m_headroom_packet.py"),
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def _reject_forbidden_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    forbidden = (
        "rars-v9",
        "rars-v10-development",
        "future_method_holdout",
        "oracle_audit",
    )
    if any(token in lowered for token in forbidden):
        raise ValueError(f"V11 refuses forbidden {label} path: {path}")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_commit(source_commit)
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must use canonical path: {canonical}")
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
        raise ValueError("V11 diagnostic requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != (
        "FROZEN_BEFORE_FIRST_V11_DIAGNOSTIC_RUN"
    ):
        raise ValueError("Unexpected V11 protocol identity or status")
    records: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = (repo_root / relative).resolve(strict=True)
        records[str(relative)] = file_record(path)
    return protocol, records


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("Refusing to reuse a non-empty V11 diagnostic output")
    path.mkdir(parents=True, exist_ok=True)


def _metric_summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(value)) for name, value in metrics.items()}


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    for path, label in (
        (args.design_role_dir, "design input"),
        (args.v6_packet_root, "V6 lineage"),
        (args.output_dir, "output"),
    ):
        _reject_forbidden_path(path, label)
    protocol, source_blobs = validate_source(
        repo_root, args.protocol, args.source_commit
    )
    environment = validate_runtime(protocol)
    v6_verification = verify_v6_packet(args.v6_packet_root)
    _prepare_output(args.output_dir)
    qids, folds, arrays, input_records = load_design_bundle(
        args.design_role_dir, protocol
    )
    started_path = args.output_dir / "diagnostic_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V11_RANK_RATE_DIAGNOSTIC_STARTED",
            "source_commit": args.source_commit,
            "environment": environment,
            "source_blobs": source_blobs,
            "inputs": input_records,
            "v6_verification": v6_verification,
            "opened_roles": ["oracle_design"],
            "v9_packet_opened": False,
            "v10_packet_opened": False,
            "future_method_holdout_opened": False,
            "cutoff_training_performed": False,
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
    scoring = protocol["shared_scoring_contract"]
    alpha = float(scoring["alpha"])
    top_b = int(scoring["top_b"])
    final_k = int(protocol["frozen_index_contract"]["final_cutoff"])

    basis64 = fit_uncentered_pca_basis(residuals, rank=64)
    basis32 = basis64[:, :32]
    basis16 = basis64[:, :16]
    pca16_scales = fit_int8_scales(residuals, basis16)
    pca16_codes, pca16_quantization = encode_residuals_int8(
        residuals, basis16, pca16_scales
    )
    pca16_int8_scores = score_sidecar_candidates(
        queries,
        rows,
        lookup,
        base_scores,
        basis16,
        pca16_codes,
        pca16_scales,
        alpha=alpha,
        top_b=top_b,
    )
    fp32_scores = {
        rank: score_float_sidecar_candidates(
            queries,
            rows,
            lookup,
            base_scores,
            residuals,
            basis64[:, :rank],
            alpha=alpha,
            top_b=top_b,
        )
        for rank in (16, 32, 64)
    }

    int4_scales = fit_int4_scales(residuals, basis32)
    int4_codes, int4_quantization = encode_residuals_int4(
        residuals, basis32, int4_scales
    )
    if int4_codes.shape[1] != 16:
        raise AssertionError("Packed rank-32 int4 payload is not 16 bytes")
    int4_scores = score_int4_sidecar_candidates(
        queries,
        rows,
        lookup,
        base_scores,
        basis32,
        int4_codes,
        int4_scales,
        alpha=alpha,
        top_b=top_b,
    )

    import faiss  # Imported only after the pinned runtime contract passes.

    rpq = protocol["rpq_training"]
    rpq_scores: dict[int, np.ndarray] = {}
    rpq_codebooks: dict[int, np.ndarray] = {}
    rpq_diagnostics: dict[int, dict[str, Any]] = {}
    for rank in (32, 64):
        coefficients = np.ascontiguousarray(
            residuals @ basis64[:, :rank], dtype=np.float32
        )
        codes, codebooks, diagnostic = fit_faiss_product_quantizer(
            coefficients,
            faiss,
            subquantizers=int(rpq["subquantizers"]),
            bits=int(rpq["bits_per_subquantizer"]),
            iterations=int(rpq["iterations"]),
            seed=int(rpq["seed"]),
            max_points_per_centroid=int(rpq["maximum_points_per_centroid"]),
        )
        if codes.shape[1] != 16:
            raise AssertionError(f"Rank-{rank} RPQ payload is not 16 bytes")
        rpq_scores[rank] = score_product_sidecar_candidates(
            queries,
            rows,
            lookup,
            base_scores,
            basis64[:, :rank],
            codes,
            codebooks,
            alpha=alpha,
            top_b=top_b,
        )
        rpq_codebooks[rank] = codebooks
        rpq_diagnostics[rank] = diagnostic

    score_matrices = {
        "base": base_scores,
        "pca_rank16_int8": pca16_int8_scores,
        "pca_rank16_fp32": fp32_scores[16],
        "pca_rank32_fp32": fp32_scores[32],
        "pca_rank64_fp32": fp32_scores[64],
        "pca_rank32_int4": int4_scores,
        "pca_rank32_rpq16x8": rpq_scores[32],
        "pca_rank64_rpq16x8": rpq_scores[64],
        "same_candidate_exact": teacher_scores,
    }
    metrics = {
        name: per_query_metrics(
            scores, rows, labels, relevant_counts, k=final_k
        )
        for name, scores in score_matrices.items()
    }
    pca_recall = metrics["pca_rank16_int8"]["recall"]
    capacity = paired_inference(
        metrics["pca_rank64_fp32"]["recall"],
        pca_recall,
        **_inference_kwargs(protocol, "rank64_fp32_vs_pca"),
    )
    encoding = paired_inference(
        metrics["pca_rank64_rpq16x8"]["recall"],
        pca_recall,
        **_inference_kwargs(protocol, "rank64_rpq_vs_pca"),
    )
    rpq_vs_base = paired_inference(
        metrics["pca_rank64_rpq16x8"]["recall"],
        metrics["base"]["recall"],
        **_inference_kwargs(protocol, "rank64_rpq_vs_base"),
    )
    fold_gains = [
        float(
            np.mean(
                metrics["pca_rank64_rpq16x8"]["recall"][folds == fold]
                - pca_recall[folds == fold]
            )
        )
        for fold in range(int(protocol["data_policy"]["cross_validation"]["fold_count"]))
    ]
    gap_recovery = candidate_gap_recovery(
        metrics["pca_rank64_rpq16x8"]["recall"],
        metrics["base"]["recall"],
        metrics["same_candidate_exact"]["recall"],
    )
    decision = rank_rate_decision(
        rank64_fp32_vs_pca=capacity,
        rank64_rpq_vs_pca=encoding,
        rank64_rpq_vs_base=rpq_vs_base,
        fold_gains_over_pca=fold_gains,
        gap_recovery=gap_recovery,
        pca_mrr=float(np.mean(metrics["pca_rank16_int8"]["mrr"])),
        rpq_mrr=float(np.mean(metrics["pca_rank64_rpq16x8"]["mrr"])),
        pca_ndcg=float(np.mean(metrics["pca_rank16_int8"]["ndcg"])),
        rpq_ndcg=float(np.mean(metrics["pca_rank64_rpq16x8"]["ndcg"])),
        thresholds=protocol["diagnostic_gate"],
    )

    arrays_to_write = {
        "pca_basis_rank64.float32.npy": basis64.astype(np.float32),
        "pca_rank16_scales.float32.npy": pca16_scales.astype(np.float32),
        "pca_rank32_int4_scales.float32.npy": int4_scales.astype(np.float32),
        "rpq_rank32_codebooks.float32.npy": rpq_codebooks[32].astype(np.float32),
        "rpq_rank64_codebooks.float32.npy": rpq_codebooks[64].astype(np.float32),
        "per_query_base_recall_at_10.float64.npy": metrics["base"]["recall"],
        "per_query_pca_rank16_int8_recall_at_10.float64.npy": pca_recall,
        "per_query_pca_rank16_fp32_recall_at_10.float64.npy": metrics["pca_rank16_fp32"]["recall"],
        "per_query_pca_rank32_fp32_recall_at_10.float64.npy": metrics["pca_rank32_fp32"]["recall"],
        "per_query_pca_rank64_fp32_recall_at_10.float64.npy": metrics["pca_rank64_fp32"]["recall"],
        "per_query_pca_rank32_int4_recall_at_10.float64.npy": metrics["pca_rank32_int4"]["recall"],
        "per_query_pca_rank32_rpq_recall_at_10.float64.npy": metrics["pca_rank32_rpq16x8"]["recall"],
        "per_query_pca_rank64_rpq_recall_at_10.float64.npy": metrics["pca_rank64_rpq16x8"]["recall"],
        "per_query_same_candidate_exact_recall_at_10.float64.npy": metrics["same_candidate_exact"]["recall"],
    }
    for filename, values in arrays_to_write.items():
        atomic_save(args.output_dir / filename, np.asarray(values))

    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V11_RANK_RATE_DIAGNOSTIC_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "evidence_tier": protocol["evidence_boundary"]["tier"],
        "query_count": len(qids),
        "candidate_residual_count": int(len(residuals)),
        "metrics": {name: _metric_summary(value) for name, value in metrics.items()},
        "comparisons": {
            "rank64_fp32_vs_pca_rank16_int8": capacity,
            "rank64_rpq_vs_pca_rank16_int8": encoding,
            "rank64_rpq_vs_base": rpq_vs_base,
        },
        "rank64_rpq_fold_gains_over_pca": fold_gains,
        "rank64_rpq_candidate_gap_recovery_fraction": gap_recovery,
        "rank64_headroom_retention_fraction": decision[
            "rank64_headroom_retention_fraction"
        ],
        "pca_rank16_quantization": pca16_quantization,
        "pca_rank32_int4_quantization": int4_quantization,
        "rpq_diagnostics": {
            "rank32": rpq_diagnostics[32],
            "rank64": rpq_diagnostics[64],
        },
        "decision": decision,
        "opened_roles": ["oracle_design"],
        "v9_packet_opened": False,
        "v10_packet_opened": False,
        "future_method_holdout_opened": False,
        "cutoff_training_performed": False,
        "fresh_confirmation_access_authorized": False,
        "interpretation": (
            "This is a fixed architecture screen. A GO permits only writing a "
            "separate cutoff-aware CA-RPQ development protocol on fresh "
            "development data; it is not algorithm confirmation."
        ),
    }
    result_path = args.output_dir / "rank_rate_result.json"
    atomic_json(result_path, result)
    atomic_json(
        args.output_dir / "diagnostic_freeze.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V11_RANK_RATE_DIAGNOSTIC_CLOSED",
            "source_commit": args.source_commit,
            "formal_decision": decision["decision"],
            "shared_scoring_contract": protocol["shared_scoring_contract"],
            "rpq_training": protocol["rpq_training"],
            "diagnostic_gate": protocol["diagnostic_gate"],
            "rank_rate_result": file_record(result_path),
            "cutoff_training_performed": False,
            "old_holdout_reuse_authorized": False,
            "fresh_confirmation_access_authorized": False,
        },
    )
    outputs = {
        filename: file_record(args.output_dir / filename)
        for filename in protocol["required_outputs"]
        if filename not in ("diagnostic_started.json", "diagnostic_complete.json")
    }
    atomic_json(
        args.output_dir / "diagnostic_complete.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V11_RANK_RATE_DIAGNOSTIC_COMPLETE",
            "source_commit": args.source_commit,
            "formal_decision": decision["decision"],
            "started": file_record(started_path),
            "outputs": outputs,
            "cutoff_training_performed": False,
            "old_holdout_reuse_authorized": False,
        },
    )
    missing = [
        filename
        for filename in protocol["required_outputs"]
        if not (args.output_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"Required V11 outputs missing: {missing}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-role-dir", type=Path, required=True)
    parser.add_argument("--v6-packet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
