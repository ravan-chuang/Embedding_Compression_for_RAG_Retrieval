#!/usr/bin/env python3
"""Run the frozen V15 cross-fitted selective uniform-RPQ gate development."""

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
    fit_faiss_product_quantizer,
    paired_inference,
    score_product_sidecar_candidates,
)
from rars_v13_signed_score_core import assign_product_codes  # noqa: E402
from rars_v15_selective_gate_core import (  # noqa: E402
    FEATURE_NAMES,
    PROTOCOL_ID,
    apply_query_gate,
    fit_weighted_ridge_gate,
    gate_utility_scores,
    query_gate_features,
    select_calibrated_threshold,
    selective_gate_decision,
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
from verify_rars_v13_committed_closure import verify_closure as verify_v13_closure  # noqa: E402
from verify_rars_v13_signed_score_rpq_packet import verify_packet as verify_v13_packet  # noqa: E402
from verify_rars_v14_committed_closure import verify_closure as verify_v14_closure  # noqa: E402


CANONICAL_PROTOCOL = Path("protocols/rars_v15_cross_fitted_selective_rpq_gate_v1.json")
V13_PROTOCOL = Path("protocols/rars_v13_signed_score_distilled_rpq_v1.json")
V13_SOURCE_COMMIT = "d8cb761c289fe17ea2c2bfb92059e8b5553cfd74"
SOURCE_FILES = (
    CANONICAL_PROTOCOL,
    Path("scripts/rars_v15_selective_gate_core.py"),
    Path("scripts/evaluate_rars_v15_selective_gate.py"),
    Path("scripts/verify_rars_v15_selective_gate_packet.py"),
    Path("scripts/verify_rars_v14_committed_closure.py"),
    Path("scripts/verify_rars_v13_committed_closure.py"),
    Path("scripts/verify_rars_v13_signed_score_rpq_packet.py"),
    Path("scripts/train_rars_v13_signed_score_rpq.py"),
    Path("scripts/rars_v13_signed_score_core.py"),
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
        raise ValueError(f"V15 requires canonical protocol path: {canonical}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or status:
        raise ValueError("V15 development requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != (
        "FROZEN_BEFORE_FIRST_V15_DEVELOPMENT_RUN"
    ):
        raise ValueError("Unexpected V15 protocol identity or status")
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


def _serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }


def _load_parent_metrics(
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
        values = {
            metric: np.load(
                packet_root / f"per_query_{prefix}_{metric}_at_10.float64.npy",
                allow_pickle=False,
            )
            for metric in METRICS
        }
        if any(value.shape != (len(qids),) or value.dtype != np.float64 for value in values.values()):
            raise ValueError(f"V13 parent metric contract changed for {prefix}")
        return values

    return {
        "base": load("base"),
        "same_candidate_exact": load("same_candidate_exact"),
        "pca16": load("pca16"),
        "uniform": {seed: load(f"unsupervised_seed{seed}") for seed in seeds},
    }


def _fit_uniform_geometry(
    residuals: np.ndarray,
    lookup: np.ndarray,
    training_queries: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    method = protocol["uniform_sidecar"]
    training_rows = np.unique(lookup[training_queries].reshape(-1))
    basis = fit_uncentered_pca_basis(
        residuals[training_rows], rank=int(method["rank"])
    )
    coefficients = np.asarray(residuals @ basis, dtype=np.float32)
    return training_rows, basis, coefficients


def _fit_uniform_quantizer(
    coefficients: np.ndarray,
    training_rows: np.ndarray,
    protocol: dict[str, Any],
    faiss_module: Any,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = protocol["uniform_sidecar"]
    training = protocol["rpq_training"]
    _, codebooks, summary = fit_faiss_product_quantizer(
        coefficients[training_rows],
        faiss_module,
        subquantizers=int(method["subquantizers"]),
        bits=int(method["bits_per_subquantizer"]),
        iterations=int(training["iterations"]),
        seed=seed,
        max_points_per_centroid=int(training["maximum_points_per_centroid"]),
    )
    codes = assign_product_codes(coefficients, codebooks)
    return codebooks, codes, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = validate_source(
        repo_root, args.protocol, args.source_commit
    )
    environment = validate_runtime(protocol)
    if environment["faiss_version"] != protocol["execution_environment_contract"]["faiss_version"]:
        raise ValueError("V15 Faiss version differs from the protocol")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty V15 output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v13_closure = verify_v13_closure(
        repo_root / "results/rars_v13_signed_score_rpq", repo_root
    )
    v14_closure = verify_v14_closure(
        repo_root / "results/rars_v14_anisotropic_rate_rpq", repo_root
    )
    if v13_closure["formal_decision"] != protocol["parent_evidence"]["v13_formal_decision"]:
        raise ValueError("Committed V13 parent decision changed")
    if v14_closure["formal_decision"] != protocol["parent_evidence"]["v14_formal_decision"]:
        raise ValueError("Committed V14 parent decision changed")
    v13_verification = verify_v13_packet(args.v13_packet_root, repo_root)
    if v13_verification["source_commit"] != V13_SOURCE_COMMIT:
        raise ValueError("V13 Drive packet source commit changed")
    v13_protocol = read_json(repo_root / V13_PROTOCOL)
    qids, folds, arrays, bundle_records = load_bundle(
        args.v13_bundle_root, v13_protocol, repo_root, V13_SOURCE_COMMIT
    )
    seeds = [int(value) for value in protocol["rpq_training"]["seeds"]]
    primary_seed = int(protocol["rpq_training"]["primary_seed"])
    parent = _load_parent_metrics(args.v13_packet_root, qids, folds, seeds)
    parent_payload = args.v13_packet_root / "full_corpus_signed_score_assignments.uint8.memmap"
    _verify_record(
        parent_payload,
        read_json(args.v13_packet_root / "development_result.json")["final_fit"][
            "full_corpus_codes"
        ]["record"],
        "V13 uniform full-corpus payload",
    )

    started_path = args.output_dir / "development_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V15_SELECTIVE_GATE_DEVELOPMENT_STARTED",
            "source_commit": args.source_commit,
            "environment": environment,
            "source_blobs": source_blobs,
            "inputs": {
                **bundle_records,
                "v13_packet_result": file_record(
                    args.v13_packet_root / "development_result.json"
                ),
                "v13_uniform_payload": file_record(parent_payload),
            },
            "v13_committed_closure": v13_closure,
            "v14_committed_closure": v14_closure,
            "v13_packet_verification": v13_verification,
            "evidence_tier": protocol["evidence_boundary"]["tier"],
            "labels_used_for_gate_fitting": True,
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
    top_b = int(protocol["uniform_sidecar"]["top_b"])
    fold_count = int(protocol["cross_validation"]["outer_fold_count"])
    gate_cfg = protocol["gate"]
    all_indices = np.arange(len(qids), dtype=np.int64)
    base_metrics = per_query_metrics(
        base_scores, rows, labels, relevant_counts, k=final_k
    )
    for metric in METRICS:
        if not np.array_equal(base_metrics[metric], parent["base"][metric]):
            raise ValueError(f"V15 Base metric no longer matches V13: {metric}")

    selective = {
        metric: np.full((len(seeds), len(qids)), np.nan, dtype=np.float64)
        for metric in METRICS
    }
    gate_features = np.full(
        (len(seeds), len(qids), len(FEATURE_NAMES)), np.nan, dtype=np.float64
    )
    gate_scores = np.full((len(seeds), len(qids)), np.nan, dtype=np.float64)
    gate_applied = np.zeros((len(seeds), len(qids)), dtype=np.uint8)
    audit_features = np.full(
        (fold_count, len(seeds), len(qids), len(FEATURE_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    audit_sidecar_metrics = np.full(
        (fold_count, len(seeds), len(qids), len(METRICS)),
        np.nan,
        dtype=np.float64,
    )
    diagnostics: list[dict[str, Any]] = []

    for outer_fold in range(fold_count):
        heldout = np.flatnonzero(folds == outer_fold)
        representation_train = np.flatnonzero(folds != outer_fold)
        calibration_fold = (outer_fold + 1) % fold_count
        calibration = np.flatnonzero(folds == calibration_fold)
        gate_fit = np.flatnonzero((folds != outer_fold) & (folds != calibration_fold))
        training_rows, basis, coefficients = _fit_uniform_geometry(
            residuals, lookup, representation_train, protocol
        )
        for seed_index, seed in enumerate(seeds):
            codebooks, codes, rpq_summary = _fit_uniform_quantizer(
                coefficients,
                training_rows,
                protocol,
                faiss,
                seed,
            )
            representation_summary = {
                "training_query_count": int(len(representation_train)),
                "training_residual_count": int(len(training_rows)),
                "rpq": rpq_summary,
            }
            sidecar_scores = score_product_sidecar_candidates(
                queries,
                rows,
                lookup,
                base_scores,
                basis,
                codes,
                codebooks,
                alpha=float(protocol["uniform_sidecar"]["alpha"]),
                top_b=top_b,
            )
            features = query_gate_features(
                base_scores,
                sidecar_scores,
                rows,
                final_k=final_k,
                top_b=top_b,
            )
            audit_features[outer_fold, seed_index] = features
            all_sidecar_metrics = per_query_metrics(
                sidecar_scores, rows, labels, relevant_counts, k=final_k
            )
            for metric_index, metric in enumerate(METRICS):
                audit_sidecar_metrics[outer_fold, seed_index, :, metric_index] = (
                    all_sidecar_metrics[metric]
                )
            fit_base = {metric: base_metrics[metric][gate_fit] for metric in METRICS}
            fit_sidecar = {
                metric: all_sidecar_metrics[metric][gate_fit] for metric in METRICS
            }
            model = fit_weighted_ridge_gate(
                features[gate_fit],
                fit_sidecar["recall"] - fit_base["recall"],
                ridge=float(gate_cfg["ridge"]),
                neutral_weight=float(gate_cfg["neutral_weight"]),
                harm_weight=float(gate_cfg["harm_weight"]),
                minimum_scale=float(gate_cfg["minimum_feature_scale"]),
            )
            utility = gate_utility_scores(features, model)
            calibration_base = {
                metric: base_metrics[metric][calibration] for metric in METRICS
            }
            calibration_sidecar = {
                metric: all_sidecar_metrics[metric][calibration]
                for metric in METRICS
            }
            threshold = select_calibrated_threshold(
                utility[calibration],
                calibration_base,
                calibration_sidecar,
                quantile_grid=[float(value) for value in gate_cfg["threshold_quantiles"]],
                minimum_coverage=float(gate_cfg["calibration_minimum_coverage"]),
                maximum_coverage=float(gate_cfg["calibration_maximum_coverage"]),
                minimum_mrr_change=float(
                    gate_cfg["calibration_minimum_mrr_change_vs_always_on"]
                ),
                minimum_ndcg_change=float(
                    gate_cfg["calibration_minimum_ndcg_change_vs_always_on"]
                ),
            )
            heldout_base = {
                metric: base_metrics[metric][heldout] for metric in METRICS
            }
            heldout_sidecar = {
                metric: all_sidecar_metrics[metric][heldout] for metric in METRICS
            }
            for metric in METRICS:
                if not np.array_equal(
                    heldout_sidecar[metric], parent["uniform"][seed][metric][heldout]
                ):
                    raise ValueError(
                        f"V15 failed to reproduce V13 uniform {metric}, fold={outer_fold}, seed={seed}"
                    )
            applied = apply_query_gate(utility[heldout], float(threshold["threshold"]))
            for metric in METRICS:
                selective[metric][seed_index, heldout] = np.where(
                    applied,
                    heldout_sidecar[metric],
                    heldout_base[metric],
                )
            gate_features[seed_index, heldout] = features[heldout]
            gate_scores[seed_index, heldout] = utility[heldout]
            gate_applied[seed_index, heldout] = applied.astype(np.uint8)
            diagnostics.append(
                {
                    "outer_fold": outer_fold,
                    "calibration_fold": calibration_fold,
                    "seed": seed,
                    "gate_fit_query_count": int(len(gate_fit)),
                    "calibration_query_count": int(len(calibration)),
                    "heldout_query_count": int(len(heldout)),
                    "representation": representation_summary,
                    "gate_model": _serializable_model(model),
                    "calibration": threshold,
                    "heldout_coverage": float(np.mean(applied)),
                    "heldout_uniform_metrics": _metric_summary(heldout_sidecar),
                    "heldout_selective_metrics": {
                        metric: float(np.mean(selective[metric][seed_index, heldout]))
                        for metric in METRICS
                    },
                }
            )

    if any(np.any(~np.isfinite(value)) for value in selective.values()):
        raise ValueError("V15 OOF selective metric arrays are incomplete")
    if np.any(~np.isfinite(gate_features)) or np.any(~np.isfinite(gate_scores)):
        raise ValueError("V15 OOF gate arrays are incomplete")
    if np.any(~np.isfinite(audit_features)) or np.any(
        ~np.isfinite(audit_sidecar_metrics)
    ):
        raise ValueError("V15 cross-fit audit arrays are incomplete")

    # Export-only full-development gate. It cannot alter the OOF decision.
    final_training_rows, final_basis, final_coefficients = _fit_uniform_geometry(
        residuals, lookup, all_indices, protocol
    )
    final_codebooks, final_codes, final_rpq_summary = _fit_uniform_quantizer(
        final_coefficients,
        final_training_rows,
        protocol,
        faiss,
        primary_seed,
    )
    final_representation = {
        "training_query_count": int(len(all_indices)),
        "training_residual_count": int(len(final_training_rows)),
        "rpq": final_rpq_summary,
    }
    parent_basis = np.load(
        args.v13_packet_root / "final_pca_basis_rank64.float32.npy", allow_pickle=False
    )
    parent_codebooks = np.load(
        args.v13_packet_root / "final_unsupervised_codebooks.float32.npy",
        allow_pickle=False,
    )
    if not np.array_equal(final_basis, parent_basis) or not np.array_equal(
        final_codebooks, parent_codebooks
    ):
        raise ValueError("V15 final representation differs from the frozen V13 uniform sidecar")
    final_sidecar_scores = score_product_sidecar_candidates(
        queries,
        rows,
        lookup,
        base_scores,
        final_basis,
        final_codes,
        final_codebooks,
        alpha=float(protocol["uniform_sidecar"]["alpha"]),
        top_b=top_b,
    )
    final_features = query_gate_features(
        base_scores, final_sidecar_scores, rows, final_k=final_k, top_b=top_b
    )
    final_sidecar_metrics = per_query_metrics(
        final_sidecar_scores, rows, labels, relevant_counts, k=final_k
    )
    final_gain = final_sidecar_metrics["recall"] - base_metrics["recall"]
    final_oof_utility = np.full(len(qids), np.nan, dtype=np.float64)
    for fold in range(fold_count):
        heldout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        model = fit_weighted_ridge_gate(
            final_features[training],
            final_gain[training],
            ridge=float(gate_cfg["ridge"]),
            neutral_weight=float(gate_cfg["neutral_weight"]),
            harm_weight=float(gate_cfg["harm_weight"]),
            minimum_scale=float(gate_cfg["minimum_feature_scale"]),
        )
        final_oof_utility[heldout] = gate_utility_scores(final_features[heldout], model)
    final_threshold = select_calibrated_threshold(
        final_oof_utility,
        base_metrics,
        final_sidecar_metrics,
        quantile_grid=[float(value) for value in gate_cfg["threshold_quantiles"]],
        minimum_coverage=float(gate_cfg["calibration_minimum_coverage"]),
        maximum_coverage=float(gate_cfg["calibration_maximum_coverage"]),
        minimum_mrr_change=float(gate_cfg["calibration_minimum_mrr_change_vs_always_on"]),
        minimum_ndcg_change=float(gate_cfg["calibration_minimum_ndcg_change_vs_always_on"]),
    )
    final_model = fit_weighted_ridge_gate(
        final_features,
        final_gain,
        ridge=float(gate_cfg["ridge"]),
        neutral_weight=float(gate_cfg["neutral_weight"]),
        harm_weight=float(gate_cfg["harm_weight"]),
        minimum_scale=float(gate_cfg["minimum_feature_scale"]),
    )
    global_model_bytes = int(
        np.asarray(final_model["mean"], dtype=np.float64).nbytes
        + np.asarray(final_model["scale"], dtype=np.float64).nbytes
        + np.asarray(final_model["coefficients"], dtype=np.float64).nbytes
        + np.asarray([final_threshold["threshold"]], dtype=np.float64).nbytes
    )

    primary_index = seeds.index(primary_seed)
    primary_uniform = parent["uniform"][primary_seed]
    primary_selective = {
        metric: selective[metric][primary_index] for metric in METRICS
    }
    comparisons = {
        "selective_vs_uniform_rpq": paired_inference(
            primary_selective["recall"],
            primary_uniform["recall"],
            **_inference_kwargs(protocol, "primary_vs_uniform_rpq"),
        ),
        "selective_vs_base": paired_inference(
            primary_selective["recall"],
            parent["base"]["recall"],
            **_inference_kwargs(protocol, "primary_vs_base"),
        ),
        "selective_vs_pca16": paired_inference(
            primary_selective["recall"],
            parent["pca16"]["recall"],
            **_inference_kwargs(protocol, "primary_vs_pca16"),
        ),
    }
    seed_gains = [
        float(np.mean(selective["recall"][index] - parent["uniform"][seed]["recall"]))
        for index, seed in enumerate(seeds)
    ]
    fold_gains = [
        float(
            np.mean(
                primary_selective["recall"][folds == fold]
                - primary_uniform["recall"][folds == fold]
            )
        )
        for fold in range(fold_count)
    ]
    primary_difference = primary_selective["recall"] - primary_uniform["recall"]
    applied_coverages = [float(np.mean(gate_applied[index])) for index in range(len(seeds))]
    decision = selective_gate_decision(
        primary_vs_uniform=comparisons["selective_vs_uniform_rpq"],
        primary_vs_base=comparisons["selective_vs_base"],
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        uniform_mrr=float(np.mean(primary_uniform["mrr"])),
        selective_mrr=float(np.mean(primary_selective["mrr"])),
        uniform_ndcg=float(np.mean(primary_uniform["ndcg"])),
        selective_ndcg=float(np.mean(primary_selective["ndcg"])),
        applied_coverages=applied_coverages,
        improved_queries=int(np.sum(primary_difference > 0)),
        harmed_queries=int(np.sum(primary_difference < 0)),
        parent_payload_bytes_per_document=int(
            protocol["storage_contract"]["parent_payload_bytes_per_document"]
        ),
        extra_document_bytes=0,
        global_model_bytes=global_model_bytes,
        thresholds=protocol["development_gate"],
    )
    gap = candidate_gap_recovery(
        primary_selective["recall"],
        parent["base"]["recall"],
        parent["same_candidate_exact"]["recall"],
    )
    oracle = np.maximum(primary_uniform["recall"], parent["base"]["recall"])

    output_names: list[str] = []
    qids_path = args.output_dir / "query_ids.utf8.txt"
    qids_path.write_text("\n".join(qids) + "\n", encoding="utf-8")
    output_names.append(qids_path.name)
    atomic_save(args.output_dir / "fold_ids.int64.npy", folds)
    output_names.append("fold_ids.int64.npy")
    for prefix in ("base", "same_candidate_exact", "pca16"):
        output_names += _save_metric_arrays(args.output_dir, prefix, parent[prefix])
    for seed_index, seed in enumerate(seeds):
        output_names += _save_metric_arrays(
            args.output_dir, f"uniform_seed{seed}", parent["uniform"][seed]
        )
        output_names += _save_metric_arrays(
            args.output_dir,
            f"selective_seed{seed}",
            {metric: selective[metric][seed_index] for metric in METRICS},
        )
        for prefix, array in (
            ("gate_features", gate_features[seed_index]),
            ("gate_utility", gate_scores[seed_index]),
            ("gate_applied", gate_applied[seed_index]),
        ):
            name = f"oof_{prefix}_seed{seed}.npy"
            atomic_save(args.output_dir / name, array)
            output_names.append(name)
    diagnostics_path = args.output_dir / "fold_seed_gate_diagnostics.json"
    atomic_json(diagnostics_path, diagnostics)
    output_names.append(diagnostics_path.name)
    for name, array in (
        ("crossfit_gate_features.float64.npy", audit_features),
        ("crossfit_sidecar_metrics.float64.npy", audit_sidecar_metrics),
    ):
        atomic_save(args.output_dir / name, array)
        output_names.append(name)
    exports = {
        "final_gate_features.float64.npy": final_features,
        "final_oof_gate_utility.float64.npy": final_oof_utility,
        "final_sidecar_metrics.float64.npy": np.column_stack(
            [final_sidecar_metrics[metric] for metric in METRICS]
        ),
        "final_gate_feature_mean.float64.npy": np.asarray(final_model["mean"], dtype=np.float64),
        "final_gate_feature_scale.float64.npy": np.asarray(final_model["scale"], dtype=np.float64),
        "final_gate_coefficients.float64.npy": np.asarray(final_model["coefficients"], dtype=np.float64),
        "final_gate_threshold.float64.npy": np.asarray([final_threshold["threshold"]], dtype=np.float64),
    }
    for name, value in exports.items():
        atomic_save(args.output_dir / name, value)
        output_names.append(name)

    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V15_SELECTIVE_GATE_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "evidence_tier": protocol["evidence_boundary"]["tier"],
        "query_count": len(qids),
        "metrics": {
            "base": _metric_summary(parent["base"]),
            "same_candidate_exact": _metric_summary(parent["same_candidate_exact"]),
            "pca16": _metric_summary(parent["pca16"]),
            "uniform_primary": _metric_summary(primary_uniform),
            "selective_primary": _metric_summary(primary_selective),
        },
        "comparisons": comparisons,
        "seed_gains": seed_gains,
        "fold_gains": fold_gains,
        "applied_coverages": applied_coverages,
        "candidate_gap_recovery_fraction": gap,
        "oracle_selective_gain_over_uniform": float(np.mean(oracle - primary_uniform["recall"])),
        "decision": decision,
        "final_export": {
            "representation": final_representation,
            "parent_basis": file_record(
                args.v13_packet_root / "final_pca_basis_rank64.float32.npy"
            ),
            "parent_codebooks": file_record(
                args.v13_packet_root / "final_unsupervised_codebooks.float32.npy"
            ),
            "parent_payload": file_record(parent_payload),
            "gate_model": _serializable_model(final_model),
            "gate_threshold": final_threshold,
            "global_model_bytes": global_model_bytes,
            "additional_document_bytes": 0,
        },
        "labels_used_for_gate_fitting": True,
        "labels_used_for_representation_learning": False,
        "future_method_holdout_opened": False,
        "old_rars_holdout_opened": False,
        "fresh_query_access_authorized": False,
        "interpretation": "Outcome-informed query-gate development only. GO authorizes only writing a disjoint fresh-query protocol.",
    }
    result_path = args.output_dir / "development_result.json"
    atomic_json(result_path, result)
    output_names.append(result_path.name)
    freeze_path = args.output_dir / "method_freeze.json"
    atomic_json(
        freeze_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V15_METHOD_AND_DECISION_FROZEN",
            "source_commit": args.source_commit,
            "formal_decision": decision["decision"],
            "uniform_sidecar": protocol["uniform_sidecar"],
            "gate": protocol["gate"],
            "development_gate": protocol["development_gate"],
            "development_result": file_record(result_path),
            "fresh_query_access_authorized": False,
        },
    )
    output_names.append(freeze_path.name)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V15_SELECTIVE_GATE_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "started": file_record(started_path),
        "outputs": {
            name: file_record(args.output_dir / name)
            for name in sorted(set(output_names))
        },
        "fresh_query_access_authorized": False,
    }
    atomic_json(args.output_dir / "development_complete.json", complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-bundle-root", type=Path, required=True)
    parser.add_argument("--v13-packet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
