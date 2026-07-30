#!/usr/bin/env python3
"""Run one frozen V10 post-confirmation development configuration.

Only the historically opened ``oracle_design`` role is accepted.  The script
has no V9 input argument and refuses paths bearing V9/future-role identities.
It performs five-fold OOF comparison against the identical rank-16 int8 PCA
sidecar and never authorizes reuse of the already opened V9 holdout.
"""

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

from rars_v10_stable_core import (  # noqa: E402
    PROTOCOL_ID,
    build_objective_batch,
    fit_stable_basis,
    gradient_direction_audit,
    paired_inference,
    scalar_quantization_headroom_decision,
    score_float_sidecar_candidates,
    stable_development_decision,
)
from rars_v8_cutoff_sidecar_core import (  # noqa: E402
    CutoffPairBatch,
    candidate_gap_recovery,
    encode_residuals_int8,
    fit_int8_scales,
    fit_uncentered_pca_basis,
    mine_cutoff_pairs,
    per_query_metrics,
    query_role_balanced_weights,
    score_sidecar_candidates,
    subset_pairs,
    summarize_pairs,
)
from train_rars_v8_cutoff_sidecar import (  # noqa: E402
    atomic_json,
    atomic_save,
    exact_candidate_scores,
    file_record,
    load_design_bundle,
    prepare_output,
    read_json,
    validate_runtime,
)
from verify_rars_v6_1m_headroom_packet import (  # noqa: E402
    verify_packet as verify_v6_packet,
)


CANONICAL_PROTOCOL = Path(
    "protocols/rars_v10_pca_anchored_harm_constrained_v1.json"
)
SOURCE_FILES = (
    CANONICAL_PROTOCOL,
    Path("scripts/rars_v10_stable_core.py"),
    Path("scripts/train_rars_v10_stable_sidecar.py"),
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
    forbidden = ("rars-v9", "future_method_holdout", "oracle_audit")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"V10 refuses forbidden {label} path: {path}")


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
        raise ValueError("V10 development requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != (
        "FROZEN_BEFORE_FIRST_V10_DEVELOPMENT_RUN"
    ):
        raise ValueError("Unexpected V10 protocol identity or status")
    records: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = (repo_root / relative).resolve(strict=True)
        records[str(relative)] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return protocol, records


def _first_pairs(batch: CutoffPairBatch, limit: int) -> CutoffPairBatch:
    if limit <= 0:
        raise ValueError("audit pair limit must be positive")
    count = min(limit, len(batch))
    values = {
        field: np.asarray(getattr(batch, field))[:count].copy()
        for field in batch.__dataclass_fields__
    }
    values["balanced_weight"] = query_role_balanced_weights(
        values["query"], values["kind"], values["raw_weight"]
    )
    return CutoffPairBatch(**values)


def _metric_summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(value)) for name, value in metrics.items()}


def _history_monotone(history: list[dict[str, Any]]) -> bool:
    accepted = [
        float(row["post_retraction_loss"])
        for row in history
        if row["accepted"]
    ]
    return all(
        later <= earlier + 1e-12
        for earlier, later in zip(accepted, accepted[1:])
    )


def _optimizer_arguments(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    objective = protocol["objective"]
    optimization = protocol["optimization"]
    objective_kwargs = {
        "alpha": float(protocol["method"]["alpha"]),
        "distillation_weight": float(objective["distillation_weight"]),
        "cutoff_weight": float(objective["cutoff_weight"]),
        "harm_weight": float(objective["tail_harm_weight"]),
        "anchor_weight": float(objective["pca_anchor_weight"]),
        "huber_delta": float(objective["huber_delta"]),
        "cutoff_temperature": float(objective["cutoff_temperature"]),
        "margin_floor": float(objective["margin_floor"]),
        "harm_scale": float(objective["harm_scale"]),
        "cvar_fraction": float(objective["cvar_fraction"]),
    }
    optimizer_kwargs = {
        "maximum_steps": int(optimization["maximum_steps"]),
        "initial_step_size": float(optimization["initial_step_size"]),
        "backtracking_factor": float(optimization["backtracking_factor"]),
        "armijo_constant": float(optimization["armijo_constant"]),
        "maximum_backtracks": int(optimization["maximum_backtracks"]),
        "maximum_principal_angle": float(
            optimization["maximum_principal_angle_degrees_from_pca"]
        ),
        "gradient_tolerance": float(optimization["gradient_tolerance"]),
    }
    return objective_kwargs, optimizer_kwargs


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
    prepare_output(args.output_dir)
    qids, folds, arrays, input_records = load_design_bundle(
        args.design_role_dir, protocol
    )
    started_path = args.output_dir / "development_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V10_DEVELOPMENT_STARTED",
            "source_commit": args.source_commit,
            "environment": environment,
            "source_blobs": source_blobs,
            "inputs": input_records,
            "v6_verification": v6_verification,
            "opened_roles": ["oracle_design"],
            "v9_files_opened": False,
            "future_method_holdout_opened": False,
            "configuration_count": 1,
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

    method = protocol["method"]
    mining = protocol["pair_mining"]
    pairs = mine_cutoff_pairs(
        rows,
        lookup,
        base_scores,
        teacher_scores,
        labels,
        final_k=int(protocol["frozen_index_contract"]["final_cutoff"]),
        top_b=int(method["top_b"]),
        protection_window=int(mining["protection_window"]),
        max_challengers_per_positive=int(
            mining["maximum_challengers_per_positive"]
        ),
        margin_temperature=float(mining["margin_temperature"]),
        damage_scale=float(mining["damage_scale"]),
        promotion_mass=float(mining["promotion_total_loss_mass"]),
    )
    if not len(pairs):
        raise ValueError("V10 found no cutoff pairs")
    pair_support = summarize_pairs(pairs)
    atomic_json(args.output_dir / "pair_support.json", pair_support)

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
    pca_fp32_scores = score_float_sidecar_candidates(
        queries,
        rows,
        lookup,
        base_scores,
        residuals,
        pca_basis,
        alpha=float(method["alpha"]),
        top_b=int(method["top_b"]),
    )
    objective_kwargs, optimizer_kwargs = _optimizer_arguments(protocol)
    audit_contract = protocol["optimizer_audit"]
    audit_limit = int(audit_contract["pair_limit"])
    audit_epsilon = float(audit_contract["finite_difference_epsilon"])
    audit_tolerance = float(
        audit_contract["maximum_relative_directional_derivative_error"]
    )

    oof_scores = np.full(base_scores.shape, np.nan, dtype=np.float32)
    fold_results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    fold_gains: list[float] = []
    fold_count = int(protocol["data_policy"]["cross_validation"]["fold_count"])
    final_k = int(protocol["frozen_index_contract"]["final_cutoff"])
    for fold in range(fold_count):
        train_queries = np.flatnonzero(folds != fold)
        validation_queries = np.flatnonzero(folds == fold)
        training_pairs = subset_pairs(pairs, train_queries)
        objective_batch = build_objective_batch(
            queries, residuals, training_pairs, pca_basis
        )
        audit_batch = build_objective_batch(
            queries, residuals, _first_pairs(training_pairs, audit_limit), pca_basis
        )
        audit = gradient_direction_audit(
            pca_basis,
            audit_batch,
            pca_basis,
            objective_kwargs,
            epsilon=audit_epsilon,
            maximum_relative_error=audit_tolerance,
        )
        audit["fit"] = f"fold_{fold}"
        audits.append(audit)
        if audit["status"] != "PASS":
            raise ValueError(f"Fold {fold} gradient audit failed: {audit}")
        basis, history = fit_stable_basis(
            objective_batch,
            pca_basis,
            objective_kwargs,
            **optimizer_kwargs,
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
        oof_scores[validation_queries] = fold_scores
        v10_metrics = per_query_metrics(
            fold_scores,
            rows[validation_queries],
            labels[validation_queries],
            relevant_counts[validation_queries],
            k=final_k,
        )
        pca_fold_metrics = per_query_metrics(
            pca_scores[validation_queries],
            rows[validation_queries],
            labels[validation_queries],
            relevant_counts[validation_queries],
            k=final_k,
        )
        gain = float(
            np.mean(v10_metrics["recall"] - pca_fold_metrics["recall"])
        )
        fold_gains.append(gain)
        fold_results.append(
            {
                "fold": fold,
                "training_query_count": int(len(train_queries)),
                "validation_query_count": int(len(validation_queries)),
                "training_pair_support": summarize_pairs(training_pairs),
                "gradient_audit": audit,
                "accepted_losses_monotone": _history_monotone(history),
                "initial_loss": float(history[0]["post_retraction_loss"]),
                "final_loss": float(
                    [row for row in history if row["accepted"]][-1][
                        "post_retraction_loss"
                    ]
                ),
                "accepted_steps": int(sum(row["accepted"] for row in history) - 1),
                "maximum_principal_angle_degrees": float(
                    max(row["maximum_principal_angle_degrees"] for row in history)
                ),
                "quantization": quantization,
                "validation_v10": _metric_summary(v10_metrics),
                "validation_pca": _metric_summary(pca_fold_metrics),
                "v10_minus_pca_recall_at_10": gain,
                "optimization_history": history,
            }
        )
    if not np.all(np.isfinite(oof_scores)):
        raise AssertionError("V10 OOF score matrix was not filled exactly once")

    full_batch = build_objective_batch(queries, residuals, pairs, pca_basis)
    full_audit_batch = build_objective_batch(
        queries, residuals, _first_pairs(pairs, audit_limit), pca_basis
    )
    full_audit = gradient_direction_audit(
        pca_basis,
        full_audit_batch,
        pca_basis,
        objective_kwargs,
        epsilon=audit_epsilon,
        maximum_relative_error=audit_tolerance,
    )
    full_audit["fit"] = "full_development"
    audits.append(full_audit)
    if full_audit["status"] != "PASS":
        raise ValueError(f"Full gradient audit failed: {full_audit}")
    final_basis, final_history = fit_stable_basis(
        full_batch,
        pca_basis,
        objective_kwargs,
        **optimizer_kwargs,
    )
    final_scales = fit_int8_scales(residuals, final_basis)
    _, final_quantization = encode_residuals_int8(
        residuals, final_basis, final_scales
    )
    atomic_json(
        args.output_dir / "optimizer_audit.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V10_OPTIMIZER_AUDIT_COMPLETE",
            "all_gradient_audits_pass": all(
                audit["status"] == "PASS" for audit in audits
            ),
            "all_accepted_losses_monotone": all(
                result["accepted_losses_monotone"] for result in fold_results
            )
            and _history_monotone(final_history),
            "audits": audits,
            "full_optimization_history": final_history,
        },
    )

    base_metrics = per_query_metrics(
        base_scores, rows, labels, relevant_counts, k=final_k
    )
    pca_metrics = per_query_metrics(
        pca_scores, rows, labels, relevant_counts, k=final_k
    )
    pca_fp32_metrics = per_query_metrics(
        pca_fp32_scores, rows, labels, relevant_counts, k=final_k
    )
    v10_metrics = per_query_metrics(
        oof_scores, rows, labels, relevant_counts, k=final_k
    )
    teacher_metrics = per_query_metrics(
        teacher_scores, rows, labels, relevant_counts, k=final_k
    )
    inference = protocol["inference"]
    inference_kwargs = {
        "bootstrap_replicates": int(inference["bootstrap_replicates"]),
        "bootstrap_seed": int(inference["bootstrap_seed"]),
        "randomization_replicates": int(inference["randomization_replicates"]),
        "randomization_seed": int(inference["randomization_seed"]),
        "confidence": float(inference["confidence"]),
    }
    v10_vs_base = paired_inference(
        v10_metrics["recall"], base_metrics["recall"], **inference_kwargs
    )
    v10_vs_pca = paired_inference(
        v10_metrics["recall"], pca_metrics["recall"], **inference_kwargs
    )
    avq = protocol["avq_scalar_headroom_diagnostic"]
    pca_fp32_vs_int8 = paired_inference(
        pca_fp32_metrics["recall"],
        pca_metrics["recall"],
        bootstrap_replicates=int(avq["bootstrap_replicates"]),
        bootstrap_seed=int(avq["bootstrap_seed"]),
        randomization_replicates=int(avq["randomization_replicates"]),
        randomization_seed=int(avq["randomization_seed"]),
        confidence=float(avq["confidence"]),
    )
    avq_headroom = scalar_quantization_headroom_decision(
        pca_fp32_vs_int8, avq
    )
    gap_recovery = candidate_gap_recovery(
        v10_metrics["recall"],
        base_metrics["recall"],
        teacher_metrics["recall"],
    )
    all_audits_pass = all(audit["status"] == "PASS" for audit in audits)
    all_losses_monotone = all(
        result["accepted_losses_monotone"] for result in fold_results
    ) and _history_monotone(final_history)
    decision = stable_development_decision(
        v10_vs_base=v10_vs_base,
        v10_vs_pca=v10_vs_pca,
        fold_gains_over_pca=fold_gains,
        gap_recovery=gap_recovery,
        pca_mrr=float(np.mean(pca_metrics["mrr"])),
        v10_mrr=float(np.mean(v10_metrics["mrr"])),
        pca_ndcg=float(np.mean(pca_metrics["ndcg"])),
        v10_ndcg=float(np.mean(v10_metrics["ndcg"])),
        optimizer_audits_pass=all_audits_pass,
        accepted_losses_monotone=all_losses_monotone,
        thresholds=protocol["development_gate"],
    )

    arrays_to_write = {
        "oof_base_recall_at_10.float64.npy": base_metrics["recall"],
        "oof_pca_recall_at_10.float64.npy": pca_metrics["recall"],
        "oof_pca_fp32_recall_at_10.float64.npy": pca_fp32_metrics["recall"],
        "oof_v10_recall_at_10.float64.npy": v10_metrics["recall"],
        "oof_teacher_recall_at_10.float64.npy": teacher_metrics["recall"],
        "pca_basis_rank16.float32.npy": pca_basis.astype(np.float32),
        "v10_basis_rank16.float32.npy": final_basis.astype(np.float32),
        "pca_scales_rank16.float32.npy": pca_scales.astype(np.float32),
        "v10_scales_rank16.float32.npy": final_scales.astype(np.float32),
    }
    for filename, values in arrays_to_write.items():
        atomic_save(args.output_dir / filename, np.asarray(values))
    atomic_json(args.output_dir / "fold_results.json", fold_results)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V10_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "evidence_tier": protocol["evidence_boundary"]["tier"],
        "query_count": len(qids),
        "metrics": {
            "base": _metric_summary(base_metrics),
            "pca": _metric_summary(pca_metrics),
            "pca_rank16_fp32_coefficient_ceiling": _metric_summary(
                pca_fp32_metrics
            ),
            "v10_oof": _metric_summary(v10_metrics),
            "same_candidate_exact": _metric_summary(teacher_metrics),
        },
        "comparisons": {
            "v10_vs_base_recall_at_10": v10_vs_base,
            "v10_vs_pca_recall_at_10": v10_vs_pca,
            "pca_fp32_vs_int8_recall_at_10": pca_fp32_vs_int8,
        },
        "avq_scalar_headroom_diagnostic": avq_headroom,
        "fold_gains_over_pca": fold_gains,
        "candidate_gap_recovery_fraction": gap_recovery,
        "pair_support": pair_support,
        "pca_quantization": pca_quantization,
        "final_v10_quantization_on_development_union": final_quantization,
        "optimizer": {
            "all_gradient_audits_pass": all_audits_pass,
            "all_accepted_losses_monotone": all_losses_monotone,
            "full_initial_loss": float(final_history[0]["post_retraction_loss"]),
            "full_final_loss": float(
                [row for row in final_history if row["accepted"]][-1][
                    "post_retraction_loss"
                ]
            ),
            "full_accepted_steps": int(
                sum(row["accepted"] for row in final_history) - 1
            ),
        },
        "decision": decision,
        "opened_roles": ["oracle_design"],
        "v9_files_opened": False,
        "future_method_holdout_opened": False,
        "fresh_external_access_authorized": False,
        "interpretation": (
            "Post-V9 outcome-informed development only. V9 was not read and "
            "cannot be reused for selection or confirmation."
        ),
    }
    result_path = args.output_dir / "development_result.json"
    atomic_json(result_path, result)
    freeze = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V10_METHOD_CLOSED_AFTER_SINGLE_DEVELOPMENT_RUN",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "method": protocol["method"],
        "objective": protocol["objective"],
        "optimization": protocol["optimization"],
        "pca_basis": file_record(args.output_dir / "pca_basis_rank16.float32.npy"),
        "v10_basis": file_record(args.output_dir / "v10_basis_rank16.float32.npy"),
        "development_result": file_record(result_path),
        "v9_reuse_authorized": False,
        "fresh_external_access_authorized": False,
        "next_action": (
            "If and only if every gate passes, write a separate protocol for a "
            "genuinely fresh external dataset/model; otherwise stop this method."
        ),
    }
    atomic_json(args.output_dir / "method_freeze.json", freeze)
    outputs = {
        filename: file_record(args.output_dir / filename)
        for filename in protocol["required_outputs"]
        if filename not in ("development_started.json", "development_complete.json")
    }
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V10_DEVELOPMENT_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "started": file_record(started_path),
        "outputs": outputs,
        "v9_files_opened": False,
        "future_method_holdout_opened": False,
        "configuration_count": 1,
    }
    atomic_json(args.output_dir / "development_complete.json", complete)
    missing = [
        filename
        for filename in protocol["required_outputs"]
        if not (args.output_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"Required V10 outputs missing: {missing}")
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
