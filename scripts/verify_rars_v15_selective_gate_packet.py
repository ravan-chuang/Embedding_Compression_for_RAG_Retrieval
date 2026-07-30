#!/usr/bin/env python3
"""Independently verify a completed V15 selective-gate development packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v11_rank_rate_core import paired_inference  # noqa: E402
from rars_v15_selective_gate_core import (  # noqa: E402
    FEATURE_NAMES,
    PROTOCOL_ID,
    apply_query_gate,
    fit_weighted_ridge_gate,
    gate_utility_scores,
    select_calibrated_threshold,
    selective_gate_decision,
)
from rars_v8_cutoff_sidecar_core import candidate_gap_recovery  # noqa: E402
from verify_rars_v13_committed_closure import verify_closure as verify_v13_closure  # noqa: E402
from verify_rars_v13_signed_score_rpq_packet import verify_packet as verify_v13_packet  # noqa: E402
from verify_rars_v14_committed_closure import verify_closure as verify_v14_closure  # noqa: E402


METRICS = ("recall", "mrr", "ndcg")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 changed")


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


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} keys changed")
        for key, value in expected.items():
            _assert_close(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length changed")
        for index, value in enumerate(expected):
            _assert_close(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-15):
            raise ValueError(f"{label} changed: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def _load_metrics(root: Path, prefix: str, query_count: int) -> dict[str, np.ndarray]:
    values = {
        metric: np.load(
            root / f"per_query_{prefix}_{metric}_at_10.float64.npy",
            allow_pickle=False,
        )
        for metric in METRICS
    }
    for metric, value in values.items():
        if value.shape != (query_count,) or value.dtype != np.float64:
            raise ValueError(f"V15 {prefix} {metric} array contract changed")
        if np.any(~np.isfinite(value)) or np.any((value < 0) | (value > 1)):
            raise ValueError(f"V15 {prefix} {metric} values are invalid")
    return values


def _metric_summary(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {metric: float(np.mean(array)) for metric, array in values.items()}


def verify_packet(packet_root: Path, v13_packet_root: Path, repo_root: Path) -> dict[str, Any]:
    protocol = read_json(
        repo_root / "protocols/rars_v15_cross_fitted_selective_rpq_gate_v1.json"
    )
    complete = read_json(packet_root / "development_complete.json")
    result = read_json(packet_root / "development_result.json")
    freeze = read_json(packet_root / "method_freeze.json")
    started = read_json(packet_root / "development_started.json")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V15 protocol identity changed")
    statuses = (
        (complete, "RARS_V15_SELECTIVE_GATE_DEVELOPMENT_COMPLETE"),
        (result, "RARS_V15_SELECTIVE_GATE_DEVELOPMENT_COMPLETE"),
        (freeze, "RARS_V15_METHOD_AND_DECISION_FROZEN"),
        (started, "RARS_V15_SELECTIVE_GATE_DEVELOPMENT_STARTED"),
    )
    source_commit = str(complete.get("source_commit"))
    for payload, status in statuses:
        if payload.get("status") != status or payload.get("source_commit") != source_commit:
            raise ValueError("V15 status or source lineage changed")
    verify_record(packet_root / "development_started.json", complete["started"], "start marker")
    for name, record in complete.get("outputs", {}).items():
        verify_record(packet_root / name, record, f"V15 output {name}")
    verify_record(
        packet_root / "development_result.json",
        freeze["development_result"],
        "V15 frozen development result",
    )
    for relative, record in started.get("source_blobs", {}).items():
        verify_record(repo_root / relative, record, f"V15 source blob {relative}")

    v13_closure = verify_v13_closure(repo_root / "results/rars_v13_signed_score_rpq", repo_root)
    v14_closure = verify_v14_closure(repo_root / "results/rars_v14_anisotropic_rate_rpq", repo_root)
    v13_verification = verify_v13_packet(v13_packet_root, repo_root)
    if v13_closure["formal_decision"] != protocol["parent_evidence"]["v13_formal_decision"]:
        raise ValueError("V13 committed parent decision changed")
    if v14_closure["formal_decision"] != protocol["parent_evidence"]["v14_formal_decision"]:
        raise ValueError("V14 committed parent decision changed")
    _assert_close(started["v13_committed_closure"], v13_closure, "V13 closure")
    _assert_close(started["v14_committed_closure"], v14_closure, "V14 closure")
    _assert_close(started["v13_packet_verification"], v13_verification, "V13 packet")
    if started["labels_used_for_gate_fitting"] is not True:
        raise ValueError("V15 gate fitting did not register label use")
    if started["labels_used_for_representation_learning"] is not False:
        raise ValueError("V15 representation unexpectedly used labels")
    if started["future_method_holdout_opened"] is not False or started[
        "old_rars_holdout_opened"
    ] is not False:
        raise ValueError("V15 opened a prohibited holdout")

    qids = (packet_root / "query_ids.utf8.txt").read_text(encoding="utf-8").splitlines()
    parent_qids = (v13_packet_root / "query_ids.utf8.txt").read_text(encoding="utf-8").splitlines()
    folds = np.load(packet_root / "fold_ids.int64.npy", allow_pickle=False)
    parent_folds = np.load(v13_packet_root / "fold_ids.int64.npy", allow_pickle=False)
    query_count = int(protocol["input_contract"]["query_count"])
    if qids != parent_qids or len(qids) != query_count or len(set(qids)) != query_count:
        raise ValueError("V15 query identities changed")
    if folds.dtype != np.int64 or folds.shape != (query_count,) or not np.array_equal(folds, parent_folds):
        raise ValueError("V15 fold identities changed")
    if np.bincount(folds, minlength=5).tolist() != protocol["input_contract"]["fold_counts"]:
        raise ValueError("V15 fold counts changed")

    seeds = [int(value) for value in protocol["rpq_training"]["seeds"]]
    primary_seed = int(protocol["rpq_training"]["primary_seed"])
    primary_index = seeds.index(primary_seed)
    base = _load_metrics(packet_root, "base", query_count)
    exact = _load_metrics(packet_root, "same_candidate_exact", query_count)
    pca16 = _load_metrics(packet_root, "pca16", query_count)
    uniform = {seed: _load_metrics(packet_root, f"uniform_seed{seed}", query_count) for seed in seeds}
    selective = {seed: _load_metrics(packet_root, f"selective_seed{seed}", query_count) for seed in seeds}
    parent_comparators = {
        "base": base,
        "same_candidate_exact": exact,
        "pca16": pca16,
    }
    for prefix, values in parent_comparators.items():
        for metric in METRICS:
            parent = np.load(
                v13_packet_root / f"per_query_{prefix}_{metric}_at_10.float64.npy",
                allow_pickle=False,
            )
            if not np.array_equal(values[metric], parent):
                raise ValueError(f"V15 parent comparator changed: {prefix}/{metric}")
    for seed in seeds:
        for metric in METRICS:
            parent = np.load(
                v13_packet_root / f"per_query_unsupervised_seed{seed}_{metric}_at_10.float64.npy",
                allow_pickle=False,
            )
            if not np.array_equal(uniform[seed][metric], parent):
                raise ValueError(f"V15 uniform parent changed: seed={seed}/{metric}")

    feature_count = len(FEATURE_NAMES)
    diagnostics = read_json(packet_root / "fold_seed_gate_diagnostics.json")
    if len(diagnostics) != 5 * len(seeds):
        raise ValueError("V15 fold/seed diagnostic count changed")
    audit_features = np.load(
        packet_root / "crossfit_gate_features.float64.npy", allow_pickle=False
    )
    audit_sidecar = np.load(
        packet_root / "crossfit_sidecar_metrics.float64.npy", allow_pickle=False
    )
    if audit_features.shape != (5, len(seeds), query_count, feature_count):
        raise ValueError("V15 cross-fit feature audit shape changed")
    if audit_sidecar.shape != (5, len(seeds), query_count, len(METRICS)):
        raise ValueError("V15 cross-fit metric audit shape changed")
    if audit_features.dtype != np.float64 or audit_sidecar.dtype != np.float64:
        raise ValueError("V15 cross-fit audit dtype changed")
    if np.any(~np.isfinite(audit_features)) or np.any(~np.isfinite(audit_sidecar)):
        raise ValueError("V15 cross-fit audit arrays contain non-finite values")
    feature_arrays: dict[int, np.ndarray] = {}
    utility_arrays: dict[int, np.ndarray] = {}
    applied_arrays: dict[int, np.ndarray] = {}
    for seed in seeds:
        features = np.load(packet_root / f"oof_gate_features_seed{seed}.npy", allow_pickle=False)
        utility = np.load(packet_root / f"oof_gate_utility_seed{seed}.npy", allow_pickle=False)
        applied = np.load(packet_root / f"oof_gate_applied_seed{seed}.npy", allow_pickle=False)
        if features.shape != (query_count, feature_count) or features.dtype != np.float64:
            raise ValueError(f"V15 gate feature contract changed for seed {seed}")
        if utility.shape != (query_count,) or utility.dtype != np.float64:
            raise ValueError(f"V15 gate utility contract changed for seed {seed}")
        if applied.shape != (query_count,) or applied.dtype != np.uint8 or not set(np.unique(applied)).issubset({0, 1}):
            raise ValueError(f"V15 gate application contract changed for seed {seed}")
        if np.any(~np.isfinite(features)) or np.any(~np.isfinite(utility)):
            raise ValueError(f"V15 gate arrays contain non-finite values for seed {seed}")
        feature_arrays[seed] = features
        utility_arrays[seed] = utility
        applied_arrays[seed] = applied

    seen: set[tuple[int, int]] = set()
    gate_cfg = protocol["gate"]
    for row in diagnostics:
        fold = int(row["outer_fold"])
        seed = int(row["seed"])
        key = (fold, seed)
        if key in seen or fold not in range(5) or seed not in seeds:
            raise ValueError("V15 fold/seed diagnostic identity changed")
        seen.add(key)
        seed_index = seeds.index(seed)
        heldout = np.flatnonzero(folds == fold)
        calibration_fold = (fold + 1) % 5
        calibration = np.flatnonzero(folds == calibration_fold)
        gate_fit = np.flatnonzero((folds != fold) & (folds != calibration_fold))
        if int(row["calibration_fold"]) != calibration_fold:
            raise ValueError("V15 calibration fold rule changed")
        if int(row["gate_fit_query_count"]) != len(gate_fit):
            raise ValueError("V15 gate-fit row count changed")
        if int(row["calibration_query_count"]) != len(calibration) or int(
            row["heldout_query_count"]
        ) != len(heldout):
            raise ValueError("V15 calibration or held-out row count changed")
        features = audit_features[fold, seed_index]
        sidecar_metrics = {
            metric: audit_sidecar[fold, seed_index, :, metric_index]
            for metric_index, metric in enumerate(METRICS)
        }
        recomputed_model = fit_weighted_ridge_gate(
            features[gate_fit],
            sidecar_metrics["recall"][gate_fit] - base["recall"][gate_fit],
            ridge=float(gate_cfg["ridge"]),
            neutral_weight=float(gate_cfg["neutral_weight"]),
            harm_weight=float(gate_cfg["harm_weight"]),
            minimum_scale=float(gate_cfg["minimum_feature_scale"]),
        )
        serializable_model = {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in recomputed_model.items()
        }
        _assert_close(row["gate_model"], serializable_model, "V15 gate model")
        utility = gate_utility_scores(features, recomputed_model)
        calibration_result = select_calibrated_threshold(
            utility[calibration],
            {metric: base[metric][calibration] for metric in METRICS},
            {metric: sidecar_metrics[metric][calibration] for metric in METRICS},
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
        _assert_close(row["calibration"], calibration_result, "V15 calibration")
        if not np.array_equal(features[heldout], feature_arrays[seed][heldout]):
            raise ValueError(f"V15 OOF features changed for fold={fold}, seed={seed}")
        recomputed_utility = utility[heldout]
        if not np.array_equal(recomputed_utility, utility_arrays[seed][heldout]):
            raise ValueError(f"V15 gate utilities changed for fold={fold}, seed={seed}")
        recomputed_applied = apply_query_gate(
            recomputed_utility, float(calibration_result["threshold"])
        ).astype(np.uint8)
        if not np.array_equal(recomputed_applied, applied_arrays[seed][heldout]):
            raise ValueError(f"V15 gate mask changed for fold={fold}, seed={seed}")
        if not np.isclose(np.mean(recomputed_applied), row["heldout_coverage"], rtol=0.0, atol=1e-15):
            raise ValueError("V15 held-out coverage changed")
        for metric in METRICS:
            expected = np.where(
                recomputed_applied.astype(bool),
                sidecar_metrics[metric][heldout],
                base[metric][heldout],
            )
            if not np.array_equal(sidecar_metrics[metric][heldout], uniform[seed][metric][heldout]):
                raise ValueError(f"V15 reproduced uniform metric changed: fold={fold}, seed={seed}/{metric}")
            if not np.array_equal(expected, selective[seed][metric][heldout]):
                raise ValueError(f"V15 selective metric changed: fold={fold}, seed={seed}/{metric}")
    if seen != {(fold, seed) for fold in range(5) for seed in seeds}:
        raise ValueError("V15 fold/seed diagnostics are incomplete")

    primary_uniform = uniform[primary_seed]
    primary_selective = selective[primary_seed]
    comparisons = {
        "selective_vs_uniform_rpq": paired_inference(
            primary_selective["recall"], primary_uniform["recall"],
            **_inference_kwargs(protocol, "primary_vs_uniform_rpq"),
        ),
        "selective_vs_base": paired_inference(
            primary_selective["recall"], base["recall"],
            **_inference_kwargs(protocol, "primary_vs_base"),
        ),
        "selective_vs_pca16": paired_inference(
            primary_selective["recall"], pca16["recall"],
            **_inference_kwargs(protocol, "primary_vs_pca16"),
        ),
    }
    seed_gains = [
        float(np.mean(selective[seed]["recall"] - uniform[seed]["recall"]))
        for seed in seeds
    ]
    fold_gains = [
        float(np.mean((primary_selective["recall"] - primary_uniform["recall"])[folds == fold]))
        for fold in range(5)
    ]
    primary_difference = primary_selective["recall"] - primary_uniform["recall"]
    applied_coverages = [float(np.mean(applied_arrays[seed])) for seed in seeds]

    final_mean = np.load(packet_root / "final_gate_feature_mean.float64.npy", allow_pickle=False)
    final_scale = np.load(packet_root / "final_gate_feature_scale.float64.npy", allow_pickle=False)
    final_coefficients = np.load(packet_root / "final_gate_coefficients.float64.npy", allow_pickle=False)
    final_threshold = np.load(packet_root / "final_gate_threshold.float64.npy", allow_pickle=False)
    final_features = np.load(packet_root / "final_gate_features.float64.npy", allow_pickle=False)
    final_oof_utility = np.load(packet_root / "final_oof_gate_utility.float64.npy", allow_pickle=False)
    final_sidecar_matrix = np.load(packet_root / "final_sidecar_metrics.float64.npy", allow_pickle=False)
    if final_features.shape != (query_count, feature_count) or final_features.dtype != np.float64:
        raise ValueError("V15 final feature audit contract changed")
    if final_oof_utility.shape != (query_count,) or final_oof_utility.dtype != np.float64:
        raise ValueError("V15 final OOF utility audit contract changed")
    if final_sidecar_matrix.shape != (query_count, len(METRICS)) or final_sidecar_matrix.dtype != np.float64:
        raise ValueError("V15 final sidecar metric audit contract changed")
    if any(np.any(~np.isfinite(value)) for value in (final_features, final_oof_utility, final_sidecar_matrix)):
        raise ValueError("V15 final export audit arrays contain non-finite values")
    final_sidecar_metrics = {
        metric: final_sidecar_matrix[:, index]
        for index, metric in enumerate(METRICS)
    }
    final_gain = final_sidecar_metrics["recall"] - base["recall"]
    recomputed_oof_utility = np.full(query_count, np.nan, dtype=np.float64)
    for fold in range(5):
        heldout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fold_model = fit_weighted_ridge_gate(
            final_features[training],
            final_gain[training],
            ridge=float(gate_cfg["ridge"]),
            neutral_weight=float(gate_cfg["neutral_weight"]),
            harm_weight=float(gate_cfg["harm_weight"]),
            minimum_scale=float(gate_cfg["minimum_feature_scale"]),
        )
        recomputed_oof_utility[heldout] = gate_utility_scores(
            final_features[heldout], fold_model
        )
    if not np.array_equal(recomputed_oof_utility, final_oof_utility):
        raise ValueError("V15 final OOF utility changed")
    recomputed_threshold = select_calibrated_threshold(
        recomputed_oof_utility,
        base,
        final_sidecar_metrics,
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
    recomputed_final_model = fit_weighted_ridge_gate(
        final_features,
        final_gain,
        ridge=float(gate_cfg["ridge"]),
        neutral_weight=float(gate_cfg["neutral_weight"]),
        harm_weight=float(gate_cfg["harm_weight"]),
        minimum_scale=float(gate_cfg["minimum_feature_scale"]),
    )
    if final_mean.shape != (feature_count,) or final_scale.shape != (feature_count,):
        raise ValueError("V15 final feature transform shape changed")
    if final_coefficients.shape != (feature_count + 1,) or final_threshold.shape != (1,):
        raise ValueError("V15 final gate parameter shape changed")
    if any(value.dtype != np.float64 for value in (final_mean, final_scale, final_coefficients, final_threshold)):
        raise ValueError("V15 final gate parameter dtype changed")
    if any(np.any(~np.isfinite(value)) for value in (final_mean, final_scale, final_coefficients, final_threshold)) or np.any(final_scale <= 0):
        raise ValueError("V15 final gate parameters are invalid")
    export = result["final_export"]
    _assert_close(
        export["gate_threshold"], recomputed_threshold, "V15 final threshold"
    )
    _assert_close(
        export["gate_model"],
        {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in recomputed_final_model.items()
        },
        "V15 final gate model",
    )
    if export["gate_model"]["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("V15 final gate feature order changed")
    if not np.array_equal(final_mean, np.asarray(export["gate_model"]["mean"], dtype=np.float64)):
        raise ValueError("V15 exported gate mean changed")
    if not np.array_equal(final_scale, np.asarray(export["gate_model"]["scale"], dtype=np.float64)):
        raise ValueError("V15 exported gate scale changed")
    if not np.array_equal(final_coefficients, np.asarray(export["gate_model"]["coefficients"], dtype=np.float64)):
        raise ValueError("V15 exported gate coefficients changed")
    if final_threshold[0] != float(export["gate_threshold"]["threshold"]):
        raise ValueError("V15 exported threshold changed")
    global_model_bytes = int(final_mean.nbytes + final_scale.nbytes + final_coefficients.nbytes + final_threshold.nbytes)
    if global_model_bytes != int(export["global_model_bytes"]):
        raise ValueError("V15 global model byte count changed")
    verify_record(
        v13_packet_root / "full_corpus_signed_score_assignments.uint8.memmap",
        export["parent_payload"],
        "V13 parent uniform payload",
    )
    verify_record(
        v13_packet_root / "final_pca_basis_rank64.float32.npy",
        export["parent_basis"],
        "V13 parent PCA64 basis",
    )
    verify_record(
        v13_packet_root / "final_unsupervised_codebooks.float32.npy",
        export["parent_codebooks"],
        "V13 parent uniform codebooks",
    )
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
        parent_payload_bytes_per_document=int(protocol["storage_contract"]["parent_payload_bytes_per_document"]),
        extra_document_bytes=int(export["additional_document_bytes"]),
        global_model_bytes=global_model_bytes,
        thresholds=protocol["development_gate"],
    )
    recomputed = {
        "metrics": {
            "base": _metric_summary(base),
            "same_candidate_exact": _metric_summary(exact),
            "pca16": _metric_summary(pca16),
            "uniform_primary": _metric_summary(primary_uniform),
            "selective_primary": _metric_summary(primary_selective),
        },
        "comparisons": comparisons,
        "seed_gains": seed_gains,
        "fold_gains": fold_gains,
        "applied_coverages": applied_coverages,
        "candidate_gap_recovery_fraction": candidate_gap_recovery(
            primary_selective["recall"], base["recall"], exact["recall"]
        ),
        "oracle_selective_gain_over_uniform": float(
            np.mean(np.maximum(primary_uniform["recall"], base["recall"]) - primary_uniform["recall"])
        ),
        "decision": decision,
    }
    for key, value in recomputed.items():
        _assert_close(result[key], value, f"result.{key}")
    if result["formal_decision"] != decision["decision"] or complete["formal_decision"] != decision["decision"] or freeze["formal_decision"] != decision["decision"]:
        raise ValueError("V15 formal decision changed")
    if any(result[key] is not False for key in ("future_method_holdout_opened", "old_rars_holdout_opened", "fresh_query_access_authorized")):
        raise ValueError("V15 result overclaims access or authorization")
    return {
        "status": "RARS_V15_SELECTIVE_GATE_PACKET_VERIFIED",
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "formal_decision": decision["decision"],
        "query_count": query_count,
        "primary_recall_gain_over_uniform": comparisons["selective_vs_uniform_rpq"]["mean_difference"],
        "primary_coverage": applied_coverages[primary_index],
        "global_model_bytes": global_model_bytes,
        "verified_output_count": len(complete["outputs"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--v13-packet-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(verify_packet(args.packet_root, args.v13_packet_root, args.repo_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
