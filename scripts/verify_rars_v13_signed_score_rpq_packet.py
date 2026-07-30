#!/usr/bin/env python3
"""Independently verify a completed RARS-v13 development packet."""

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
from rars_v13_signed_score_core import (  # noqa: E402
    PROTOCOL_ID,
    signed_score_decision,
    deterministic_fold_ids,
)
from rars_v8_cutoff_sidecar_core import candidate_gap_recovery  # noqa: E402


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
        raise ValueError(f"{label} hash changed")


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


def _assert_same(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-15):
            raise ValueError(f"{label} changed: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def verify_packet(packet_root: Path, repo_root: Path) -> dict[str, Any]:
    protocol = read_json(
        repo_root / "protocols/rars_v13_signed_score_distilled_rpq_v1.json"
    )
    complete = read_json(packet_root / "development_complete.json")
    result = read_json(packet_root / "development_result.json")
    freeze = read_json(packet_root / "method_freeze.json")
    started = read_json(packet_root / "development_started.json")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V13 protocol identity changed")
    if complete.get("status") != "RARS_V13_FRESH_DEVELOPMENT_COMPLETE":
        raise ValueError("V13 packet is incomplete")
    if result.get("status") != "RARS_V13_FRESH_DEVELOPMENT_COMPLETE":
        raise ValueError("V13 result is incomplete")
    if freeze.get("status") != "RARS_V13_METHOD_AND_DECISION_FROZEN":
        raise ValueError("V13 method freeze is incomplete")
    if started.get("status") != "RARS_V13_FRESH_DEVELOPMENT_STARTED":
        raise ValueError("V13 start marker is invalid")
    source_commit = complete.get("source_commit")
    for payload in (result, freeze, started):
        if payload.get("source_commit") != source_commit:
            raise ValueError("V13 source commit is inconsistent")
    verify_record(
        packet_root / "development_started.json", complete["started"], "start marker"
    )
    for name, record in complete.get("outputs", {}).items():
        verify_record(packet_root / name, record, f"output {name}")
    for relative, record in started.get("source_blobs", {}).items():
        verify_record(repo_root / relative, record, f"source blob {relative}")

    qids = (packet_root / "query_ids.utf8.txt").read_text(encoding="utf-8").splitlines()
    folds = np.load(packet_root / "fold_ids.int64.npy", allow_pickle=False)
    target = int(protocol["fresh_query_freeze"]["target_query_count"])
    if len(qids) != target or len(set(qids)) != target:
        raise ValueError("V13 packet qids changed")
    if folds.dtype != np.int64 or not np.array_equal(folds, deterministic_fold_ids(qids)):
        raise ValueError("V13 packet fold ids changed")
    prior: set[str] = set()
    for relative in protocol["fresh_query_freeze"]["prior_qid_sources"]:
        path = repo_root / relative
        if path.suffix == ".txt":
            values = path.read_text(encoding="utf-8").splitlines()
        else:
            value = read_json(path)
            values = value.get("query_ids") if isinstance(value, dict) else value
        prior.update(str(item) for item in values)
    if len(prior) != int(
        protocol["fresh_query_freeze"]["expected_unique_excluded_qids"]
    ):
        raise ValueError("V13 exclusion registry count changed")
    if prior.intersection(qids):
        raise ValueError("V13 packet contains a historical MS MARCO dev qid")

    def load(prefix: str, metric: str) -> np.ndarray:
        path = packet_root / f"per_query_{prefix}_{metric}_at_10.float64.npy"
        value = np.load(path, allow_pickle=False)
        if value.shape != (target,) or value.dtype != np.float64:
            raise ValueError(f"Per-query array contract changed: {path.name}")
        if np.any(~np.isfinite(value)) or np.any((value < 0) | (value > 1)):
            raise ValueError(f"Per-query values are invalid: {path.name}")
        return value

    seeds = [int(value) for value in protocol["rpq_training"]["seeds"]]
    primary_seed = int(protocol["rpq_training"]["primary_seed"])
    base = {name: load("base", name) for name in ("recall", "mrr", "ndcg")}
    exact = {
        name: load("same_candidate_exact", name)
        for name in ("recall", "mrr", "ndcg")
    }
    pca16 = {name: load("pca16", name) for name in ("recall", "mrr", "ndcg")}
    unsupervised = {
        seed: {
            name: load(f"unsupervised_seed{seed}", name)
            for name in ("recall", "mrr", "ndcg")
        }
        for seed in seeds
    }
    challenger = {
        seed: {
            name: load(f"signed_score_seed{seed}", name)
            for name in ("recall", "mrr", "ndcg")
        }
        for seed in seeds
    }
    primary = paired_inference(
        challenger[primary_seed]["recall"],
        unsupervised[primary_seed]["recall"],
        **_inference_kwargs(protocol, "primary_vs_unsupervised"),
    )
    versus_pca16 = paired_inference(
        challenger[primary_seed]["recall"],
        pca16["recall"],
        **_inference_kwargs(protocol, "primary_vs_pca16"),
    )
    versus_base = paired_inference(
        challenger[primary_seed]["recall"],
        base["recall"],
        **_inference_kwargs(protocol, "primary_vs_base"),
    )
    seed_gains = [
        float(np.mean(challenger[seed]["recall"] - unsupervised[seed]["recall"]))
        for seed in seeds
    ]
    fold_gains = [
        float(
            np.mean(
                challenger[primary_seed]["recall"][folds == fold]
                - unsupervised[primary_seed]["recall"][folds == fold]
            )
        )
        for fold in range(5)
    ]
    gap = candidate_gap_recovery(
        challenger[primary_seed]["recall"], base["recall"], exact["recall"]
    )
    diagnostics = read_json(packet_root / "fold_seed_diagnostics.json")
    signed_diagnostics = diagnostics.get("signed_score", [])
    pca_diagnostics = diagnostics.get("pca16", [])
    if len(signed_diagnostics) != 5 * len(seeds) or len(pca_diagnostics) != 5:
        raise ValueError("V13 fold/seed diagnostic count changed")
    all_nonincreasing = all(
        row["update_summary"]["objective_nonincreasing"]
        for row in signed_diagnostics
    ) and bool(result["final_fit"]["update_summary"]["objective_nonincreasing"])
    maximum_drift = max(
        [
            float(row["update_summary"]["maximum_centroid_drift_fraction"])
            for row in signed_diagnostics
        ]
        + [float(result["final_fit"]["update_summary"]["maximum_centroid_drift_fraction"])]
    )
    assignment_changes = sum(
        int(row["update_summary"]["assignment_changes"])
        for row in signed_diagnostics
    ) + int(result["final_fit"]["update_summary"]["assignment_changes"])
    code_path = (
        packet_root / "full_corpus_signed_score_assignments.uint8.memmap"
    )
    expected_payload = int(protocol["full_corpus_sidecar"]["payload_bytes"])
    if code_path.stat().st_size != expected_payload:
        raise ValueError("V13 full-corpus payload size changed")
    code_shape = tuple(protocol["full_corpus_sidecar"]["code_shape"])
    codes = np.memmap(code_path, dtype=np.uint8, mode="r", shape=code_shape)
    histograms = np.stack(
        [np.bincount(codes[:, block], minlength=256) for block in range(code_shape[1])]
    ).astype(np.int64)
    occupied = np.sum(histograms > 0, axis=1)
    minimum_occupied = int(
        protocol["full_corpus_sidecar"]["minimum_occupied_centroids_per_block"]
    )
    if int(occupied.min()) < minimum_occupied:
        raise ValueError("V13 full-corpus codes are degenerate")
    registered_codes = result["final_fit"]["full_corpus_codes"]
    if registered_codes["code_histograms"] != histograms.tolist():
        raise ValueError("V13 full-corpus code histograms changed")
    if registered_codes["occupied_centroids_per_block"] != occupied.tolist():
        raise ValueError("V13 full-corpus centroid occupancy changed")
    if registered_codes["minimum_occupied_centroids_per_block"] != int(
        occupied.min()
    ):
        raise ValueError("V13 minimum centroid occupancy changed")
    if registered_codes["maximum_occupied_centroids_per_block"] != int(
        occupied.max()
    ):
        raise ValueError("V13 maximum centroid occupancy changed")
    if bool(result["all_objectives_nonincreasing"]) is not bool(all_nonincreasing):
        raise ValueError("V13 objective audit flag does not recompute")
    recomputed = signed_score_decision(
        primary_vs_unsupervised=primary,
        primary_vs_pca16=versus_pca16,
        primary_vs_base=versus_base,
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        candidate_gap_recovery=gap,
        unsupervised_mrr=float(np.mean(unsupervised[primary_seed]["mrr"])),
        challenger_mrr=float(np.mean(challenger[primary_seed]["mrr"])),
        unsupervised_ndcg=float(np.mean(unsupervised[primary_seed]["ndcg"])),
        challenger_ndcg=float(np.mean(challenger[primary_seed]["ndcg"])),
        payload_bytes_per_document=16,
        full_corpus_codes_materialized=True,
        all_objectives_nonincreasing=all_nonincreasing,
        maximum_centroid_drift_fraction=maximum_drift,
        assignment_changes=assignment_changes,
        thresholds=protocol["development_gate"],
    )
    if recomputed != result.get("decision"):
        raise ValueError("V13 formal decision does not recompute exactly")
    if not (
        complete.get("formal_decision")
        == result.get("formal_decision")
        == freeze.get("formal_decision")
        == recomputed["decision"]
    ):
        raise ValueError("V13 formal decision is inconsistent")
    for key, observed in (
        ("seed_gains", seed_gains),
        ("fold_gains", fold_gains),
    ):
        registered = result[key]
        if not np.allclose(observed, registered, rtol=0.0, atol=1e-15):
            raise ValueError(f"V13 {key} changed")
    _assert_same(gap, result["candidate_gap_recovery_fraction"], "gap recovery")
    for method, arrays in (
        ("base", base),
        ("same_candidate_exact", exact),
        ("pca16", pca16),
        ("unsupervised_primary", unsupervised[primary_seed]),
        ("signed_score_primary", challenger[primary_seed]),
    ):
        for metric, values in arrays.items():
            _assert_same(
                float(np.mean(values)), result["metrics"][method][metric], f"{method} {metric}"
            )
    basis = np.load(packet_root / "final_pca_basis_rank64.float32.npy", allow_pickle=False)
    initial_books = np.load(
        packet_root / "final_unsupervised_codebooks.float32.npy", allow_pickle=False
    )
    signed_books = np.load(
        packet_root / "final_signed_score_codebooks.float32.npy", allow_pickle=False
    )
    pca16_basis = np.load(
        packet_root / "final_pca16_basis.float32.npy", allow_pickle=False
    )
    pca16_scales = np.load(
        packet_root / "final_pca16_scales.float32.npy", allow_pickle=False
    )
    if basis.shape != (384, 64) or basis.dtype != np.float32:
        raise ValueError("V13 final basis contract changed")
    if not np.allclose(basis.T @ basis, np.eye(64), rtol=0.0, atol=2e-4):
        raise ValueError("V13 final basis is not orthonormal")
    if pca16_basis.shape != (384, 16) or pca16_basis.dtype != np.float32:
        raise ValueError("V13 PCA16 basis contract changed")
    if pca16_scales.shape != (16,) or pca16_scales.dtype != np.float32:
        raise ValueError("V13 PCA16 scale contract changed")
    if np.any(~np.isfinite(pca16_scales)) or np.any(pca16_scales <= 0):
        raise ValueError("V13 PCA16 scales are invalid")
    for label, books in (("initial", initial_books), ("signed", signed_books)):
        if books.shape != (16, 256, 4) or books.dtype != np.float32:
            raise ValueError(f"V13 {label} codebook contract changed")
        if np.any(~np.isfinite(books)):
            raise ValueError(f"V13 {label} codebooks contain non-finite values")
    return {
        "status": "RARS_V13_PACKET_VERIFIED",
        "source_commit": source_commit,
        "formal_decision": recomputed["decision"],
        "query_count": target,
        "primary_gain": primary["mean_difference"],
        "primary_ci": [primary["lower"], primary["upper"]],
        "primary_randomization_p_value": primary[
            "randomization_p_value_one_sided"
        ],
        "seed_gains": seed_gains,
        "fold_gains": fold_gains,
        "candidate_gap_recovery_fraction": gap,
        "full_corpus_payload_bytes": expected_payload,
        "verified_output_count": len(complete["outputs"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            verify_packet(args.packet_root, args.repo_root),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
