#!/usr/bin/env python3
"""Independently verify a completed RARS-v12 development packet."""

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
from rars_v12_ca_rpq_core import (  # noqa: E402
    PROTOCOL_ID,
    ca_rpq_decision,
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
    protocol = read_json(repo_root / "protocols/rars_v12_anchored_cutoff_rpq_v1.json")
    complete = read_json(packet_root / "development_complete.json")
    result = read_json(packet_root / "development_result.json")
    freeze = read_json(packet_root / "method_freeze.json")
    started = read_json(packet_root / "development_started.json")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V12 protocol identity changed")
    if complete.get("status") != "RARS_V12_FRESH_DEVELOPMENT_COMPLETE":
        raise ValueError("V12 packet is incomplete")
    if result.get("status") != "RARS_V12_FRESH_DEVELOPMENT_COMPLETE":
        raise ValueError("V12 result is incomplete")
    if freeze.get("status") != "RARS_V12_METHOD_AND_DECISION_FROZEN":
        raise ValueError("V12 method freeze is incomplete")
    if started.get("status") != "RARS_V12_FRESH_DEVELOPMENT_STARTED":
        raise ValueError("V12 start marker is invalid")
    source_commit = complete.get("source_commit")
    for payload in (result, freeze, started):
        if payload.get("source_commit") != source_commit:
            raise ValueError("V12 source commit is inconsistent")
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
        raise ValueError("V12 packet qids changed")
    if folds.dtype != np.int64 or not np.array_equal(folds, deterministic_fold_ids(qids)):
        raise ValueError("V12 packet fold ids changed")
    prior: set[str] = set()
    for relative in protocol["fresh_query_freeze"]["prior_qid_sources"]:
        prior.update(str(value) for value in read_json(repo_root / relative))
    if prior.intersection(qids):
        raise ValueError("V12 packet contains a historical MS MARCO dev qid")

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
    unsupervised = {
        seed: {
            name: load(f"unsupervised_seed{seed}", name)
            for name in ("recall", "mrr", "ndcg")
        }
        for seed in seeds
    }
    challenger = {
        seed: {
            name: load(f"ca_rpq_seed{seed}", name)
            for name in ("recall", "mrr", "ndcg")
        }
        for seed in seeds
    }
    primary = paired_inference(
        challenger[primary_seed]["recall"],
        unsupervised[primary_seed]["recall"],
        **_inference_kwargs(protocol, "primary_vs_unsupervised"),
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
    if len(diagnostics) != 5 * len(seeds):
        raise ValueError("V12 fold/seed diagnostic count changed")
    all_nonincreasing = all(
        row["update_summary"]["fixed_assignment_objective_after"]
        <= row["update_summary"]["fixed_assignment_objective_before"] + 1e-8
        for row in diagnostics
    ) and (
        result["final_fit"]["update_summary"][
            "fixed_assignment_objective_after"
        ]
        <= result["final_fit"]["update_summary"][
            "fixed_assignment_objective_before"
        ]
        + 1e-8
    )
    maximum_drift = max(
        [float(row["update_summary"]["maximum_centroid_drift_fraction"]) for row in diagnostics]
        + [float(result["final_fit"]["update_summary"]["maximum_centroid_drift_fraction"])]
    )
    code_path = packet_root / "full_corpus_ca_rpq_codes.uint8.memmap"
    expected_payload = int(protocol["full_corpus_sidecar"]["payload_bytes"])
    if code_path.stat().st_size != expected_payload:
        raise ValueError("V12 full-corpus payload size changed")
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
        raise ValueError("V12 full-corpus codes are degenerate")
    registered_codes = result["final_fit"]["full_corpus_codes"]
    if registered_codes["code_histograms"] != histograms.tolist():
        raise ValueError("V12 full-corpus code histograms changed")
    if registered_codes["occupied_centroids_per_block"] != occupied.tolist():
        raise ValueError("V12 full-corpus centroid occupancy changed")
    if registered_codes["minimum_occupied_centroids_per_block"] != int(
        occupied.min()
    ):
        raise ValueError("V12 minimum centroid occupancy changed")
    if registered_codes["maximum_occupied_centroids_per_block"] != int(
        occupied.max()
    ):
        raise ValueError("V12 maximum centroid occupancy changed")
    if bool(result["all_objectives_nonincreasing"]) is not bool(all_nonincreasing):
        raise ValueError("V12 objective audit flag does not recompute")
    recomputed = ca_rpq_decision(
        primary_vs_unsupervised=primary,
        primary_vs_base=versus_base,
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        candidate_gap_recovery=gap,
        unsupervised_mrr=float(np.mean(unsupervised[primary_seed]["mrr"])),
        ca_mrr=float(np.mean(challenger[primary_seed]["mrr"])),
        unsupervised_ndcg=float(np.mean(unsupervised[primary_seed]["ndcg"])),
        ca_ndcg=float(np.mean(challenger[primary_seed]["ndcg"])),
        payload_bytes_per_document=16,
        full_corpus_codes_materialized=True,
        all_objectives_nonincreasing=all_nonincreasing,
        maximum_centroid_drift_fraction=maximum_drift,
        thresholds=protocol["development_gate"],
    )
    if recomputed != result.get("decision"):
        raise ValueError("V12 formal decision does not recompute exactly")
    if not (
        complete.get("formal_decision")
        == result.get("formal_decision")
        == freeze.get("formal_decision")
        == recomputed["decision"]
    ):
        raise ValueError("V12 formal decision is inconsistent")
    for key, observed in (
        ("seed_gains", seed_gains),
        ("fold_gains", fold_gains),
    ):
        registered = result[key]
        if not np.allclose(observed, registered, rtol=0.0, atol=1e-15):
            raise ValueError(f"V12 {key} changed")
    _assert_same(gap, result["candidate_gap_recovery_fraction"], "gap recovery")
    for method, arrays in (
        ("base", base),
        ("same_candidate_exact", exact),
        ("unsupervised_primary", unsupervised[primary_seed]),
        ("ca_rpq_primary", challenger[primary_seed]),
    ):
        for metric, values in arrays.items():
            _assert_same(
                float(np.mean(values)), result["metrics"][method][metric], f"{method} {metric}"
            )
    basis = np.load(packet_root / "final_pca_basis_rank64.float32.npy", allow_pickle=False)
    initial_books = np.load(
        packet_root / "final_unsupervised_codebooks.float32.npy", allow_pickle=False
    )
    ca_books = np.load(
        packet_root / "final_ca_rpq_codebooks.float32.npy", allow_pickle=False
    )
    if basis.shape != (384, 64) or basis.dtype != np.float32:
        raise ValueError("V12 final basis contract changed")
    if not np.allclose(basis.T @ basis, np.eye(64), rtol=0.0, atol=2e-4):
        raise ValueError("V12 final basis is not orthonormal")
    for label, books in (("initial", initial_books), ("CA", ca_books)):
        if books.shape != (16, 256, 4) or books.dtype != np.float32:
            raise ValueError(f"V12 {label} codebook contract changed")
        if np.any(~np.isfinite(books)):
            raise ValueError(f"V12 {label} codebooks contain non-finite values")
    return {
        "status": "RARS_V12_PACKET_VERIFIED",
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
