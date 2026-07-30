#!/usr/bin/env python3
"""Independently verify a completed V14 anisotropic rate-RPQ packet."""

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
from rars_v14_anisotropic_rate_core import (  # noqa: E402
    PROTOCOL_ID,
    anisotropic_rate_decision,
    multi_seed_consensus,
    unpack_variable_codes,
)
from rars_v8_cutoff_sidecar_core import candidate_gap_recovery  # noqa: E402


PAYLOAD = "full_corpus_qw_ar_rpq_codes.uint8.memmap"
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
    if isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-15):
            raise ValueError(f"{label} changed: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def verify_packet(packet_root: Path, repo_root: Path) -> dict[str, Any]:
    protocol = read_json(
        repo_root
        / "protocols/rars_v14_query_whitened_anisotropic_rate_rpq_diagnostic_v1.json"
    )
    complete = read_json(packet_root / "diagnostic_complete.json")
    result = read_json(packet_root / "diagnostic_result.json")
    freeze = read_json(packet_root / "diagnostic_freeze.json")
    started = read_json(packet_root / "diagnostic_started.json")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V14 protocol identity changed")
    statuses = (
        (complete, "RARS_V14_ANISOTROPIC_RATE_DIAGNOSTIC_COMPLETE"),
        (result, "RARS_V14_ANISOTROPIC_RATE_DIAGNOSTIC_COMPLETE"),
        (freeze, "RARS_V14_METHOD_AND_DECISION_FROZEN"),
        (started, "RARS_V14_ANISOTROPIC_RATE_DIAGNOSTIC_STARTED"),
    )
    source_commit = complete.get("source_commit")
    for payload, status in statuses:
        if payload.get("status") != status or payload.get("source_commit") != source_commit:
            raise ValueError("V14 status or source lineage changed")
    verify_record(packet_root / "diagnostic_started.json", complete["started"], "start marker")
    for name, record in complete["outputs"].items():
        verify_record(packet_root / name, record, f"V14 output {name}")
    for relative, record in started["source_blobs"].items():
        verify_record(repo_root / relative, record, f"V14 source blob {relative}")
    if started["labels_used_for_representation_learning"] is not False:
        raise ValueError("V14 representation unexpectedly used labels")
    if started["future_method_holdout_opened"] is not False or started[
        "old_rars_holdout_opened"
    ] is not False:
        raise ValueError("V14 opened a prohibited holdout")

    qids = (packet_root / "query_ids.utf8.txt").read_text(encoding="utf-8").splitlines()
    folds = np.load(packet_root / "fold_ids.int64.npy", allow_pickle=False)
    target = int(protocol["input_contract"]["query_count"])
    if len(qids) != target or len(set(qids)) != target:
        raise ValueError("V14 query identities changed")
    if folds.shape != (target,) or folds.dtype != np.int64:
        raise ValueError("V14 fold array contract changed")
    if np.bincount(folds, minlength=5).tolist() != protocol["input_contract"]["fold_counts"]:
        raise ValueError("V14 fold counts changed")

    def load(prefix: str) -> dict[str, np.ndarray]:
        output = {
            metric: np.load(
                packet_root / f"per_query_{prefix}_{metric}_at_10.float64.npy",
                allow_pickle=False,
            )
            for metric in METRICS
        }
        if any(value.shape != (target,) or value.dtype != np.float64 for value in output.values()):
            raise ValueError(f"V14 metric contract changed for {prefix}")
        if any(np.any(~np.isfinite(value)) or np.any((value < 0) | (value > 1)) for value in output.values()):
            raise ValueError(f"V14 metric values are invalid for {prefix}")
        return output

    seeds = [int(value) for value in protocol["quantizer_training"]["seeds"]]
    primary_seed = int(protocol["quantizer_training"]["primary_seed"])
    base = load("base")
    exact = load("same_candidate_exact")
    pca16 = load("pca16")
    uniform_whitened = load("uniform_whitened_primary")
    uniform = {seed: load(f"v13_uniform_seed{seed}") for seed in seeds}
    challenger = {seed: load(f"anisotropic_seed{seed}") for seed in seeds}
    primary = {
        "anisotropic_vs_v13_uniform_rpq": paired_inference(
            challenger[primary_seed]["recall"],
            uniform[primary_seed]["recall"],
            **_inference_kwargs(protocol, "primary_vs_v13_uniform_rpq"),
        ),
        "anisotropic_vs_uniform_whitened": paired_inference(
            challenger[primary_seed]["recall"],
            uniform_whitened["recall"],
            **_inference_kwargs(protocol, "primary_vs_uniform_whitened"),
        ),
        "anisotropic_vs_pca16": paired_inference(
            challenger[primary_seed]["recall"],
            pca16["recall"],
            **_inference_kwargs(protocol, "primary_vs_pca16"),
        ),
        "anisotropic_vs_base": paired_inference(
            challenger[primary_seed]["recall"],
            base["recall"],
            **_inference_kwargs(protocol, "primary_vs_base"),
        ),
    }
    seed_gains = [
        float(np.mean(challenger[seed]["recall"] - uniform[seed]["recall"]))
        for seed in seeds
    ]
    fold_gains = [
        float(
            np.mean(
                challenger[primary_seed]["recall"][folds == fold]
                - uniform[primary_seed]["recall"][folds == fold]
            )
        )
        for fold in range(5)
    ]
    consensus = multi_seed_consensus(
        np.stack([challenger[seed]["recall"] for seed in seeds]),
        np.stack([uniform[seed]["recall"] for seed in seeds]),
    )
    gap = candidate_gap_recovery(
        challenger[primary_seed]["recall"], base["recall"], exact["recall"]
    )
    allocation = np.load(packet_root / "final_bit_allocation.int64.npy", allow_pickle=False)
    basis = np.load(packet_root / "final_pca_basis_rank64.float32.npy", allow_pickle=False)
    transforms = np.load(
        packet_root / "final_query_metric_transforms.float32.npy", allow_pickle=False
    )
    books = np.load(packet_root / "final_codebooks.float32.npy", allow_pickle=False)
    offsets = np.load(packet_root / "final_codebook_offsets.int64.npy", allow_pickle=False)
    if allocation.shape != (16,) or allocation.dtype != np.int64 or allocation.sum() != 128:
        raise ValueError("V14 final allocation contract changed")
    if basis.shape != (384, 64) or basis.dtype != np.float32:
        raise ValueError("V14 final basis contract changed")
    if transforms.shape != (16, 4, 4) or transforms.dtype != np.float32:
        raise ValueError("V14 final metric contract changed")
    if books.ndim != 2 or books.shape[1] != 4 or books.dtype != np.float32:
        raise ValueError("V14 final codebook contract changed")
    if offsets.shape != (17,) or offsets.dtype != np.int64 or offsets[-1] != len(books):
        raise ValueError("V14 final offsets contract changed")
    if not np.array_equal(np.diff(offsets), 1 << allocation):
        raise ValueError("V14 codebook sizes differ from allocated bits")
    if not np.allclose(basis.T @ basis, np.eye(64), rtol=0.0, atol=2e-4):
        raise ValueError("V14 basis is not orthonormal")
    if np.any(~np.isfinite(transforms)) or np.any(np.diagonal(transforms, axis1=1, axis2=2) <= 0):
        raise ValueError("V14 query transforms are invalid")

    full = result["final_fit"]["full_corpus_codes"]
    payload = packet_root / PAYLOAD
    if payload.stat().st_size != int(protocol["full_corpus_sidecar"]["payload_bytes"]):
        raise ValueError("V14 full payload byte count changed")
    if full["record"]["sha256"] != sha256_file(payload):
        raise ValueError("V14 full payload hash changed")
    registered_histograms = [
        np.asarray(value, dtype=np.int64) for value in full["code_histograms"]
    ]
    if len(registered_histograms) != 16:
        raise ValueError("V14 histogram block count changed")
    observed_histograms = [
        np.zeros(1 << int(bits), dtype=np.int64) for bits in allocation
    ]
    payload_codes = np.memmap(
        payload, dtype=np.uint8, mode="r", shape=(1_000_000, 16)
    )
    for start in range(0, 1_000_000, 100_000):
        decoded = unpack_variable_codes(
            np.asarray(payload_codes[start : start + 100_000]), allocation
        )
        for block in range(16):
            observed_histograms[block] += np.bincount(
                decoded[:, block], minlength=len(observed_histograms[block])
            )
    for block, histogram in enumerate(registered_histograms):
        if len(histogram) != 1 << int(allocation[block]) or int(histogram.sum()) != 1_000_000:
            raise ValueError("V14 registered code histogram changed")
        if not np.array_equal(histogram, observed_histograms[block]):
            raise ValueError("V14 full payload no longer matches its histogram")
    allocations = result["allocations"]
    recomputed = anisotropic_rate_decision(
        primary_vs_uniform_rpq=primary["anisotropic_vs_v13_uniform_rpq"],
        primary_vs_uniform_whitened=primary["anisotropic_vs_uniform_whitened"],
        primary_vs_pca16=primary["anisotropic_vs_pca16"],
        primary_vs_base=primary["anisotropic_vs_base"],
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        candidate_gap_recovery=gap,
        uniform_rpq_mrr=float(np.mean(uniform[primary_seed]["mrr"])),
        challenger_mrr=float(np.mean(challenger[primary_seed]["mrr"])),
        uniform_rpq_ndcg=float(np.mean(uniform[primary_seed]["ndcg"])),
        challenger_ndcg=float(np.mean(challenger[primary_seed]["ndcg"])),
        consensus=consensus,
        allocations=allocations,
        payload_bytes_per_document=int(full["payload_bytes_per_document"]),
        full_corpus_codes_materialized=True,
        thresholds=protocol["diagnostic_gate"],
    )
    if recomputed != result["decision"]:
        raise ValueError("V14 formal decision does not recompute")
    if len({complete["formal_decision"], result["formal_decision"], freeze["formal_decision"], recomputed["decision"]}) != 1:
        raise ValueError("V14 formal decisions disagree")
    for key, observed in (("seed_gains", seed_gains), ("fold_gains", fold_gains)):
        if not np.allclose(observed, result[key], rtol=0.0, atol=1e-15):
            raise ValueError(f"V14 {key} changed")
    if consensus != result["multi_seed_consensus"]:
        raise ValueError("V14 consensus support changed")
    _assert_close(gap, result["candidate_gap_recovery_fraction"], "gap recovery")
    for name, comparison in primary.items():
        if comparison != result["comparisons"][name]:
            raise ValueError(f"V14 comparison changed: {name}")
    return {
        "status": "RARS_V14_PACKET_VERIFIED",
        "source_commit": source_commit,
        "formal_decision": recomputed["decision"],
        "query_count": target,
        "primary_gain": primary["anisotropic_vs_v13_uniform_rpq"]["mean_difference"],
        "primary_ci95": [
            primary["anisotropic_vs_v13_uniform_rpq"]["lower"],
            primary["anisotropic_vs_v13_uniform_rpq"]["upper"],
        ],
        "seed_gains": seed_gains,
        "fold_gains": fold_gains,
        "final_bit_allocation": allocation.tolist(),
        "full_corpus_payload_bytes": payload.stat().st_size,
        "verified_output_count": len(complete["outputs"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(verify_packet(args.packet_root, args.repo_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
