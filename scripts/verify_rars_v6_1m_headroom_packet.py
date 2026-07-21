#!/usr/bin/env python3
"""Verify the durable RARS-v6 1M headroom packet before v7 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.rars_v6_headroom_core import (
        decompose_recall_gaps,
        diagnostic_gate_decision,
        known_positive_recall_at_k,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from rars_v6_headroom_core import (
        decompose_recall_gaps,
        diagnostic_gate_decision,
        known_positive_recall_at_k,
    )


PROTOCOL_ID = "rars_v6_1m_headroom_v1"
SOURCE_COMMIT = "26a7717b964eed979b3bf7a3149d0d24e9bce3f1"
STATUS = "RARS_V6_1M_HEADROOM_COMPLETE"
DECISION = "GO_TO_V6_LOSS_IMPLEMENTATION"
QUERY_COUNT = 2307
ANALYSIS_K = 200


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


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} SHA-256 changed")


def _close(left: float, right: float, tolerance: float = 1e-15) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _load_array(
    packet_root: Path, filename: str, *, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> np.ndarray:
    value = np.load(packet_root / filename, allow_pickle=False)
    if value.dtype != dtype or value.shape != shape:
        raise ValueError(
            f"{filename} has {value.dtype}/{value.shape}; expected {dtype}/{shape}"
        )
    return value


def _verify_identity(result: dict[str, Any], complete: dict[str, Any]) -> None:
    for payload, label in ((result, "result"), (complete, "complete")):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{label} protocol identity changed")
        if payload.get("status") != STATUS:
            raise ValueError(f"{label} completion status changed")
        if payload.get("source_commit") != SOURCE_COMMIT:
            raise ValueError(f"{label} source commit changed")
        if payload.get("formal_decision") != DECISION:
            raise ValueError(f"{label} formal decision changed")
        for flag in ("training_performed", "adapter_used", "rars_used"):
            if payload.get(flag) is not False:
                raise ValueError(f"{label} unexpectedly reports {flag}")
        if payload.get("future_or_audit_role_opened") is not False:
            raise ValueError(f"{label} reports forbidden role access")
    if complete.get("corpus_tensor_persisted") is not False:
        raise ValueError("V6 reports a persisted corpus tensor")
    if result.get("evidence_role") != "oracle_design":
        raise ValueError("V6 evidence role changed")
    if result.get("signal_gate", {}).get("training_authorized") is not False:
        raise ValueError("V6 incorrectly authorizes training under its own protocol")


def verify_packet(packet_root: Path) -> dict[str, Any]:
    packet_root = packet_root.resolve()
    result_path = packet_root / "headroom_result.json"
    complete_path = packet_root / "headroom_complete.json"
    result = read_json(result_path)
    complete = read_json(complete_path)
    _verify_identity(result, complete)
    verify_record(result_path, complete["result"], "headroom_result.json")
    if result.get("outputs") != complete.get("outputs"):
        raise ValueError("Result and complete output registries differ")
    for filename, record in complete["outputs"].items():
        verify_record(packet_root / filename, record, filename)

    row_shape = (QUERY_COUNT, ANALYSIS_K)
    score_shape = row_shape
    recall_shape = (QUERY_COUNT,)
    rows = {
        name: _load_array(packet_root, name, dtype=np.dtype("int64"), shape=row_shape)
        for name in (
            "base_pq_top_rows.int64.npy",
            "ivf_exact_top_rows.int64.npy",
            "full_exact_top_rows.int64.npy",
        )
    }
    for name in (
        "base_pq_top_scores.float32.npy",
        "ivf_exact_top_scores.float32.npy",
        "full_exact_top_scores.float32.npy",
    ):
        scores = _load_array(
            packet_root, name, dtype=np.dtype("float32"), shape=score_shape
        )
        if np.any(~np.isfinite(scores)):
            raise ValueError(f"{name} contains non-finite values")
    _load_array(
        packet_root,
        "probed_ivf_lists.int64.npy",
        dtype=np.dtype("int64"),
        shape=(QUERY_COUNT, 16),
    )

    recalls = {
        method: {
            k: _load_array(
                packet_root,
                f"{method}_recall_at_{k}.float64.npy",
                dtype=np.dtype("float64"),
                shape=recall_shape,
            )
            for k in (10, 100)
        }
        for method in ("base_pq", "ivf_exact", "full_exact")
    }
    for method, by_k in recalls.items():
        result_name = "same_ivf_exact" if method == "ivf_exact" else method
        for k, values in by_k.items():
            if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError(f"{method} Recall@{k} is invalid")
            reported = result["mean_recall"][result_name][f"recall_at_{k}"]
            if not _close(float(np.mean(values)), float(reported)):
                raise ValueError(f"{method} Recall@{k} mean cannot be recomputed")

    decomposition = decompose_recall_gaps(
        recalls["full_exact"][100],
        recalls["ivf_exact"][100],
        recalls["base_pq"][100],
    )
    decomposition["qrels_corpus_coverage"] = float(
        result["qrels_mapping"]["qrels_corpus_coverage"]
    )
    for key, value in result["recall_gap_decomposition"].items():
        if key == "qrels_corpus_coverage":
            continue
        if isinstance(value, float) and not _close(decomposition[key], value):
            raise ValueError(f"Recall decomposition changed: {key}")
        if isinstance(value, int) and decomposition[key] != value:
            raise ValueError(f"Recall decomposition changed: {key}")

    uncapped = result["flip_support"]["uncapped"]
    gate = diagnostic_gate_decision(
        pq_specific_r100_gap=decomposition["pq_specific_r100_gap"],
        uncapped_triplets=int(uncapped["triplets"]),
        distinct_flip_queries=int(uncapped["distinct_flip_queries"]),
        effective_sample_size=float(uncapped["effective_sample_size"]),
        max_query_weight_share=float(uncapped["max_query_weight_share"]),
        qrels_corpus_coverage=float(decomposition["qrels_corpus_coverage"]),
    )
    if gate != result["signal_gate"] or gate != complete["signal_gate"]:
        raise ValueError("V6 signal gate cannot be recomputed")
    if gate["failed_gates"] or gate["decision"] != DECISION:
        raise ValueError("V6 packet does not contain the required GO")

    return {
        "status": "RARS_V6_1M_HEADROOM_PACKET_VERIFIED",
        "protocol_id": PROTOCOL_ID,
        "source_commit": SOURCE_COMMIT,
        "formal_decision": DECISION,
        "result": file_record(result_path),
        "query_count": QUERY_COUNT,
        "base_pq_recall_at_10": float(np.mean(recalls["base_pq"][10])),
        "base_pq_recall_at_100": float(np.mean(recalls["base_pq"][100])),
        "same_ivf_exact_recall_at_100": float(np.mean(recalls["ivf_exact"][100])),
        "pq_specific_recall_at_100_gap": float(
            decomposition["pq_specific_r100_gap"]
        ),
        "uncapped_flip_triplets": int(uncapped["triplets"]),
        "distinct_flip_queries": int(uncapped["distinct_flip_queries"]),
        "flip_effective_sample_size": float(uncapped["effective_sample_size"]),
        "max_query_weight_share": float(uncapped["max_query_weight_share"]),
        "verified_output_count": int(len(complete["outputs"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(verify_packet(parse_args().packet_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
