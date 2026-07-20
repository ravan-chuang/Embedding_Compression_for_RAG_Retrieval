#!/usr/bin/env python3
"""Verify the committed thin closure packet for the RARS-v5 100K pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_ID = "rars_v5_pq_aware_100k_pilot_v1"
SOURCE_COMMIT = "93105ae1895974e28b34952c2dd777f037c6e0bf"
RUN_ID = "3820883ba4aae9c5b58f815f9a42ec484d793f171bd22eb45d9f1c70df9b7dd9"
DECISION = "STOP_PQ_AWARE_100K_PILOT"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} SHA-256 changed")


def close(left: float, right: float, tolerance: float = 1e-15) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def notebook_source_parity(clean: Path, executed: Path) -> None:
    clean_notebook = read_json(clean)
    executed_notebook = read_json(executed)
    clean_cells = clean_notebook["cells"]
    executed_cells = executed_notebook["cells"]
    if len(clean_cells) != len(executed_cells):
        raise ValueError("Executed notebook cell count differs from the clean notebook")
    execution_counts: list[int] = []
    for index, (left, right) in enumerate(zip(clean_cells, executed_cells)):
        for field in ("cell_type", "id", "source"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Executed notebook source differs at cell {index}: {field}")
        if right["cell_type"] == "code":
            execution_counts.append(int(right["execution_count"]))
            if any(output.get("output_type") == "error" for output in right["outputs"]):
                raise ValueError(f"Executed notebook contains an error at cell {index}")
    if execution_counts != list(range(1, len(execution_counts) + 1)):
        raise ValueError("Executed notebook cells were not run once in order")
    rendered = json.dumps(executed_notebook, allow_nan=False)
    if DECISION not in rendered or RUN_ID not in rendered:
        raise ValueError("Executed notebook does not preserve the formal report")


def verify_packet(packet_root: Path) -> dict[str, Any]:
    packet_root = packet_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    protocol = read_json(repo_root / "protocols/rars_v5_pq_aware_100k_pilot_v1.json")
    started = read_json(packet_root / "training_started.json")
    history = read_json(packet_root / "training_history.json")
    result = read_json(packet_root / "pilot_result.json")
    complete = read_json(packet_root / "training_complete.json")
    per_query = read_json(packet_root / "per_query_recall_at_100.json")
    audit = read_json(packet_root / "artifact_audit.json")

    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Canonical protocol identity changed")
    identities = {started.get("run_id"), result.get("run_id"), complete.get("run_id")}
    if identities != {RUN_ID}:
        raise ValueError("Run identity changed")
    if complete.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Source commit changed")
    if result.get("formal_decision") != DECISION or complete.get("formal_decision") != DECISION:
        raise ValueError("Formal decision changed")
    if started.get("protocol_id") != PROTOCOL_ID or result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Artifact protocol identity changed")

    for field, filename in (
        ("started", "training_started.json"),
        ("history", "training_history.json"),
        ("result", "pilot_result.json"),
    ):
        verify_record(packet_root / filename, complete[field], filename)

    if [row.get("epoch") for row in history] != list(range(9)):
        raise ValueError("Training history must contain identity epoch 0 and epochs 1-8")
    if history[0].get("loss") is not None:
        raise ValueError("Epoch zero must be the exact untrained identity checkpoint")
    for row in history[1:]:
        values = (
            row["loss"],
            row["hard_pq_recall_at_10"],
            row["hard_pq_recall_at_100"],
            row["adapted_fp32_recall_at_100"],
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Training history contains a non-finite value")
    best = max(
        history,
        key=lambda row: (
            row["hard_pq_recall_at_100"],
            row["hard_pq_recall_at_10"],
            row["adapted_fp32_recall_at_100"],
            -row["epoch"],
        ),
    )
    if int(result["selected_epoch"]) != int(best["epoch"]):
        raise ValueError("Selected epoch does not follow the frozen lexicographic rule")

    base = [float(value) for value in per_query["base_recall_at_100"]]
    adapter = [float(value) for value in per_query["adapter_recall_at_100"]]
    delta = [float(value) for value in per_query["delta"]]
    query_count = int(result["selection"]["query_count"])
    if not (len(base) == len(adapter) == len(delta) == query_count == 728):
        raise ValueError("Per-query Recall arrays have the wrong length")
    recomputed_delta = [right - left for left, right in zip(base, adapter)]
    if recomputed_delta != delta:
        raise ValueError("Committed per-query deltas are inconsistent")
    mean_base = math.fsum(base) / query_count
    mean_adapter = math.fsum(adapter) / query_count
    mean_gain = math.fsum(delta) / query_count
    improved = sum(value > 0 for value in delta)
    harmed = sum(value < 0 for value in delta)
    selection = result["selection"]
    if not close(mean_base, selection["base_pq_recall_at_100"]):
        raise ValueError("Base Recall@100 cannot be recomputed")
    if not close(mean_adapter, selection["adapter_hard_pq_recall_at_100"]):
        raise ValueError("Adapter Recall@100 cannot be recomputed")
    if not close(mean_gain, selection["recall_at_100_gain"]):
        raise ValueError("Recall@100 gain cannot be recomputed")
    if improved != selection["improved_queries"] or harmed != selection["harmed_queries"]:
        raise ValueError("Improved/harmed query support cannot be recomputed")

    gate = protocol["stage_a_gate"]
    teacher = float(selection["teacher_exact_recall_at_100"])
    gap = teacher - mean_base
    recovery = mean_gain / gap if gap > 0 else 0.0
    required_support = max(
        int(gate["minimum_improved_queries"]),
        math.ceil(float(gate["minimum_improved_query_fraction"]) * query_count),
    )
    recomputed_gates = {
        "minimum_recall_at_100_gain": mean_gain
        >= float(gate["minimum_hard_pq_recall_at_100_gain"]),
        "bootstrap_lower_above_zero": float(selection["paired_bootstrap"]["lower"]) > 0.0,
        "minimum_gap_recovery": recovery
        >= float(gate["minimum_teacher_gap_recovery_fraction"]),
        "minimum_improved_query_support": improved >= required_support,
        "recall_at_10_guardrail": float(selection["adapter_hard_pq_recall_at_10"])
        >= float(selection["base_pq_recall_at_10"])
        - float(gate["maximum_hard_pq_recall_at_10_drop"]),
        "adapted_fp32_guardrail": float(selection["adapted_fp32_recall_at_100"])
        >= teacher - float(gate["maximum_adapted_fp32_recall_at_100_drop"]),
    }
    if recomputed_gates != result["gates"]:
        raise ValueError("Formal gates cannot be recomputed from the frozen protocol")
    if all(recomputed_gates.values()):
        raise ValueError("STOP decision is inconsistent with all gates passing")

    if audit.get("audit_status") != "FULL_EXPORTED_ARTIFACT_AUDIT_PASS":
        raise ValueError("Full-export audit status changed")
    if audit.get("run_id") != RUN_ID or audit.get("formal_decision") != DECISION:
        raise ValueError("Full-export audit identity changed")
    if audit.get("recomputed_selection", {}).get("improved_queries") != improved:
        raise ValueError("Full-export audit support differs from committed arrays")
    for filename, record in complete["outputs"].items():
        audited = audit["artifacts"].get(filename)
        if audited is None:
            raise ValueError(f"Full-export audit omitted {filename}")
        if audited["registered_bytes"] != record["bytes"]:
            raise ValueError(f"Full-export byte record changed: {filename}")
        if audited["registered_sha256"] != record["sha256"]:
            raise ValueError(f"Full-export hash record changed: {filename}")

    notebook_source_parity(
        repo_root / "notebooks/MSMARCO_RARS_v5_PQ_Aware_100K_Pilot.ipynb",
        packet_root
        / "executed_notebook/MSMARCO_RARS_v5_PQ_Aware_100K_Pilot.ipynb",
    )

    closure = read_json(packet_root / "closure_manifest.json")
    if closure.get("run_id") != RUN_ID or closure.get("formal_decision") != DECISION:
        raise ValueError("Closure manifest identity changed")
    expected_files = {
        path.relative_to(packet_root).as_posix()
        for path in packet_root.rglob("*")
        if path.is_file() and path.name != "closure_manifest.json"
    }
    if set(closure.get("files", {})) != expected_files:
        raise ValueError("Closure manifest file inventory is incomplete")
    for relative, record in closure["files"].items():
        verify_record(packet_root / relative, record, f"closure file {relative}")

    return {
        "status": "RARS_V5_PQ_AWARE_100K_CLOSURE_VERIFIED",
        "formal_decision": DECISION,
        "run_id": RUN_ID,
        "selected_epoch": int(result["selected_epoch"]),
        "query_count": query_count,
        "recall_at_100_gain": mean_gain,
        "improved_queries": improved,
        "harmed_queries": harmed,
        "failed_gates": [name for name, passed in recomputed_gates.items() if not passed],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/rars_v5_pq_aware_100k_pilot",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(verify_packet(parse_args().packet_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
