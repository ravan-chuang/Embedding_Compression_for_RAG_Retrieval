#!/usr/bin/env python3
"""Verify the committed RARS-v2.2 FP32 replication closure packet."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "results" / "rars_v2_2_fp32_replication"
DEFAULT_SOURCE_NOTEBOOK = (
    ROOT / "notebooks" / "MSMARCO_RARS_v2_2_FP32_Replication.ipynb"
)
AGGREGATE_NAME = "aggregate-00a0dee30767"
EXPECTED_TRAINING_COMMIT = "bb9b106e69b9a453756fd800665f701614ce67b3"
EXPECTED_CONTROL_COMMIT = "00a0dee30767b04b8c650c28d63f4f662ef61517"
EXPECTED_DECISION = "UNSTABLE_NO_QAT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET)
    parser.add_argument(
        "--source-notebook",
        type=Path,
        default=DEFAULT_SOURCE_NOTEBOOK,
    )
    parser.add_argument("--write-closure-manifest", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_record(path: Path, record: dict[str, Any]) -> None:
    assert path.is_file(), f"Missing registered artifact: {path}"
    assert path.stat().st_size == record["bytes"], path
    assert sha256_file(path) == record["sha256"], path


def normalized_notebook_sources(notebook: dict[str, Any]) -> bytes:
    cells = [
        {"cell_type": cell["cell_type"], "source": cell.get("source", [])}
        for cell in notebook["cells"]
    ]
    return json.dumps(
        cells,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def read_npy_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        assert handle.read(6) == b"\x93NUMPY", path
        major, minor = struct.unpack("BB", handle.read(2))
        if major == 1:
            header_length = struct.unpack("<H", handle.read(2))[0]
        elif major in {2, 3}:
            header_length = struct.unpack("<I", handle.read(4))[0]
        else:
            raise AssertionError(f"Unsupported NPY version {major}.{minor}: {path}")
        header = handle.read(header_length).decode("latin1").strip()
    value = ast.literal_eval(header)
    assert isinstance(value, dict), path
    return value


def local_runner_path(packet_root: Path, original: str) -> Path:
    marker = "/fp32-replication-v1/"
    assert marker in original, original
    relative = original.split(marker, 1)[1]
    if relative.startswith("aggregate-"):
        return packet_root / relative
    if relative.startswith(("seed43-fp32/", "seed44-fp32/")):
        return packet_root / "seeds" / relative
    return packet_root / "provenance" / relative


def evidence_files(packet_root: Path) -> list[Path]:
    excluded = {"README.md", "closure_manifest.json"}
    return sorted(
        path
        for path in packet_root.rglob("*")
        if path.is_file() and path.name not in excluded
    )


def build_closure_manifest(
    packet_root: Path,
    source_notebook: Path,
) -> dict[str, Any]:
    executed_path = (
        packet_root
        / "executed_notebook"
        / "MSMARCO_RARS_v2_2_FP32_Replication.ipynb"
    )
    executed = load_json(executed_path)
    source = load_json(source_notebook)
    executed_sources = normalized_notebook_sources(executed)
    source_sources = normalized_notebook_sources(source)
    assert executed_sources == source_sources

    code_cells = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
    source_digest = hashlib.sha256(executed_sources).hexdigest()
    records = {
        path.relative_to(packet_root).as_posix(): file_record(path)
        for path in evidence_files(packet_root)
    }
    return {
        "schema_version": 1,
        "protocol_id": "rars_v2_2_fp32_replication_v1",
        "status": "REPLICATION_COMPLETE",
        "decision": EXPECTED_DECISION,
        "training_source_commit": EXPECTED_TRAINING_COMMIT,
        "control_commit": EXPECTED_CONTROL_COMMIT,
        "imported_on": "2026-07-19",
        "source": {
            "drive_root": (
                "/content/drive/MyDrive/rag-pq-checkpoints/"
                "rars-v2.2-fp32-msmarco/bb9b106e69b9/fp32-replication-v1"
            ),
            "drive_root_folder_id": "1FWkHct3p0oCXqKxamMW81-Y26r3PqmYA",
            "drive_aggregate_folder_id": "1jlf_OidrQA1dkeU7b4GtVFV_cXl6ODHz",
        },
        "executed_notebook": {
            "path": executed_path.relative_to(packet_root).as_posix(),
            "sha256": sha256_file(executed_path),
            "source_cells_sha256": source_digest,
            "source_cells_match_committed_notebook": True,
            "code_execution_counts": [
                cell.get("execution_count") for cell in code_cells
            ],
            "error_output_count": sum(
                output.get("output_type") == "error"
                for cell in code_cells
                for output in cell.get("outputs", [])
            ),
        },
        "files": records,
    }


def write_closure_manifest(packet_root: Path, source_notebook: Path) -> Path:
    manifest = build_closure_manifest(packet_root, source_notebook)
    path = packet_root / "closure_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def verify_packet(packet_root: Path, source_notebook: Path) -> dict[str, Any]:
    aggregate = packet_root / AGGREGATE_NAME
    complete = load_json(aggregate / "replication_complete.json")
    decision = load_json(aggregate / "replication_decision.json")
    summary = load_json(aggregate / "replication_summary.json")
    runner = load_json(packet_root / "provenance" / "replication_runner_manifest.json")

    assert complete["status"] == "REPLICATION_COMPLETE"
    assert complete["decision"] == EXPECTED_DECISION
    assert len(complete["outputs"]) == 19
    for name, record in complete["outputs"].items():
        verify_record(aggregate / name, record)

    assert decision["decision"] == EXPECTED_DECISION
    assert decision["qat_protocol_definition_authorized"] is False
    assert decision["query_count"] == 1019
    assert decision["positive_support_queries_required_per_heldout_seed_and_contrast"] == 11
    assert decision["failed_conditions"] == [
        "heldout_seeds_each_have_required_positive_query_support"
    ]
    assert summary["decision"] == EXPECTED_DECISION
    assert summary["evidence_boundary"]["development_only"] is True
    assert summary["decision_conditions"] == decision["conditions"]

    per_seed = {row["seed"]: row for row in summary["per_seed"]}
    assert set(per_seed) == {42, 43, 44}
    assert per_seed[43]["vs_pca_fp32"]["improved_queries"] == 12
    assert per_seed[44]["vs_pca_fp32"]["improved_queries"] == 10
    for row in per_seed.values():
        for comparator in ("vs_base", "vs_pca_fp32"):
            counts = row[comparator]
            assert sum(counts.values()) == 1019

    assert runner["training_source_commit"] == EXPECTED_TRAINING_COMMIT
    assert runner["control_commit"] == EXPECTED_CONTROL_COMMIT
    for record in runner["artifacts"]:
        verify_record(local_runner_path(packet_root, record["path"]), record)

    seed_directories = {
        42: packet_root / "seeds" / "seed42-fp32-stage-a",
        43: packet_root / "seeds" / "seed43-fp32",
        44: packet_root / "seeds" / "seed44-fp32",
    }
    for seed, directory in seed_directories.items():
        marker = load_json(directory / "training_complete.json")
        assert marker["status"] == "TRAINING_COMPLETE"
        assert marker["run_id"] == per_seed[seed]["run_id"]
        for name, record in marker["outputs"].items():
            verify_record(directory / name, record)

    expected_npy = {
        "selection_base_per_query.float64.npy": ((1019,), "<f8"),
        "selection_pca_fp32_per_query.float64.npy": ((1019,), "<f8"),
        "selection_pca_bounded_per_query.float64.npy": ((1019,), "<f8"),
        "selection_v2_2_seed_42_per_query.float64.npy": ((1019,), "<f8"),
        "selection_v2_2_seed_43_per_query.float64.npy": ((1019,), "<f8"),
        "selection_v2_2_seed_44_per_query.float64.npy": ((1019,), "<f8"),
        "selection_v2_2_by_seed.float64.npy": ((3, 1019), "<f8"),
        "gain_over_base_by_seed.float64.npy": ((3, 1019), "<f8"),
        "gain_over_pca_fp32_by_seed.float64.npy": ((3, 1019), "<f8"),
        "selection_v2_2_heldout_mean_per_query.float64.npy": ((1019,), "<f8"),
        "selection_v2_2_all_seed_mean_per_query.float64.npy": ((1019,), "<f8"),
        "paired_bootstrap_statistics.float64.npy": ((20000, 10), "<f8"),
    }
    for name, (shape, descriptor) in expected_npy.items():
        header = read_npy_header(aggregate / name)
        assert header["shape"] == shape, name
        assert header["descr"] == descriptor, name
        assert header["fortran_order"] is False, name

    with (aggregate / "per_query_replication.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert sum(1 for _ in csv.reader(handle)) == 1020
    with (aggregate / "per_seed_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert sum(1 for _ in csv.reader(handle)) == 4

    executed_path = (
        packet_root
        / "executed_notebook"
        / "MSMARCO_RARS_v2_2_FP32_Replication.ipynb"
    )
    executed = load_json(executed_path)
    source = load_json(source_notebook)
    assert normalized_notebook_sources(executed) == normalized_notebook_sources(source)
    code_cells = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
    assert [cell.get("execution_count") for cell in code_cells] == list(range(1, 9))
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]

    closure_path = packet_root / "closure_manifest.json"
    if closure_path.exists():
        closure = load_json(closure_path)
        assert closure["decision"] == EXPECTED_DECISION
        expected_files = {
            path.relative_to(packet_root).as_posix(): file_record(path)
            for path in evidence_files(packet_root)
        }
        assert closure["files"] == expected_files
        assert closure["executed_notebook"]["error_output_count"] == 0
        assert closure["executed_notebook"][
            "source_cells_match_committed_notebook"
        ] is True

    heldout = summary["groups"]["heldout_replication_seeds"]
    return {
        "status": complete["status"],
        "decision": decision["decision"],
        "registered_aggregate_outputs": len(complete["outputs"]),
        "runner_artifacts": len(runner["artifacts"]),
        "verified_seeds": sorted(seed_directories),
        "query_count": summary["query_count"],
        "heldout_mean_recall_at_10": heldout["v2_2_fp32_recall_at_10"]["mean"],
        "heldout_gain_over_pca_fp32": heldout["gain_over_pca_fp32"]["mean"],
    }


def main() -> None:
    args = parse_args()
    if args.write_closure_manifest:
        path = write_closure_manifest(args.packet_root, args.source_notebook)
        print(f"Wrote {path}")
    report = verify_packet(args.packet_root, args.source_notebook)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
