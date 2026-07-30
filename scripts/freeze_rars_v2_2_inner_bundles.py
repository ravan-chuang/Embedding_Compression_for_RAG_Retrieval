#!/usr/bin/env python3
"""Audit and freeze only the inner MS MARCO bundles for RARS-v2.2.

The v2.1 builder historically parsed a 6,980-query qrels container.  This
freezer does not pretend that access never happened.  It proves which query
rows entered each inner bundle, records the broader-container access honestly,
and emits role-specific manifests that the v2.2 trainer can validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rars_v2_2_core import (
    FIT_ROLE_ID,
    PROTOCOL_ID,
    SELECTION_ROLE_ID,
    canonical_sha256,
    file_record,
    sha256_file,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    payload = read_json(path)
    qids = [str(value) for value in payload["query_ids"]]
    rows = np.asarray(payload["query_rows"], dtype=np.int64)
    if len(qids) != len(rows):
        raise ValueError(f"Split qid/row lengths disagree: {path}")
    if len(set(qids)) != len(qids):
        raise ValueError(f"Split contains duplicate qids: {path}")
    if len(set(rows.tolist())) != len(rows):
        raise ValueError(f"Split contains duplicate query rows: {path}")
    if np.any(rows < 0):
        raise ValueError(f"Split contains negative query rows: {path}")
    return qids, rows


def inner_partition(qids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the already-established v2.1 80/20 inner partition."""

    is_selection = np.asarray([
        int(hashlib.sha256(f"rars-v2.1-inner:{qid}".encode()).hexdigest()[:16], 16)
        % 5 == 0
        for qid in qids
    ])
    fit = np.flatnonzero(~is_selection)
    selection = np.flatnonzero(is_selection)
    if not len(fit) or not len(selection):
        raise ValueError("Inner split is empty")
    return fit, selection


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def text_values_sha256(values: list[str]) -> str:
    return canonical_sha256(values)


def _assert_disjoint(
    left_name: str,
    left_qids: list[str],
    left_rows: np.ndarray,
    right_name: str,
    right_qids: list[str],
    right_rows: np.ndarray,
) -> dict[str, int]:
    qid_overlap = len(set(left_qids) & set(right_qids))
    row_overlap = len(set(left_rows.tolist()) & set(right_rows.tolist()))
    if qid_overlap or row_overlap:
        raise ValueError(
            f"{left_name}/{right_name} overlap: "
            f"{qid_overlap} qids, {row_overlap} rows"
        )
    return {"qid_overlap": qid_overlap, "row_overlap": row_overlap}


def _verify_source_bundle(bundle_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Source bundle has no file records: {bundle_dir}")
    for filename, record in files.items():
        path = bundle_dir / filename
        if not path.exists():
            raise ValueError(f"Source bundle file is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Source bundle byte count changed: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Source bundle hash changed: {path}")
    return manifest, sha256_file(manifest_path)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.source_commit) != 40 or any(
        value not in "0123456789abcdef" for value in args.source_commit
    ):
        raise ValueError("--source-commit must be exact lowercase 40-hex")
    repo_root = Path(__file__).resolve().parents[1]
    source_builder_sha256 = sha256_file(
        repo_root / "scripts/build_msmarco_rars_v2_boundary_bundles.py"
    )
    bundle_freezer_sha256 = sha256_file(Path(__file__).resolve())
    protocol_sha256 = sha256_file(
        repo_root / "protocols/rars_v2_2_boundary_loss_development_v1.json"
    )
    train_qids, train_rows = load_split(args.train_split)
    outer_qids, outer_rows = load_split(args.outer_validation_split)
    test_qids, test_rows = load_split(args.clean_test_split)
    fit_indices, selection_indices = inner_partition(train_qids)
    roles = {
        FIT_ROLE_ID: (
            [train_qids[index] for index in fit_indices],
            train_rows[fit_indices],
            "train",
        ),
        SELECTION_ROLE_ID: (
            [train_qids[index] for index in selection_indices],
            train_rows[selection_indices],
            "validation",
        ),
    }
    overlap = {
        "inner_train_vs_inner_validation": _assert_disjoint(
            FIT_ROLE_ID,
            roles[FIT_ROLE_ID][0],
            roles[FIT_ROLE_ID][1],
            SELECTION_ROLE_ID,
            roles[SELECTION_ROLE_ID][0],
            roles[SELECTION_ROLE_ID][1],
        )
    }
    for role_id, (qids, rows, _) in roles.items():
        overlap[f"{role_id}_vs_burned_outer"] = _assert_disjoint(
            role_id, qids, rows, "burned_outer", outer_qids, outer_rows
        )
        overlap[f"{role_id}_vs_clean_test"] = _assert_disjoint(
            role_id, qids, rows, "clean_test", test_qids, test_rows
        )

    split_audit = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_commit": args.source_commit,
        "source_builder_sha256": source_builder_sha256,
        "bundle_freezer_sha256": bundle_freezer_sha256,
        "protocol_sha256": protocol_sha256,
        "source_train_split": file_record(args.train_split),
        "source_outer_validation_split": file_record(args.outer_validation_split),
        "source_clean_test_split": file_record(args.clean_test_split),
        "roles": {
            role_id: {
                "query_count": len(qids),
                "query_ids_sha256": text_values_sha256(qids),
                "query_rows_sha256": array_sha256(rows),
                "evidence_status": "DEVELOPMENT_ONLY",
            }
            for role_id, (qids, rows, _) in roles.items()
        },
        "burned_outer_validation": {
            "query_count": len(outer_qids),
            "query_ids_sha256": text_values_sha256(outer_qids),
            "query_rows_sha256": array_sha256(outer_rows),
            "evidence_status": "BURNED_DEVELOPMENT_ONLY",
            "selection_allowed": False,
            "confirmatory_claim_allowed": False,
        },
        "clean_test": {
            "query_count": len(test_qids),
            "query_ids_sha256": text_values_sha256(test_qids),
            "query_rows_sha256": array_sha256(test_rows),
            "used_by_v2_2": False,
        },
        "pairwise_overlap": overlap,
        "all_required_assertions_passed": True,
    }
    audit_path = args.bundle_root / "v2_2_split_audit.json"
    atomic_json(audit_path, split_audit)
    split_audit_sha256 = sha256_file(audit_path)

    query_vectors = np.load(args.query_vectors, mmap_mode="r")
    frozen_roles: dict[str, Any] = {}
    for role_id, (qids, rows, split_role) in roles.items():
        bundle_dir = args.bundle_root / role_id
        source_manifest, source_manifest_sha256 = _verify_source_bundle(bundle_dir)
        bundle_queries = np.load(
            bundle_dir / "query_vectors.float32.npy", mmap_mode="r"
        )
        if bundle_queries.shape != (len(rows), query_vectors.shape[1]):
            raise ValueError(f"Unexpected query array shape in {role_id}")
        expected_queries = np.asarray(query_vectors[rows], dtype=np.float32)
        if not np.array_equal(np.asarray(bundle_queries), expected_queries):
            raise ValueError(f"Bundle query order/content does not match {role_id}")
        ann_rows = np.load(bundle_dir / "ann_rows.int64.npy", mmap_mode="r")
        labels = np.load(
            bundle_dir / "candidate_relevance.uint8.npy", mmap_mode="r"
        )
        if ann_rows.shape != labels.shape or ann_rows.shape[0] != len(qids):
            raise ValueError(f"Candidate arrays do not match {role_id}")
        query_manifest_path = bundle_dir / "query_manifest.json"
        atomic_json(
            query_manifest_path,
            {
                "role_id": role_id,
                "query_ids": qids,
                "query_rows": rows.tolist(),
            },
        )
        v2_manifest = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_commit": args.source_commit,
            "source_builder_sha256": source_builder_sha256,
            "bundle_freezer_sha256": bundle_freezer_sha256,
            "protocol_sha256": protocol_sha256,
            "role_id": role_id,
            "split_role": split_role,
            "evidence_status": "DEVELOPMENT_ONLY",
            "query_count": len(qids),
            "candidate_count": int(ann_rows.shape[1]),
            "query_ids_sha256": text_values_sha256(qids),
            "query_rows_sha256": array_sha256(rows),
            "split_audit_sha256": split_audit_sha256,
            "source_bundle_manifest_sha256": source_manifest_sha256,
            "source_bundle_protocol_id": source_manifest.get("protocol_id"),
            "query_manifest": file_record(query_manifest_path),
            "files": {
                filename: file_record(bundle_dir / filename)
                for filename in source_manifest["files"]
            },
            "data_access": {
                "shared_6980_qrels_container_opened_by_source_builder": True,
                "closed_test_query_ids_loaded_by_source_builder": True,
                "closed_test_relevance_values_parsed_by_source_builder": True,
                "closed_test_relevance_values_used": False,
                "closed_test_outcomes_computed": False,
                "outer_query_ids_loaded_for_split_audit": True,
                "outer_relevance_values_used": False,
                "outer_outcomes_used": False,
            },
        }
        manifest_path = bundle_dir / "v2_2_manifest.json"
        atomic_json(manifest_path, v2_manifest)
        frozen_roles[role_id] = {
            "manifest": file_record(manifest_path),
            "query_count": len(qids),
        }

    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_commit": args.source_commit,
        "status": "INNER_BUNDLES_FROZEN",
        "split_audit": file_record(audit_path),
        "roles": frozen_roles,
        "outer_validation_built_or_read_by_freezer": False,
        "outer_validation_evidence_status": "BURNED_DEVELOPMENT_ONLY",
    }
    atomic_json(args.bundle_root / "v2_2_freeze_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--query-vectors", required=True, type=Path)
    parser.add_argument("--train-split", required=True, type=Path)
    parser.add_argument("--outer-validation-split", required=True, type=Path)
    parser.add_argument("--clean-test-split", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2))


if __name__ == "__main__":
    main()
