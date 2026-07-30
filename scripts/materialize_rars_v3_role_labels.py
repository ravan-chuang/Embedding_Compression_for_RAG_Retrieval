#!/usr/bin/env python3
"""Materialize frozen parent labels for one permitted RARS-v3 role.

Design labels may be materialized after the qrels-free candidate freeze. Audit
labels additionally require a fully verified evaluator ``design_freeze.json``.
The future-method holdout is not an accepted role.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_msmarco_rars_v3_oracle_bundles import (
    CANONICAL_PROTOCOL,
    _load_and_verify_parent_bundle,
    _validate_canonical_repository,
    _verify_record,
    atomic_json,
    atomic_save,
)
from rars_v3_oracle_core import (
    AUDIT_ROLE_ID,
    DESIGN_ROLE_ID,
    PROTOCOL_ID,
    array_sha256,
    build_run_fingerprint,
    canonical_sha256,
    file_record,
    read_json,
    sha256_file,
    split_development_qids,
)


LABEL_FILENAMES = (
    "candidate_relevance.uint8.npy",
    "relevant_counts.int32.npy",
    "v3_role_labels_started.json",
    "v3_role_labels_manifest.json",
)


def _record_matches(path: Path, record: dict[str, Any], label: str) -> None:
    _verify_record(path, record, label)


def _validate_candidate_freeze(
    bundle_root: Path,
    *,
    role_id: str,
    protocol: dict[str, Any],
    source_commit: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    summary_path = bundle_root / "v3_oracle_bundle_freeze_summary.json"
    summary = read_json(summary_path)
    if summary.get("status") != "V3_QRELS_FREE_CANDIDATE_BUNDLES_FROZEN":
        raise ValueError("Candidate bundle freeze is incomplete")
    if summary.get("protocol_id") != PROTOCOL_ID or summary.get("source_commit") != source_commit:
        raise ValueError("Candidate bundle freeze source/protocol mismatch")
    if summary.get("qrels_opened_or_parsed") is not False:
        raise ValueError("Candidate bundle freeze was not qrels-free")
    if summary.get("parent_label_values_loaded_or_sliced") is not False:
        raise ValueError("Candidate builder touched parent label values")
    if summary.get("parent_label_payload_bytes_read") is not False:
        raise ValueError("Candidate builder read parent label payload bytes")
    if summary.get("parent_candidate_payloads_hash_verified") is not True:
        raise ValueError("Candidate payload hashes were not fully verified")
    _record_matches(
        bundle_root / "v3_oracle_bundle_build_started.json",
        summary["started"],
        "candidate-freeze start marker",
    )
    _record_matches(
        bundle_root / "v3_oracle_split_audit.json",
        summary["split_audit"],
        "candidate split audit",
    )
    future = summary.get("future_method_holdout", {})
    if any(
        future.get(field) is not False
        for field in ("candidate_arrays_created", "labels_materialized", "metrics_computed")
    ):
        raise ValueError("Future-method holdout was accessed by candidate freeze")
    _record_matches(
        bundle_root / "future_method_holdout/v3_identity_manifest.json",
        future["identity_manifest"],
        "future-method identity manifest",
    )

    role_dir = bundle_root / role_id
    manifest_path = role_dir / "v3_candidate_manifest.json"
    summary_record = summary["roles"][role_id]["candidate_manifest"]
    _record_matches(manifest_path, summary_record, f"{role_id} candidate manifest")
    manifest = read_json(manifest_path)
    expected_hashes = {
        "builder_sha256": sha256_file(
            repo_root / "scripts/build_msmarco_rars_v3_oracle_bundles.py"
        ),
        "core_sha256": sha256_file(repo_root / "scripts/rars_v3_oracle_core.py"),
        "protocol_sha256": sha256_file(repo_root / CANONICAL_PROTOCOL),
        "parent_v2_2_manifest_sha256": protocol["parent_lineage"][
            "parent_inner_train_manifest_sha256"
        ],
        "parent_v2_source_manifest_sha256": protocol["parent_lineage"][
            "parent_inner_train_source_manifest_sha256"
        ],
        "parent_query_manifest_sha256": protocol["parent_lineage"][
            "parent_inner_train_query_manifest_sha256"
        ],
        "doc_ids_sha256": protocol["parent_lineage"]["frozen_doc_ids_sha256"],
    }
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("role_id") != role_id:
        raise ValueError("Candidate role manifest identity mismatch")
    if manifest.get("source_commit") != source_commit:
        raise ValueError("Candidate role manifest source commit mismatch")
    for field, expected in expected_hashes.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Candidate role lineage mismatch: {field}")
    label_contract = manifest.get("label_artifacts", {})
    if any(value is not False for value in label_contract.values()):
        raise ValueError("Candidate manifest was not frozen label-free")
    data_access = manifest.get("data_access", {})
    for field in (
        "qrels_opened_or_parsed",
        "parent_label_values_loaded_or_sliced",
        "faiss_imported_or_search_performed",
        "pca_fit_or_score_recomputation_performed",
        "future_method_holdout_candidate_arrays_created",
    ):
        if data_access.get(field) is not False:
            raise ValueError(f"Candidate data-access contract failed: {field}")
    for filename, record in manifest.get("files", {}).items():
        if Path(filename).name != filename:
            raise ValueError("Unsafe candidate filename registration")
        _record_matches(role_dir / filename, record, f"candidate file {filename}")
    query_manifest_path = role_dir / "query_manifest.json"
    _record_matches(query_manifest_path, manifest["query_manifest"], "role query manifest")
    query_manifest = read_json(query_manifest_path)
    qids = [str(value) for value in query_manifest["query_ids"]]
    parent_indices = np.load(role_dir / "parent_role_indices.int64.npy", mmap_mode="r")
    if canonical_sha256(qids) != manifest["query_ids_sha256"]:
        raise ValueError("Candidate role query ID hash changed")
    if array_sha256(parent_indices) != manifest["parent_role_indices_sha256"]:
        raise ValueError("Candidate parent-role indices changed")
    if list(np.asarray(parent_indices, dtype=np.int64)) != query_manifest[
        "parent_inner_train_indices"
    ]:
        raise ValueError("Candidate query manifest/parent indices disagree")
    if len(qids) != int(protocol["data_policy"]["roles"][role_id]["query_count"]):
        raise ValueError("Candidate role count differs from protocol")
    return manifest, query_manifest, np.asarray(parent_indices, dtype=np.int64)


def _verify_registered_outputs(directory: Path, records: dict[str, Any]) -> None:
    if not isinstance(records, dict) or not records:
        raise ValueError("Design freeze has no registered outputs")
    root = directory.resolve()
    for filename, record in records.items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe design-freeze output registration: {filename}")
        path = (root / relative).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"Unsafe design-freeze output registration: {filename}")
        _record_matches(path, record, f"design output {filename}")


def _validate_design_freeze(
    design_freeze_path: Path,
    *,
    bundle_root: Path,
    protocol: dict[str, Any],
    source_commit: str,
    repo_root: Path,
) -> dict[str, Any]:
    freeze = read_json(design_freeze_path)
    if freeze.get("status") != "DESIGN_ARTIFACTS_FROZEN_BEFORE_AUDIT_LOAD":
        raise ValueError("Audit labels require a complete design freeze")
    if freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("source_commit") != source_commit:
        raise ValueError("Design freeze source/protocol mismatch")
    if freeze.get("audit_bundle_loaded_before_this_freeze") is not False:
        raise ValueError("Audit bundle was loaded before the design freeze")
    if freeze.get("audit_role_labels_materialized_before_this_freeze", False) is not False:
        raise ValueError("Audit labels were materialized before the design freeze")

    run_fingerprint = freeze.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or len(run_fingerprint) != 64:
        raise ValueError("Design freeze lacks a canonical run fingerprint")
    started_path = design_freeze_path.parent / "oracle_started.json"
    started = read_json(started_path)
    if started.get("protocol_id") != PROTOCOL_ID or started.get("source_commit") != source_commit:
        raise ValueError("Oracle start marker source/protocol mismatch")
    if started.get("run_fingerprint") != run_fingerprint:
        raise ValueError("Oracle start/design-freeze fingerprint mismatch")
    payload = started.get("fingerprint_payload")
    if not isinstance(payload, dict) or build_run_fingerprint(payload) != run_fingerprint:
        raise ValueError("Oracle run fingerprint cannot be reproduced")

    canonical_sources = {
        "protocol_sha256": repo_root / CANONICAL_PROTOCOL,
        "builder_sha256": repo_root / "scripts/build_msmarco_rars_v3_oracle_bundles.py",
        "label_materializer_sha256": Path(__file__).resolve(),
        "core_sha256": repo_root / "scripts/rars_v3_oracle_core.py",
        "evaluator_sha256": repo_root
        / "scripts/evaluate_rars_v3_oracle_first_feasibility.py",
    }
    source_hashes = freeze.get("source_hashes", {})
    for field, path in canonical_sources.items():
        if source_hashes.get(field) != sha256_file(path):
            raise ValueError(f"Design-freeze source hash mismatch: {field}")

    design_candidate_path = bundle_root / DESIGN_ROLE_ID / "v3_candidate_manifest.json"
    audit_candidate_path = bundle_root / AUDIT_ROLE_ID / "v3_candidate_manifest.json"
    design_labels_path = bundle_root / DESIGN_ROLE_ID / "v3_role_labels_manifest.json"
    registrations = {
        "design_bundle_manifest": design_candidate_path,
        "audit_bundle_manifest_registered_but_arrays_unloaded": audit_candidate_path,
        "design_role_labels_manifest": design_labels_path,
    }
    for field, path in registrations.items():
        record = freeze.get(field)
        if not isinstance(record, dict):
            raise ValueError(f"Design freeze lacks registration: {field}")
        _record_matches(path, record, field)
    # Recursively verify the already-open design inputs and labels.  Only the
    # audit *manifest* is registered above; audit arrays remain untouched.
    design_candidate, _, _ = _validate_candidate_freeze(
        bundle_root,
        role_id=DESIGN_ROLE_ID,
        protocol=protocol,
        source_commit=source_commit,
        repo_root=repo_root,
    )
    design_labels = read_json(design_labels_path)
    if (
        design_labels.get("status")
        != "ROLE_LABELS_MATERIALIZED_FROM_FROZEN_PARENT"
        or design_labels.get("role_id") != DESIGN_ROLE_ID
        or design_labels.get("source_commit") != source_commit
        or design_labels.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("Design role-label manifest identity changed")
    _record_matches(
        design_candidate_path,
        design_labels["candidate_manifest"],
        "design labels candidate manifest",
    )
    if design_labels.get("parent_role_indices_sha256") != design_candidate.get(
        "parent_role_indices_sha256"
    ):
        raise ValueError("Design label/candidate parent indices disagree")
    _record_matches(
        bundle_root / DESIGN_ROLE_ID / "v3_role_labels_started.json",
        design_labels["started"],
        "design role-label start marker",
    )
    for filename, record in design_labels.get("files", {}).items():
        if Path(filename).name != filename:
            raise ValueError("Unsafe design role-label filename")
        _record_matches(
            bundle_root / DESIGN_ROLE_ID / filename,
            record,
            f"design role-label file {filename}",
        )
    _verify_registered_outputs(design_freeze_path.parent, freeze.get("registered_outputs"))
    return freeze


def _assert_label_outputs_absent(role_dir: Path) -> None:
    present = [filename for filename in LABEL_FILENAMES if (role_dir / filename).exists()]
    if present:
        raise ValueError(f"Refusing to reuse partial/existing role labels: {present}")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.role not in {DESIGN_ROLE_ID, AUDIT_ROLE_ID}:
        raise ValueError("Only oracle_design and oracle_audit labels are permitted")
    repo_root = Path(__file__).resolve().parents[1]
    protocol = _validate_canonical_repository(repo_root, args.protocol, args.source_commit)
    if args.role == AUDIT_ROLE_ID:
        if args.design_freeze is None:
            raise ValueError("Audit labels require --design-freeze")
        # Only the design-freeze and its small manifest registrations are read
        # before this point; no audit candidate array may be hashed or loaded.
        _validate_design_freeze(
            args.design_freeze,
            bundle_root=args.bundle_root,
            protocol=protocol,
            source_commit=args.source_commit,
            repo_root=repo_root,
        )
    candidate_manifest, query_manifest, parent_indices = _validate_candidate_freeze(
        args.bundle_root,
        role_id=args.role,
        protocol=protocol,
        source_commit=args.source_commit,
        repo_root=repo_root,
    )
    role_dir = args.bundle_root / args.role
    _assert_label_outputs_absent(role_dir)
    audit_dir = args.bundle_root / AUDIT_ROLE_ID
    if args.role == DESIGN_ROLE_ID:
        _assert_label_outputs_absent(audit_dir)
        if args.design_freeze is not None:
            raise ValueError("Design labels must not accept a design-freeze input")
        design_freeze_record = None
    else:
        assert args.design_freeze is not None
        design_freeze_record = file_record(args.design_freeze)

    parent_manifest, _, parent_qids, _, _ = _load_and_verify_parent_bundle(
        args.parent_inner_train_bundle,
        protocol,
        verify_label_payloads=True,
    )
    expected_indices = split_development_qids(parent_qids)[args.role]
    if not np.array_equal(parent_indices, expected_indices):
        raise ValueError("Role parent indices no longer match the frozen split")
    expected_qids = [parent_qids[int(index)] for index in parent_indices]
    if expected_qids != [str(value) for value in query_manifest["query_ids"]]:
        raise ValueError("Role query IDs no longer match parent labels")

    parent_labels_path = args.parent_inner_train_bundle / "candidate_relevance.uint8.npy"
    parent_counts_path = args.parent_inner_train_bundle / "relevant_counts.int32.npy"
    parent_labels = np.load(parent_labels_path, mmap_mode="r")
    parent_counts = np.load(parent_counts_path, mmap_mode="r")
    parent_query_count = int(parent_manifest["query_count"])
    candidate_count = int(parent_manifest["candidate_count"])
    if parent_labels.shape != (parent_query_count, candidate_count):
        raise ValueError("Frozen parent candidate-label shape changed")
    if parent_counts.shape != (parent_query_count,):
        raise ValueError("Frozen parent relevant-count shape changed")
    labels = np.asarray(parent_labels[parent_indices], dtype=np.uint8)
    counts = np.asarray(parent_counts[parent_indices], dtype=np.int32)
    expected_shape = (
        int(candidate_manifest["query_count"]),
        int(candidate_manifest["candidate_count"]),
    )
    if labels.shape != expected_shape or counts.shape != (expected_shape[0],):
        raise ValueError("Materialized role-label shapes differ from candidate bundle")

    materializer_sha256 = sha256_file(Path(__file__).resolve())
    started_path = role_dir / "v3_role_labels_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "ROLE_LABEL_MATERIALIZATION_STARTED",
            "role_id": args.role,
            "source_commit": args.source_commit,
            "materializer_sha256": materializer_sha256,
            "candidate_manifest": file_record(role_dir / "v3_candidate_manifest.json"),
            "design_freeze": design_freeze_record,
            "qrels_opened_or_parsed": False,
            "future_method_holdout_label_values_sliced_or_interpreted": False,
        },
    )
    labels_path = role_dir / "candidate_relevance.uint8.npy"
    counts_path = role_dir / "relevant_counts.int32.npy"
    atomic_save(labels_path, labels)
    atomic_save(counts_path, counts)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "ROLE_LABELS_MATERIALIZED_FROM_FROZEN_PARENT",
        "role_id": args.role,
        "source_commit": args.source_commit,
        "query_count": expected_shape[0],
        "candidate_count": expected_shape[1],
        "query_ids_sha256": canonical_sha256(expected_qids),
        "parent_role_indices_sha256": array_sha256(parent_indices),
        "protocol_sha256": sha256_file(args.protocol),
        "materializer_sha256": materializer_sha256,
        "parent_v2_2_manifest_sha256": protocol["parent_lineage"][
            "parent_inner_train_manifest_sha256"
        ],
        "candidate_manifest": file_record(role_dir / "v3_candidate_manifest.json"),
        "started": file_record(started_path),
        "design_freeze": design_freeze_record,
        "files": {
            labels_path.name: file_record(labels_path),
            counts_path.name: file_record(counts_path),
        },
        "label_source": {
            "candidate_relevance": file_record(parent_labels_path),
            "relevant_counts": file_record(parent_counts_path),
            "selection": "exact frozen parent-role indices",
            "qrels_opened_or_parsed": False,
            "future_method_holdout_label_values_sliced_or_interpreted": False,
        },
        "audit_release": {
            "design_freeze_required": args.role == AUDIT_ROLE_ID,
            "design_freeze_verified": args.role == AUDIT_ROLE_ID,
        },
    }
    manifest_path = role_dir / "v3_role_labels_manifest.json"
    atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": file_record(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--parent-inner-train-bundle", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=(DESIGN_ROLE_ID, AUDIT_ROLE_ID))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / CANONICAL_PROTOCOL,
    )
    parser.add_argument("--design-freeze", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(materialize(parse_args()), indent=2))


if __name__ == "__main__":
    main()
