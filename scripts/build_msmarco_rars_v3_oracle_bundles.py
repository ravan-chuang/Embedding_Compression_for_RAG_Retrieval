#!/usr/bin/env python3
"""Freeze qrels-free RARS-v3 candidate roles from the frozen v2.2 bundle.

This script never imports FAISS, opens qrels, fits PCA, or retrieves candidates.
It verifies every registered qrels-free v2.2 ``inner_train`` candidate artifact,
applies the frozen RARS-v3 hash split, and copies deterministic design/audit
subsets. Label payload bytes are first verified and selected later by
``materialize_rars_v3_role_labels.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rars_v3_oracle_core import (
    AUDIT_ROLE_ID,
    DESIGN_ROLE_ID,
    FUTURE_ROLE_ID,
    PROTOCOL_ID,
    array_sha256,
    canonical_sha256,
    design_fold_ids,
    file_record,
    read_json,
    sha256_file,
    split_development_qids,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


CANONICAL_PROTOCOL = Path("protocols/rars_v3_oracle_first_feasibility_v1.json")
CANONICAL_V2_2_SPLIT_AUDIT = Path(
    "results/rars_v2_2_fp32_replication/provenance/"
    "input-audit-00a0dee30767/v2_2_split_audit.json"
)
CANONICAL_CLOSED_INNER_VALIDATION = Path(
    "results/rars_v2_2_fp32_replication/provenance/"
    "input-audit-00a0dee30767/inner_validation/query_manifest.json"
)
CANONICAL_OUTER_VALIDATION = Path("splits/msmarco_rars_validation_split.json")
CANONICAL_CLEAN_TEST = Path("splits/msmarco_rars_test_split.json")

PARENT_REQUIRED_FILES = {
    "query_vectors.float32.npy",
    "ann_rows.int64.npy",
    "ann_scores.float32.npy",
    "candidate_relevance.uint8.npy",
    "relevant_counts.int32.npy",
    "candidate_doc_rows.int64.npy",
    "ann_residual_rows.int64.npy",
    "pca_scores.float32.npy",
    "rars_scores.float32.npy",
    "candidate_residuals.float32.npy",
}
PARENT_QRELS_FREE_ARRAY_FILES = PARENT_REQUIRED_FILES - {
    "candidate_relevance.uint8.npy",
    "relevant_counts.int32.npy",
    "rars_scores.float32.npy",
}
CANDIDATE_OUTPUT_FILES = (
    "query_vectors.float32.npy",
    "ann_rows.int64.npy",
    "candidate_doc_ids.int64.npy",
    "ann_scores.float32.npy",
    "candidate_doc_rows.int64.npy",
    "ann_residual_rows.int64.npy",
    "pca_scores.float32.npy",
    "candidate_residuals.float32.npy",
    "parent_role_indices.int64.npy",
)


def _validate_exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def _validate_clean_git_head(repo_root: Path, expected_commit: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if actual != expected_commit:
        raise ValueError(f"Git HEAD {actual} does not match {expected_commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if status.strip():
        raise ValueError("RARS-v3 candidate freeze requires a clean Git worktree")


def _validate_canonical_repository(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> dict[str, Any]:
    _validate_exact_commit(source_commit)
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    supplied = protocol_path.resolve(strict=True)
    if supplied != canonical:
        raise ValueError(f"Protocol must be the canonical repository path: {canonical}")
    _validate_clean_git_head(repo_root, source_commit)
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected RARS-v3 protocol")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_ORACLE_RUN":
        raise ValueError("RARS-v3 protocol is not in its preregistered frozen state")
    return protocol


def _verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing registered {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Registered {label} byte count changed: {path}")
    actual = sha256_file(path)
    if actual != str(record["sha256"]):
        raise ValueError(f"Registered {label} hash changed: {actual}")


def _verify_size_without_reading_payload(
    path: Path, record: dict[str, Any], label: str
) -> None:
    if not path.is_file():
        raise ValueError(f"Missing registered {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Registered {label} byte count changed: {path}")


def _verify_exact_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing registered {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Registered {label} hash changed: {actual}")
    return file_record(path)


def _load_query_identity(path: Path) -> tuple[list[str], np.ndarray]:
    payload = read_json(path)
    qids = [str(value) for value in payload["query_ids"]]
    rows = np.asarray(payload["query_rows"], dtype=np.int64)
    if len(qids) != len(rows):
        raise ValueError(f"Query identity length mismatch: {path}")
    if len(qids) != len(set(qids)):
        raise ValueError(f"Duplicate query IDs: {path}")
    if len(rows) != len(set(rows.tolist())) or np.any(rows < 0):
        raise ValueError(f"Duplicate or negative query rows: {path}")
    return qids, rows


def _newline_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _numeric_sorted_newline_sha256(values: list[str]) -> str:
    return _newline_sha256(sorted(values, key=lambda value: int(value)))


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
            f"{left_name}/{right_name} overlap: {qid_overlap} qids, {row_overlap} rows"
        )
    return {"qid_overlap": qid_overlap, "row_overlap": row_overlap}


def _role_identity(
    qids: list[str], rows: np.ndarray, indices: np.ndarray
) -> tuple[list[str], np.ndarray]:
    return [qids[int(index)] for index in indices], rows[indices]


def _validate_closed_identities(
    repo_root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, Any]]:
    lineage = protocol["parent_lineage"]
    split_audit_path = repo_root / CANONICAL_V2_2_SPLIT_AUDIT
    split_audit_record = _verify_exact_hash(
        split_audit_path,
        lineage["parent_v2_2_split_audit_sha256"],
        "parent v2.2 split audit",
    )
    historical = read_json(split_audit_path)
    specifications = {
        "v2_2_inner_validation": (
            repo_root / CANONICAL_CLOSED_INNER_VALIDATION,
            lineage["closed_inner_validation_query_manifest_sha256"],
            historical["roles"]["inner_validation"],
        ),
        "burned_outer": (
            repo_root / CANONICAL_OUTER_VALIDATION,
            lineage["outer_validation_split_sha256"],
            historical["burned_outer_validation"],
        ),
        "clean_test": (
            repo_root / CANONICAL_CLEAN_TEST,
            lineage["clean_test_split_sha256"],
            historical["clean_test"],
        ),
    }
    identities: dict[str, tuple[list[str], np.ndarray]] = {}
    records: dict[str, Any] = {"v2_2_split_audit": split_audit_record}
    for name, (path, expected_hash, registered_identity) in specifications.items():
        records[name] = _verify_exact_hash(path, expected_hash, name)
        qids, rows = _load_query_identity(path)
        if len(qids) != int(registered_identity["query_count"]):
            raise ValueError(f"Closed-role query count changed: {name}")
        if canonical_sha256(qids) != registered_identity["query_ids_sha256"]:
            raise ValueError(f"Closed-role query ID hash changed: {name}")
        if array_sha256(rows) != registered_identity["query_rows_sha256"]:
            raise ValueError(f"Closed-role query-row hash changed: {name}")
        identities[name] = (qids, rows)
    return identities, records


def _safe_manifest_filename(filename: str) -> str:
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Unsafe parent manifest filename: {filename!r}")
    return filename


def _load_and_verify_parent_bundle(
    bundle_dir: Path,
    protocol: dict[str, Any],
    *,
    verify_label_payloads: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[str], np.ndarray, dict[str, np.ndarray]]:
    lineage = protocol["parent_lineage"]
    manifest_path = bundle_dir / "v2_2_manifest.json"
    source_manifest_path = bundle_dir / "manifest.json"
    query_manifest_path = bundle_dir / "query_manifest.json"
    _verify_exact_hash(
        manifest_path,
        lineage["parent_inner_train_manifest_sha256"],
        "parent v2.2 manifest",
    )
    _verify_exact_hash(
        source_manifest_path,
        lineage["parent_inner_train_source_manifest_sha256"],
        "parent v2 source manifest",
    )
    _verify_exact_hash(
        query_manifest_path,
        lineage["parent_inner_train_query_manifest_sha256"],
        "parent query manifest",
    )
    manifest = read_json(manifest_path)
    source_manifest = read_json(source_manifest_path)
    if manifest.get("protocol_id") != lineage["parent_v2_2_protocol_id"]:
        raise ValueError("Parent v2.2 protocol ID changed")
    if manifest.get("source_commit") != lineage["parent_training_commit"]:
        raise ValueError("Parent v2.2 training commit changed")
    if manifest.get("role_id") != "inner_train" or source_manifest.get("role_id") != "inner_train":
        raise ValueError("Parent bundle is not frozen inner_train")
    if manifest.get("source_bundle_manifest_sha256") != lineage[
        "parent_inner_train_source_manifest_sha256"
    ]:
        raise ValueError("Parent source-manifest registration changed")
    files = manifest.get("files")
    source_files = source_manifest.get("files")
    if not isinstance(files, dict) or not isinstance(source_files, dict):
        raise ValueError("Parent manifests do not register file maps")
    if set(files) != set(source_files) or not PARENT_REQUIRED_FILES.issubset(files):
        raise ValueError("Parent v2.2/source manifest file sets changed")
    files_to_hash = (
        set(files)
        if verify_label_payloads
        else set(PARENT_QRELS_FREE_ARRAY_FILES)
    )
    for filename, record in files.items():
        safe_name = _safe_manifest_filename(str(filename))
        if record.get("bytes") != source_files[filename].get("bytes") or record.get(
            "sha256"
        ) != source_files[filename].get("sha256"):
            raise ValueError(f"Parent manifest registrations disagree: {filename}")
        if filename in files_to_hash:
            _verify_record(bundle_dir / safe_name, record, f"parent file {safe_name}")
        elif filename in {"candidate_relevance.uint8.npy", "relevant_counts.int32.npy"}:
            _verify_size_without_reading_payload(
                bundle_dir / safe_name,
                record,
                f"parent label file {safe_name}",
            )
    query_record = manifest.get("query_manifest", {})
    _verify_record(query_manifest_path, query_record, "parent query manifest")
    qids, rows = _load_query_identity(query_manifest_path)
    if len(qids) != int(protocol["data_policy"]["source_pool"]["query_count"]):
        raise ValueError("Parent v2.2 development query count changed")
    if len(qids) != int(manifest["query_count"]):
        raise ValueError("Parent manifest/query identity count mismatch")
    if canonical_sha256(qids) != manifest["query_ids_sha256"]:
        raise ValueError("Parent ordered query ID hash changed")
    if array_sha256(rows) != manifest["query_rows_sha256"]:
        raise ValueError("Parent query-row hash changed")

    # Label-derived arrays and the qrel-trained RARS scores are neither hashed
    # nor loaded by the qrels-free candidate builder. Their exact registrations
    # are already bound by the two exact parent-manifest hashes; the separate
    # role-label process verifies their payload bytes immediately before use.
    arrays = {
        filename: np.load(bundle_dir / filename, mmap_mode="r")
        for filename in PARENT_QRELS_FREE_ARRAY_FILES
    }
    query_count = len(qids)
    candidate_count = int(manifest["candidate_count"])
    matrix_shape = (query_count, candidate_count)
    if arrays["query_vectors.float32.npy"].shape[0] != query_count:
        raise ValueError("Parent query-vector row count changed")
    for filename in (
        "ann_rows.int64.npy",
        "ann_scores.float32.npy",
        "ann_residual_rows.int64.npy",
        "pca_scores.float32.npy",
    ):
        if arrays[filename].shape != matrix_shape:
            raise ValueError(f"Parent matrix shape changed: {filename}")
    candidate_rows = arrays["candidate_doc_rows.int64.npy"]
    residuals = arrays["candidate_residuals.float32.npy"]
    if candidate_rows.ndim != 1 or residuals.shape[0] != len(candidate_rows):
        raise ValueError("Parent residual union shape changed")
    lookup = np.asarray(arrays["ann_residual_rows.int64.npy"])
    if np.any(lookup < 0) or np.any(lookup >= len(candidate_rows)):
        raise ValueError("Parent residual lookup is out of range")
    ann_rows = np.asarray(arrays["ann_rows.int64.npy"])
    if not np.array_equal(np.asarray(candidate_rows)[lookup], ann_rows):
        raise ValueError("Parent residual lookup no longer maps to ANN rows")
    return manifest, source_manifest, qids, rows, arrays


def _prepare_empty_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("Refusing to reuse a non-empty RARS-v3 candidate output root")
    output_root.mkdir(parents=True, exist_ok=True)


def _freeze_role(
    *,
    role_id: str,
    qids: list[str],
    query_rows: np.ndarray,
    parent_indices: np.ndarray,
    parent_arrays: dict[str, np.ndarray],
    doc_ids: np.memmap,
    output_root: Path,
    manifest_lineage: dict[str, Any],
) -> dict[str, Any]:
    role_dir = output_root / role_id
    role_dir.mkdir(parents=True, exist_ok=False)
    parent_indices = np.asarray(parent_indices, dtype=np.int64)
    ann_rows = np.asarray(parent_arrays["ann_rows.int64.npy"][parent_indices], dtype=np.int64)
    if np.any(ann_rows < 0) or np.any(ann_rows >= len(doc_ids)):
        raise ValueError(f"Frozen candidate row is out of range for {role_id}")
    parent_lookup = np.asarray(
        parent_arrays["ann_residual_rows.int64.npy"][parent_indices], dtype=np.int64
    )
    selected_parent_local, inverse = np.unique(parent_lookup, return_inverse=True)
    remapped_lookup = inverse.reshape(parent_lookup.shape).astype(np.int64, copy=False)
    role_candidate_rows = np.asarray(
        parent_arrays["candidate_doc_rows.int64.npy"][selected_parent_local],
        dtype=np.int64,
    )
    if not np.array_equal(role_candidate_rows[remapped_lookup], ann_rows):
        raise ValueError(f"Residual remap failed for {role_id}")

    outputs: dict[str, np.ndarray] = {
        "query_vectors.float32.npy": np.asarray(
            parent_arrays["query_vectors.float32.npy"][parent_indices], dtype=np.float32
        ),
        "ann_rows.int64.npy": ann_rows,
        "candidate_doc_ids.int64.npy": np.asarray(doc_ids[ann_rows], dtype=np.int64),
        "ann_scores.float32.npy": np.asarray(
            parent_arrays["ann_scores.float32.npy"][parent_indices], dtype=np.float32
        ),
        "candidate_doc_rows.int64.npy": role_candidate_rows,
        "ann_residual_rows.int64.npy": remapped_lookup,
        "pca_scores.float32.npy": np.asarray(
            parent_arrays["pca_scores.float32.npy"][parent_indices], dtype=np.float32
        ),
        "candidate_residuals.float32.npy": np.asarray(
            parent_arrays["candidate_residuals.float32.npy"][selected_parent_local],
            dtype=np.float32,
        ),
        "parent_role_indices.int64.npy": parent_indices,
    }
    if tuple(outputs) != CANDIDATE_OUTPUT_FILES:
        raise AssertionError("Internal candidate output contract changed")
    output_paths: dict[str, Path] = {}
    for filename, value in outputs.items():
        path = role_dir / filename
        atomic_save(path, value)
        output_paths[filename] = path

    query_manifest_path = role_dir / "query_manifest.json"
    query_manifest: dict[str, Any] = {
        "role_id": role_id,
        "query_ids": qids,
        "query_rows": query_rows.tolist(),
        "parent_inner_train_indices": parent_indices.tolist(),
    }
    if role_id == DESIGN_ROLE_ID:
        query_manifest["diagnostic_fold_ids"] = design_fold_ids(qids).tolist()
    atomic_json(query_manifest_path, query_manifest)

    candidate_count = int(ann_rows.shape[1])
    manifest = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "role_id": role_id,
        "split_role": "design" if role_id == DESIGN_ROLE_ID else "audit",
        "evidence_status": "DEVELOPMENT_ONLY",
        "source_commit": manifest_lineage["source_commit"],
        "query_count": len(qids),
        "candidate_count": candidate_count,
        "unique_candidate_document_count": len(role_candidate_rows),
        "query_ids_sha256": canonical_sha256(qids),
        "query_rows_sha256": array_sha256(query_rows),
        "parent_role_indices_sha256": array_sha256(parent_indices),
        "split_audit_sha256": manifest_lineage["split_audit_sha256"],
        "builder_sha256": manifest_lineage["builder_sha256"],
        "core_sha256": manifest_lineage["core_sha256"],
        "protocol_sha256": manifest_lineage["protocol_sha256"],
        "parent_v2_2_manifest_sha256": manifest_lineage[
            "parent_v2_2_manifest_sha256"
        ],
        "parent_v2_source_manifest_sha256": manifest_lineage[
            "parent_v2_source_manifest_sha256"
        ],
        "parent_query_manifest_sha256": manifest_lineage[
            "parent_query_manifest_sha256"
        ],
        "doc_ids_sha256": manifest_lineage["doc_ids_sha256"],
        "query_manifest": file_record(query_manifest_path),
        "files": {name: file_record(path) for name, path in output_paths.items()},
        "label_artifacts": {
            "materialized_by_candidate_builder": False,
            "candidate_relevance_present_at_freeze": False,
            "relevant_counts_present_at_freeze": False,
            "role_labels_manifest_present_at_freeze": False,
        },
        "data_access": {
            "qrels_opened_or_parsed": False,
            "parent_label_values_loaded_or_sliced": False,
            "faiss_imported_or_search_performed": False,
            "pca_fit_or_score_recomputation_performed": False,
            "candidates_and_scores_inherited_from_frozen_parent": True,
            "future_method_holdout_candidate_arrays_created": False,
            "v2_2_inner_validation_values_used": False,
            "outer_relevance_values_used": False,
            "clean_test_relevance_values_used": False,
            "nq_relevance_values_used": False,
            "trec_relevance_values_used": False,
            "future_method_holdout_relevance_values_used": False,
        },
    }
    manifest_path = role_dir / "v3_candidate_manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        "query_count": len(qids),
        "candidate_count": candidate_count,
        "unique_candidate_document_count": len(role_candidate_rows),
        "candidate_manifest": file_record(manifest_path),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    protocol = _validate_canonical_repository(repo_root, args.protocol, args.source_commit)
    lineage = protocol["parent_lineage"]
    closed_identities, closed_records = _validate_closed_identities(repo_root, protocol)
    (
        parent_manifest,
        source_manifest,
        parent_qids,
        parent_rows,
        parent_arrays,
    ) = _load_and_verify_parent_bundle(args.parent_inner_train_bundle, protocol)

    role_indices = split_development_qids(parent_qids)
    identities = {
        role_id: _role_identity(parent_qids, parent_rows, indices)
        for role_id, indices in role_indices.items()
    }
    for role_id, (qids, _) in identities.items():
        registered = protocol["data_policy"]["roles"][role_id]
        if len(qids) != int(registered["query_count"]):
            raise ValueError(f"Registered query count changed for {role_id}")
        if _newline_sha256(qids) != registered["source_order_newline_qid_sha256"]:
            raise ValueError(f"Source-order qid hash changed for {role_id}")
        if _numeric_sorted_newline_sha256(qids) != registered[
            "numeric_sorted_newline_qid_sha256"
        ]:
            raise ValueError(f"Sorted qid hash changed for {role_id}")

    all_identities = {**identities, **closed_identities}
    overlap: dict[str, Any] = {}
    identity_names = list(all_identities)
    for left_index, left_name in enumerate(identity_names):
        left_qids, left_rows = all_identities[left_name]
        for right_name in identity_names[left_index + 1 :]:
            right_qids, right_rows = all_identities[right_name]
            overlap[f"{left_name}_vs_{right_name}"] = _assert_disjoint(
                left_name,
                left_qids,
                left_rows,
                right_name,
                right_qids,
                right_rows,
            )

    doc_ids_record = _verify_exact_hash(
        args.doc_ids, lineage["frozen_doc_ids_sha256"], "frozen document IDs"
    )
    if args.doc_ids.stat().st_size != int(args.n_docs) * np.dtype(np.int64).itemsize:
        raise ValueError("Frozen document-ID byte count does not match --n-docs")
    doc_ids = np.memmap(args.doc_ids, dtype=np.int64, mode="r", shape=(args.n_docs,))

    _prepare_empty_output_root(args.output_root)
    builder_sha256 = sha256_file(Path(__file__).resolve())
    core_sha256 = sha256_file(repo_root / "scripts/rars_v3_oracle_core.py")
    protocol_sha256 = sha256_file(args.protocol)
    started_path = args.output_root / "v3_oracle_bundle_build_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "V3_QRELS_FREE_CANDIDATE_FREEZE_STARTED",
            "source_commit": args.source_commit,
            "builder_sha256": builder_sha256,
            "core_sha256": core_sha256,
            "protocol": file_record(args.protocol),
            "parent_v2_2_manifest": file_record(
                args.parent_inner_train_bundle / "v2_2_manifest.json"
            ),
            "parent_v2_source_manifest": file_record(
                args.parent_inner_train_bundle / "manifest.json"
            ),
            "parent_query_manifest": file_record(
                args.parent_inner_train_bundle / "query_manifest.json"
            ),
            "doc_ids": doc_ids_record,
            "qrels_input_accepted": False,
            "parent_label_payload_bytes_read": False,
            "faiss_or_pca_recomputation_allowed": False,
        },
    )

    split_audit = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "source_commit": args.source_commit,
        "builder_sha256": builder_sha256,
        "core_sha256": core_sha256,
        "protocol_sha256": protocol_sha256,
        "parent_v2_2_manifest": file_record(
            args.parent_inner_train_bundle / "v2_2_manifest.json"
        ),
        "parent_v2_source_manifest": file_record(
            args.parent_inner_train_bundle / "manifest.json"
        ),
        "parent_query_manifest": file_record(
            args.parent_inner_train_bundle / "query_manifest.json"
        ),
        "closed_identity_inputs": closed_records,
        "roles": {
            role_id: {
                "query_count": len(qids),
                "query_ids_canonical_sha256": canonical_sha256(qids),
                "query_ids_source_order_newline_sha256": _newline_sha256(qids),
                "query_ids_numeric_sorted_newline_sha256": _numeric_sorted_newline_sha256(qids),
                "query_rows_sha256": array_sha256(rows),
                "parent_role_indices_sha256": array_sha256(role_indices[role_id]),
                "candidate_retrieval_performed": False,
                "labels_materialized": False,
            }
            for role_id, (qids, rows) in identities.items()
        },
        "pairwise_overlap": overlap,
        "qrels_opened_or_parsed": False,
        "parent_label_payload_bytes_read": False,
        "parent_label_values_loaded_or_sliced": False,
        "all_required_assertions_passed": True,
    }
    split_audit_path = args.output_root / "v3_oracle_split_audit.json"
    atomic_json(split_audit_path, split_audit)
    split_audit_sha256 = sha256_file(split_audit_path)

    future_qids, future_rows = identities[FUTURE_ROLE_ID]
    future_dir = args.output_root / FUTURE_ROLE_ID
    future_dir.mkdir(parents=True, exist_ok=False)
    future_query_manifest = future_dir / "query_manifest.json"
    atomic_json(
        future_query_manifest,
        {
            "role_id": FUTURE_ROLE_ID,
            "query_ids": future_qids,
            "query_rows": future_rows.tolist(),
            "parent_inner_train_indices": role_indices[FUTURE_ROLE_ID].tolist(),
            "candidate_arrays_created": False,
            "labels_materialized": False,
            "metrics_computed": False,
        },
    )
    future_identity_path = future_dir / "v3_identity_manifest.json"
    atomic_json(
        future_identity_path,
        {
            "schema_version": 2,
            "protocol_id": PROTOCOL_ID,
            "role_id": FUTURE_ROLE_ID,
            "evidence_status": "DEVELOPMENT_ONLY_V3_METHOD_HOLDOUT",
            "source_commit": args.source_commit,
            "query_count": len(future_qids),
            "query_ids_sha256": canonical_sha256(future_qids),
            "query_rows_sha256": array_sha256(future_rows),
            "parent_role_indices_sha256": array_sha256(role_indices[FUTURE_ROLE_ID]),
            "split_audit_sha256": split_audit_sha256,
            "query_manifest": file_record(future_query_manifest),
            "candidate_arrays_created": False,
            "labels_materialized": False,
            "metrics_computed": False,
        },
    )

    manifest_lineage = {
        "source_commit": args.source_commit,
        "split_audit_sha256": split_audit_sha256,
        "builder_sha256": builder_sha256,
        "core_sha256": core_sha256,
        "protocol_sha256": protocol_sha256,
        "parent_v2_2_manifest_sha256": lineage["parent_inner_train_manifest_sha256"],
        "parent_v2_source_manifest_sha256": lineage[
            "parent_inner_train_source_manifest_sha256"
        ],
        "parent_query_manifest_sha256": lineage[
            "parent_inner_train_query_manifest_sha256"
        ],
        "doc_ids_sha256": lineage["frozen_doc_ids_sha256"],
    }
    role_summaries = {}
    for role_id in (DESIGN_ROLE_ID, AUDIT_ROLE_ID):
        qids, rows = identities[role_id]
        role_summaries[role_id] = _freeze_role(
            role_id=role_id,
            qids=qids,
            query_rows=rows,
            parent_indices=role_indices[role_id],
            parent_arrays=parent_arrays,
            doc_ids=doc_ids,
            output_root=args.output_root,
            manifest_lineage=manifest_lineage,
        )

    summary = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "status": "V3_QRELS_FREE_CANDIDATE_BUNDLES_FROZEN",
        "source_commit": args.source_commit,
        "started": file_record(started_path),
        "split_audit": file_record(split_audit_path),
        "roles": role_summaries,
        "future_method_holdout": {
            "query_count": len(future_qids),
            "identity_manifest": file_record(future_identity_path),
            "candidate_arrays_created": False,
            "labels_materialized": False,
            "metrics_computed": False,
        },
        "parent_candidate_payloads_hash_verified": True,
        "qrels_opened_or_parsed": False,
        "parent_label_payload_bytes_read": False,
        "parent_label_values_loaded_or_sliced": False,
        "faiss_imported_or_search_performed": False,
        "pca_fit_or_score_recomputation_performed": False,
        "closed_role_outcomes_computed": False,
    }
    summary_path = args.output_root / "v3_oracle_bundle_freeze_summary.json"
    atomic_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-inner-train-bundle", required=True, type=Path)
    parser.add_argument("--doc-ids", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / CANONICAL_PROTOCOL,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--n-docs", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
