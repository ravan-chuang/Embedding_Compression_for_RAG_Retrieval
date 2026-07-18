#!/usr/bin/env python3
"""Run the frozen RARS-v3 matched-access oracle feasibility gate."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rars_v3_oracle_core import (
    ALLOWED_TIERS,
    AUDIT_ROLE_ID,
    DESIGN_ROLE_ID,
    PROTOCOL_ID,
    build_run_fingerprint,
    array_sha256,
    canonical_sha256,
    candidate_relevance_ceiling,
    compression_recovery_diagnostics,
    decide_oracle_gate,
    design_fold_ids,
    exact_accessed_byte_oracle,
    exact_residual_scores,
    file_record,
    fit_progressive_pca,
    gain_diagnostics,
    paired_bootstrap_mean_delta,
    progressive_tier_scores,
    read_json,
    recall_at_k_per_query,
    sha256_file,
    topk_membership,
    validate_bundle_manifest,
)


CANONICAL_PROTOCOL_RELATIVE = Path(
    "protocols/rars_v3_oracle_first_feasibility_v1.json"
)
CANONICAL_SOURCE_RELATIVES = {
    "protocol": CANONICAL_PROTOCOL_RELATIVE,
    "builder": Path("scripts/build_msmarco_rars_v3_oracle_bundles.py"),
    "label_materializer": Path("scripts/materialize_rars_v3_role_labels.py"),
    "core": Path("scripts/rars_v3_oracle_core.py"),
    "evaluator": Path("scripts/evaluate_rars_v3_oracle_first_feasibility.py"),
}
BASELINE_NAMES = (
    "base_boundary_top20_rank32_else0",
    "design_candidate_exposure_top20_rank32_else0",
    "frozen_pca_rank16_int8",
    "residual_norm_top20_rank32_else0",
    "uniform_progressive_rank16_int8",
)
RANDOM_SEEDS = tuple(range(10))
ROLE_LABEL_FILENAMES = (
    "candidate_relevance.uint8.npy",
    "relevant_counts.int32.npy",
    "v3_role_labels_started.json",
    "v3_role_labels_manifest.json",
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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        for chunk in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
            output_handle.write(chunk)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    temporary.replace(destination)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo_root, stderr=subprocess.STDOUT
    )


def _git_text(repo_root: Path, *arguments: str) -> str:
    return _git_bytes(repo_root, *arguments).decode("utf-8").strip()


def _validate_exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def validate_clean_git_head(repo_root: Path, expected_commit: str) -> None:
    actual = _git_text(repo_root, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ValueError(f"Git HEAD {actual} does not match {expected_commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if status.strip():
        raise ValueError("RARS-v3 oracle execution requires a clean Git worktree")


def validate_canonical_protocol_and_sources(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> dict[str, Any]:
    """Bind every executable input to its canonical path and source Git blob."""

    canonical_protocol = (repo_root / CANONICAL_PROTOCOL_RELATIVE).resolve()
    if protocol_path.resolve() != canonical_protocol:
        raise ValueError(
            "--protocol must be the canonical repository protocol path: "
            f"{canonical_protocol}"
        )
    records: dict[str, Any] = {}
    blob_records: dict[str, Any] = {}
    for name, relative in CANONICAL_SOURCE_RELATIVES.items():
        path = (repo_root / relative).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"Missing canonical {name} source: {path}")
        local_bytes = path.read_bytes()
        try:
            committed_bytes = _git_bytes(
                repo_root, "show", f"{source_commit}:{relative.as_posix()}"
            )
            blob_oid = _git_text(
                repo_root, "rev-parse", f"{source_commit}:{relative.as_posix()}"
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"Canonical {name} source is not present in {source_commit}"
            ) from error
        if local_bytes != committed_bytes:
            raise ValueError(
                f"Canonical {name} source differs from its registered Git blob"
            )
        records[f"{name}_sha256"] = _sha256_bytes(local_bytes)
        blob_records[name] = {
            "repository_path": relative.as_posix(),
            "sha256": _sha256_bytes(local_bytes),
            "git_blob_oid": blob_oid,
        }
    records["git_blob_records"] = blob_records
    return records


def _verify_manifest_files(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Bundle has no registered files: {bundle_dir}")
    for filename, record in files.items():
        path = bundle_dir / filename
        if not path.exists():
            raise ValueError(f"Missing bundle file: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Bundle byte count changed: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Bundle hash changed: {path}")


def _newline_sha256(values: list[str]) -> str:
    import hashlib

    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _numeric_sorted_newline_sha256(values: list[str]) -> str:
    return _newline_sha256(sorted(values, key=lambda value: int(value)))


def _validate_manifest_lineage(
    manifest: dict[str, Any],
    query_manifest: dict[str, Any],
    *,
    expected_role_id: str,
    protocol: dict[str, Any],
    source_commit: str,
    source_hashes: dict[str, Any],
    split_audit_sha256: str,
) -> None:
    validate_bundle_manifest(manifest, expected_role_id=expected_role_id)
    if manifest.get("source_commit") != source_commit:
        raise ValueError("Bundle source commit does not match the oracle run")
    exact_hashes = {
        "builder_sha256": source_hashes["builder_sha256"],
        "core_sha256": source_hashes["core_sha256"],
        "protocol_sha256": source_hashes["protocol_sha256"],
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
        "split_audit_sha256": split_audit_sha256,
    }
    for field, expected in exact_hashes.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Bundle lineage mismatch for {field}")
    registered = protocol["data_policy"]["roles"][expected_role_id]
    qids = [str(value) for value in query_manifest.get("query_ids", [])]
    rows = np.asarray(query_manifest.get("query_rows", []), dtype=np.int64)
    if len(qids) != int(registered["query_count"]):
        raise ValueError("Bundle role query count differs from the protocol")
    if manifest.get("query_count") != len(qids):
        raise ValueError("Bundle manifest/query-manifest counts disagree")
    if manifest.get("query_ids_sha256") != canonical_sha256(qids):
        raise ValueError("Bundle ordered query identity hash changed")
    if manifest.get("query_rows_sha256") != array_sha256(rows):
        raise ValueError("Bundle query-row hash changed")
    if _newline_sha256(qids) != registered["source_order_newline_qid_sha256"]:
        raise ValueError("Bundle source-order query hash differs from protocol")
    if (
        _numeric_sorted_newline_sha256(qids)
        != registered["numeric_sorted_newline_qid_sha256"]
    ):
        raise ValueError("Bundle sorted query hash differs from protocol")
    if expected_role_id == DESIGN_ROLE_ID:
        registered_folds = np.asarray(
            query_manifest.get("diagnostic_fold_ids", []), dtype=np.uint8
        )
        recomputed = design_fold_ids(qids)
        if not np.array_equal(registered_folds, recomputed):
            raise ValueError("Design diagnostic fold IDs changed")
        if np.bincount(recomputed, minlength=5).tolist() != protocol["data_policy"][
            "design_fold_counts"
        ]:
            raise ValueError("Design diagnostic fold counts changed")


def _require_dtype(name: str, value: np.ndarray, expected: np.dtype[Any]) -> None:
    if np.asarray(value).dtype != np.dtype(expected):
        raise ValueError(
            f"{name} must have dtype {np.dtype(expected)}, found {np.asarray(value).dtype}"
        )


def _validate_bundle_arrays(arrays: dict[str, Any]) -> None:
    exact_dtypes = {
        "queries": np.float32,
        "ann_rows": np.int64,
        "document_ids": np.int64,
        "ann_scores": np.float32,
        "candidate_doc_rows": np.int64,
        "residual_lookup": np.int64,
        "pca_scores": np.float32,
        "residuals": np.float32,
        "parent_role_indices": np.int64,
    }
    for name, dtype in exact_dtypes.items():
        _require_dtype(name, arrays[name], dtype)
    query_count, candidate_count = arrays["ann_scores"].shape
    if arrays["queries"].ndim != 2 or arrays["queries"].shape[0] != query_count:
        raise ValueError("Query matrix does not match bundle query count")
    for key in ("ann_rows", "document_ids", "residual_lookup", "pca_scores"):
        if arrays[key].shape != (query_count, candidate_count):
            raise ValueError(f"{key} does not match the candidate shape")
    if arrays["candidate_doc_rows"].ndim != 1:
        raise ValueError("Candidate document rows must be one-dimensional")
    if arrays["residuals"].shape != (
        len(arrays["candidate_doc_rows"]),
        arrays["queries"].shape[1],
    ):
        raise ValueError("Candidate residual rows do not match document rows")
    if arrays["parent_role_indices"].shape != (query_count,):
        raise ValueError("Parent role indices must have one entry per query")
    for key in ("queries", "ann_scores", "pca_scores", "residuals"):
        if not np.all(np.isfinite(np.asarray(arrays[key]))):
            raise ValueError(f"{key} contains non-finite values")
    if np.any(np.asarray(arrays["ann_rows"]) < 0):
        raise ValueError("ANN rows must be non-negative")
    candidate_rows = np.asarray(arrays["candidate_doc_rows"], dtype=np.int64)
    if np.any(candidate_rows < 0) or len(candidate_rows) != len(set(candidate_rows.tolist())):
        raise ValueError("Candidate document rows must be unique and non-negative")
    lookup = np.asarray(arrays["residual_lookup"], dtype=np.int64)
    if np.any(lookup < 0) or np.any(lookup >= len(candidate_rows)):
        raise ValueError("Residual lookup is outside candidate residual bounds")
    if not np.array_equal(candidate_rows[lookup], np.asarray(arrays["ann_rows"])):
        raise ValueError("Residual lookup does not reconstruct ANN document rows")
    docids = np.asarray(arrays["document_ids"], dtype=np.int64)
    for query_index, row in enumerate(docids):
        if len(row) != len(set(row.tolist())):
            raise ValueError(
                f"Candidate document IDs are not unique for query {query_index}"
            )
    parent_indices = np.asarray(arrays["parent_role_indices"], dtype=np.int64)
    if np.any(parent_indices < 0) or len(parent_indices) != len(
        set(parent_indices.tolist())
    ):
        raise ValueError("Parent role indices must be unique and non-negative")


def load_bundle(
    bundle_dir: Path,
    *,
    expected_role_id: str,
    protocol: dict[str, Any],
    source_commit: str,
    source_hashes: dict[str, Any],
    split_audit_sha256: str,
) -> dict[str, Any]:
    manifest_path = bundle_dir / "v3_candidate_manifest.json"
    manifest = read_json(manifest_path)
    _verify_manifest_files(bundle_dir, manifest)
    mapping = {
        "query_vectors.float32.npy": "queries",
        "ann_rows.int64.npy": "ann_rows",
        "candidate_doc_ids.int64.npy": "document_ids",
        "ann_scores.float32.npy": "ann_scores",
        "candidate_doc_rows.int64.npy": "candidate_doc_rows",
        "ann_residual_rows.int64.npy": "residual_lookup",
        "pca_scores.float32.npy": "pca_scores",
        "parent_role_indices.int64.npy": "parent_role_indices",
    }
    arrays: dict[str, Any] = {
        target: np.load(bundle_dir / filename, mmap_mode="r")
        for filename, target in mapping.items()
    }
    arrays["residuals"] = np.load(
        bundle_dir / "candidate_residuals.float32.npy", mmap_mode="r"
    )
    query_manifest_path = bundle_dir / "query_manifest.json"
    query_manifest_record = manifest.get("query_manifest")
    if not isinstance(query_manifest_record, dict):
        raise ValueError("Candidate manifest lacks query-manifest registration")
    _record_matches(query_manifest_path, query_manifest_record, "query manifest")
    query_manifest = read_json(query_manifest_path)
    _validate_manifest_lineage(
        manifest,
        query_manifest,
        expected_role_id=expected_role_id,
        protocol=protocol,
        source_commit=source_commit,
        source_hashes=source_hashes,
        split_audit_sha256=split_audit_sha256,
    )
    arrays["query_manifest"] = query_manifest
    arrays["manifest"] = manifest
    arrays["manifest_path"] = manifest_path
    query_count, candidate_count = arrays["ann_scores"].shape
    if query_count != int(manifest["query_count"]):
        raise ValueError("Bundle query count disagrees with manifest")
    if candidate_count != int(manifest["candidate_count"]):
        raise ValueError("Bundle candidate count disagrees with manifest")
    _validate_bundle_arrays(arrays)
    parent_indices = np.asarray(arrays["parent_role_indices"], dtype=np.int64)
    if manifest.get("parent_role_indices_sha256") != array_sha256(parent_indices):
        raise ValueError("Candidate parent-role index hash changed")
    query_parent_indices = np.asarray(
        query_manifest.get("parent_inner_train_indices", []), dtype=np.int64
    )
    if not np.array_equal(parent_indices, query_parent_indices):
        raise ValueError("Candidate/query manifests disagree on parent-role indices")
    return arrays


def _record_matches(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.exists() or not path.is_file():
        raise ValueError(f"Missing registered {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Registered {label} hash changed")


def load_role_labels(
    bundle: dict[str, Any],
    label_manifest_path: Path,
    *,
    expected_role_id: str,
    protocol: dict[str, Any],
    source_commit: str,
    source_hashes: dict[str, Any],
    design_freeze_path: Path | None = None,
) -> dict[str, Any]:
    """Attach only separately materialized, hash-bound candidate labels."""

    manifest = read_json(label_manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Role-label artifact uses an unexpected protocol")
    if manifest.get("role_id") != expected_role_id:
        raise ValueError("Role-label artifact role does not match the candidate bundle")
    if manifest.get("source_commit") != source_commit:
        raise ValueError("Role-label source commit changed")
    candidate_record = manifest.get("candidate_manifest") or manifest.get(
        "candidate_bundle_manifest"
    )
    if not isinstance(candidate_record, dict):
        raise ValueError("Role-label manifest lacks candidate-manifest lineage")
    _record_matches(bundle["manifest_path"], candidate_record, "candidate manifest")
    if manifest.get("materializer_sha256") != source_hashes[
        "label_materializer_sha256"
    ]:
        raise ValueError("Role-label materializer source hash changed")
    if manifest.get("protocol_sha256") != source_hashes["protocol_sha256"]:
        raise ValueError("Role-label protocol hash changed")
    if manifest.get("parent_v2_2_manifest_sha256") != protocol[
        "parent_lineage"
    ]["parent_inner_train_manifest_sha256"]:
        raise ValueError("Role-label parent manifest lineage changed")
    expected_parent_indices = array_sha256(
        np.asarray(bundle["parent_role_indices"], dtype=np.int64)
    )
    if manifest.get("parent_role_indices_sha256") != expected_parent_indices:
        raise ValueError("Role-label parent-row selection changed")
    expected_query_count = int(bundle["manifest"]["query_count"])
    expected_candidate_count = int(bundle["manifest"]["candidate_count"])
    if int(manifest.get("query_count", -1)) != expected_query_count:
        raise ValueError("Role-label query count differs from candidate bundle")
    if int(manifest.get("candidate_count", -1)) != expected_candidate_count:
        raise ValueError("Role-label candidate count differs from candidate bundle")
    expected_qids = [
        str(value) for value in bundle["query_manifest"].get("query_ids", [])
    ]
    if manifest.get("query_ids_sha256") != canonical_sha256(expected_qids):
        raise ValueError("Role-label query identity hash changed")
    if expected_role_id == AUDIT_ROLE_ID:
        if design_freeze_path is None:
            raise ValueError("Audit labels require a durable design freeze")
        freeze_record = manifest.get("design_freeze")
        if not isinstance(freeze_record, dict):
            raise ValueError("Audit labels lack design-freeze lineage")
        _record_matches(design_freeze_path, freeze_record, "design freeze")
        audit_release = manifest.get("audit_release", {})
        if not (
            manifest.get("materialized_after_design_freeze") is True
            or audit_release.get("design_freeze_verified") is True
        ):
            raise ValueError("Audit labels were not certified post-freeze")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Role-label manifest lacks registered arrays")
    expected_files = {
        "candidate_relevance.uint8.npy": np.uint8,
        "relevant_counts.int32.npy": np.int32,
    }
    arrays: dict[str, np.ndarray] = {}
    for filename, dtype in expected_files.items():
        record = files.get(filename)
        if not isinstance(record, dict):
            raise ValueError(f"Role-label manifest lacks {filename}")
        path = label_manifest_path.parent / filename
        _record_matches(path, record, filename)
        value = np.load(path, mmap_mode="r")
        _require_dtype(filename, value, dtype)
        arrays[filename] = value
    labels = arrays["candidate_relevance.uint8.npy"]
    counts = arrays["relevant_counts.int32.npy"]
    if labels.shape != bundle["ann_scores"].shape:
        raise ValueError("Candidate relevance shape differs from candidate arrays")
    if counts.shape != (len(labels),) or np.any(counts <= 0):
        raise ValueError("Relevant counts must be positive with one value per query")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("Candidate relevance must be binary")
    if np.any(np.sum(labels, axis=1) > counts):
        raise ValueError("Candidate positives exceed registered total positives")
    bundle["labels"] = labels
    bundle["relevant_counts"] = counts
    bundle["label_manifest"] = manifest
    bundle["label_manifest_path"] = label_manifest_path
    return manifest


def capture_environment(protocol: dict[str, Any]) -> tuple[dict[str, Any], str]:
    contract = protocol.get("execution_environment_contract")
    if not isinstance(contract, dict):
        raise ValueError("Protocol lacks execution_environment_contract")
    python_version = ".".join(str(value) for value in sys.version_info[:3])
    if python_version != str(contract.get("python_version")):
        raise ValueError(
            f"Python {python_version} differs from protocol "
            f"{contract.get('python_version')}"
        )
    if np.__version__ != str(contract.get("numpy_version")):
        raise ValueError(
            f"NumPy {np.__version__} differs from protocol "
            f"{contract.get('numpy_version')}"
        )
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    numpy_config = stream.getvalue()
    if contract.get("blas_configuration_must_be_recorded") is True and not numpy_config:
        raise ValueError("NumPy/BLAS configuration could not be recorded")
    environment = {
        "contract": contract,
        "python_version": python_version,
        "python_full": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "numpy_module": str(Path(np.__file__).resolve()),
        "numpy_config_sha256": _sha256_bytes(numpy_config.encode("utf-8")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "thread_environment": {
            name: os.environ.get(name, "UNSET")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    return environment, numpy_config


def fit_uncentered_progressive_svd(
    residuals: np.ndarray,
    *,
    rank: int,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the protocol's uncentered rank-32 SVD and retain its spectrum."""

    fitted = fit_progressive_pca(
        residuals, rank=rank, max_samples=max_samples, seed=seed
    )
    if len(fitted) != 4:
        raise ValueError("Unexpected fit_progressive_pca return contract")
    basis, scales, sample_rows, singular_values = fitted
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.ndim != 1 or len(singular_values) < rank:
        raise ValueError("Uncentered SVD did not return a complete spectrum")
    if not np.all(np.isfinite(singular_values)) or np.any(singular_values < 0):
        raise ValueError("Uncentered SVD spectrum is invalid")
    if np.any(singular_values[1:] > singular_values[:-1] + 1e-10):
        raise ValueError("Uncentered SVD singular values are not descending")
    gaps = singular_values[:-1] - singular_values[1:]
    return (
        np.asarray(basis, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
        np.asarray(sample_rows, dtype=np.int64),
        singular_values,
        gaps.astype(np.float64),
    )


def _registered_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe registered output path: {name}")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"Registered output escapes output directory: {name}")
    return path


def verify_registered_outputs(
    output_dir: Path, registered: dict[str, Any]
) -> None:
    if not isinstance(registered, dict) or not registered:
        raise ValueError("No registered outputs were provided")
    for name, record in registered.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid output record for {name}")
        path = _registered_path(output_dir, name)
        _record_matches(path, record, f"registered output {name}")


def verify_design_freeze(
    output_dir: Path,
    *,
    protocol: dict[str, Any],
    source_commit: str,
    source_hashes: dict[str, Any],
) -> dict[str, Any]:
    path = output_dir / "design_freeze.json"
    freeze = read_json(path)
    if freeze.get("status") != "DESIGN_ARTIFACTS_FROZEN_BEFORE_AUDIT_LOAD":
        raise ValueError("Design freeze status is invalid")
    if freeze.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Design freeze protocol changed")
    if freeze.get("source_commit") != source_commit:
        raise ValueError("Design freeze source commit changed")
    if freeze.get("audit_bundle_loaded_before_this_freeze") is not False:
        raise ValueError("Design freeze does not certify audit isolation")
    if freeze.get("audit_role_labels_materialized_before_this_freeze") is not False:
        raise ValueError("Audit labels were materialized before design freeze")
    frozen_hashes = freeze.get("source_hashes")
    if frozen_hashes != source_hashes:
        raise ValueError("Design-freeze source hashes differ from current Git blobs")
    verify_registered_outputs(output_dir, freeze.get("registered_outputs", {}))
    fingerprint_payload = freeze.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise ValueError("Design freeze lacks its fingerprint payload")
    if build_run_fingerprint(fingerprint_payload) != freeze.get("run_fingerprint"):
        raise ValueError("Design run fingerprint is not canonical")
    expected_contracts = {
        "progressive_representation": protocol["progressive_representation"],
        "eligible_baselines": protocol["registered_matched_baselines"],
        "matched_access_oracle": protocol["matched_access_oracle"],
        "access_gate": protocol["access_gate"],
        "bootstrap": protocol["metric_contract"]["bootstrap"],
        "budgets_bytes_per_query": protocol["matched_access_oracle"][
            "budget_curve_bytes_per_query"
        ],
    }
    if freeze.get("contracts") != expected_contracts:
        raise ValueError("Design-frozen protocol contracts changed")
    selection = read_json(output_dir / "design_primary_comparator.json")
    if freeze.get("selected_primary_comparator") != selection.get("selected"):
        raise ValueError("Design-freeze comparator differs from registered selection")
    fold_gains = selection.get("design_fold_gains")
    if (
        not isinstance(fold_gains, list)
        or len(fold_gains) != 5
        or not np.all(np.isfinite(np.asarray(fold_gains, dtype=np.float64)))
    ):
        raise ValueError("Registered design-fold gains are invalid")
    if freeze.get("design_fold_gains") != fold_gains:
        raise ValueError("Design-freeze fold gains differ from registered selection")
    environment_artifact = read_json(output_dir / "execution_environment.json")
    numpy_config_record = environment_artifact.pop("numpy_config", None)
    if not isinstance(numpy_config_record, dict):
        raise ValueError("Execution environment lacks NumPy-config registration")
    _record_matches(
        output_dir / "numpy_config.txt", numpy_config_record, "NumPy configuration"
    )
    if freeze.get("execution_environment") != environment_artifact:
        raise ValueError("Design-freeze execution environment changed")
    if freeze.get("design_bundle_manifest") != fingerprint_payload.get(
        "design_candidate_manifest"
    ):
        raise ValueError("Design candidate registration changed after fingerprinting")
    if freeze.get("design_role_labels_manifest") != fingerprint_payload.get(
        "design_role_labels_manifest"
    ):
        raise ValueError("Design label registration changed after fingerprinting")
    if freeze.get(
        "audit_bundle_manifest_registered_but_arrays_unloaded"
    ) != fingerprint_payload.get("audit_candidate_manifest_registration"):
        raise ValueError("Audit candidate registration changed after fingerprinting")
    if freeze.get("future_method_holdout_accessed") is not False:
        raise ValueError("Design freeze does not certify future-holdout isolation")
    return freeze


def _lineage_files(bundle_root: Path) -> dict[str, Path]:
    return {
        "bundle_build_started": bundle_root / "v3_oracle_bundle_build_started.json",
        "split_audit": bundle_root / "v3_oracle_split_audit.json",
        "future_identity": (
            bundle_root
            / "future_method_holdout"
            / "v3_identity_manifest.json"
        ),
        "bundle_freeze_summary": (
            bundle_root / "v3_oracle_bundle_freeze_summary.json"
        ),
    }


def validate_bundle_root_lineage(
    bundle_root: Path,
    *,
    design_manifest_path: Path | None,
    audit_manifest_path: Path | None,
    protocol: dict[str, Any],
    source_commit: str,
    source_hashes: dict[str, Any],
) -> dict[str, Any]:
    paths = _lineage_files(bundle_root)
    for name, path in paths.items():
        if not path.exists():
            raise ValueError(f"Missing {name} lineage artifact: {path}")
    started = read_json(paths["bundle_build_started"])
    split = read_json(paths["split_audit"])
    future = read_json(paths["future_identity"])
    summary = read_json(paths["bundle_freeze_summary"])
    for label, payload in (
        ("bundle build", started),
        ("split audit", split),
        ("future identity", future),
        ("bundle freeze", summary),
    ):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{label} protocol lineage changed")
        if payload.get("source_commit") != source_commit:
            raise ValueError(f"{label} source-commit lineage changed")
    if summary.get("status") != "V3_QRELS_FREE_CANDIDATE_BUNDLES_FROZEN":
        raise ValueError("Candidate bundle freeze summary status is invalid")
    required_false_flags = (
        "qrels_opened_or_parsed",
        "parent_label_payload_bytes_read",
        "parent_label_values_loaded_or_sliced",
        "faiss_imported_or_search_performed",
        "pca_fit_or_score_recomputation_performed",
        "closed_role_outcomes_computed",
    )
    for field in required_false_flags:
        if summary.get(field) is not False:
            raise ValueError(f"Candidate freeze does not certify {field}=false")
    if summary.get("parent_candidate_payloads_hash_verified") is not True:
        raise ValueError("Candidate payload hashes were not fully verified")
    if started.get("qrels_input_accepted") is not False:
        raise ValueError("Candidate-freeze start marker accepted a qrels input")
    if started.get("parent_label_payload_bytes_read") is not False:
        raise ValueError("Candidate-freeze start marker permits parent label reads")
    for field in (
        "qrels_opened_or_parsed",
        "parent_label_payload_bytes_read",
        "parent_label_values_loaded_or_sliced",
    ):
        if split.get(field) is not False:
            raise ValueError(f"Split audit does not certify {field}=false")
    if future.get("role_id") != "future_method_holdout":
        raise ValueError("Future identity role changed")
    for field in ("candidate_arrays_created", "labels_materialized", "metrics_computed"):
        if future.get(field) is not False:
            raise ValueError(f"Future holdout access flag is not false: {field}")
    future_summary = summary.get("future_method_holdout")
    if not isinstance(future_summary, dict):
        raise ValueError("Bundle freeze summary lacks future holdout registration")
    for field in ("candidate_arrays_created", "labels_materialized", "metrics_computed"):
        if future_summary.get(field) is not False:
            raise ValueError(f"Future summary access flag is not false: {field}")
    split_record = summary.get("split_audit")
    future_record = future_summary.get("identity_manifest")
    if not isinstance(split_record, dict):
        raise ValueError("Bundle freeze summary lacks split-audit registration")
    if not isinstance(future_record, dict):
        raise ValueError("Bundle freeze summary lacks future identity registration")
    _record_matches(paths["split_audit"], split_record, "split audit")
    _record_matches(paths["future_identity"], future_record, "future identity")
    roles = summary.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Bundle freeze summary lacks role registrations")
    role_paths = {
        DESIGN_ROLE_ID: design_manifest_path,
        AUDIT_ROLE_ID: audit_manifest_path,
    }
    registrations: dict[str, dict[str, Any]] = {}
    for role_id, path in role_paths.items():
        role = roles.get(role_id)
        if not isinstance(role, dict):
            raise ValueError(f"Bundle freeze summary lacks {role_id}")
        record = role.get("candidate_manifest") or role.get("manifest")
        if not isinstance(record, dict):
            raise ValueError(f"Bundle freeze summary lacks {role_id} manifest record")
        registrations[role_id] = record
        if path is not None:
            _record_matches(path, record, f"{role_id} candidate manifest")
    if split.get("all_required_assertions_passed") is not True:
        raise ValueError("Split audit did not pass every assertion")

    # Recompute the three v3 role identities rather than trusting a self-consistent
    # split summary.  The future role is identity-only; its candidate arrays remain
    # forbidden and absent.
    role_query_manifests = {
        DESIGN_ROLE_ID: bundle_root / DESIGN_ROLE_ID / "query_manifest.json",
        AUDIT_ROLE_ID: bundle_root / AUDIT_ROLE_ID / "query_manifest.json",
        "future_method_holdout": (
            bundle_root / "future_method_holdout" / "query_manifest.json"
        ),
    }
    role_identities: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for role_id, query_path in role_query_manifests.items():
        query_payload = read_json(query_path)
        qids = [str(value) for value in query_payload.get("query_ids", [])]
        rows = np.asarray(query_payload.get("query_rows", []), dtype=np.int64)
        parent_indices = np.asarray(
            query_payload.get("parent_inner_train_indices", []), dtype=np.int64
        )
        registered = protocol["data_policy"]["roles"][role_id]
        if len(qids) != int(registered["query_count"]):
            raise ValueError(f"{role_id} query count differs from protocol")
        if rows.shape != (len(qids),) or parent_indices.shape != (len(qids),):
            raise ValueError(f"{role_id} identity arrays have invalid shapes")
        if len(set(qids)) != len(qids):
            raise ValueError(f"{role_id} contains duplicate query IDs")
        if (
            np.any(rows < 0)
            or np.any(parent_indices < 0)
            or len(set(rows.tolist())) != len(rows)
            or len(set(parent_indices.tolist())) != len(parent_indices)
        ):
            raise ValueError(f"{role_id} identity rows must be unique and non-negative")
        if _newline_sha256(qids) != registered["source_order_newline_qid_sha256"]:
            raise ValueError(f"{role_id} source-order query hash changed")
        if (
            _numeric_sorted_newline_sha256(qids)
            != registered["numeric_sorted_newline_qid_sha256"]
        ):
            raise ValueError(f"{role_id} sorted query hash changed")
        role_identities[role_id] = (qids, rows, parent_indices)
        split_role = split.get("roles", {}).get(role_id)
        expected_split_role = {
            "query_count": len(qids),
            "query_ids_canonical_sha256": canonical_sha256(qids),
            "query_ids_source_order_newline_sha256": _newline_sha256(qids),
            "query_ids_numeric_sorted_newline_sha256": (
                _numeric_sorted_newline_sha256(qids)
            ),
            "query_rows_sha256": array_sha256(rows),
            "parent_role_indices_sha256": array_sha256(parent_indices),
            "candidate_retrieval_performed": False,
            "labels_materialized": False,
        }
        if split_role != expected_split_role:
            raise ValueError(f"Split-audit identity changed for {role_id}")
    future_qids, future_rows, future_parent_indices = role_identities[
        "future_method_holdout"
    ]
    if int(future.get("query_count", -1)) != len(future_qids):
        raise ValueError("Future identity query count changed")
    if future.get("query_ids_sha256") != canonical_sha256(future_qids):
        raise ValueError("Future identity query hash changed")
    if future.get("query_rows_sha256") != array_sha256(future_rows):
        raise ValueError("Future identity row hash changed")
    if future.get("parent_role_indices_sha256") != array_sha256(
        future_parent_indices
    ):
        raise ValueError("Future identity parent-index hash changed")
    future_query_record = future.get("query_manifest")
    if not isinstance(future_query_record, dict):
        raise ValueError("Future identity lacks query-manifest registration")
    _record_matches(
        role_query_manifests["future_method_holdout"],
        future_query_record,
        "future query manifest",
    )
    future_query_payload = read_json(role_query_manifests["future_method_holdout"])
    for field in ("candidate_arrays_created", "labels_materialized", "metrics_computed"):
        if future_query_payload.get(field) is not False:
            raise ValueError(f"Future query-manifest access flag is not false: {field}")
    future_dir = bundle_root / "future_method_holdout"
    expected_future_files = {"query_manifest.json", "v3_identity_manifest.json"}
    actual_future_files = {path.name for path in future_dir.iterdir()}
    if actual_future_files != expected_future_files or any(
        not path.is_file() for path in future_dir.iterdir()
    ):
        raise ValueError("Future holdout directory contains unexpected artifacts")
    role_names = list(role_identities)
    for left_index, left_name in enumerate(role_names):
        left_qids, left_rows, left_parent = role_identities[left_name]
        for right_name in role_names[left_index + 1 :]:
            right_qids, right_rows, right_parent = role_identities[right_name]
            if set(left_qids) & set(right_qids):
                raise ValueError(f"{left_name}/{right_name} query-ID overlap")
            if set(left_rows.tolist()) & set(right_rows.tolist()):
                raise ValueError(f"{left_name}/{right_name} query-row overlap")
            if set(left_parent.tolist()) & set(right_parent.tolist()):
                raise ValueError(f"{left_name}/{right_name} parent-index overlap")
    pairwise = split.get("pairwise_overlap")
    if not isinstance(pairwise, dict) or not pairwise:
        raise ValueError("Split audit lacks pairwise-overlap registrations")
    all_identity_names = (
        DESIGN_ROLE_ID,
        AUDIT_ROLE_ID,
        "future_method_holdout",
        "v2_2_inner_validation",
        "burned_outer",
        "clean_test",
    )
    expected_pair_names = {
        f"{left}_vs_{right}"
        for left, right in itertools.combinations(all_identity_names, 2)
    }
    if set(pairwise) != expected_pair_names:
        raise ValueError("Split audit does not register the exact 15 role pairs")
    for name, values in pairwise.items():
        if not isinstance(values, dict):
            raise ValueError(f"Invalid pairwise-overlap record: {name}")
        if int(values.get("qid_overlap", -1)) != 0 or int(
            values.get("row_overlap", -1)
        ) != 0:
            raise ValueError(f"Split overlap is nonzero: {name}")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "records": {name: file_record(path) for name, path in paths.items()},
        "candidate_manifest_registrations": registrations,
        "split_audit_sha256": sha256_file(paths["split_audit"]),
        "future_identity_sha256": sha256_file(paths["future_identity"]),
        "bundle_freeze_summary_sha256": sha256_file(
            paths["bundle_freeze_summary"]
        ),
    }


def _tier_index(tier: int) -> int:
    return ALLOWED_TIERS.index(tier)


def _select_top_by_priority(
    priority: np.ndarray,
    document_ids: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    values = np.asarray(priority, dtype=np.float64)
    docids = np.asarray(document_ids, dtype=np.int64)
    if values.shape != docids.shape or values.ndim != 2:
        raise ValueError("Priority and document IDs must be matching matrices")
    if not 0 < count <= values.shape[1]:
        raise ValueError("Invalid policy selection count")
    selected = np.zeros(values.shape, dtype=bool)
    for query_index in range(len(values)):
        order = np.lexsort((docids[query_index], -values[query_index]))
        selected[query_index, order[:count]] = True
    return selected


def binary_rank32_policy_scores(
    tier_scores: np.ndarray,
    document_ids: np.ndarray,
    priority: np.ndarray,
    *,
    top_b: int,
    selected_count: int,
) -> np.ndarray:
    scores = np.asarray(tier_scores)
    result = np.asarray(scores[:, _tier_index(0), :], dtype=np.float32).copy()
    chosen = _select_top_by_priority(
        np.asarray(priority)[:, :top_b],
        np.asarray(document_ids)[:, :top_b],
        count=selected_count,
    )
    rank32 = scores[:, _tier_index(32), :top_b]
    result[:, :top_b][chosen] = rank32[chosen]
    return result


def _residual_norm_priority(bundle: dict[str, Any], *, top_b: int) -> np.ndarray:
    lookup = np.asarray(bundle["residual_lookup"][:, :top_b], dtype=np.int64)
    residual = np.asarray(bundle["residuals"][lookup], dtype=np.float32)
    return np.linalg.norm(residual, axis=2).astype(np.float32)


def _boundary_priority(bundle: dict[str, Any], *, final_k: int, top_b: int) -> np.ndarray:
    scores = np.asarray(bundle["ann_scores"], dtype=np.float32)
    document_ids = np.asarray(bundle["document_ids"], dtype=np.int64)
    boundary = np.empty(len(scores), dtype=np.float32)
    for query_index in range(len(scores)):
        order = np.lexsort((document_ids[query_index], -scores[query_index]))
        boundary[query_index] = scores[query_index, order[final_k - 1]]
    return -np.abs(scores[:, :top_b] - boundary[:, None])


def _exposure_counts(document_ids: np.ndarray, *, top_b: int) -> dict[int, int]:
    unique, counts = np.unique(
        np.asarray(document_ids[:, :top_b], dtype=np.int64), return_counts=True
    )
    return {
        int(document_id): int(count)
        for document_id, count in zip(unique, counts, strict=True)
    }


def _exposure_priority(
    document_ids: np.ndarray, exposure: dict[int, int], *, top_b: int
) -> np.ndarray:
    selected = np.asarray(document_ids[:, :top_b], dtype=np.int64)
    return np.asarray(
        [[exposure.get(int(value), 0) for value in row] for row in selected],
        dtype=np.float32,
    )


def build_baselines(
    bundle: dict[str, Any],
    tier_scores: np.ndarray,
    *,
    exposure: dict[int, int],
    final_k: int,
    top_b: int,
) -> dict[str, np.ndarray]:
    document_ids = np.asarray(bundle["document_ids"])
    residual_priority = _residual_norm_priority(bundle, top_b=top_b)
    boundary_priority = _boundary_priority(
        bundle, final_k=final_k, top_b=top_b
    )
    exposure_priority = _exposure_priority(document_ids, exposure, top_b=top_b)
    return {
        "frozen_pca_rank16_int8": np.asarray(bundle["pca_scores"], dtype=np.float32),
        "uniform_progressive_rank16_int8": np.asarray(
            tier_scores[:, _tier_index(16), :], dtype=np.float32
        ),
        "residual_norm_top20_rank32_else0": binary_rank32_policy_scores(
            tier_scores,
            document_ids,
            residual_priority,
            top_b=top_b,
            selected_count=top_b // 2,
        ),
        "base_boundary_top20_rank32_else0": binary_rank32_policy_scores(
            tier_scores,
            document_ids,
            boundary_priority,
            top_b=top_b,
            selected_count=top_b // 2,
        ),
        "design_candidate_exposure_top20_rank32_else0": binary_rank32_policy_scores(
            tier_scores,
            document_ids,
            exposure_priority,
            top_b=top_b,
            selected_count=top_b // 2,
        ),
    }


def baseline_recalls(
    baselines: dict[str, np.ndarray], bundle: dict[str, Any], *, final_k: int
) -> dict[str, np.ndarray]:
    return {
        name: recall_at_k_per_query(
            scores,
            bundle["document_ids"],
            bundle["labels"],
            bundle["relevant_counts"],
            k=final_k,
        )
        for name, scores in baselines.items()
    }


def select_primary_baseline(
    recalls: dict[str, np.ndarray], *, accessed_bytes: int
) -> dict[str, Any]:
    if set(recalls) != set(BASELINE_NAMES):
        raise ValueError("Registered matched-baseline set changed")
    rows = [
        {
            "method": name,
            "recall_at_10": float(np.mean(values)),
            "accessed_code_bytes_per_query": accessed_bytes,
        }
        for name, values in recalls.items()
    ]
    rows.sort(
        key=lambda row: (
            -row["recall_at_10"],
            row["accessed_code_bytes_per_query"],
            row["method"],
        )
    )
    return {"selected": rows[0], "registered_results": rows}


def random_baseline_summary(
    bundle: dict[str, Any],
    tier_scores: np.ndarray,
    *,
    final_k: int,
    top_b: int,
) -> dict[str, Any]:
    rows = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        priority = rng.random((len(bundle["ann_scores"]), top_b))
        scores = binary_rank32_policy_scores(
            tier_scores,
            bundle["document_ids"],
            priority,
            top_b=top_b,
            selected_count=top_b // 2,
        )
        recall = recall_at_k_per_query(
            scores,
            bundle["document_ids"],
            bundle["labels"],
            bundle["relevant_counts"],
            k=final_k,
        )
        rows.append({"seed": seed, "recall_at_10": float(np.mean(recall))})
    values = np.asarray([row["recall_at_10"] for row in rows])
    return {
        "policy": "random_top20_rank32_else0",
        "seeds": rows,
        "mean_recall_at_10": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
    }


def membership_event_diagnostics(
    base_scores: np.ndarray,
    exact_scores: np.ndarray,
    oracle_membership: np.ndarray,
    document_ids: np.ndarray,
    labels: np.ndarray,
    *,
    final_k: int,
) -> dict[str, float | int]:
    base = topk_membership(base_scores, document_ids, k=final_k)
    exact = topk_membership(exact_scores, document_ids, k=final_k)
    oracle = np.asarray(oracle_membership, dtype=bool)
    relevance = np.asarray(labels) > 0
    relevant_drops = exact & ~base & relevance
    intrusions = base & ~exact & ~relevance
    base_relevant = base & relevance
    drop_count = int(np.sum(relevant_drops))
    intrusion_count = int(np.sum(intrusions))
    base_relevant_count = int(np.sum(base_relevant))
    return {
        "compression_relevant_drop_events": drop_count,
        "recovered_relevant_drop_events": int(np.sum(relevant_drops & oracle)),
        "relevant_drop_recovery_rate": (
            0.0 if drop_count == 0 else float(np.sum(relevant_drops & oracle)) / drop_count
        ),
        "compression_nonrelevant_intrusions": intrusion_count,
        "removed_nonrelevant_intrusions": int(np.sum(intrusions & ~oracle)),
        "intrusion_removal_rate": (
            0.0
            if intrusion_count == 0
            else float(np.sum(intrusions & ~oracle)) / intrusion_count
        ),
        "base_relevant_events": base_relevant_count,
        "harmed_base_relevant_events": int(np.sum(base_relevant & ~oracle)),
        "base_relevant_harm_rate": (
            0.0
            if base_relevant_count == 0
            else float(np.sum(base_relevant & ~oracle)) / base_relevant_count
        ),
    }


def _brute_force_query_oracle(
    tier_scores: np.ndarray,
    labels: np.ndarray,
    document_ids: np.ndarray,
    *,
    final_k: int,
    top_b: int,
    budget_bytes: int,
) -> tuple[int, int]:
    best_hits = -1
    best_cost = budget_bytes + 1
    costs = tuple(int(value) for value in ALLOWED_TIERS)
    for assignment in itertools.product(range(len(costs)), repeat=top_b):
        cost = sum(costs[index] for index in assignment)
        if cost > budget_bytes:
            continue
        scores = np.asarray(tier_scores[0, 0], dtype=np.float32).copy()
        for candidate_index, tier_index in enumerate(assignment):
            scores[candidate_index] = tier_scores[0, tier_index, candidate_index]
        order = np.lexsort((document_ids[0], -scores))[:final_k]
        hits = int(np.sum(labels[0, order] > 0))
        if hits > best_hits or (hits == best_hits and cost < best_cost):
            best_hits = hits
            best_cost = cost
    return best_hits, best_cost


def exact_solver_preflight() -> dict[str, Any]:
    """Prove the production DP agrees with exhaustive search on fixed cases."""

    cases = [
        {
            "name": "promotion_under_16_bytes",
            "base": [0.90, 0.80, 0.70, 0.60, 0.50],
            "increments": [
                [0.00, 0.00, 0.00],
                [0.00, 0.22, 0.01],
                [0.00, 0.28, 0.02],
                [0.01, 0.31, 0.03],
            ],
            "labels": [0, 1, 0, 0, 0],
            "docids": [50, 40, 30, 20, 10],
            "budget": 16,
        },
        {
            "name": "stable_tie_and_minimum_cost",
            "base": [0.80, 0.80, 0.75, 0.70, 0.65],
            "increments": [
                [0.00, 0.00, 0.00],
                [0.02, 0.00, 0.03],
                [0.02, 0.00, 0.04],
                [0.02, 0.00, 0.05],
            ],
            "labels": [0, 0, 1, 0, 0],
            "docids": [20, 10, 30, 40, 50],
            "budget": 32,
        },
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        base = np.asarray([case["base"]], dtype=np.float32)
        tiers = np.repeat(base[:, None, :], len(ALLOWED_TIERS), axis=1)
        increments = np.asarray(case["increments"], dtype=np.float32)
        tiers[0, :, :3] += increments
        labels = np.asarray([case["labels"]], dtype=np.uint8)
        docids = np.asarray([case["docids"]], dtype=np.int64)
        counts = np.asarray([max(1, int(np.sum(labels)))], dtype=np.int32)
        budget = int(case["budget"])
        expected_hits, expected_cost = _brute_force_query_oracle(
            tiers,
            labels,
            docids,
            final_k=2,
            top_b=3,
            budget_bytes=budget,
        )
        actual = exact_accessed_byte_oracle(
            tiers,
            ALLOWED_TIERS,
            labels,
            docids,
            counts,
            final_k=2,
            top_b=3,
            budget_bytes=budget,
        )
        passed = (
            int(actual.hits_at_k[0]) == expected_hits
            and int(actual.accessed_bytes[0]) == expected_cost
        )
        results.append(
            {
                "name": case["name"],
                "input_sha256": canonical_sha256(case),
                "budget_bytes": budget,
                "brute_force_hits": expected_hits,
                "brute_force_minimum_bytes": expected_cost,
                "solver_hits": int(actual.hits_at_k[0]),
                "solver_accessed_bytes": int(actual.accessed_bytes[0]),
                "solver_rate_assignments": actual.rate_assignments[0].tolist(),
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in results):
        raise ValueError("Exact accessed-byte solver failed brute-force preflight")
    return {
        "schema_version": 1,
        "status": "EXACT_SOLVER_BRUTE_FORCE_PREFLIGHT_PASSED",
        "tier_bytes": list(ALLOWED_TIERS),
        "cases": results,
    }


def paired_bootstrap_statistics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[dict[str, Any], np.ndarray]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("Bootstrap inputs must have exactly matching shapes")
    if (
        left_array.ndim != 1
        or not len(left_array)
        or not np.all(np.isfinite(left_array))
        or not np.all(np.isfinite(right_array))
    ):
        raise ValueError("Bootstrap inputs must be finite non-empty vectors")
    if isinstance(replicates, bool) or int(replicates) != replicates or replicates <= 0:
        raise ValueError("Bootstrap replicates must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be in (0, 1)")
    delta = left_array - right_array
    rng = np.random.Generator(np.random.PCG64(seed))
    statistics = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        end = min(start + 256, replicates)
        indices = rng.integers(0, len(delta), size=(end - start, len(delta)))
        statistics[start:end] = np.mean(delta[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    summary = {
        "replicates": replicates,
        "seed": seed,
        "rng": "NumPy Generator(PCG64)",
        "confidence": confidence,
        "quantile_method": "linear",
        "point_estimate": float(np.mean(delta)),
        "lower": float(np.quantile(statistics, tail, method="linear")),
        "upper": float(np.quantile(statistics, 1.0 - tail, method="linear")),
    }
    return summary, statistics


def validate_oracle0_contract(
    oracle: Any,
    *,
    base_recall: np.ndarray,
    base_scores: np.ndarray,
    document_ids: np.ndarray,
    final_k: int,
) -> dict[str, bool]:
    """Prove the zero-byte curve point is exactly the frozen base ranking."""

    base_membership = topk_membership(base_scores, document_ids, k=final_k)
    if not np.array_equal(np.asarray(oracle.recall_at_k), np.asarray(base_recall)):
        raise AssertionError("Oracle0 recall does not exactly reproduce Base")
    if not np.array_equal(np.asarray(oracle.topk_membership), base_membership):
        raise AssertionError("Oracle0 Top-k membership does not reproduce Base")
    if np.any(np.asarray(oracle.accessed_bytes) != 0) or np.any(
        np.asarray(oracle.rate_assignments) != 0
    ):
        raise AssertionError("Oracle0 used nonzero code bytes")
    return {
        "passed": True,
        "recall_exactly_matches_base": True,
        "topk_membership_exactly_matches_base": True,
        "all_accessed_bytes_zero": True,
        "all_rate_assignments_zero": True,
    }


def summarize_oracle_budget_curve(
    oracles: dict[str, Any],
    oracle_names: tuple[str, ...],
    budgets: list[int],
    comparator_recall: np.ndarray,
) -> dict[str, Any]:
    """Report both the full curve and its comparator-relative gain per byte."""

    if len(oracle_names) != len(budgets) or set(oracle_names) != set(oracles):
        raise ValueError("Oracle names and registered budgets do not match")
    comparator_mean = float(np.mean(np.asarray(comparator_recall, dtype=np.float64)))
    curve: dict[str, Any] = {}
    for name, budget in zip(oracle_names, budgets, strict=True):
        mean_accessed = float(np.mean(np.asarray(oracles[name].accessed_bytes)))
        mean_recall = float(np.mean(np.asarray(oracles[name].recall_at_k)))
        gain = mean_recall - comparator_mean
        curve[name] = {
            "budget_bytes_per_query": int(budget),
            "mean_accessed_bytes": mean_accessed,
            "mean_recall_at_10": mean_recall,
            "mean_recall_gain_over_primary_comparator": gain,
            "gain_per_mean_accessed_byte": (
                None if mean_accessed == 0.0 else gain / mean_accessed
            ),
            "gain_per_budget_byte": None if budget == 0 else gain / budget,
        }
    return curve


def scores_from_rate_assignments(
    tier_scores: np.ndarray, rate_assignments: np.ndarray, *, top_b: int
) -> np.ndarray:
    scores = np.asarray(tier_scores)
    rates = np.asarray(rate_assignments)
    if rates.shape != (scores.shape[0], top_b):
        raise ValueError("Rate assignments do not match the correctable candidates")
    result = np.asarray(scores[:, _tier_index(0), :], dtype=np.float32).copy()
    for tier_index, tier in enumerate(ALLOWED_TIERS):
        mask = rates == tier
        if np.any(mask):
            selected = result[:, :top_b]
            tier_values = scores[:, tier_index, :top_b]
            selected[mask] = tier_values[mask]
    return result


def per_document_influence_rows(
    tier_scores: np.ndarray,
    oracle_recall: np.ndarray,
    rate_assignments: np.ndarray,
    document_ids: np.ndarray,
    labels: np.ndarray,
    relevant_counts: np.ndarray,
    exact_scores: np.ndarray,
    *,
    final_k: int,
    top_b: int,
) -> list[dict[str, Any]]:
    """Aggregate leave-one-code-read marginal Recall influence by document."""

    chosen_scores = scores_from_rate_assignments(
        tier_scores, rate_assignments, top_b=top_b
    )
    base_scores = np.asarray(tier_scores[:, _tier_index(0), :], dtype=np.float32)
    docids = np.asarray(document_ids, dtype=np.int64)
    rates = np.asarray(rate_assignments, dtype=np.int16)
    aggregates: dict[int, dict[str, Any]] = {}
    for query_index in range(len(chosen_scores)):
        denominator = float(relevant_counts[query_index])
        for candidate_index in range(top_b):
            document_id = int(docids[query_index, candidate_index])
            row = aggregates.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "candidate_exposures": 0,
                    "relevant_exposures": 0,
                    "nonzero_assignments": 0,
                    "total_accessed_bytes": 0,
                    "positive_marginal_recall_mass": 0.0,
                    "negative_marginal_recall_mass": 0.0,
                    "exact_score_error_reduction_sum": 0.0,
                },
            )
            row["candidate_exposures"] += 1
            row["relevant_exposures"] += int(labels[query_index, candidate_index] > 0)
            rate = int(rates[query_index, candidate_index])
            if rate <= 0:
                continue
            row["nonzero_assignments"] += 1
            row["total_accessed_bytes"] += rate
            ablated = chosen_scores[query_index : query_index + 1].copy()
            ablated[0, candidate_index] = base_scores[query_index, candidate_index]
            ablated_recall = recall_at_k_per_query(
                ablated,
                docids[query_index : query_index + 1],
                labels[query_index : query_index + 1],
                np.asarray([relevant_counts[query_index]], dtype=np.int32),
                k=final_k,
            )[0]
            marginal = float(oracle_recall[query_index] - ablated_recall)
            row["positive_marginal_recall_mass"] += max(marginal, 0.0)
            row["negative_marginal_recall_mass"] += max(-marginal, 0.0)
            before_error = abs(
                float(base_scores[query_index, candidate_index])
                - float(exact_scores[query_index, candidate_index])
            )
            after_error = abs(
                float(chosen_scores[query_index, candidate_index])
                - float(exact_scores[query_index, candidate_index])
            )
            row["exact_score_error_reduction_sum"] += before_error - after_error
    rows = [aggregates[key] for key in sorted(aggregates)]
    for row in rows:
        row["mean_bytes_per_exposure"] = (
            float(row["total_accessed_bytes"]) / row["candidate_exposures"]
        )
        row["positive_marginal_recall_per_accessed_byte"] = (
            0.0
            if row["total_accessed_bytes"] == 0
            else float(row["positive_marginal_recall_mass"])
            / row["total_accessed_bytes"]
        )
    return rows


def _output_records(output_dir: Path, paths: list[Path]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    root = output_dir.resolve()
    for path in paths:
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError(f"Output is not inside output directory: {path}")
        name = resolved.relative_to(root).as_posix()
        records[name] = file_record(resolved)
    return records


def _external_file_records(
    bundle_dir: Path, manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    records: dict[str, Any] = {
        "candidate_manifest": file_record(manifest_path),
        "query_manifest": file_record(bundle_dir / "query_manifest.json"),
        "candidate_arrays": {},
    }
    for filename, record in manifest.get("files", {}).items():
        path = bundle_dir / filename
        _record_matches(path, record, f"external candidate file {filename}")
        records["candidate_arrays"][filename] = file_record(path)
    return records


def _external_label_records(label_manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(label_manifest_path)
    records: dict[str, Any] = {
        "label_manifest": file_record(label_manifest_path),
        "label_arrays": {},
    }
    started_record = manifest.get("started")
    if isinstance(started_record, dict):
        started_path = label_manifest_path.parent / "v3_role_labels_started.json"
        _record_matches(started_path, started_record, "role-label start marker")
        records["started"] = file_record(started_path)
    for filename, record in manifest.get("files", {}).items():
        path = label_manifest_path.parent / filename
        _record_matches(path, record, f"external label file {filename}")
        records["label_arrays"][filename] = file_record(path)
    return records


def verify_external_inputs(value: Any) -> None:
    if isinstance(value, dict) and {"path", "bytes", "sha256"} <= set(value):
        _record_matches(Path(value["path"]), value, "external input")
        return
    if isinstance(value, dict):
        for child in value.values():
            verify_external_inputs(child)
    elif isinstance(value, list):
        for child in value:
            verify_external_inputs(child)


def _copy_provenance(
    output_dir: Path,
    sources: dict[str, Path],
) -> list[Path]:
    copied: list[Path] = []
    for name, source in sources.items():
        if not source.exists() or not source.is_file():
            raise ValueError(f"Missing provenance source {name}: {source}")
        destination = output_dir / "provenance" / f"{name}.json"
        atomic_copy(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise AssertionError(f"Provenance copy changed bytes for {name}")
        copied.append(destination)
    return copied


def _design_provenance_sources(
    bundle_root: Path,
    design_bundle: Path,
    audit_bundle: Path,
    design_label_manifest: Path,
) -> dict[str, Path]:
    lineage = _lineage_files(bundle_root)
    return {
        "bundle_build_started": lineage["bundle_build_started"],
        "split_audit": lineage["split_audit"],
        "bundle_freeze_summary": lineage["bundle_freeze_summary"],
        "future_identity_manifest": lineage["future_identity"],
        "future_query_manifest": (
            bundle_root / "future_method_holdout" / "query_manifest.json"
        ),
        "design_candidate_manifest": (
            design_bundle / "v3_candidate_manifest.json"
        ),
        "design_query_manifest": design_bundle / "query_manifest.json",
        "audit_candidate_manifest": audit_bundle / "v3_candidate_manifest.json",
        "audit_query_manifest": audit_bundle / "query_manifest.json",
        "design_role_labels_manifest": design_label_manifest,
    }


def _write_invalid(
    output_dir: Path,
    *,
    phase: str,
    started_path: Path,
    error: BaseException,
) -> None:
    if not started_path.exists():
        return
    atomic_json(
        output_dir / "oracle_invalid.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "INVALID",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "started": file_record(started_path),
        },
    )


def _materialize_audit_labels(
    *,
    repo_root: Path,
    bundle_root: Path,
    parent_inner_train_bundle: Path,
    protocol_path: Path,
    output_dir: Path,
    source_commit: str,
) -> Path:
    label_manifest = bundle_root / AUDIT_ROLE_ID / "v3_role_labels_manifest.json"
    if label_manifest.exists():
        raise ValueError("Audit role labels existed before the audit phase materializer")
    command = [
        sys.executable,
        str(repo_root / CANONICAL_SOURCE_RELATIVES["label_materializer"]),
        "--bundle-root",
        str(bundle_root),
        "--parent-inner-train-bundle",
        str(parent_inner_train_bundle),
        "--role",
        AUDIT_ROLE_ID,
        "--source-commit",
        source_commit,
        "--protocol",
        str(protocol_path),
        "--design-freeze",
        str(output_dir / "design_freeze.json"),
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    if not label_manifest.exists():
        raise ValueError("Audit label materializer did not produce its manifest")
    return label_manifest


def _prepare_design_output(
    output_dir: Path,
    *,
    reuse_complete: bool,
    protocol: dict[str, Any],
    source_commit: str,
    source_hashes: dict[str, Any],
) -> dict[str, Any] | None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if reuse_complete and (output_dir / "design_freeze.json").exists():
            freeze = verify_design_freeze(
                output_dir,
                protocol=protocol,
                source_commit=source_commit,
                source_hashes=source_hashes,
            )
            verify_external_inputs(freeze.get("external_inputs", {}))
            return freeze
        raise ValueError("Refusing to reuse a non-empty partial design directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return None


def _reuse_audit_complete(
    output_dir: Path,
    *,
    protocol: dict[str, Any],
    source_commit: str,
    source_hashes: dict[str, Any],
) -> dict[str, Any] | None:
    complete_path = output_dir / "oracle_complete.json"
    if not complete_path.exists():
        return None
    freeze = verify_design_freeze(
        output_dir,
        protocol=protocol,
        source_commit=source_commit,
        source_hashes=source_hashes,
    )
    complete = read_json(complete_path)
    if complete.get("status") != "ORACLE_COMPLETE":
        raise ValueError("Complete marker status is invalid")
    if complete.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Complete marker protocol changed")
    if complete.get("source_commit") != source_commit:
        raise ValueError("Complete marker source commit changed")
    if complete.get("run_fingerprint") != freeze.get("run_fingerprint"):
        raise ValueError("Complete run fingerprint differs from design freeze")
    freeze_record = complete.get("design_freeze")
    if not isinstance(freeze_record, dict):
        raise ValueError("Complete marker lacks design-freeze registration")
    _record_matches(
        output_dir / "design_freeze.json", freeze_record, "complete design freeze"
    )
    verify_registered_outputs(output_dir, complete.get("outputs", {}))
    verify_external_inputs(freeze.get("external_inputs", {}))
    verify_external_inputs(complete.get("external_inputs", {}))
    summary = read_json(output_dir / "oracle_summary.json")
    if summary.get("status") != "ORACLE_COMPLETE":
        raise ValueError("Reused oracle summary status is invalid")
    if summary.get("run_fingerprint") != freeze.get("run_fingerprint"):
        raise ValueError("Reused oracle summary fingerprint changed")
    return summary


def _common_context(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    _validate_exact_commit(args.source_commit)
    repo_root = Path(__file__).resolve().parents[1]
    validate_clean_git_head(repo_root, args.source_commit)
    source_hashes = validate_canonical_protocol_and_sources(
        repo_root, args.protocol, args.source_commit
    )
    protocol = read_json(args.protocol)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected RARS-v3 protocol")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_ORACLE_RUN":
        raise ValueError("RARS-v3 protocol was not frozen before execution")
    if protocol.get("method_revision_allowed") is not False:
        raise ValueError("RARS-v3 protocol permits method revision")
    if protocol.get("outcome_informed_revision_allowed") is not False:
        raise ValueError("RARS-v3 protocol permits outcome-informed revision")
    environment, numpy_config = capture_environment(protocol)
    return repo_root, protocol, source_hashes, environment, numpy_config


def run_design_phase(args: argparse.Namespace) -> dict[str, Any]:
    (
        repo_root,
        protocol,
        source_hashes,
        environment,
        numpy_config,
    ) = _common_context(args)
    if args.design_bundle is None or args.design_label_manifest is None:
        raise ValueError(
            "Design phase requires --design-bundle and --design-label-manifest"
        )
    design_bundle_dir = args.design_bundle.resolve()
    design_label_manifest = args.design_label_manifest.resolve()
    bundle_root = (
        args.bundle_root.resolve()
        if args.bundle_root is not None
        else design_bundle_dir.parent
    )
    audit_bundle_dir = (
        args.audit_bundle.resolve()
        if args.audit_bundle is not None
        else bundle_root / AUDIT_ROLE_ID
    )
    design_manifest_path = design_bundle_dir / "v3_candidate_manifest.json"
    audit_manifest_path = audit_bundle_dir / "v3_candidate_manifest.json"
    lineage = validate_bundle_root_lineage(
        bundle_root,
        design_manifest_path=design_manifest_path,
        audit_manifest_path=audit_manifest_path,
        protocol=protocol,
        source_commit=args.source_commit,
        source_hashes=source_hashes,
    )
    present_audit_labels = [
        str(audit_bundle_dir / filename)
        for filename in ROLE_LABEL_FILENAMES
        if (audit_bundle_dir / filename).exists()
    ]
    if present_audit_labels:
        raise ValueError(
            "Audit role labels already exist before design freeze: "
            f"{present_audit_labels}"
        )
    fingerprint_payload = {
        "protocol_id": PROTOCOL_ID,
        "phase_contract": "DESIGN_THEN_AUDIT_V1",
        "source_commit": args.source_commit,
        "source_hashes": source_hashes,
        "design_candidate_manifest": file_record(design_manifest_path),
        "design_role_labels_manifest": file_record(design_label_manifest),
        "audit_candidate_manifest_registration": lineage[
            "candidate_manifest_registrations"
        ][AUDIT_ROLE_ID],
        "lineage": lineage["records"],
        "configuration": {
            "progressive_representation": protocol["progressive_representation"],
            "eligible_baselines": protocol["registered_matched_baselines"],
            "matched_access_oracle": protocol["matched_access_oracle"],
            "access_gate": protocol["access_gate"],
            "bootstrap": protocol["metric_contract"]["bootstrap"],
            "environment_contract": protocol["execution_environment_contract"],
        },
    }
    fingerprint = build_run_fingerprint(fingerprint_payload)
    reused = _prepare_design_output(
        args.output_dir,
        reuse_complete=args.reuse_complete,
        protocol=protocol,
        source_commit=args.source_commit,
        source_hashes=source_hashes,
    )
    if reused is not None:
        return reused
    started_path = args.output_dir / "oracle_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "ORACLE_DESIGN_STARTED",
            "phase": "design",
            "source_commit": args.source_commit,
            "run_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "audit_candidate_arrays_loaded": False,
            "audit_role_labels_materialized": False,
            "future_method_holdout_accessed": False,
        },
    )
    try:
        design = load_bundle(
            design_bundle_dir,
            expected_role_id=DESIGN_ROLE_ID,
            protocol=protocol,
            source_commit=args.source_commit,
            source_hashes=source_hashes,
            split_audit_sha256=lineage["split_audit_sha256"],
        )
        load_role_labels(
            design,
            design_label_manifest,
            expected_role_id=DESIGN_ROLE_ID,
            protocol=protocol,
            source_commit=args.source_commit,
            source_hashes=source_hashes,
        )
        final_k = int(protocol["frozen_retrieval"]["final_k"])
        top_b = int(protocol["frozen_retrieval"]["correction_depth"])
        representation = protocol["progressive_representation"]
        rank = int(representation["rank"])
        if rank != 32 or tuple(representation["tiers_code_bytes"]) != ALLOWED_TIERS:
            raise ValueError("Progressive rank/tier contract changed")
        basis, scales, sample_rows, singular_values, singular_gaps = (
            fit_uncentered_progressive_svd(
                design["residuals"],
                rank=rank,
                max_samples=int(representation["sample_count_maximum"]),
                seed=int(representation["sample_seed"]),
            )
        )
        paths: list[Path] = [started_path]

        def save_array(filename: str, value: np.ndarray) -> Path:
            path = args.output_dir / filename
            atomic_save(path, np.asarray(value))
            paths.append(path)
            return path

        basis_path = save_array("progressive_svd_rank32.float32.npy", basis)
        scales_path = save_array(
            "progressive_svd_rank32_scales.float32.npy", scales
        )
        sample_path = save_array(
            "progressive_svd_sample_rows.int64.npy", sample_rows
        )
        singular_path = save_array(
            "progressive_svd_singular_values.float64.npy", singular_values
        )
        gap_path = save_array(
            "progressive_svd_adjacent_gaps.float64.npy", singular_gaps
        )
        environment_path = args.output_dir / "execution_environment.json"
        numpy_config_path = args.output_dir / "numpy_config.txt"
        atomic_text(numpy_config_path, numpy_config)
        paths.append(numpy_config_path)
        atomic_json(
            environment_path,
            {
                **environment,
                "numpy_config": file_record(numpy_config_path),
            },
        )
        paths.append(environment_path)
        design_tiers = progressive_tier_scores(
            design["queries"],
            design["ann_scores"],
            design["residual_lookup"],
            design["residuals"],
            basis,
            scales,
            alpha=float(representation["alpha"]),
            top_b=top_b,
        )
        exposure = _exposure_counts(design["document_ids"], top_b=top_b)
        exposure_ids = np.asarray(sorted(exposure), dtype=np.int64)
        exposure_counts = np.asarray(
            [exposure[int(value)] for value in exposure_ids], dtype=np.int32
        )
        save_array("design_exposure_doc_ids.int64.npy", exposure_ids)
        save_array("design_exposure_counts.int32.npy", exposure_counts)
        design_baselines = build_baselines(
            design,
            design_tiers,
            exposure=exposure,
            final_k=final_k,
            top_b=top_b,
        )
        if set(design_baselines) != set(BASELINE_NAMES):
            raise ValueError("The five eligible design baselines changed")
        design_recalls = baseline_recalls(
            design_baselines, design, final_k=final_k
        )
        baseline_artifacts: dict[str, Any] = {}
        for name in sorted(BASELINE_NAMES):
            score_path = save_array(
                f"design_baseline_{name}_scores.float32.npy",
                np.asarray(design_baselines[name], dtype=np.float32),
            )
            recall_path = save_array(
                f"design_baseline_{name}_recall_at_10.float64.npy",
                np.asarray(design_recalls[name], dtype=np.float64),
            )
            baseline_artifacts[name] = {
                "scores": file_record(score_path),
                "per_query_recall": file_record(recall_path),
            }
        primary_budget = int(
            protocol["matched_access_oracle"]["primary_budget_bytes_per_query"]
        )
        selection = select_primary_baseline(
            design_recalls, accessed_bytes=primary_budget
        )
        selected_name = str(selection["selected"]["method"])
        design_oracle = exact_accessed_byte_oracle(
            design_tiers,
            ALLOWED_TIERS,
            design["labels"],
            design["document_ids"],
            design["relevant_counts"],
            final_k=final_k,
            top_b=top_b,
            budget_bytes=primary_budget,
        )
        oracle_scores = scores_from_rate_assignments(
            design_tiers, design_oracle.rate_assignments, top_b=top_b
        )
        oracle_paths = {
            "scores": save_array(
                "design_oracle16_scores.float32.npy", oracle_scores
            ),
            "per_query_recall": save_array(
                "design_oracle16_recall_at_10.float64.npy",
                design_oracle.recall_at_k.astype(np.float64),
            ),
            "rates": save_array(
                "design_oracle16_rates.int16.npy",
                design_oracle.rate_assignments.astype(np.int16),
            ),
            "accessed_bytes": save_array(
                "design_oracle16_accessed_bytes.int32.npy",
                design_oracle.accessed_bytes.astype(np.int32),
            ),
            "topk_membership": save_array(
                "design_oracle16_topk_membership.bool.npy",
                design_oracle.topk_membership.astype(np.bool_),
            ),
        }
        qids = [str(value) for value in design["query_manifest"]["query_ids"]]
        folds = design_fold_ids(qids)
        save_array("design_fold_ids.uint8.npy", folds.astype(np.uint8))
        fold_gains = [
            float(
                np.mean(
                    design_oracle.recall_at_k[folds == fold]
                    - design_recalls[selected_name][folds == fold]
                )
            )
            for fold in range(5)
        ]
        preflight_path = args.output_dir / "exact_solver_bruteforce_preflight.json"
        atomic_json(preflight_path, exact_solver_preflight())
        paths.append(preflight_path)
        selection_path = args.output_dir / "design_primary_comparator.json"
        atomic_json(
            selection_path,
            {
                **selection,
                "eligible_baseline_artifacts": baseline_artifacts,
                "design_oracle16_artifacts": {
                    name: file_record(path) for name, path in oracle_paths.items()
                },
                "design_fold_gains": fold_gains,
                "design_fold_counts": np.bincount(folds, minlength=5).tolist(),
                "audit_reselection_allowed": False,
                "random_secondary": random_baseline_summary(
                    design, design_tiers, final_k=final_k, top_b=top_b
                ),
            },
        )
        paths.append(selection_path)
        provenance_paths = _copy_provenance(
            args.output_dir,
            _design_provenance_sources(
                bundle_root,
                design_bundle_dir,
                audit_bundle_dir,
                design_label_manifest,
            ),
        )
        paths.extend(provenance_paths)
        design_external = {
            "design_candidate": _external_file_records(
                design_bundle_dir, design["manifest"], design_manifest_path
            ),
            "design_labels": _external_label_records(design_label_manifest),
            "audit_candidate_manifest": file_record(audit_manifest_path),
            "audit_query_manifest": file_record(
                audit_bundle_dir / "query_manifest.json"
            ),
            "lineage": lineage["records"],
        }
        freeze_payload = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "DESIGN_ARTIFACTS_FROZEN_BEFORE_AUDIT_LOAD",
            "source_commit": args.source_commit,
            "run_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "source_hashes": source_hashes,
            "execution_environment": environment,
            "contracts": {
                "progressive_representation": representation,
                "eligible_baselines": protocol["registered_matched_baselines"],
                "matched_access_oracle": protocol["matched_access_oracle"],
                "access_gate": protocol["access_gate"],
                "bootstrap": protocol["metric_contract"]["bootstrap"],
                "budgets_bytes_per_query": protocol["matched_access_oracle"][
                    "budget_curve_bytes_per_query"
                ],
            },
            "uncentered_svd": {
                "centered": False,
                "rank": rank,
                "full_spectrum_count": len(singular_values),
                "rank32_boundary_adjacent_gap": (
                    None if len(singular_values) <= rank else float(singular_gaps[rank - 1])
                ),
                "basis": file_record(basis_path),
                "scales": file_record(scales_path),
                "sample_rows": file_record(sample_path),
                "singular_values": file_record(singular_path),
                "adjacent_gaps": file_record(gap_path),
            },
            "selected_primary_comparator": selection["selected"],
            "design_fold_gains": fold_gains,
            "candidate_manifest_registrations": lineage[
                "candidate_manifest_registrations"
            ],
            "design_bundle_manifest": file_record(design_manifest_path),
            "audit_bundle_manifest_registered_but_arrays_unloaded": file_record(
                audit_manifest_path
            ),
            "design_role_labels_manifest": file_record(design_label_manifest),
            "audit_role_labels_expected_path": str(audit_label_manifest),
            "audit_bundle_loaded_before_this_freeze": False,
            "audit_role_labels_materialized_before_this_freeze": False,
            "future_method_holdout_accessed": False,
            "external_inputs": design_external,
            "registered_outputs": _output_records(args.output_dir, paths),
        }
        freeze_path = args.output_dir / "design_freeze.json"
        atomic_json(freeze_path, freeze_payload)
        return freeze_payload
    except BaseException as error:
        _write_invalid(
            args.output_dir,
            phase="design",
            started_path=started_path,
            error=error,
        )
        raise


def run_audit_phase(args: argparse.Namespace) -> dict[str, Any]:
    (
        repo_root,
        protocol,
        source_hashes,
        environment,
        _,
    ) = _common_context(args)
    if args.audit_bundle is None or args.parent_inner_train_bundle is None:
        raise ValueError(
            "Audit phase requires --audit-bundle and --parent-inner-train-bundle"
        )
    audit_bundle_dir = args.audit_bundle.resolve()
    bundle_root = (
        args.bundle_root.resolve()
        if args.bundle_root is not None
        else audit_bundle_dir.parent
    )
    design_bundle_dir = bundle_root / DESIGN_ROLE_ID
    audit_label_paths = tuple(
        audit_bundle_dir / filename
        for filename in ROLE_LABEL_FILENAMES
    )
    if args.reuse_complete and (args.output_dir / "oracle_complete.json").exists():
        # A Drive-persisted complete run may be reused after a fresh /content
        # rematerialization.  Recreate the exact audit label slice only after
        # recursively verifying the durable design freeze.
        freeze = verify_design_freeze(
            args.output_dir,
            protocol=protocol,
            source_commit=args.source_commit,
            source_hashes=source_hashes,
        )
        verify_external_inputs(freeze.get("external_inputs", {}))
        if freeze.get("execution_environment") != environment:
            raise ValueError("Audit environment differs from the design freeze")
        label_manifest_path = audit_bundle_dir / "v3_role_labels_manifest.json"
        if not label_manifest_path.exists():
            partial = [str(path) for path in audit_label_paths if path.exists()]
            if partial:
                raise ValueError(
                    "Refusing to recreate audit labels over partial artifacts: "
                    f"{partial}"
                )
            _materialize_audit_labels(
                repo_root=repo_root,
                bundle_root=bundle_root,
                parent_inner_train_bundle=args.parent_inner_train_bundle.resolve(),
                protocol_path=args.protocol,
                output_dir=args.output_dir,
                source_commit=args.source_commit,
            )
        reused = _reuse_audit_complete(
            args.output_dir,
            protocol=protocol,
            source_commit=args.source_commit,
            source_hashes=source_hashes,
        )
        if reused is None:
            raise AssertionError("Complete-reuse marker disappeared during verification")
        return reused
    # This recursive verification is deliberately the first audit-data action.
    freeze = verify_design_freeze(
        args.output_dir,
        protocol=protocol,
        source_commit=args.source_commit,
        source_hashes=source_hashes,
    )
    verify_external_inputs(freeze.get("external_inputs", {}))
    if freeze.get("execution_environment") != environment:
        raise ValueError("Audit environment differs from the design freeze")
    if (args.output_dir / "oracle_complete.json").exists():
        raise ValueError("Use --reuse-complete for an existing complete audit")
    if (args.output_dir / "oracle_audit_started.json").exists() or (
        args.output_dir / "oracle_invalid.json"
    ).exists():
        raise ValueError("Refusing to reuse a partial or invalid audit phase")
    present_audit_labels = [str(path) for path in audit_label_paths if path.exists()]
    if present_audit_labels:
        raise ValueError(
            "Audit role labels existed before the audit start marker: "
            f"{present_audit_labels}"
        )
    started_path = args.output_dir / "oracle_audit_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "ORACLE_AUDIT_STARTED_AFTER_DESIGN_FREEZE",
            "phase": "audit",
            "source_commit": args.source_commit,
            "run_fingerprint": freeze["run_fingerprint"],
            "design_freeze": file_record(args.output_dir / "design_freeze.json"),
            "design_registered_outputs_verified": True,
            "audit_role_labels_present_before_start": False,
            "future_method_holdout_accessed": False,
        },
    )
    try:
        audit_label_manifest = _materialize_audit_labels(
            repo_root=repo_root,
            bundle_root=bundle_root,
            parent_inner_train_bundle=args.parent_inner_train_bundle.resolve(),
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
        )
        audit_started = read_json(started_path)
        started_freeze_record = audit_started.get("design_freeze")
        if not isinstance(started_freeze_record, dict):
            raise ValueError("Audit start marker lacks design-freeze registration")
        _record_matches(
            args.output_dir / "design_freeze.json",
            started_freeze_record,
            "audit-start design freeze",
        )
        if audit_started.get("run_fingerprint") != freeze.get("run_fingerprint"):
            raise ValueError("Audit start fingerprint differs from design freeze")
        if audit_started.get("audit_role_labels_present_before_start") is not False:
            raise ValueError("Audit start marker does not certify label isolation")
        design_manifest_path = design_bundle_dir / "v3_candidate_manifest.json"
        audit_manifest_path = audit_bundle_dir / "v3_candidate_manifest.json"
        lineage = validate_bundle_root_lineage(
            bundle_root,
            design_manifest_path=design_manifest_path,
            audit_manifest_path=audit_manifest_path,
            protocol=protocol,
            source_commit=args.source_commit,
            source_hashes=source_hashes,
        )
        expected_audit_record = freeze[
            "audit_bundle_manifest_registered_but_arrays_unloaded"
        ]
        _record_matches(
            audit_manifest_path, expected_audit_record, "design-registered audit bundle"
        )
        audit = load_bundle(
            audit_bundle_dir,
            expected_role_id=AUDIT_ROLE_ID,
            protocol=protocol,
            source_commit=args.source_commit,
            source_hashes=source_hashes,
            split_audit_sha256=lineage["split_audit_sha256"],
        )
        load_role_labels(
            audit,
            audit_label_manifest,
            expected_role_id=AUDIT_ROLE_ID,
            protocol=protocol,
            source_commit=args.source_commit,
            source_hashes=source_hashes,
            design_freeze_path=args.output_dir / "design_freeze.json",
        )
        basis = np.load(
            args.output_dir / "progressive_svd_rank32.float32.npy", mmap_mode="r"
        )
        scales = np.load(
            args.output_dir / "progressive_svd_rank32_scales.float32.npy",
            mmap_mode="r",
        )
        _require_dtype("frozen progressive basis", basis, np.float32)
        _require_dtype("frozen progressive scales", scales, np.float32)
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(scales)):
            raise ValueError("Frozen progressive representation is non-finite")
        final_k = int(protocol["frozen_retrieval"]["final_k"])
        top_b = int(protocol["frozen_retrieval"]["correction_depth"])
        candidate_k = int(protocol["frozen_retrieval"]["candidate_k"])
        representation = protocol["progressive_representation"]
        audit_tiers = progressive_tier_scores(
            audit["queries"],
            audit["ann_scores"],
            audit["residual_lookup"],
            audit["residuals"],
            basis,
            scales,
            alpha=float(representation["alpha"]),
            top_b=top_b,
        )
        exposure_ids = np.load(
            args.output_dir / "design_exposure_doc_ids.int64.npy", mmap_mode="r"
        )
        exposure_counts = np.load(
            args.output_dir / "design_exposure_counts.int32.npy", mmap_mode="r"
        )
        exposure = {
            int(document_id): int(count)
            for document_id, count in zip(exposure_ids, exposure_counts, strict=True)
        }
        baselines = build_baselines(
            audit,
            audit_tiers,
            exposure=exposure,
            final_k=final_k,
            top_b=top_b,
        )
        recalls = baseline_recalls(baselines, audit, final_k=final_k)
        uniform_scores = {
            f"Uniform{tier}": np.asarray(
                audit_tiers[:, _tier_index(tier), :], dtype=np.float32
            )
            for tier in ALLOWED_TIERS
        }
        uniform_recalls = {
            name: recall_at_k_per_query(
                scores,
                audit["document_ids"],
                audit["labels"],
                audit["relevant_counts"],
                k=final_k,
            )
            for name, scores in uniform_scores.items()
        }
        selection = read_json(args.output_dir / "design_primary_comparator.json")
        selected_name = str(selection["selected"]["method"])
        if selected_name not in baselines:
            raise ValueError("Design-selected comparator is absent on audit")
        comparator_scores = baselines[selected_name]
        comparator_recall = recalls[selected_name]
        base_scores = np.asarray(audit["ann_scores"], dtype=np.float32)
        base_recall = recall_at_k_per_query(
            base_scores,
            audit["document_ids"],
            audit["labels"],
            audit["relevant_counts"],
            k=final_k,
        )
        exact40_scores = exact_residual_scores(
            audit["queries"],
            base_scores,
            audit["residual_lookup"],
            audit["residuals"],
            top_b=top_b,
        )
        exact100_scores = exact_residual_scores(
            audit["queries"],
            base_scores,
            audit["residual_lookup"],
            audit["residuals"],
            top_b=candidate_k,
        )
        exact40_recall = recall_at_k_per_query(
            exact40_scores,
            audit["document_ids"],
            audit["labels"],
            audit["relevant_counts"],
            k=final_k,
        )
        exact100_recall = recall_at_k_per_query(
            exact100_scores,
            audit["document_ids"],
            audit["labels"],
            audit["relevant_counts"],
            k=final_k,
        )
        budgets = [
            int(value)
            for value in protocol["matched_access_oracle"][
                "budget_curve_bytes_per_query"
            ]
        ]
        if budgets != [0, top_b * 8, top_b * 16, top_b * 32]:
            raise ValueError("Registered oracle budget curve changed")
        oracle_names = ("Oracle0", "Oracle8", "Oracle16", "Oracle32")
        oracles = {
            name: exact_accessed_byte_oracle(
                audit_tiers,
                ALLOWED_TIERS,
                audit["labels"],
                audit["document_ids"],
                audit["relevant_counts"],
                final_k=final_k,
                top_b=top_b,
                budget_bytes=budget,
            )
            for name, budget in zip(oracle_names, budgets, strict=True)
        }
        oracle0_contract = validate_oracle0_contract(
            oracles["Oracle0"],
            base_recall=base_recall,
            base_scores=base_scores,
            document_ids=audit["document_ids"],
            final_k=final_k,
        )
        recoveries: dict[str, Any] = {"base_relative": {}, "comparator_relative": {}}
        for name, oracle in oracles.items():
            recoveries["base_relative"][name] = compression_recovery_diagnostics(
                base_scores,
                exact40_scores,
                oracle.recall_at_k,
                base_recall,
                exact40_recall,
                oracle.topk_membership,
                audit["document_ids"],
                k=final_k,
                reference_name="base",
            )
            recoveries["comparator_relative"][name] = (
                compression_recovery_diagnostics(
                    comparator_scores,
                    exact40_scores,
                    oracle.recall_at_k,
                    comparator_recall,
                    exact40_recall,
                    oracle.topk_membership,
                    audit["document_ids"],
                    k=final_k,
                    reference_name=selected_name,
                )
            )
        bootstrap_contract = protocol["metric_contract"]["bootstrap"]
        bootstrap, bootstrap_statistics = paired_bootstrap_statistics(
            oracles["Oracle16"].recall_at_k,
            comparator_recall,
            replicates=int(bootstrap_contract["replicates"]),
            seed=int(bootstrap_contract["seed"]),
            confidence=float(bootstrap_contract["confidence"]),
        )
        base8 = recoveries["base_relative"]["Oracle8"]
        base16 = recoveries["base_relative"]["Oracle16"]
        comparator8 = recoveries["comparator_relative"]["Oracle8"]
        comparator16 = recoveries["comparator_relative"]["Oracle16"]
        decision = decide_oracle_gate(
            oracle_recall=oracles["Oracle16"].recall_at_k,
            comparator_recall=comparator_recall,
            exact40_recall=exact40_recall,
            base_recall=base_recall,
            base_relative_cfr8=float(base8["counterfactual_recovery_fraction"]),
            base_relative_cfr16=float(base16["counterfactual_recovery_fraction"]),
            base_relative_alignment16=float(
                base16["positive_gain_mass_with_exact_distance_reduction_fraction"]
            ),
            comparator_relative_cfr8=float(
                comparator8["counterfactual_recovery_fraction"]
            ),
            comparator_relative_cfr16=float(
                comparator16["counterfactual_recovery_fraction"]
            ),
            comparator_relative_alignment16=float(
                comparator16[
                    "positive_gain_mass_with_exact_distance_reduction_fraction"
                ]
            ),
            design_fold_gains=freeze["design_fold_gains"],
            bootstrap=bootstrap,
            thresholds=protocol["access_gate"],
        )
        paths: list[Path] = [started_path]

        def save_array(filename: str, value: np.ndarray) -> Path:
            path = args.output_dir / filename
            atomic_save(path, np.asarray(value))
            paths.append(path)
            return path

        save_array("audit_base_recall_at_10.float64.npy", base_recall)
        save_array("audit_exact40_recall_at_10.float64.npy", exact40_recall)
        save_array("audit_exact100_recall_at_10.float64.npy", exact100_recall)
        save_array(
            "audit_primary_comparator_recall_at_10.float64.npy", comparator_recall
        )
        for name in sorted(BASELINE_NAMES):
            save_array(
                f"audit_baseline_{name}_scores.float32.npy", baselines[name]
            )
            save_array(
                f"audit_baseline_{name}_recall_at_10.float64.npy", recalls[name]
            )
        for name in ("Uniform0", "Uniform8", "Uniform16", "Uniform32"):
            slug = name.casefold()
            save_array(
                f"audit_{slug}_scores.float32.npy", uniform_scores[name]
            )
            save_array(
                f"audit_{slug}_recall_at_10.float64.npy", uniform_recalls[name]
            )
        for name, oracle in oracles.items():
            slug = name.casefold()
            save_array(
                f"audit_{slug}_recall_at_10.float64.npy", oracle.recall_at_k
            )
            save_array(
                f"audit_{slug}_rates.int16.npy", oracle.rate_assignments
            )
            save_array(
                f"audit_{slug}_accessed_bytes.int32.npy", oracle.accessed_bytes
            )
            save_array(
                f"audit_{slug}_topk_membership.bool.npy", oracle.topk_membership
            )
        save_array(
            "audit_oracle16_vs_comparator_bootstrap.float64.npy",
            bootstrap_statistics,
        )
        influence_path = args.output_dir / "audit_document_influence.csv"
        atomic_csv(
            influence_path,
            per_document_influence_rows(
                audit_tiers,
                oracles["Oracle16"].recall_at_k,
                oracles["Oracle16"].rate_assignments,
                audit["document_ids"],
                audit["labels"],
                audit["relevant_counts"],
                exact40_scores,
                final_k=final_k,
                top_b=top_b,
            ),
        )
        paths.append(influence_path)
        audit_label_provenance = args.output_dir / "provenance/audit_role_labels_manifest.json"
        atomic_copy(audit_label_manifest, audit_label_provenance)
        paths.append(audit_label_provenance)
        mean_recalls = {
            "base": float(np.mean(base_recall)),
            "primary_comparator": float(np.mean(comparator_recall)),
            "Exact40": float(np.mean(exact40_recall)),
            "Exact100": float(np.mean(exact100_recall)),
            "QrelCandidate40": float(
                np.mean(
                    candidate_relevance_ceiling(
                        audit["labels"],
                        audit["relevant_counts"],
                        k=final_k,
                        depth=top_b,
                    )
                )
            ),
            "QrelCandidate100": float(
                np.mean(
                    candidate_relevance_ceiling(
                        audit["labels"],
                        audit["relevant_counts"],
                        k=final_k,
                        depth=candidate_k,
                    )
                )
            ),
            **{
                name: float(np.mean(values))
                for name, values in uniform_recalls.items()
            },
            **{
                name: float(np.mean(oracle.recall_at_k))
                for name, oracle in oracles.items()
            },
        }
        oracle_budget_curve = summarize_oracle_budget_curve(
            oracles, oracle_names, budgets, comparator_recall
        )
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "ORACLE_COMPLETE",
            "phase": "audit",
            "source_commit": args.source_commit,
            "run_fingerprint": freeze["run_fingerprint"],
            "formal_decision": decision["decision"],
            "evidence_status": "DEVELOPMENT_ONLY_V3_AUDIT_NOT_EXTERNAL_CONFIRMATION",
            "selected_primary_comparator": freeze["selected_primary_comparator"],
            "mean_recall_at_10": mean_recalls,
            "eligible_audit_baselines": {
                name: float(np.mean(values)) for name, values in recalls.items()
            },
            "oracle_budget_curve": oracle_budget_curve,
            "oracle0_contract": oracle0_contract,
            "counterfactual_recovery": recoveries,
            "bootstrap": bootstrap,
            "gate": decision,
            "membership_events_oracle16_vs_base": membership_event_diagnostics(
                base_scores,
                exact40_scores,
                oracles["Oracle16"].topk_membership,
                audit["document_ids"],
                audit["labels"],
                final_k=final_k,
            ),
            "membership_events_oracle16_vs_primary_comparator": (
                membership_event_diagnostics(
                    comparator_scores,
                    exact40_scores,
                    oracles["Oracle16"].topk_membership,
                    audit["document_ids"],
                    audit["labels"],
                    final_k=final_k,
                )
            ),
            "random_secondary": random_baseline_summary(
                audit, audit_tiers, final_k=final_k, top_b=top_b
            ),
            "training_performed": False,
            "static_storage_oracle_performed": False,
            "persistent_storage_claim_allowed": False,
            "future_method_holdout_accessed": False,
            "outer_or_test_outcomes_accessed": False,
        }
        summary_path = args.output_dir / "oracle_summary.json"
        atomic_json(summary_path, summary)
        paths.append(summary_path)
        external_inputs = {
            "audit_candidate": _external_file_records(
                audit_bundle_dir, audit["manifest"], audit_manifest_path
            ),
            "audit_labels": _external_label_records(audit_label_manifest),
            "lineage": lineage["records"],
        }
        complete = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "ORACLE_COMPLETE",
            "formal_decision": decision["decision"],
            "source_commit": args.source_commit,
            "run_fingerprint": freeze["run_fingerprint"],
            "design_freeze": file_record(args.output_dir / "design_freeze.json"),
            "outputs": _output_records(args.output_dir, paths),
            "external_inputs": external_inputs,
        }
        atomic_json(args.output_dir / "oracle_complete.json", complete)
        return summary
    except BaseException as error:
        _write_invalid(
            args.output_dir,
            phase="audit",
            started_path=started_path,
            error=error,
        )
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.phase == "design":
        return run_design_phase(args)
    if args.phase == "audit":
        return run_audit_phase(args)
    raise ValueError(f"Unsupported phase: {args.phase}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("design", "audit"))
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--design-bundle", type=Path)
    parser.add_argument("--design-label-manifest", type=Path)
    parser.add_argument("--audit-bundle", type=Path)
    parser.add_argument("--parent-inner-train-bundle", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reuse-complete", action="store_true")
    args = parser.parse_args()
    if args.phase == "design":
        if args.design_bundle is None or args.design_label_manifest is None:
            parser.error(
                "design requires --design-bundle and --design-label-manifest"
            )
        if args.parent_inner_train_bundle is not None:
            parser.error("design must not receive --parent-inner-train-bundle")
    else:
        if args.audit_bundle is None or args.parent_inner_train_bundle is None:
            parser.error(
                "audit requires --audit-bundle and --parent-inner-train-bundle"
            )
        if args.design_label_manifest is not None:
            parser.error("audit materializes its own post-freeze role labels")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
