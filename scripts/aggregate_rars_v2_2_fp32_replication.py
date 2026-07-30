#!/usr/bin/env python3
"""Aggregate the frozen RARS-v2.2 three-seed FP32 replication.

Seed 42 is the observed Stage-A development result.  Seeds 43 and 44 are the
held-out replication seeds and therefore own the primary replication decision.
This script is deliberately CPU-only: it verifies completed training artifacts,
recomputes all seed-independent comparators from the frozen inner-validation
bundle, and performs shared-index paired query bootstraps.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v2_2_core import (  # noqa: E402
    FIT_ROLE_ID,
    PROTOCOL_ID,
    SELECTION_ROLE_ID,
    build_run_fingerprint,
    canonical_sha256,
    file_record,
    load_pca_warm_start,
    pca_fp32_scores,
    score_candidates_fp32,
    sha256_file,
)
from train_boundary_loss_sidecar import (  # noqa: E402
    recall_at_k_per_query,
)
from train_boundary_loss_sidecar_v2_2 import load_bundle  # noqa: E402


EXPECTED_SEEDS = (42, 43, 44)
HELDOUT_SEEDS = (43, 44)
REPLICATION_PROTOCOL_ID = "rars_v2_2_fp32_replication_v1"
MINIMUM_GAIN_OVER_BASE = 0.01135
MINIMUM_GAIN_OVER_PCA = 0.005
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260717

STABLE_DECISION = "STABLE_GO_TO_QAT"
UNSTABLE_DECISION = "UNSTABLE_NO_QAT"
STOP_DECISION = "STOP"

REQUIRED_RUN_OUTPUTS = {
    "training_started.json",
    "query_projection.float32.npy",
    "document_projection.float32.npy",
    "selection_base_per_query.float64.npy",
    "selection_v2_2_per_query.float64.npy",
    "training_history.json",
    "selection_metrics.json",
    "training_summary.json",
}


class ReplicationInvalid(ValueError):
    """Raised when a replication artifact fails a frozen lineage contract."""


@dataclass(frozen=True)
class RunArtifact:
    seed: int
    directory: Path
    run_id: str
    fingerprint_payload: dict[str, Any]
    configuration: dict[str, Any]
    summary: dict[str, Any]
    metrics: dict[str, Any]
    environment: dict[str, Any]
    base_per_query: np.ndarray
    v2_per_query: np.ndarray
    complete_sha256: str


@dataclass(frozen=True)
class RunPreflight:
    seed: int
    directory: Path
    run_id: str
    complete: dict[str, Any]
    complete_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplicationInvalid(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ReplicationInvalid(f"Cannot read valid JSON from {path}: {exc}") from exc


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_file_record(path: Path, record: Any, *, label: str) -> None:
    _require(isinstance(record, dict), f"{label} has no file-record object")
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        expected_bytes = int(record["bytes"])
        expected_sha256 = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplicationInvalid(f"{label} has a malformed file record") from exc
    _require(len(expected_sha256) == 64, f"{label} has a malformed SHA-256")
    _require(path.stat().st_size == expected_bytes, f"{label} byte count changed")
    _require(sha256_file(path) == expected_sha256, f"{label} SHA-256 changed")


def _load_float_array(path: Path, *, label: str) -> np.ndarray:
    try:
        value = np.asarray(np.load(path, allow_pickle=False))
    except (OSError, ValueError) as exc:
        raise ReplicationInvalid(f"Cannot load {label}: {exc}") from exc
    _require(value.dtype == np.float64, f"{label} must have dtype float64")
    _require(value.ndim == 1 and len(value) > 0, f"{label} must be non-empty 1D")
    _require(np.all(np.isfinite(value)), f"{label} contains non-finite values")
    _require(
        bool(np.all((value >= 0.0) & (value <= 1.0))),
        f"{label} contains values outside [0, 1]",
    )
    return value.copy()


def _same_json(left: Any, right: Any) -> bool:
    return canonical_sha256(left) == canonical_sha256(right)


def _float_matches(left: Any, right: float, *, atol: float = 1e-12) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return bool(np.isclose(value, right, rtol=0.0, atol=atol))


def _preflight_run_completion(run_dir: Path, *, expected_seed: int) -> RunPreflight:
    """Verify completion and every output hash without opening metric payloads."""

    run_dir = run_dir.resolve()
    complete_path = run_dir / "training_complete.json"
    complete = _read_json(complete_path)
    _require(isinstance(complete, dict), f"Seed {expected_seed} completion is invalid")
    _require(complete.get("protocol_id") == PROTOCOL_ID, "Run protocol mismatch")
    _require(complete.get("status") == "TRAINING_COMPLETE", "Run is incomplete")
    run_id = str(complete.get("run_id", ""))
    _require(len(run_id) == 64, f"Seed {expected_seed} has invalid run_id")
    outputs = complete.get("outputs")
    _require(isinstance(outputs, dict), f"Seed {expected_seed} lacks output records")
    missing = sorted(REQUIRED_RUN_OUTPUTS - set(outputs))
    _require(not missing, f"Seed {expected_seed} lacks outputs: {', '.join(missing)}")
    for filename, record in outputs.items():
        _require(
            isinstance(filename, str) and Path(filename).name == filename,
            f"Seed {expected_seed} output name is unsafe: {filename!r}",
        )
        _verify_file_record(
            run_dir / filename,
            record,
            label=f"seed {expected_seed} output {filename}",
        )
    return RunPreflight(
        seed=expected_seed,
        directory=run_dir,
        run_id=run_id,
        complete=complete,
        complete_sha256=sha256_file(complete_path),
    )


def _load_run_artifact(preflight: RunPreflight) -> RunArtifact:
    """Open metrics only after every registered seed has passed preflight."""

    expected_seed = preflight.seed
    run_dir = preflight.directory
    run_id = preflight.run_id

    started = _read_json(run_dir / "training_started.json")
    summary = _read_json(run_dir / "training_summary.json")
    metrics = _read_json(run_dir / "selection_metrics.json")
    history = _read_json(run_dir / "training_history.json")
    for name, payload in (
        ("training_started", started),
        ("training_summary", summary),
        ("selection_metrics", metrics),
    ):
        _require(isinstance(payload, dict), f"Seed {expected_seed} {name} is invalid")
    _require(isinstance(history, list) and history, "Training history is empty")
    _require(started.get("status") == "TRAINING_STARTED", "Started status mismatch")
    _require(summary.get("status") == "FP32_STAGE_A_COMPLETE", "Summary status mismatch")
    _require(
        started.get("protocol_id") == summary.get("protocol_id") == PROTOCOL_ID,
        "Run protocol lineage disagrees",
    )
    _require(
        started.get("run_id") == summary.get("run_id") == run_id,
        "Run IDs disagree across completion artifacts",
    )
    fingerprint = started.get("fingerprint_payload")
    _require(isinstance(fingerprint, dict), "Started artifact lacks fingerprint payload")
    _require(
        _same_json(fingerprint, summary.get("fingerprint_payload")),
        "Started and summary fingerprints disagree",
    )
    try:
        expected_run_id = build_run_fingerprint(fingerprint)
    except (ValueError, TypeError, KeyError) as exc:
        raise ReplicationInvalid(f"Seed {expected_seed} fingerprint is invalid: {exc}") from exc
    _require(expected_run_id == run_id, "run_id does not match canonical fingerprint")
    configuration = fingerprint.get("configuration")
    _require(isinstance(configuration, dict), "Fingerprint lacks configuration")
    _require(
        _same_json(configuration, summary.get("configuration")),
        "Summary configuration differs from fingerprint configuration",
    )
    _require(
        int(configuration.get("seed", -1)) == expected_seed,
        f"Expected seed {expected_seed}, found {configuration.get('seed')!r}",
    )
    _require(
        _same_json(metrics, summary.get("selection")),
        "Selection metrics differ from the summary",
    )
    _require(summary.get("quantization") == "none", "FP32 run contains quantization")
    _require(summary.get("document_codes_written") is False, "Run wrote document codes")
    _require(summary.get("query_gate_present") is False, "Run contains a query gate")
    data_access = started.get("data_access")
    _require(isinstance(data_access, dict), "Run lacks data-access disclosure")
    _require(data_access.get("outer_validation") is False, "Run accessed outer validation")
    _require(data_access.get("closed_test") is False, "Run accessed closed test")
    environment = started.get("environment")
    _require(isinstance(environment, dict), "Run lacks environment record")

    base = _load_float_array(
        run_dir / "selection_base_per_query.float64.npy",
        label=f"seed {expected_seed} Base per-query array",
    )
    v2 = _load_float_array(
        run_dir / "selection_v2_2_per_query.float64.npy",
        label=f"seed {expected_seed} v2.2 per-query array",
    )
    _require(base.shape == v2.shape, "Base and v2.2 query arrays disagree")

    try:
        query_projection = np.load(
            run_dir / "query_projection.float32.npy", allow_pickle=False, mmap_mode="r"
        )
        document_projection = np.load(
            run_dir / "document_projection.float32.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
    except (OSError, ValueError) as exc:
        raise ReplicationInvalid("Cannot load learned projections") from exc
    rank = int(configuration.get("rank", -1))
    _require(
        query_projection.dtype == document_projection.dtype == np.float32,
        "Learned projections must be float32",
    )
    _require(
        query_projection.ndim == 2
        and query_projection.shape == document_projection.shape
        and query_projection.shape[1] == rank,
        "Learned projection shapes disagree with configured rank",
    )
    _require(
        bool(np.all(np.isfinite(query_projection)))
        and bool(np.all(np.isfinite(document_projection))),
        "Learned projections contain non-finite values",
    )

    base_mean = float(np.mean(base))
    v2_mean = float(np.mean(v2))
    _require(int(metrics.get("query_count", -1)) == len(v2), "Metric query count changed")
    _require(_float_matches(metrics.get("base_recall_at_10"), base_mean), "Base mean mismatch")
    _require(_float_matches(metrics.get("v2_2_fp32_recall_at_10"), v2_mean), "v2.2 mean mismatch")
    _require(
        _float_matches(metrics.get("gain_over_base"), v2_mean - base_mean),
        "Recorded Base gain does not match per-query arrays",
    )
    selected_epoch = int(summary.get("selected_epoch", -1))
    history_epochs = {int(row.get("epoch", -1)) for row in history if isinstance(row, dict)}
    _require(selected_epoch in history_epochs, "Selected epoch is absent from history")

    return RunArtifact(
        seed=expected_seed,
        directory=run_dir,
        run_id=run_id,
        fingerprint_payload=copy.deepcopy(fingerprint),
        configuration=copy.deepcopy(configuration),
        summary=copy.deepcopy(summary),
        metrics=copy.deepcopy(metrics),
        environment=copy.deepcopy(environment),
        base_per_query=base,
        v2_per_query=v2,
        complete_sha256=preflight.complete_sha256,
    )


def _normalized_fingerprint(run: RunArtifact) -> dict[str, Any]:
    payload = copy.deepcopy(run.fingerprint_payload)
    configuration = payload.get("configuration")
    _require(isinstance(configuration, dict), "Fingerprint lacks configuration")
    _require("seed" in configuration, "Fingerprint configuration lacks seed")
    configuration.pop("seed")
    return payload


def _load_replication_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    _require(isinstance(protocol, dict), "Replication protocol is not an object")
    _require(
        protocol.get("protocol_id") == REPLICATION_PROTOCOL_ID,
        "Replication protocol ID mismatch",
    )
    _require(protocol.get("parent_protocol_id") == PROTOCOL_ID, "Parent protocol mismatch")
    seed_policy = protocol.get("seed_policy")
    _require(isinstance(seed_policy, dict), "Replication protocol lacks seed policy")
    _require(
        tuple(seed_policy.get("registered_seeds", ())) == EXPECTED_SEEDS,
        "Replication protocol seeds changed",
    )
    _require(
        tuple(seed_policy.get("heldout_primary_seeds", ())) == HELDOUT_SEEDS,
        "Held-out seed policy changed",
    )
    bootstrap = protocol.get("paired_joint_query_bootstrap")
    _require(isinstance(bootstrap, dict), "Replication protocol lacks bootstrap contract")
    _require(int(bootstrap.get("resamples", -1)) == BOOTSTRAP_REPLICATES, "Bootstrap count changed")
    _require(int(bootstrap.get("bootstrap_seed", -1)) == BOOTSTRAP_SEED, "Bootstrap seed changed")
    heldout_thresholds = protocol.get("decision_procedure", {}).get(
        "heldout_primary_thresholds", {}
    )
    _require(
        _float_matches(
            heldout_thresholds.get("heldout_mean_gain_over_base_at_least"),
            MINIMUM_GAIN_OVER_BASE,
            atol=0.0,
        ),
        "Held-out Base threshold changed",
    )
    _require(
        _float_matches(
            heldout_thresholds.get("heldout_mean_gain_over_pca_at_least"),
            MINIMUM_GAIN_OVER_PCA,
            atol=0.0,
        ),
        "Held-out PCA threshold changed",
    )
    return protocol


def _verify_run_invariants(
    runs: dict[int, RunArtifact],
    *,
    protocol: dict[str, Any],
    train_manifest_path: Path,
    selection_manifest_path: Path,
    pca_basis_path: Path,
    pca_config_path: Path,
) -> dict[str, Any]:
    _require(tuple(sorted(runs)) == EXPECTED_SEEDS, "Exact seeds 42, 43, and 44 are required")
    _require(len({run.directory for run in runs.values()}) == 3, "Run directories must be unique")
    _require(len({run.run_id for run in runs.values()}) == 3, "Run IDs must be unique")
    reference = runs[42]
    normalized = _normalized_fingerprint(reference)
    for seed in EXPECTED_SEEDS[1:]:
        _require(
            _same_json(normalized, _normalized_fingerprint(runs[seed])),
            f"Seed {seed} fingerprint differs by more than the seed",
        )
    configuration = reference.configuration
    frozen_configuration = protocol.get("frozen_training_configuration")
    _require(isinstance(frozen_configuration, dict), "Protocol lacks frozen configuration")
    expected_configuration = copy.deepcopy(frozen_configuration)
    expected_configuration.pop("epoch_selection", None)
    expected_configuration["seed"] = 42
    _require(
        _same_json(configuration, expected_configuration),
        "Run configuration differs from the frozen replication protocol",
    )

    fingerprint = reference.fingerprint_payload
    lineage = protocol.get("execution_lineage")
    _require(isinstance(lineage, dict), "Protocol lacks execution lineage")
    registered_hashes = lineage.get("source_hashes")
    _require(isinstance(registered_hashes, dict), "Protocol lacks registered hashes")
    registered_commit = str(lineage.get("training_source_commit", ""))
    _require(
        fingerprint.get("source_commit") == registered_commit,
        "Run source commit differs from replication protocol",
    )
    actual_records = {
        "train_bundle_manifest_sha256": sha256_file(train_manifest_path),
        "selection_bundle_manifest_sha256": sha256_file(selection_manifest_path),
        "pca_basis_sha256": sha256_file(pca_basis_path),
        "pca_config_sha256": sha256_file(pca_config_path),
        "trainer_sha256": sha256_file(SCRIPT_DIR / "train_boundary_loss_sidecar_v2_2.py"),
        "core_sha256": sha256_file(SCRIPT_DIR / "rars_v2_2_core.py"),
    }
    for key, actual in actual_records.items():
        _require(fingerprint.get(key) == actual, f"Fingerprint {key} does not match the file")
    protocol_hash_map = {
        "trainer_sha256": actual_records["trainer_sha256"],
        "core_sha256": actual_records["core_sha256"],
        "inner_train_manifest_sha256": actual_records[
            "train_bundle_manifest_sha256"
        ],
        "inner_validation_manifest_sha256": actual_records[
            "selection_bundle_manifest_sha256"
        ],
        "pca_basis_sha256": actual_records["pca_basis_sha256"],
        "pca_config_sha256": actual_records["pca_config_sha256"],
    }
    for key, actual in protocol_hash_map.items():
        _require(registered_hashes.get(key) == actual, f"Registered {key} changed")
    control_hash_map = {
        "aggregator_sha256": sha256_file(Path(__file__).resolve()),
        "metric_helper_sha256": sha256_file(
            SCRIPT_DIR / "train_boundary_loss_sidecar.py"
        ),
    }
    for key, actual in control_hash_map.items():
        _require(registered_hashes.get(key) == actual, f"Registered {key} changed")
    audited_seed42 = protocol.get("audited_seed_42")
    _require(isinstance(audited_seed42, dict), "Protocol lacks audited seed-42 anchor")
    _require(reference.run_id == audited_seed42.get("run_id"), "Seed-42 run ID changed")
    verified_outputs = audited_seed42.get("verified_outputs")
    _require(isinstance(verified_outputs, dict), "Protocol lacks seed-42 output hashes")
    complete_outputs = _read_json(reference.directory / "training_complete.json").get(
        "outputs", {}
    )
    for filename, expected_record in verified_outputs.items():
        actual_record = complete_outputs.get(filename)
        _require(isinstance(actual_record, dict), f"Seed 42 lacks audited {filename}")
        _require(
            int(actual_record.get("bytes", -1)) == int(expected_record.get("bytes", -2))
            and actual_record.get("sha256") == expected_record.get("sha256"),
            f"Seed-42 audited output changed: {filename}",
        )
    return configuration


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_query_manifest(
    selection_bundle_dir: Path, manifest: dict[str, Any], *, query_count: int
) -> tuple[list[str], np.ndarray]:
    record = manifest.get("query_manifest")
    path = selection_bundle_dir / "query_manifest.json"
    _verify_file_record(path, record, label="selection query manifest")
    payload = _read_json(path)
    _require(isinstance(payload, dict), "Query manifest is not an object")
    _require(payload.get("role_id") == SELECTION_ROLE_ID, "Query manifest role changed")
    qids_raw = payload.get("query_ids")
    rows_raw = payload.get("query_rows")
    _require(isinstance(qids_raw, list) and isinstance(rows_raw, list), "Query manifest fields missing")
    qids = [str(value) for value in qids_raw]
    try:
        rows = np.asarray(rows_raw, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplicationInvalid("Query rows are invalid") from exc
    _require(len(qids) == len(rows) == query_count, "Query manifest length changed")
    _require(len(set(qids)) == query_count, "Query IDs are not unique")
    _require(len(set(rows.tolist())) == query_count, "Query rows are not unique")
    _require(bool(np.all(rows >= 0)), "Query rows contain negative values")
    _require(canonical_sha256(qids) == manifest.get("query_ids_sha256"), "Query ID hash changed")
    _require(_array_sha256(rows) == manifest.get("query_rows_sha256"), "Query-row hash changed")
    return qids, rows


def _nested_manifest_value(payload: dict[str, Any], key: str) -> Any:
    matches: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if key in value:
                matches.append(value[key])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    _require(matches, f"Environment manifest lacks required field {key}")
    first = matches[0]
    _require(
        all(_same_json(first, value) for value in matches[1:]),
        f"Environment manifest has conflicting {key} values",
    )
    return first


def _validate_environment_manifest(
    path: Path,
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    runs: dict[int, RunArtifact],
) -> dict[str, Any]:
    payload = _read_json(path)
    _require(isinstance(payload, dict), "Environment manifest is not an object")
    required = (
        "schema_version",
        "protocol_id",
        "training_source_commit",
        "seed42_run_id",
        "trainer_sha256",
        "core_sha256",
        "aggregator_sha256",
        "metric_helper_sha256",
        "python_version",
        "numpy_version",
        "torch_version",
        "torch_cuda_version",
        "device",
        "faiss_version",
        "gpu_name",
        "compute_capability",
        "cudnn_version",
        "cuda_driver_version",
        "deterministic_algorithms_supported_and_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cublas_workspace_config",
        "pip_freeze_sha256",
        "protocol_sha256",
    )
    values = {key: _nested_manifest_value(payload, key) for key in required}
    _require(int(values["schema_version"]) >= 1, "Environment schema version is invalid")
    _require(values["protocol_id"] == REPLICATION_PROTOCOL_ID, "Environment protocol mismatch")
    lineage = protocol["execution_lineage"]
    registered_hashes = lineage["source_hashes"]
    _require(
        values["training_source_commit"] == lineage["training_source_commit"],
        "Environment source commit changed",
    )
    _require(values["seed42_run_id"] == runs[42].run_id, "Environment seed42 run ID changed")
    _require(values["trainer_sha256"] == registered_hashes["trainer_sha256"], "Environment trainer hash changed")
    _require(values["core_sha256"] == registered_hashes["core_sha256"], "Environment core hash changed")
    _require(values["aggregator_sha256"] == registered_hashes["aggregator_sha256"], "Environment aggregator hash changed")
    _require(values["metric_helper_sha256"] == registered_hashes["metric_helper_sha256"], "Environment metric-helper hash changed")
    _require(values["protocol_sha256"] == sha256_file(protocol_path), "Environment protocol hash changed")

    environment_contract = protocol.get("execution_environment_contract")
    _require(
        isinstance(environment_contract, dict),
        "Protocol lacks execution environment contract",
    )
    known_versions = {
        "python_version": environment_contract.get("python_version"),
        "numpy_version": environment_contract.get("numpy_version"),
        "torch_version": environment_contract.get("torch_version"),
        "torch_cuda_version": environment_contract.get("torch_cuda_version"),
        "device": environment_contract.get("device"),
    }
    for key, expected in known_versions.items():
        _require(str(values[key]) == expected, f"Environment {key} differs from seed 42")
    for key in (
        "faiss_version",
        "gpu_name",
        "compute_capability",
        "cudnn_version",
        "cuda_driver_version",
        "cublas_workspace_config",
    ):
        _require(
            isinstance(values[key], str) and bool(values[key].strip()),
            f"Environment {key} must be an explicit non-empty string",
        )
    _require(
        str(environment_contract.get("gpu_name_must_contain"))
        in str(values["gpu_name"]),
        "Held-out replication GPU differs from the registered contract",
    )
    _require(
        str(values["compute_capability"])
        == str(environment_contract.get("compute_capability")),
        "Held-out replication compute capability differs from the registered contract",
    )
    for key in (
        "deterministic_algorithms_supported_and_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
    ):
        _require(type(values[key]) is bool, f"Environment {key} must be boolean")
    _require(
        values["deterministic_algorithms_supported_and_enabled"]
        is environment_contract.get(
            "deterministic_algorithms_supported_and_enabled"
        ),
        "Held-out replication requires deterministic algorithms",
    )
    _require(
        values["cudnn_benchmark"] is environment_contract.get("cudnn_benchmark"),
        "Held-out replication requires cudnn_benchmark=False",
    )
    _require(
        values["cublas_workspace_config"]
        == environment_contract.get("cublas_workspace_config"),
        "Held-out replication CUBLAS_WORKSPACE_CONFIG differs from the registered contract",
    )
    pip_hash = str(values["pip_freeze_sha256"])
    _require(
        len(pip_hash) == 64 and all(character in "0123456789abcdef" for character in pip_hash),
        "Environment pip_freeze_sha256 is invalid",
    )

    seed42_recorded_environment = runs[42].environment
    for seed, run in runs.items():
        recorded = run.environment
        _require(
            _same_json(recorded, seed42_recorded_environment),
            f"Seed {seed} full training environment differs from seed 42",
        )
        _require(
            str(recorded.get("python", "")).split()[0] == values["python_version"],
            f"Seed {seed} Python version differs from environment manifest",
        )
        for manifest_key, run_key in (
            ("numpy_version", "numpy"),
            ("torch_version", "torch"),
            ("torch_cuda_version", "cuda"),
            ("device", "device"),
        ):
            _require(
                str(recorded.get(run_key)) == str(values[manifest_key]),
                f"Seed {seed} {run_key} differs from environment manifest",
            )
    reference_environment = runs[42].environment
    for seed in HELDOUT_SEEDS:
        _require(
            _same_json(reference_environment, runs[seed].environment),
            f"Seed {seed} recorded environment differs from seed 42",
        )
    return payload


def _recompute_comparators(
    selection_bundle: dict[str, Any],
    *,
    pca_basis_path: Path,
    pca_config_path: Path,
    configuration: dict[str, Any],
) -> dict[str, np.ndarray]:
    rank = int(configuration["rank"])
    top_b = int(configuration["top_b"])
    final_k = int(configuration["final_k"])
    batch_size = int(configuration["score_batch_size"])
    max_correction = float(configuration["max_correction"])
    dimension = int(selection_bundle["queries"].shape[1])
    try:
        query_init, document_init, alpha = load_pca_warm_start(
            pca_basis_path,
            pca_config_path,
            dimension=dimension,
            rank=rank,
            top_b=top_b,
        )
        basis = np.asarray(np.load(pca_basis_path, allow_pickle=False), dtype=np.float32)
        base_scores = np.asarray(selection_bundle["ann_scores"], dtype=np.float32)
        pca_scores = pca_fp32_scores(
            selection_bundle,
            basis,
            alpha=alpha,
            top_b=top_b,
            batch_size=batch_size,
        )
        bounded_scores = score_candidates_fp32(
            selection_bundle,
            query_init,
            document_init,
            top_b=top_b,
            max_correction=max_correction,
            batch_size=batch_size,
        )
        labels = selection_bundle["labels"]
        counts = selection_bundle["relevant_counts"]
        return {
            "base": recall_at_k_per_query(base_scores, labels, counts, k=final_k),
            "pca_fp32": recall_at_k_per_query(pca_scores, labels, counts, k=final_k),
            "pca_bounded": recall_at_k_per_query(
                bounded_scores, labels, counts, k=final_k
            ),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReplicationInvalid(f"Comparator reconstruction failed: {exc}") from exc


def _validate_run_metrics(
    runs: dict[int, RunArtifact], comparators: dict[str, np.ndarray]
) -> None:
    base = comparators["base"]
    pca = comparators["pca_fp32"]
    bounded = comparators["pca_bounded"]
    for seed, run in runs.items():
        _require(run.v2_per_query.shape == base.shape, f"Seed {seed} query count changed")
        _require(
            np.array_equal(run.base_per_query, base),
            f"Seed {seed} Base array differs from frozen recomputation",
        )
        metrics = run.metrics
        v2_mean = float(np.mean(run.v2_per_query))
        expected = {
            "base_recall_at_10": float(np.mean(base)),
            "pca_fp32_recall_at_10": float(np.mean(pca)),
            "pca_parameter_warm_start_bounded_recall_at_10": float(np.mean(bounded)),
            "v2_2_fp32_recall_at_10": v2_mean,
            "gain_over_base": v2_mean - float(np.mean(base)),
            "gain_over_pca_fp32": v2_mean - float(np.mean(pca)),
        }
        for key, value in expected.items():
            _require(
                _float_matches(metrics.get(key), value),
                f"Seed {seed} recorded {key} differs from frozen recomputation",
            )
        passes_base = expected["gain_over_base"] >= MINIMUM_GAIN_OVER_BASE
        passes_pca = expected["gain_over_pca_fp32"] >= MINIMUM_GAIN_OVER_PCA
        _require(metrics.get("passes_base_gain_gate") is passes_base, "Base gate flag changed")
        _require(metrics.get("passes_pca_gain_gate") is passes_pca, "PCA gate flag changed")


def _contrast_counts(delta: np.ndarray) -> dict[str, int]:
    return {
        "improved_queries": int(np.sum(delta > 0.0)),
        "harmed_queries": int(np.sum(delta < 0.0)),
        "unchanged_queries": int(np.sum(delta == 0.0)),
    }


def build_per_seed_metrics(
    runs: dict[int, RunArtifact], comparators: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = comparators["base"]
    pca = comparators["pca_fp32"]
    for seed in EXPECTED_SEEDS:
        run = runs[seed]
        base_delta = run.v2_per_query - base
        pca_delta = run.v2_per_query - pca
        gain_base = float(np.mean(base_delta))
        gain_pca = float(np.mean(pca_delta))
        rows.append({
            "seed": seed,
            "run_id": run.run_id,
            "selected_epoch": int(run.summary["selected_epoch"]),
            "v2_2_fp32_recall_at_10": float(np.mean(run.v2_per_query)),
            "gain_over_base": gain_base,
            "gain_over_pca_fp32": gain_pca,
            "passes_base_gain_gate": gain_base >= MINIMUM_GAIN_OVER_BASE,
            "passes_pca_gain_gate": gain_pca >= MINIMUM_GAIN_OVER_PCA,
            "joint_pass": (
                gain_base >= MINIMUM_GAIN_OVER_BASE
                and gain_pca >= MINIMUM_GAIN_OVER_PCA
            ),
            "vs_base": _contrast_counts(base_delta),
            "vs_pca_fp32": _contrast_counts(pca_delta),
        })
    return rows


def _scalar_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(len(array) >= 2, "A seed summary needs at least two values")
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def build_group_summaries(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {int(row["seed"]): row for row in per_seed}

    def summarize(seeds: tuple[int, ...]) -> dict[str, Any]:
        return {
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "v2_2_fp32_recall_at_10": _scalar_summary([
                float(indexed[seed]["v2_2_fp32_recall_at_10"]) for seed in seeds
            ]),
            "gain_over_base": _scalar_summary([
                float(indexed[seed]["gain_over_base"]) for seed in seeds
            ]),
            "gain_over_pca_fp32": _scalar_summary([
                float(indexed[seed]["gain_over_pca_fp32"]) for seed in seeds
            ]),
        }

    return {
        "all_seeds": summarize(EXPECTED_SEEDS),
        "heldout_replication_seeds": summarize(HELDOUT_SEEDS),
    }


def paired_bootstrap_shared(
    v2_by_seed: np.ndarray,
    base: np.ndarray,
    pca: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 256,
) -> tuple[dict[str, Any], np.ndarray]:
    """Bootstrap every paired contrast with the same sampled query indices."""

    values = np.asarray(v2_by_seed, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    pca = np.asarray(pca, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] == 3, "Need a [3, Q] seed array")
    _require(base.shape == pca.shape == (values.shape[1],), "Comparator shapes disagree")
    _require(replicates > 0 and chunk_size > 0, "Bootstrap sizes must be positive")
    contrasts: dict[str, tuple[np.ndarray, float]] = {}
    for index, seed_value in enumerate(EXPECTED_SEEDS):
        contrasts[f"seed_{seed_value}_minus_base"] = (
            values[index] - base,
            MINIMUM_GAIN_OVER_BASE,
        )
        contrasts[f"seed_{seed_value}_minus_pca_fp32"] = (
            values[index] - pca,
            MINIMUM_GAIN_OVER_PCA,
        )
    heldout_mean = np.mean(values[1:], axis=0)
    all_mean = np.mean(values, axis=0)
    contrasts.update({
        "heldout_mean_minus_base": (heldout_mean - base, MINIMUM_GAIN_OVER_BASE),
        "heldout_mean_minus_pca_fp32": (heldout_mean - pca, MINIMUM_GAIN_OVER_PCA),
        "all_seed_mean_minus_base": (all_mean - base, MINIMUM_GAIN_OVER_BASE),
        "all_seed_mean_minus_pca_fp32": (all_mean - pca, MINIMUM_GAIN_OVER_PCA),
    })
    samples = {
        name: np.empty(replicates, dtype=np.float64) for name in contrasts
    }
    rng = np.random.default_rng(seed)
    query_count = values.shape[1]
    written = 0
    while written < replicates:
        count = min(chunk_size, replicates - written)
        indices = rng.integers(0, query_count, size=(count, query_count))
        for name, (delta, _) in contrasts.items():
            samples[name][written:written + count] = delta[indices].mean(axis=1)
        written += count
    output: dict[str, Any] = {}
    contrast_order = list(contrasts)
    statistics = np.column_stack([samples[name] for name in contrast_order])
    for name, (delta, threshold) in contrasts.items():
        distribution = samples[name]
        low, high = np.quantile(
            distribution, [0.025, 0.975], method="linear"
        )
        output[name] = {
            "difference": float(np.mean(delta)),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "probability_difference_gt_zero": float(np.mean(distribution > 0.0)),
            "practical_threshold": threshold,
            "probability_difference_gt_practical_threshold": float(
                np.mean(distribution > threshold)
            ),
        }
    summary = {
        "schema_version": 1,
        "sampling_unit": "query",
        "query_count": query_count,
        "replicates": replicates,
        "seed": seed,
        "shared_query_indices_across_all_contrasts": True,
        "seed_uncertainty_included": False,
        "interval": "two-sided percentile 95% with numpy method=linear",
        "statistics_columns": contrast_order,
        "contrasts": output,
    }
    return summary, statistics


def make_replication_decision(
    per_seed: list[dict[str, Any]],
    groups: dict[str, Any],
    bootstrap: dict[str, Any],
    *,
    query_count: int,
) -> dict[str, Any]:
    indexed = {int(row["seed"]): row for row in per_seed}
    support_required = int(math.ceil(0.01 * query_count))
    joint_pass_count = int(sum(bool(row["joint_pass"]) for row in per_seed))
    heldout_each_positive_direction = all(
        float(indexed[seed][contrast]) > 0.0
        for seed in HELDOUT_SEEDS
        for contrast in ("gain_over_base", "gain_over_pca_fp32")
    )
    heldout_each_positive_support = all(
        int(indexed[seed][contrast]["improved_queries"]) >= support_required
        for seed in HELDOUT_SEEDS
        for contrast in ("vs_base", "vs_pca_fp32")
    )
    heldout = groups["heldout_replication_seeds"]
    all_seeds = groups["all_seeds"]
    contrasts = bootstrap["contrasts"]
    conditions = {
        "all_three_seeds_joint_pass": joint_pass_count == len(EXPECTED_SEEDS),
        "heldout_seeds_each_positive_vs_both_baselines": heldout_each_positive_direction,
        "heldout_seeds_each_have_required_positive_query_support": (
            heldout_each_positive_support
        ),
        "heldout_mean_meets_base_threshold": (
            heldout["gain_over_base"]["mean"] >= MINIMUM_GAIN_OVER_BASE
        ),
        "heldout_mean_meets_pca_threshold": (
            heldout["gain_over_pca_fp32"]["mean"] >= MINIMUM_GAIN_OVER_PCA
        ),
        "all_seed_mean_meets_base_threshold": (
            all_seeds["gain_over_base"]["mean"] >= MINIMUM_GAIN_OVER_BASE
        ),
        "all_seed_mean_meets_pca_threshold": (
            all_seeds["gain_over_pca_fp32"]["mean"] >= MINIMUM_GAIN_OVER_PCA
        ),
        "heldout_base_ci95_low_gt_zero": (
            contrasts["heldout_mean_minus_base"]["ci95_low"] > 0.0
        ),
        "heldout_pca_ci95_low_gt_zero": (
            contrasts["heldout_mean_minus_pca_fp32"]["ci95_low"] > 0.0
        ),
    }
    heldout_primary_thresholds_pass = (
        conditions["heldout_mean_meets_base_threshold"]
        and conditions["heldout_mean_meets_pca_threshold"]
    )
    if all(conditions.values()):
        decision = STABLE_DECISION
        qat_authorized = True
    elif heldout_primary_thresholds_pass:
        decision = UNSTABLE_DECISION
        qat_authorized = False
    else:
        decision = STOP_DECISION
        qat_authorized = False
    return {
        "schema_version": 1,
        "decision": decision,
        "qat_protocol_definition_authorized": qat_authorized,
        "seed_42_role": "observed_stage_a_development_seed",
        "primary_replication_seeds": list(HELDOUT_SEEDS),
        "joint_pass_count": joint_pass_count,
        "joint_pass_required": len(EXPECTED_SEEDS),
        "query_count": query_count,
        "positive_support_fraction": 0.01,
        "positive_support_queries_required_per_heldout_seed_and_contrast": (
            support_required
        ),
        "conditions": conditions,
        "failed_conditions": [name for name, passed in conditions.items() if not passed],
    }


def _atomic_write_per_query_csv(
    path: Path,
    qids: list[str],
    query_rows: np.ndarray,
    comparators: dict[str, np.ndarray],
    v2_by_seed: np.ndarray,
) -> None:
    fields = [
        "query_id",
        "query_row",
        "base_recall_at_10",
        "pca_fp32_recall_at_10",
        "pca_bounded_recall_at_10",
    ]
    for seed in EXPECTED_SEEDS:
        fields.extend([
            f"v2_seed_{seed}_recall_at_10",
            f"v2_seed_{seed}_minus_base",
            f"v2_seed_{seed}_minus_pca_fp32",
        ])
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, qid in enumerate(qids):
            row: dict[str, Any] = {
                "query_id": qid,
                "query_row": int(query_rows[index]),
                "base_recall_at_10": float(comparators["base"][index]),
                "pca_fp32_recall_at_10": float(comparators["pca_fp32"][index]),
                "pca_bounded_recall_at_10": float(comparators["pca_bounded"][index]),
            }
            for seed_index, seed in enumerate(EXPECTED_SEEDS):
                value = float(v2_by_seed[seed_index, index])
                row[f"v2_seed_{seed}_recall_at_10"] = value
                row[f"v2_seed_{seed}_minus_base"] = (
                    value - float(comparators["base"][index])
                )
                row[f"v2_seed_{seed}_minus_pca_fp32"] = (
                    value - float(comparators["pca_fp32"][index])
                )
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_per_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "run_id",
        "selected_epoch",
        "v2_2_fp32_recall_at_10",
        "gain_over_base",
        "gain_over_pca_fp32",
        "passes_base_gain_gate",
        "passes_pca_gain_gate",
        "joint_pass",
        "improved_queries_vs_base",
        "harmed_queries_vs_base",
        "unchanged_queries_vs_base",
        "improved_queries_vs_pca_fp32",
        "harmed_queries_vs_pca_fp32",
        "unchanged_queries_vs_pca_fp32",
    ]
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({
                "seed": item["seed"],
                "run_id": item["run_id"],
                "selected_epoch": item["selected_epoch"],
                "v2_2_fp32_recall_at_10": item["v2_2_fp32_recall_at_10"],
                "gain_over_base": item["gain_over_base"],
                "gain_over_pca_fp32": item["gain_over_pca_fp32"],
                "passes_base_gain_gate": item["passes_base_gain_gate"],
                "passes_pca_gain_gate": item["passes_pca_gain_gate"],
                "joint_pass": item["joint_pass"],
                "improved_queries_vs_base": item["vs_base"]["improved_queries"],
                "harmed_queries_vs_base": item["vs_base"]["harmed_queries"],
                "unchanged_queries_vs_base": item["vs_base"]["unchanged_queries"],
                "improved_queries_vs_pca_fp32": item["vs_pca_fp32"]["improved_queries"],
                "harmed_queries_vs_pca_fp32": item["vs_pca_fp32"]["harmed_queries"],
                "unchanged_queries_vs_pca_fp32": item["vs_pca_fp32"]["unchanged_queries"],
            })
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _prepare_output_directory(output_dir: Path, inputs: list[Path]) -> None:
    resolved = output_dir.resolve()
    _require(resolved not in {path.resolve() for path in inputs}, "Output overlaps an input")
    if output_dir.exists():
        _require(output_dir.is_dir(), "Output path is not a directory")
        _require(not any(output_dir.iterdir()), "Output directory must be empty")
    else:
        output_dir.mkdir(parents=True)


def aggregate_replication(
    *,
    train_bundle_dir: Path,
    selection_bundle_dir: Path,
    pca_basis_path: Path,
    pca_config_path: Path,
    replication_protocol_path: Path,
    environment_manifest_path: Path,
    run_dirs: dict[int, Path],
    output_dir: Path,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate, aggregate, and persist one exact 42/43/44 replication."""

    try:
        train_bundle_dir = Path(train_bundle_dir).resolve()
        selection_bundle_dir = Path(selection_bundle_dir).resolve()
        pca_basis_path = Path(pca_basis_path).resolve()
        pca_config_path = Path(pca_config_path).resolve()
        replication_protocol_path = Path(replication_protocol_path).resolve()
        environment_manifest_path = Path(environment_manifest_path).resolve()
        output_dir = Path(output_dir)
        _require(tuple(sorted(run_dirs)) == EXPECTED_SEEDS, "Exact run seed keys are required")
        protocol = _load_replication_protocol(replication_protocol_path)
        _require(
            bootstrap_replicates == BOOTSTRAP_REPLICATES,
            "Replication requires exactly 20,000 bootstrap replicates",
        )
        _require(
            bootstrap_seed == BOOTSTRAP_SEED,
            "Replication bootstrap seed changed",
        )
        # Preserve the sealed-pair rule: verify every completion and output hash
        # before opening either held-out seed's metric payload.
        preflights = {
            seed: _preflight_run_completion(Path(run_dirs[seed]), expected_seed=seed)
            for seed in EXPECTED_SEEDS
        }
        runs = {
            seed: _load_run_artifact(preflights[seed])
            for seed in EXPECTED_SEEDS
        }
        train_manifest_path = train_bundle_dir / "v2_2_manifest.json"
        selection_manifest_path = selection_bundle_dir / "v2_2_manifest.json"
        train_bundle = load_bundle(train_bundle_dir, expected_role_id=FIT_ROLE_ID)
        selection_bundle = load_bundle(
            selection_bundle_dir, expected_role_id=SELECTION_ROLE_ID
        )
        manifest = selection_bundle["manifest"]
        configuration = _verify_run_invariants(
            runs,
            protocol=protocol,
            train_manifest_path=train_manifest_path,
            selection_manifest_path=selection_manifest_path,
            pca_basis_path=pca_basis_path,
            pca_config_path=pca_config_path,
        )
        lineage = protocol["execution_lineage"]
        _require(
            train_bundle["manifest"].get("split_audit_sha256")
            == manifest.get("split_audit_sha256")
            == lineage.get("split_audit_sha256"),
            "Re-materialized bundle split-audit hash changed",
        )
        _require(
            train_bundle["manifest"].get("source_commit")
            == manifest.get("source_commit")
            == lineage.get("training_source_commit"),
            "Re-materialized bundle source commit differs from frozen lineage",
        )
        split_audit_path = selection_bundle_dir.parent / "v2_2_split_audit.json"
        _require(split_audit_path.is_file(), "Re-materialized split-audit file is missing")
        _require(
            sha256_file(split_audit_path) == lineage.get("split_audit_sha256"),
            "Re-materialized split-audit file hash changed",
        )
        environment_manifest = _validate_environment_manifest(
            environment_manifest_path,
            protocol_path=replication_protocol_path,
            protocol=protocol,
            runs=runs,
        )
        query_count = int(selection_bundle["ann_scores"].shape[0])
        _require(
            query_count == int(protocol["metric_contract"]["query_count"]),
            "Selection query count differs from replication protocol",
        )
        qids, query_rows = _load_query_manifest(
            selection_bundle_dir, manifest, query_count=query_count
        )
        comparators = _recompute_comparators(
            selection_bundle,
            pca_basis_path=pca_basis_path,
            pca_config_path=pca_config_path,
            configuration=configuration,
        )
        _validate_run_metrics(runs, comparators)
        v2_by_seed = np.stack(
            [runs[seed].v2_per_query for seed in EXPECTED_SEEDS]
        ).astype(np.float64, copy=False)
        per_seed = build_per_seed_metrics(runs, comparators)
        groups = build_group_summaries(per_seed)
        bootstrap, bootstrap_statistics = paired_bootstrap_shared(
            v2_by_seed,
            comparators["base"],
            comparators["pca_fp32"],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        decision = make_replication_decision(
            per_seed, groups, bootstrap, query_count=query_count
        )

        _prepare_output_directory(
            output_dir,
            [
                train_bundle_dir,
                selection_bundle_dir,
                pca_basis_path,
                pca_config_path,
                replication_protocol_path,
                environment_manifest_path,
            ]
            + [runs[seed].directory for seed in EXPECTED_SEEDS],
        )
        input_manifest = {
            "schema_version": 1,
            "protocol_id": REPLICATION_PROTOCOL_ID,
            "parent_protocol_id": PROTOCOL_ID,
            "replication_protocol": file_record(replication_protocol_path),
            "aggregator": file_record(Path(__file__).resolve()),
            "metric_helper": file_record(
                SCRIPT_DIR / "train_boundary_loss_sidecar.py"
            ),
            "environment_manifest": file_record(environment_manifest_path),
            "split_audit": file_record(split_audit_path),
            "train_bundle_manifest": file_record(train_manifest_path),
            "selection_bundle_manifest": file_record(selection_manifest_path),
            "query_manifest": file_record(selection_bundle_dir / "query_manifest.json"),
            "pca_basis": file_record(pca_basis_path),
            "pca_config": file_record(pca_config_path),
            "runs": {
                str(seed): {
                    "directory": str(runs[seed].directory),
                    "run_id": runs[seed].run_id,
                    "training_complete_sha256": runs[seed].complete_sha256,
                    "fingerprint_payload": runs[seed].fingerprint_payload,
                }
                for seed in EXPECTED_SEEDS
            },
            "fingerprints_invariant_except_seed": True,
            "bundle_rebuild_validation": (
                "accepted only because rebuilt manifests, split audit, and all "
                "bundle file records exactly match the frozen hashes"
            ),
            "environment": environment_manifest,
        }
        summary = {
            "schema_version": 1,
            "protocol_id": REPLICATION_PROTOCOL_ID,
            "parent_protocol_id": PROTOCOL_ID,
            "status": "FP32_THREE_SEED_REPLICATION_COMPLETE",
            "decision": decision["decision"],
            "evidence_boundary": {
                "split": "inner_validation",
                "development_only": True,
                "seed_42_observed_before_replication": True,
                "primary_replication_seeds": list(HELDOUT_SEEDS),
            },
            "query_count": query_count,
            "comparators": {
                "base_recall_at_10": float(np.mean(comparators["base"])),
                "direct_pca_fp32_recall_at_10": float(
                    np.mean(comparators["pca_fp32"])
                ),
                "bounded_pca_warm_start_recall_at_10": float(
                    np.mean(comparators["pca_bounded"])
                ),
                "direct_pca_is_decision_comparator": True,
                "bounded_pca_is_ablation_only": True,
            },
            "per_seed": per_seed,
            "groups": groups,
            "decision_conditions": decision["conditions"],
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "shared_query_indices_across_all_contrasts": True,
            },
        }

        output_names: list[str] = []

        def save_array(filename: str, value: np.ndarray) -> None:
            _atomic_save_npy(output_dir / filename, np.asarray(value))
            output_names.append(filename)

        save_array("selection_base_per_query.float64.npy", comparators["base"])
        save_array("selection_pca_fp32_per_query.float64.npy", comparators["pca_fp32"])
        save_array(
            "selection_pca_bounded_per_query.float64.npy",
            comparators["pca_bounded"],
        )
        for index, seed in enumerate(EXPECTED_SEEDS):
            save_array(
                f"selection_v2_2_seed_{seed}_per_query.float64.npy",
                v2_by_seed[index],
            )
        save_array("selection_v2_2_by_seed.float64.npy", v2_by_seed)
        save_array(
            "gain_over_base_by_seed.float64.npy",
            v2_by_seed - comparators["base"][None, :],
        )
        save_array(
            "gain_over_pca_fp32_by_seed.float64.npy",
            v2_by_seed - comparators["pca_fp32"][None, :],
        )
        save_array(
            "selection_v2_2_heldout_mean_per_query.float64.npy",
            np.mean(v2_by_seed[1:], axis=0),
        )
        save_array(
            "selection_v2_2_all_seed_mean_per_query.float64.npy",
            np.mean(v2_by_seed, axis=0),
        )
        save_array(
            "paired_bootstrap_statistics.float64.npy",
            bootstrap_statistics,
        )

        _atomic_write_per_query_csv(
            output_dir / "per_query_replication.csv",
            qids,
            query_rows,
            comparators,
            v2_by_seed,
        )
        output_names.append("per_query_replication.csv")
        _atomic_write_per_seed_csv(output_dir / "per_seed_metrics.csv", per_seed)
        output_names.append("per_seed_metrics.csv")
        for filename, payload in (
            ("replication_input_manifest.json", input_manifest),
            ("per_seed_metrics.json", {"schema_version": 1, "rows": per_seed}),
            ("paired_bootstrap.json", bootstrap),
            ("replication_decision.json", decision),
            ("replication_summary.json", summary),
        ):
            _atomic_write_json(output_dir / filename, payload)
            output_names.append(filename)
        complete = {
            "schema_version": 1,
            "protocol_id": REPLICATION_PROTOCOL_ID,
            "status": "REPLICATION_COMPLETE",
            "decision": decision["decision"],
            "outputs": {
                filename: file_record(output_dir / filename)
                for filename in output_names
            },
        }
        _atomic_write_json(output_dir / "replication_complete.json", complete)
        return summary
    except ReplicationInvalid:
        raise
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise ReplicationInvalid(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen RARS-v2.2 seeds 42/43/44 on CPU."
    )
    parser.add_argument("--train-bundle-dir", required=True, type=Path)
    parser.add_argument("--selection-bundle-dir", required=True, type=Path)
    parser.add_argument("--pca-basis", required=True, type=Path)
    parser.add_argument("--pca-config", required=True, type=Path)
    parser.add_argument(
        "--replication-protocol",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "protocols"
            / "rars_v2_2_fp32_replication_v1.json"
        ),
    )
    parser.add_argument("--environment-manifest", required=True, type=Path)
    parser.add_argument("--seed-42-dir", required=True, type=Path)
    parser.add_argument("--seed-43-dir", required=True, type=Path)
    parser.add_argument("--seed-44-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _write_invalid_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    if output_dir.exists() and output_dir.is_dir() and any(output_dir.iterdir()):
        path = output_dir.with_name(output_dir.name + ".invalid.json")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "replication_invalid.json"
    _atomic_write_json(path, payload)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = aggregate_replication(
            train_bundle_dir=args.train_bundle_dir,
            selection_bundle_dir=args.selection_bundle_dir,
            pca_basis_path=args.pca_basis,
            pca_config_path=args.pca_config,
            replication_protocol_path=args.replication_protocol,
            environment_manifest_path=args.environment_manifest,
            run_dirs={
                42: args.seed_42_dir,
                43: args.seed_43_dir,
                44: args.seed_44_dir,
            },
            output_dir=args.output_dir,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
    except ReplicationInvalid as exc:
        payload = {
            "schema_version": 1,
            "protocol_id": REPLICATION_PROTOCOL_ID,
            "parent_protocol_id": PROTOCOL_ID,
            "status": "REPLICATION_INVALID",
            "decision": "INVALID",
            "qat_protocol_definition_authorized": False,
            "error_type": "ReplicationInvalid",
            "error": str(exc),
        }
        report_path = _write_invalid_report(args.output_dir, payload)
        payload["invalid_report"] = str(report_path)
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
