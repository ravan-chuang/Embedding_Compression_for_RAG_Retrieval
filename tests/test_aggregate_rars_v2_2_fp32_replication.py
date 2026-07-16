from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_rars_v2_2_fp32_replication.py"
SPEC = importlib.util.spec_from_file_location("rars_v2_2_replication", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SOURCE_COMMIT = "b" * 40


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def _array_sha256(value: np.ndarray) -> str:
    return MODULE._array_sha256(np.asarray(value))


def _make_bundle(
    root: Path,
    role: str,
    *,
    query_count: int,
    row_offset: int,
    split_audit_sha256: str,
) -> Path:
    directory = root / role
    directory.mkdir(parents=True)
    qids = [f"{role}-q{index}" for index in range(query_count)]
    rows = np.arange(row_offset, row_offset + query_count, dtype=np.int64)
    query_manifest = {
        "role_id": role,
        "query_ids": qids,
        "query_rows": rows.tolist(),
    }
    _write_json(directory / "query_manifest.json", query_manifest)

    scores = np.tile(
        np.linspace(1.0, 0.0, 40, dtype=np.float32), (query_count, 1)
    )
    labels = np.zeros((query_count, 40), dtype=np.uint8)
    labels[:, :10] = 1
    arrays = {
        "query_vectors.float32.npy": np.zeros((query_count, 16), dtype=np.float32),
        "ann_rows.int64.npy": np.tile(
            np.arange(40, dtype=np.int64), (query_count, 1)
        ),
        "ann_scores.float32.npy": scores,
        "candidate_relevance.uint8.npy": labels,
        "relevant_counts.int32.npy": np.full(query_count, 20, dtype=np.int32),
        "ann_residual_rows.int64.npy": np.tile(
            np.arange(40, dtype=np.int64), (query_count, 1)
        ),
        "candidate_residuals.float32.npy": np.zeros((40, 16), dtype=np.float32),
    }
    for filename, value in arrays.items():
        _save(directory / filename, value)
    manifest = {
        "schema_version": 1,
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": SOURCE_COMMIT,
        "source_builder_sha256": "1" * 64,
        "bundle_freezer_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "role_id": role,
        "split_role": "train" if role == MODULE.FIT_ROLE_ID else "validation",
        "evidence_status": "DEVELOPMENT_ONLY",
        "query_count": query_count,
        "candidate_count": 40,
        "query_ids_sha256": MODULE.canonical_sha256(qids),
        "query_rows_sha256": _array_sha256(rows),
        "split_audit_sha256": split_audit_sha256,
        "source_bundle_manifest_sha256": "4" * 64,
        "source_bundle_protocol_id": "synthetic",
        "query_manifest": MODULE.file_record(directory / "query_manifest.json"),
        "files": {
            filename: MODULE.file_record(directory / filename)
            for filename in arrays
        },
        "data_access": {
            "outer_outcomes_used": False,
            "closed_test_relevance_values_used": False,
        },
    }
    _write_json(directory / "v2_2_manifest.json", manifest)
    return directory


def _configuration(seed: int, *, learning_rate: float = 0.0001) -> dict[str, Any]:
    return {
        "rank": 16,
        "top_b": 40,
        "final_k": 10,
        "epochs": 10,
        "batch_size": 2048,
        "score_batch_size": 256,
        "max_negatives_per_positive": 8,
        "promotion_mix": 0.8,
        "minimum_margin": 0.0001,
        "margin_multiplier": 1.0,
        "learning_rate": learning_rate,
        "weight_decay": 0.0001,
        "correction_l2": 0.001,
        "max_correction": 0.05,
        "max_grad_norm": 5.0,
        "seed": seed,
        "minimum_gain_over_base": MODULE.MINIMUM_GAIN_OVER_BASE,
        "minimum_gain_over_pca": MODULE.MINIMUM_GAIN_OVER_PCA,
        "device": "cuda",
    }


def _environment() -> dict[str, Any]:
    return {
        "python": "3.12.13 (synthetic)",
        "numpy": "1.26.4",
        "torch": "2.11.0+cu128",
        "cuda": "12.8",
        "device": "cuda",
    }


def _make_run(
    root: Path,
    seed: int,
    *,
    query_count: int,
    train_manifest_sha256: str,
    selection_manifest_sha256: str,
    pca_basis_sha256: str,
    pca_config_sha256: str,
    v2_value: float = 0.52,
    learning_rate: float = 0.0001,
) -> tuple[Path, dict[str, Any]]:
    directory = root / f"seed-{seed}"
    directory.mkdir(parents=True)
    config = _configuration(seed, learning_rate=learning_rate)
    fingerprint = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": SOURCE_COMMIT,
        "trainer_sha256": MODULE.sha256_file(
            ROOT / "scripts" / "train_boundary_loss_sidecar_v2_2.py"
        ),
        "core_sha256": MODULE.sha256_file(ROOT / "scripts" / "rars_v2_2_core.py"),
        "train_bundle_manifest_sha256": train_manifest_sha256,
        "selection_bundle_manifest_sha256": selection_manifest_sha256,
        "pca_basis_sha256": pca_basis_sha256,
        "pca_config_sha256": pca_config_sha256,
        "configuration": config,
    }
    run_id = MODULE.build_run_fingerprint(fingerprint)
    base = np.full(query_count, 0.5, dtype=np.float64)
    v2 = np.full(query_count, v2_value, dtype=np.float64)
    base_mean = float(np.mean(base))
    v2_mean = float(np.mean(v2))
    metrics = {
        "query_count": query_count,
        "base_recall_at_10": base_mean,
        "pca_fp32_recall_at_10": base_mean,
        "pca_parameter_warm_start_bounded_recall_at_10": base_mean,
        "v2_2_fp32_recall_at_10": v2_mean,
        "gain_over_base": v2_mean - base_mean,
        "gain_over_pca_fp32": v2_mean - base_mean,
        "improved_queries_vs_base": query_count,
        "harmed_queries_vs_base": 0,
        "unchanged_queries_vs_base": 0,
        "minimum_required_gain_over_base": MODULE.MINIMUM_GAIN_OVER_BASE,
        "minimum_required_gain_over_pca": MODULE.MINIMUM_GAIN_OVER_PCA,
        "passes_base_gain_gate": v2_mean - base_mean >= MODULE.MINIMUM_GAIN_OVER_BASE,
        "passes_pca_gain_gate": v2_mean - base_mean >= MODULE.MINIMUM_GAIN_OVER_PCA,
        "decision": "GO_TO_THREE_SEED_FP32_REPLICATION",
    }
    started = {
        "schema_version": 1,
        "protocol_id": MODULE.PROTOCOL_ID,
        "status": "TRAINING_STARTED",
        "run_id": run_id,
        "fingerprint_payload": fingerprint,
        "repo": {"head": SOURCE_COMMIT, "clean": True},
        "environment": _environment(),
        "data_access": {
            "inner_train": True,
            "inner_validation": True,
            "outer_validation": False,
            "closed_test": False,
        },
    }
    history = [
        {"epoch": 0, "selection_fp32_recall_at_10": base_mean},
        {"epoch": 1, "selection_fp32_recall_at_10": v2_mean},
    ]
    summary = {
        "schema_version": 1,
        "protocol_id": MODULE.PROTOCOL_ID,
        "status": "FP32_STAGE_A_COMPLETE",
        "decision": metrics["decision"],
        "run_id": run_id,
        "source_commit": SOURCE_COMMIT,
        "training_stage": "fp32_representation_gate",
        "rank": 16,
        "top_b": 40,
        "selected_epoch": 1,
        "completed_epochs": 1,
        "stop_reason": "MAXIMUM_EPOCHS_REACHED",
        "quantization": "none",
        "document_codes_written": False,
        "query_gate_present": False,
        "selection": metrics,
        "configuration": config,
        "fingerprint_payload": fingerprint,
        "data_access": started["data_access"],
    }
    _write_json(directory / "training_started.json", started)
    _save(
        directory / "query_projection.float32.npy",
        np.zeros((16, 16), dtype=np.float32),
    )
    _save(
        directory / "document_projection.float32.npy",
        np.zeros((16, 16), dtype=np.float32),
    )
    _save(directory / "selection_base_per_query.float64.npy", base)
    _save(directory / "selection_v2_2_per_query.float64.npy", v2)
    _write_json(directory / "training_history.json", history)
    _write_json(directory / "selection_metrics.json", metrics)
    _write_json(directory / "training_summary.json", summary)
    output_names = sorted(MODULE.REQUIRED_RUN_OUTPUTS)
    complete = {
        "schema_version": 1,
        "protocol_id": MODULE.PROTOCOL_ID,
        "status": "TRAINING_COMPLETE",
        "run_id": run_id,
        "outputs": {
            filename: MODULE.file_record(directory / filename)
            for filename in output_names
        },
    }
    _write_json(directory / "training_complete.json", complete)
    return directory, complete


def _make_fixture(
    tmp_path: Path, *, seed43_learning_rate: float = 0.0001
) -> dict[str, Any]:
    bundle_root = tmp_path / "bundles"
    split_audit_path = bundle_root / "v2_2_split_audit.json"
    _write_json(split_audit_path, {"synthetic": True})
    split_audit_sha = MODULE.sha256_file(split_audit_path)
    train = _make_bundle(
        bundle_root,
        MODULE.FIT_ROLE_ID,
        query_count=4,
        row_offset=0,
        split_audit_sha256=split_audit_sha,
    )
    selection = _make_bundle(
        bundle_root,
        MODULE.SELECTION_ROLE_ID,
        query_count=20,
        row_offset=100,
        split_audit_sha256=split_audit_sha,
    )
    basis_path = tmp_path / "pca_basis.npy"
    config_path = tmp_path / "pca_config.json"
    _save(basis_path, np.eye(16, dtype=np.float32))
    _write_json(config_path, {"rank": 16, "top_b": 40, "alpha": 0.75})
    hashes = {
        "train_manifest_sha256": MODULE.sha256_file(train / "v2_2_manifest.json"),
        "selection_manifest_sha256": MODULE.sha256_file(
            selection / "v2_2_manifest.json"
        ),
        "pca_basis_sha256": MODULE.sha256_file(basis_path),
        "pca_config_sha256": MODULE.sha256_file(config_path),
    }
    runs: dict[int, Path] = {}
    completions: dict[int, dict[str, Any]] = {}
    for seed in MODULE.EXPECTED_SEEDS:
        run, complete = _make_run(
            tmp_path / "runs",
            seed,
            query_count=20,
            learning_rate=(seed43_learning_rate if seed == 43 else 0.0001),
            **hashes,
        )
        runs[seed] = run
        completions[seed] = complete

    frozen_config = _configuration(42)
    frozen_config.pop("seed")
    frozen_config["epoch_selection"] = "highest validation recall; earliest tie"
    protocol = {
        "schema_version": 1,
        "protocol_id": MODULE.REPLICATION_PROTOCOL_ID,
        "parent_protocol_id": MODULE.PROTOCOL_ID,
        "metric_contract": {"query_count": 20},
        "seed_policy": {
            "registered_seeds": list(MODULE.EXPECTED_SEEDS),
            "heldout_primary_seeds": list(MODULE.HELDOUT_SEEDS),
        },
        "paired_joint_query_bootstrap": {
            "resamples": MODULE.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": MODULE.BOOTSTRAP_SEED,
        },
        "decision_procedure": {
            "heldout_primary_thresholds": {
                "heldout_mean_gain_over_base_at_least": MODULE.MINIMUM_GAIN_OVER_BASE,
                "heldout_mean_gain_over_pca_at_least": MODULE.MINIMUM_GAIN_OVER_PCA,
            }
        },
        "execution_lineage": {
            "training_source_commit": SOURCE_COMMIT,
            "source_hashes": {
                "aggregator_sha256": MODULE.sha256_file(SCRIPT),
                "metric_helper_sha256": MODULE.sha256_file(
                    ROOT / "scripts" / "train_boundary_loss_sidecar.py"
                ),
                "trainer_sha256": MODULE.sha256_file(
                    ROOT / "scripts" / "train_boundary_loss_sidecar_v2_2.py"
                ),
                "core_sha256": MODULE.sha256_file(
                    ROOT / "scripts" / "rars_v2_2_core.py"
                ),
                "inner_train_manifest_sha256": hashes["train_manifest_sha256"],
                "inner_validation_manifest_sha256": hashes[
                    "selection_manifest_sha256"
                ],
                "pca_basis_sha256": hashes["pca_basis_sha256"],
                "pca_config_sha256": hashes["pca_config_sha256"],
            },
            "split_audit_sha256": split_audit_sha,
        },
        "execution_environment_contract": {
            "python_version": "3.12.13",
            "numpy_version": "1.26.4",
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "device": "cuda",
            "gpu_name_must_contain": "T4",
            "compute_capability": "7.5",
            "deterministic_algorithms_supported_and_enabled": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": "UNSET",
        },
        "frozen_training_configuration": frozen_config,
        "audited_seed_42": {
            "run_id": completions[42]["run_id"],
            "verified_outputs": {
                filename: {
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
                for filename, record in completions[42]["outputs"].items()
            },
        },
    }
    protocol_path = tmp_path / "replication_protocol.json"
    _write_json(protocol_path, protocol)
    environment = {
        "schema_version": 1,
        "protocol_id": MODULE.REPLICATION_PROTOCOL_ID,
        "training_source_commit": SOURCE_COMMIT,
        "seed42_run_id": completions[42]["run_id"],
        "trainer_sha256": protocol["execution_lineage"]["source_hashes"][
            "trainer_sha256"
        ],
        "core_sha256": protocol["execution_lineage"]["source_hashes"][
            "core_sha256"
        ],
        "aggregator_sha256": protocol["execution_lineage"]["source_hashes"][
            "aggregator_sha256"
        ],
        "metric_helper_sha256": protocol["execution_lineage"]["source_hashes"][
            "metric_helper_sha256"
        ],
        "versions": {
            "python_version": "3.12.13",
            "numpy_version": "1.26.4",
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "faiss_version": "1.10.0",
        },
        "hardware": {
            "device": "cuda",
            "gpu_name": "Synthetic T4 GPU",
            "compute_capability": "7.5",
            "cudnn_version": "90000",
            "cuda_driver_version": "12.8",
        },
        "determinism": {
            "deterministic_algorithms_supported_and_enabled": True,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "cublas_workspace_config": "UNSET",
        },
        "pip_freeze_sha256": "a" * 64,
        "protocol_sha256": MODULE.sha256_file(protocol_path),
    }
    environment_path = tmp_path / "environment.json"
    _write_json(environment_path, environment)
    return {
        "train": train,
        "selection": selection,
        "basis": basis_path,
        "pca_config": config_path,
        "protocol": protocol_path,
        "environment": environment_path,
        "runs": runs,
    }


def test_end_to_end_stable_replication_writes_auditable_outputs(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "aggregate"
    summary = MODULE.aggregate_replication(
        train_bundle_dir=fixture["train"],
        selection_bundle_dir=fixture["selection"],
        pca_basis_path=fixture["basis"],
        pca_config_path=fixture["pca_config"],
        replication_protocol_path=fixture["protocol"],
        environment_manifest_path=fixture["environment"],
        run_dirs=fixture["runs"],
        output_dir=output,
    )
    assert summary["decision"] == MODULE.STABLE_DECISION
    assert summary["groups"]["all_seeds"]["gain_over_base"]["sample_sd"] == 0
    np.testing.assert_array_equal(
        np.load(output / "selection_base_per_query.float64.npy"),
        np.full(20, 0.5),
    )
    statistics = np.load(output / "paired_bootstrap_statistics.float64.npy")
    assert statistics.shape == (MODULE.BOOTSTRAP_REPLICATES, 10)
    bootstrap = json.loads((output / "paired_bootstrap.json").read_text())
    assert bootstrap["shared_query_indices_across_all_contrasts"] is True
    assert bootstrap["contrasts"]["heldout_mean_minus_pca_fp32"]["ci95_low"] > 0
    complete = json.loads((output / "replication_complete.json").read_text())
    assert complete["status"] == "REPLICATION_COMPLETE"
    for filename, record in complete["outputs"].items():
        assert MODULE.sha256_file(output / filename) == record["sha256"]


def test_fingerprint_difference_beyond_seed_is_invalid(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, seed43_learning_rate=0.0002)
    with pytest.raises(MODULE.ReplicationInvalid, match="more than the seed"):
        MODULE.aggregate_replication(
            train_bundle_dir=fixture["train"],
            selection_bundle_dir=fixture["selection"],
            pca_basis_path=fixture["basis"],
            pca_config_path=fixture["pca_config"],
            replication_protocol_path=fixture["protocol"],
            environment_manifest_path=fixture["environment"],
            run_dirs=fixture["runs"],
            output_dir=tmp_path / "aggregate",
        )


def _decision_inputs(
    *, heldout_gain: float, heldout_support: int, ci_low: float
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    per_seed = []
    for seed in MODULE.EXPECTED_SEEDS:
        gain = 0.02 if seed == 42 else heldout_gain
        per_seed.append({
            "seed": seed,
            "gain_over_base": gain,
            "gain_over_pca_fp32": gain,
            "joint_pass": (
                gain >= MODULE.MINIMUM_GAIN_OVER_BASE
                and gain >= MODULE.MINIMUM_GAIN_OVER_PCA
            ),
            "vs_base": {"improved_queries": heldout_support},
            "vs_pca_fp32": {"improved_queries": heldout_support},
        })
    metric = {
        "mean": heldout_gain,
        "sample_sd": 0.0,
        "minimum": heldout_gain,
        "maximum": heldout_gain,
    }
    all_metric = dict(metric, mean=(0.02 + 2 * heldout_gain) / 3)
    groups = {
        "heldout_replication_seeds": {
            "gain_over_base": metric,
            "gain_over_pca_fp32": metric,
        },
        "all_seeds": {
            "gain_over_base": all_metric,
            "gain_over_pca_fp32": all_metric,
        },
    }
    bootstrap = {
        "contrasts": {
            "heldout_mean_minus_base": {"ci95_low": ci_low},
            "heldout_mean_minus_pca_fp32": {"ci95_low": ci_low},
        }
    }
    return per_seed, groups, bootstrap


def test_unstable_and_stop_decisions_are_distinct() -> None:
    per_seed, groups, bootstrap = _decision_inputs(
        heldout_gain=0.02, heldout_support=0, ci_low=0.01
    )
    unstable = MODULE.make_replication_decision(
        per_seed, groups, bootstrap, query_count=1019
    )
    assert unstable["decision"] == MODULE.UNSTABLE_DECISION
    assert unstable["qat_protocol_definition_authorized"] is False

    # STOP is driven only by the held-out primary means.  An all-seed mean
    # failure after held-out passage is a remaining stability failure.
    per_seed, groups, bootstrap = _decision_inputs(
        heldout_gain=0.02, heldout_support=100, ci_low=0.01
    )
    groups["all_seeds"]["gain_over_base"]["mean"] = 0.0
    all_seed_unstable = MODULE.make_replication_decision(
        per_seed, groups, bootstrap, query_count=1019
    )
    assert all_seed_unstable["decision"] == MODULE.UNSTABLE_DECISION

    per_seed, groups, bootstrap = _decision_inputs(
        heldout_gain=0.004, heldout_support=100, ci_low=0.001
    )
    stopped = MODULE.make_replication_decision(
        per_seed, groups, bootstrap, query_count=1019
    )
    assert stopped["decision"] == MODULE.STOP_DECISION


def test_one_weak_heldout_seed_cannot_be_rescued_by_averaging() -> None:
    per_seed, groups, bootstrap = _decision_inputs(
        heldout_gain=0.02, heldout_support=100, ci_low=0.001
    )
    seed44 = next(row for row in per_seed if row["seed"] == 44)
    seed44["gain_over_base"] = 0.006
    seed44["gain_over_pca_fp32"] = 0.006
    seed44["joint_pass"] = False
    heldout_mean = (0.02 + 0.006) / 2
    all_seed_mean = (0.02 + 0.02 + 0.006) / 3
    for key in ("gain_over_base", "gain_over_pca_fp32"):
        groups["heldout_replication_seeds"][key]["mean"] = heldout_mean
        groups["all_seeds"][key]["mean"] = all_seed_mean
    result = MODULE.make_replication_decision(
        per_seed, groups, bootstrap, query_count=1019
    )
    assert heldout_mean >= MODULE.MINIMUM_GAIN_OVER_BASE
    assert result["decision"] == MODULE.UNSTABLE_DECISION
    assert result["conditions"]["all_three_seeds_joint_pass"] is False


def test_cli_writes_invalid_json(tmp_path: Path) -> None:
    output = tmp_path / "invalid-output"
    missing = tmp_path / "missing"
    exit_code = MODULE.main([
        "--train-bundle-dir",
        str(missing),
        "--selection-bundle-dir",
        str(missing),
        "--pca-basis",
        str(missing / "basis.npy"),
        "--pca-config",
        str(missing / "config.json"),
        "--environment-manifest",
        str(missing / "environment.json"),
        "--seed-42-dir",
        str(missing / "42"),
        "--seed-43-dir",
        str(missing / "43"),
        "--seed-44-dir",
        str(missing / "44"),
        "--output-dir",
        str(output),
    ])
    assert exit_code == 2
    invalid = json.loads((output / "replication_invalid.json").read_text())
    assert invalid["status"] == "REPLICATION_INVALID"
    assert invalid["decision"] == "INVALID"
    assert invalid["qat_protocol_definition_authorized"] is False
