#!/usr/bin/env python3
"""Verify a full RARS-v8 development and sidecar export.

The verifier treats the packet as evidence rather than trusting its summary
JSON.  It rechecks registered byte counts and SHA-256 hashes, reloads the OOF
vectors, recomputes the paired bootstrap comparisons and frozen decision, and
validates the full-corpus rank-16 int8 sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.rars_v8_cutoff_sidecar_core import (
        candidate_gap_recovery,
        development_decision,
        paired_bootstrap,
        validate_orthonormal_basis,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from rars_v8_cutoff_sidecar_core import (
        candidate_gap_recovery,
        development_decision,
        paired_bootstrap,
        validate_orthonormal_basis,
    )


PROTOCOL_ID = "rars_v8_cutoff_sidecar_v1"
SOURCE_COMMIT = "c9d95f15d55e7700db069da69567157f2eed469e"
DEVELOPMENT_STATUS = "RARS_V8_DEVELOPMENT_COMPLETE"
SIDECAR_STATUS = "RARS_V8_FULL_CORPUS_SIDECARS_COMPLETE"
GO_DECISION = "GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL"
QUERY_COUNT = 2307
DOCUMENT_COUNT = 1_000_000
DIMENSION = 384
RANK = 16


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
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 changed")


def _close(left: float, right: float, tolerance: float = 1e-15) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _same_payload(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    if left != right:
        raise ValueError(f"{label} differs from the registered payload")


def _identity(payload: dict[str, Any], *, status: str, label: str) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "source_commit": SOURCE_COMMIT,
        "formal_decision": GO_DECISION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{label} {key} changed")


def _load_array(
    path: Path, *, dtype: np.dtype[Any], shape: tuple[int, ...], finite: bool = True
) -> np.ndarray:
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if value.dtype != dtype or value.shape != shape:
        raise ValueError(
            f"{path.name} has {value.dtype}/{value.shape}; expected {dtype}/{shape}"
        )
    if finite and np.any(~np.isfinite(value)):
        raise ValueError(f"{path.name} contains non-finite values")
    return value


def _verify_source_blobs(started: dict[str, Any], repo_root: Path) -> None:
    paths = {
        "protocol": repo_root / "protocols/rars_v8_cutoff_sidecar_v1.json",
        "trainer": repo_root / "scripts/train_rars_v8_cutoff_sidecar.py",
        "core": repo_root / "scripts/rars_v8_cutoff_sidecar_core.py",
    }
    for name, path in paths.items():
        verify_record(path, started["source_blobs"][name], f"source blob {name}")


def _verify_development(
    packet_root: Path, protocol: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    root = packet_root / "development"
    started_path = root / "development_started.json"
    result_path = root / "development_result.json"
    freeze_path = root / "method_freeze.json"
    complete_path = root / "development_complete.json"
    started = read_json(started_path)
    result = read_json(result_path)
    freeze = read_json(freeze_path)
    complete = read_json(complete_path)

    _identity(result, status=DEVELOPMENT_STATUS, label="development result")
    _identity(complete, status=DEVELOPMENT_STATUS, label="development complete")
    if started.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Development start protocol changed")
    if started.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Development start source commit changed")
    if started.get("status") != "RARS_V8_DEVELOPMENT_STARTED":
        raise ValueError("Development start status changed")
    if freeze.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Method-freeze protocol changed")
    if freeze.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Method-freeze source commit changed")
    if freeze.get("status") != "RARS_V8_METHOD_FROZEN_AFTER_DEVELOPMENT":
        raise ValueError("Method-freeze status changed")
    if freeze.get("formal_decision") != GO_DECISION:
        raise ValueError("Method-freeze decision changed")
    for payload, label in (
        (started, "start"),
        (result, "result"),
        (complete, "complete"),
    ):
        if payload.get("future_method_holdout_opened") is not False:
            raise ValueError(f"Development {label} reports future-role access")
        if payload.get("oracle_audit_opened") is not False:
            raise ValueError(f"Development {label} reports audit-role access")
    if result.get("opened_roles") != ["oracle_design"]:
        raise ValueError("Development opened-role registry changed")
    if complete.get("full_corpus_sidecar_encoded") is not False:
        raise ValueError("Development stage reports premature full-corpus encoding")

    verify_record(started_path, complete["started"], "development_started.json")
    for filename, record in complete["outputs"].items():
        verify_record(root / filename, record, filename)
    verify_record(result_path, freeze["development_result"], "frozen result")
    verify_record(
        root / Path(freeze["pca_basis"]["path"]).name,
        freeze["pca_basis"],
        "frozen PCA basis",
    )
    verify_record(
        root / Path(freeze["rars_basis"]["path"]).name,
        freeze["rars_basis"],
        "frozen RARS basis",
    )
    _verify_source_blobs(started, repo_root)

    bases: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for name in ("pca", "rars"):
        bases[name] = _load_array(
            root / f"{name}_basis_rank16.float32.npy",
            dtype=np.dtype("float32"),
            shape=(DIMENSION, RANK),
        )
        validate_orthonormal_basis(bases[name], dimension=DIMENSION, rank=RANK)
        scales[name] = _load_array(
            root / f"{name}_scales_rank16.float32.npy",
            dtype=np.dtype("float32"),
            shape=(RANK,),
        )
        if np.any(scales[name] <= 0):
            raise ValueError(f"Development {name} scales are not positive")

    recalls = {
        name: _load_array(
            root / f"oof_{filename}_recall_at_10.float64.npy",
            dtype=np.dtype("float64"),
            shape=(QUERY_COUNT,),
        )
        for name, filename in (
            ("base", "base"),
            ("pca_oof", "pca"),
            ("rars_oof", "rars"),
            ("same_candidate_exact", "teacher"),
        )
    }
    for name, values in recalls.items():
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} Recall@10 is outside [0, 1]")
        if not _close(float(values.mean()), result["metrics"][name]["recall"]):
            raise ValueError(f"{name} Recall@10 mean cannot be recomputed")

    bootstrap = protocol["development_metrics"]
    arguments = {
        "replicates": int(bootstrap["bootstrap_replicates"]),
        "seed": int(bootstrap["bootstrap_seed"]),
    }
    comparisons = {
        "rars_vs_base_recall_at_10": paired_bootstrap(
            recalls["rars_oof"], recalls["base"], **arguments
        ),
        "pca_vs_base_recall_at_10": paired_bootstrap(
            recalls["pca_oof"], recalls["base"], **arguments
        ),
        "rars_vs_pca_recall_at_10": paired_bootstrap(
            recalls["rars_oof"], recalls["pca_oof"], **arguments
        ),
    }
    for name, recomputed in comparisons.items():
        _same_payload(recomputed, result["comparisons"][name], name)

    recovery = candidate_gap_recovery(
        recalls["rars_oof"], recalls["base"], recalls["same_candidate_exact"]
    )
    if not _close(recovery, result["candidate_gap_recovery_fraction"]):
        raise ValueError("Candidate-gap recovery cannot be recomputed")
    support = read_json(root / "pair_support.json")
    _same_payload(support, result["pair_support"], "pair support")
    if support["total_pairs"] != (
        support["promotion"]["pairs"] + support["protection"]["pairs"]
    ):
        raise ValueError("Promotion/protection pair counts do not sum")
    if not _close(
        support["promotion"]["balanced_weight_sum"]
        + support["protection"]["balanced_weight_sum"],
        1.0,
    ):
        raise ValueError("Pair-role weights do not sum to one")

    decision = development_decision(
        rars_vs_base=comparisons["rars_vs_base_recall_at_10"],
        pca_vs_base=comparisons["pca_vs_base_recall_at_10"],
        rars_vs_pca=comparisons["rars_vs_pca_recall_at_10"],
        gap_recovery=recovery,
        pair_support=support,
        thresholds=protocol["development_gate"],
    )
    _same_payload(decision, result["decision"], "frozen development decision")
    if decision["failed_gates"] or decision["decision"] != GO_DECISION:
        raise ValueError("Development packet does not contain the registered GO")

    fold_results = read_json(root / "fold_results.json")
    fold_counts = protocol["data_policy"]["cross_validation"]["fold_counts"]
    if len(fold_results) != len(fold_counts):
        raise ValueError("Fold-result count changed")
    optimization_decreased = True
    for fold_index, (fold, expected_count) in enumerate(zip(fold_results, fold_counts)):
        if fold["fold"] != fold_index:
            raise ValueError("Fold identifiers are not contiguous")
        if fold["validation_query_count"] != expected_count:
            raise ValueError("Fold validation count changed")
        if fold["training_query_count"] + expected_count != QUERY_COUNT:
            raise ValueError("Fold query accounting changed")
        initial = float(fold["optimization_initial_loss"])
        final = float(fold["optimization_final_loss"])
        if not (math.isfinite(initial) and math.isfinite(final)):
            raise ValueError("Fold optimization loss is non-finite")
        optimization_decreased &= final < initial
    final_optimization = result["final_optimization"]
    if int(final_optimization["steps"]) != int(protocol["basis_optimization"]["steps"]):
        raise ValueError("Final optimization step count changed")
    final_loss_decreased = float(final_optimization["final_loss"]) < float(
        final_optimization["initial_loss"]
    )

    return {
        "result": result,
        "complete": complete,
        "bases": bases,
        "scales": scales,
        "comparisons": comparisons,
        "gap_recovery": recovery,
        "fold_losses_all_decreased": bool(optimization_decreased),
        "final_loss_decreased": bool(final_loss_decreased),
        "verified_output_count": len(complete["outputs"]),
    }


def _verify_sidecars(packet_root: Path, development: dict[str, Any]) -> dict[str, Any]:
    root = packet_root / "sidecars"
    result_path = root / "sidecars_result.json"
    complete_path = root / "sidecars_complete.json"
    result = read_json(result_path)
    complete = read_json(complete_path)
    _identity(result, status=SIDECAR_STATUS, label="sidecar result")
    _identity(complete, status=SIDECAR_STATUS, label="sidecar complete")
    verify_record(result_path, complete["result"], "sidecars_result.json")
    if result.get("index_unchanged") is not True:
        raise ValueError("Sidecar build reports a changed index")
    if result.get("index_before") != result.get("index_after"):
        raise ValueError("Sidecar result index records differ")
    if complete.get("index_before") != complete.get("index_after"):
        raise ValueError("Sidecar complete index records differ")
    if result.get("qrels_argument_accepted") is not False:
        raise ValueError("Sidecar builder accepted a qrels argument")
    if result.get("query_argument_accepted") is not False:
        raise ValueError("Sidecar builder accepted a query argument")
    if complete.get("qrels_opened") is not False:
        raise ValueError("Sidecar completion reports qrels access")
    for payload, label in ((result, "result"), (complete, "complete")):
        if payload.get("future_method_holdout_opened") is not False:
            raise ValueError(f"Sidecar {label} reports future-role access")

    development_root = packet_root / "development"
    for key, filename in (
        ("complete", "development_complete.json"),
        ("method_freeze", "method_freeze.json"),
        ("result", "development_result.json"),
    ):
        verify_record(
            development_root / filename,
            result["development"][key],
            f"sidecar development {key}",
        )

    audit_path = packet_root / "artifact_audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else None
    summaries: dict[str, Any] = {}
    external_codes = False
    for name in ("pca", "rars"):
        method_root = root / name
        manifest_path = method_root / "manifest.json"
        manifest = read_json(manifest_path)
        verify_record(
            manifest_path,
            complete["sidecar_manifests"][name],
            f"{name} manifest",
        )
        registered_manifest = dict(result["sidecars"][name])
        registered_manifest.pop("manifest", None)
        if manifest != registered_manifest:
            raise ValueError(f"{name} manifest differs from sidecar result")
        if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("method") != name:
            raise ValueError(f"{name} manifest identity changed")
        for filename, record in manifest["files"].items():
            if filename == "codes.int8.npy" and not (method_root / filename).is_file():
                continue
            verify_record(method_root / filename, record, f"{name}/{filename}")

        basis = _load_array(
            method_root / "basis.float32.npy",
            dtype=np.dtype("float32"),
            shape=(DIMENSION, RANK),
        )
        validate_orthonormal_basis(basis, dimension=DIMENSION, rank=RANK)
        if not np.array_equal(basis, development["bases"][name]):
            raise ValueError(f"{name} full-corpus basis differs from method freeze")
        scales = _load_array(
            method_root / "scales.float32.npy",
            dtype=np.dtype("float32"),
            shape=(RANK,),
        )
        if np.any(scales <= 0):
            raise ValueError(f"{name} full-corpus scales are not positive")
        codes_path = method_root / "codes.int8.npy"
        if codes_path.is_file():
            codes = _load_array(
                codes_path,
                dtype=np.dtype("int8"),
                shape=(DOCUMENT_COUNT, RANK),
                finite=False,
            )
            minimum = int(np.min(codes))
            maximum = int(np.max(codes))
            if minimum < -127 or maximum > 127:
                raise ValueError(f"{name} codes exceed the registered int8 range")
            boundary_count = int(
                np.count_nonzero(np.abs(codes.astype(np.int16)) == 127)
            )
            code_shape = list(codes.shape)
        else:
            external_codes = True
            if (
                audit is None
                or audit.get("audit_status") != "FULL_EXPORTED_ARTIFACT_AUDIT_PASS"
            ):
                raise ValueError(
                    f"Missing {name} codes without a passing full-export audit"
                )
            relative = f"sidecars/{name}/codes.int8.npy"
            registered = audit.get("artifacts", {}).get(relative)
            if registered is None:
                raise ValueError(f"Full-export audit omitted {relative}")
            if registered["bytes"] != manifest["files"]["codes.int8.npy"]["bytes"]:
                raise ValueError(f"Audited {name} code byte count changed")
            if registered["sha256"] != manifest["files"]["codes.int8.npy"]["sha256"]:
                raise ValueError(f"Audited {name} code hash changed")
            summary = audit["full_packet_verifier_output"]["sidecars"][name]
            code_shape = summary["code_shape"]
            minimum = int(summary["code_minimum"])
            maximum = int(summary["code_maximum"])
            boundary_count = int(summary["boundary_code_count"])
            if code_shape != [DOCUMENT_COUNT, RANK] or minimum < -127 or maximum > 127:
                raise ValueError(f"Audited {name} code statistics changed")
        storage = manifest["storage"]
        representation_bytes = sum(
            int(manifest["files"][filename]["bytes"])
            for filename in (
                "basis.float32.npy",
                "scales.float32.npy",
                "codes.int8.npy",
            )
        )
        if representation_bytes != int(storage["representation_bytes"]):
            raise ValueError(f"{name} representation byte accounting changed")
        if not _close(
            representation_bytes / DOCUMENT_COUNT,
            storage["representation_bytes_per_document"],
        ):
            raise ValueError(f"{name} per-document byte accounting changed")
        if int(storage["code_payload_bytes_per_document"]) != RANK:
            raise ValueError(f"{name} code payload changed")
        quantization = manifest["quantization"]
        if int(quantization["coefficient_count"]) != DOCUMENT_COUNT * RANK:
            raise ValueError(f"{name} coefficient count changed")
        if int(quantization["saturated_coefficients"]) != 0:
            raise ValueError(f"{name} reports saturated coefficients")
        summaries[name] = {
            "code_shape": code_shape,
            "code_minimum": minimum,
            "code_maximum": maximum,
            "boundary_code_count": boundary_count,
            "representation_bytes": representation_bytes,
            "representation_bytes_per_document": float(
                storage["representation_bytes_per_document"]
            ),
        }
    return {
        "result": result,
        "complete": complete,
        "methods": summaries,
        "external_codes": external_codes,
    }


def _verify_notebook_source_parity(packet_root: Path, repo_root: Path) -> None:
    clean_path = (
        repo_root / "notebooks/MSMARCO_RARS_v8_Cutoff_Sidecar_Development.ipynb"
    )
    executed_path = (
        packet_root
        / "executed_notebook/MSMARCO_RARS_v8_Cutoff_Sidecar_Development.ipynb"
    )
    if not executed_path.is_file():
        return
    clean = read_json(clean_path)
    executed = read_json(executed_path)
    if len(clean["cells"]) != len(executed["cells"]):
        raise ValueError("Executed V8 notebook cell count changed")
    counts: list[int] = []
    for index, (left, right) in enumerate(zip(clean["cells"], executed["cells"])):
        for field in ("cell_type", "id", "source"):
            if left.get(field) != right.get(field):
                raise ValueError(
                    f"Executed V8 notebook source differs at cell {index}: {field}"
                )
        if right["cell_type"] == "code":
            count = right.get("execution_count")
            if count is None:
                raise ValueError(f"Executed V8 notebook cell {index} was not run")
            counts.append(int(count))
            if any(
                output.get("output_type") == "error"
                for output in right.get("outputs", [])
            ):
                raise ValueError(
                    f"Executed V8 notebook contains an error at cell {index}"
                )
    if counts != list(range(1, len(counts) + 1)):
        raise ValueError("Executed V8 notebook cells were not run once in order")
    rendered = json.dumps(executed, allow_nan=False)
    if GO_DECISION not in rendered or SOURCE_COMMIT not in rendered:
        raise ValueError("Executed V8 notebook omits the frozen outcome identity")


def _verify_closure_manifest(packet_root: Path) -> bool:
    path = packet_root / "closure_manifest.json"
    if not path.is_file():
        return False
    closure = read_json(path)
    if closure.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Closure-manifest protocol changed")
    if closure.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Closure-manifest source commit changed")
    if closure.get("formal_decision") != GO_DECISION:
        raise ValueError("Closure-manifest decision changed")
    expected = {
        item.relative_to(packet_root).as_posix()
        for item in packet_root.rglob("*")
        if item.is_file() and item.name != "closure_manifest.json"
    }
    if set(closure.get("files", {})) != expected:
        raise ValueError("Closure-manifest inventory is incomplete")
    for relative, record in closure["files"].items():
        verify_record(packet_root / relative, record, f"closure file {relative}")
    return True


def write_closure_manifest(packet_root: Path) -> Path:
    """Write a deterministic local inventory after the packet itself verifies."""

    packet_root = packet_root.resolve()
    verify_packet(packet_root, verify_closure=False)
    files = {
        item.relative_to(packet_root).as_posix(): {
            "bytes": int(item.stat().st_size),
            "sha256": sha256_file(item),
        }
        for item in sorted(packet_root.rglob("*"))
        if item.is_file() and item.name != "closure_manifest.json"
    }
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V8_CUTOFF_SIDECAR_CLOSURE_PACKET",
        "source_commit": SOURCE_COMMIT,
        "formal_decision": GO_DECISION,
        "packet_scope": (
            "thin closure packet with exact development outputs, small sidecar "
            "artifacts, executed notebook, and audited external 1M int8 codes"
        ),
        "files": files,
    }
    path = packet_root / "closure_manifest.json"
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def verify_packet(packet_root: Path, *, verify_closure: bool = True) -> dict[str, Any]:
    packet_root = packet_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = repo_root / "protocols/rars_v8_cutoff_sidecar_v1.json"
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Canonical protocol identity changed")
    development = _verify_development(packet_root, protocol, repo_root)
    sidecars = _verify_sidecars(packet_root, development)
    _verify_notebook_source_parity(packet_root, repo_root)
    closure_verified = (
        _verify_closure_manifest(packet_root) if verify_closure else False
    )
    result = development["result"]
    return {
        "status": (
            "RARS_V8_CUTOFF_SIDECAR_CLOSURE_VERIFIED"
            if closure_verified
            else "RARS_V8_CUTOFF_SIDECAR_FULL_PACKET_VERIFIED"
        ),
        "protocol_id": PROTOCOL_ID,
        "source_commit": SOURCE_COMMIT,
        "formal_decision": GO_DECISION,
        "query_count": QUERY_COUNT,
        "rars_recall_at_10": float(result["metrics"]["rars_oof"]["recall"]),
        "base_recall_at_10": float(result["metrics"]["base"]["recall"]),
        "pca_recall_at_10": float(result["metrics"]["pca_oof"]["recall"]),
        "rars_gain_over_base": float(
            development["comparisons"]["rars_vs_base_recall_at_10"]["mean_difference"]
        ),
        "rars_gain_over_pca": float(
            development["comparisons"]["rars_vs_pca_recall_at_10"]["mean_difference"]
        ),
        "candidate_gap_recovery_fraction": float(development["gap_recovery"]),
        "verified_development_output_count": int(development["verified_output_count"]),
        "sidecars": sidecars["methods"],
        "full_sidecar_codes_external": bool(sidecars["external_codes"]),
        "diagnostics": {
            "all_fold_recorded_losses_decreased": development[
                "fold_losses_all_decreased"
            ],
            "final_recorded_loss_decreased": development["final_loss_decreased"],
            "recorded_loss_warning": (
                "The frozen optimizer trajectory raises the recorded surrogate loss; "
                "OOF retrieval gains remain recomputable, but loss convergence "
                "must not be claimed."
            ),
        },
        "evidence_boundary": (
            "Outcome-informed five-fold OOF development on oracle_design; not an "
            "independent, prospective, or official MS MARCO evaluation."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--write-closure-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_closure_manifest:
        print(write_closure_manifest(args.packet_root))
        return
    print(json.dumps(verify_packet(args.packet_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
