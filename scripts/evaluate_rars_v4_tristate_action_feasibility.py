#!/usr/bin/env python3
"""Run the frozen RARS-v4 tri-state label/action-space Phase-0 gate."""

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

from rars_v3_oracle_core import exact_residual_scores, progressive_tier_scores
from rars_v4_tristate_action_core import (
    AUDIT_ROLE_ID,
    DESIGN_ROLE_ID,
    PROTOCOL_ID,
    action_reachability_diagnostics,
    exact_triage_action_oracle,
    final_action_decision,
    label_support_diagnostics,
    label_swap_bootstrap,
    oracle_diagnostics,
    pre_action_decision,
    validate_tristate_labels,
)


V3_PROTOCOL_ID = "rars_v3_oracle_first_feasibility_v1"
ROLE_TO_V3 = {
    DESIGN_ROLE_ID: "oracle_design",
    AUDIT_ROLE_ID: "oracle_audit",
}
CANONICAL_PROTOCOL = Path("protocols/rars_v4_tristate_action_feasibility_v1.json")
CANONICAL_SOURCES = (
    Path("scripts/rars_v4_tristate_action_core.py"),
    Path("scripts/materialize_rars_v4_tristate_labels.py"),
    Path("scripts/evaluate_rars_v4_tristate_action_feasibility.py"),
    Path("scripts/rars_v3_oracle_core.py"),
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not Path(path).is_file():
        raise ValueError(f"Missing registered {label}: {path}")
    if Path(path).stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Registered {label} hash changed")


def _validate_exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source_commit must be an exact lowercase 40-hex commit")


def _validate_clean_git_head(repo_root: Path, source_commit: str) -> None:
    _validate_exact_commit(source_commit)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if head != source_commit:
        raise ValueError(f"Expected exact Git HEAD {source_commit}, found {head}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("RARS-v4 evaluation requires a clean worktree")


def validate_protocol_and_sources(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, str]]:
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve()
    if protocol_path.resolve() != canonical:
        raise ValueError("Only the canonical RARS-v4 protocol path is allowed")
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected RARS-v4 protocol ID")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_TRISTATE_LABEL_AUDIT":
        raise ValueError("RARS-v4 protocol is not frozen before Phase-0")
    if protocol.get("method_revision_allowed") is not False or protocol.get(
        "outcome_informed_revision_allowed"
    ) is not False:
        raise ValueError("RARS-v4 protocol allows revision after outcomes")
    _validate_clean_git_head(repo_root, source_commit)
    paths = (CANONICAL_PROTOCOL, *CANONICAL_SOURCES)
    hashes = {f"{path.name}_sha256": sha256_file(repo_root / path) for path in paths}
    expected_v3_core = protocol["parent_lineage"]["v3_core_sha256"]
    if hashes["rars_v3_oracle_core.py_sha256"] != expected_v3_core:
        raise ValueError("Pinned v3 numeric-core hash changed")
    return protocol, hashes


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe registered output path: {value!r}")
    return path


def verify_v3_complete_run(
    output_dir: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Recursively verify the exact completed v3 parent run at runtime."""

    complete_path = output_dir / "oracle_complete.json"
    summary_path = output_dir / "oracle_summary.json"
    freeze_path = output_dir / "design_freeze.json"
    complete = read_json(complete_path)
    summary = read_json(summary_path)
    if complete.get("status") != "ORACLE_COMPLETE" or summary.get("status") != "ORACLE_COMPLETE":
        raise ValueError("V3 parent run is incomplete")
    expected_commit = protocol["parent_lineage"]["v3_implementation_commit"]
    if complete.get("source_commit") != expected_commit or summary.get("source_commit") != expected_commit:
        raise ValueError("V3 parent run source commit changed")
    if complete.get("protocol_id") != V3_PROTOCOL_ID:
        raise ValueError("V3 parent run protocol changed")
    if complete.get("run_fingerprint") != summary.get("run_fingerprint"):
        raise ValueError("V3 parent run fingerprint disagrees")
    _verify_record(freeze_path, complete.get("design_freeze", {}), "v3 design freeze")
    outputs = complete.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("V3 complete marker lacks registered outputs")
    for relative_name, record in outputs.items():
        relative = _safe_relative_path(str(relative_name))
        _verify_record(output_dir / relative, record, f"v3 output {relative_name}")

    expected = protocol["parent_lineage"]
    if summary.get("formal_decision") != expected["v3_observed_formal_decision"]:
        raise ValueError("V3 observed decision differs from the v4 disclosure")
    selected = summary.get("selected_primary_comparator", {}).get("method")
    if selected != expected["v3_observed_primary_comparator"]:
        raise ValueError("V3 selected comparator differs from the v4 disclosure")
    observed = expected["v3_observed_metrics"]
    actual_metrics = {
        "base_recall_at_10": summary["mean_recall_at_10"]["base"],
        "primary_comparator_recall_at_10": summary["mean_recall_at_10"][
            "primary_comparator"
        ],
        "exact40_recall_at_10": summary["mean_recall_at_10"]["Exact40"],
        "oracle16_recall_at_10": summary["mean_recall_at_10"]["Oracle16"],
        "oracle16_gain_over_comparator": summary["oracle_budget_curve"]["Oracle16"][
            "mean_recall_gain_over_primary_comparator"
        ],
        "oracle16_comparator_relative_cfr": summary["counterfactual_recovery"][
            "comparator_relative"
        ]["Oracle16"]["counterfactual_recovery_fraction"],
        "oracle16_exact40_membership_alignment": summary[
            "counterfactual_recovery"
        ]["comparator_relative"]["Oracle16"][
            "positive_gain_mass_with_exact_distance_reduction_fraction"
        ],
    }
    for name, registered in observed.items():
        if not np.isclose(float(actual_metrics[name]), float(registered), rtol=0.0, atol=1e-15):
            raise ValueError(f"V3 disclosed metric changed: {name}")
    return {
        "complete": complete,
        "summary": summary,
        "complete_record": file_record(complete_path),
        "summary_record": file_record(summary_path),
        "design_freeze_record": file_record(freeze_path),
        "run_fingerprint": complete["run_fingerprint"],
        "selected_primary_comparator": selected,
    }


def load_candidate_bundle(
    bundle_dir: Path,
    *,
    role_id: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = bundle_dir / "v3_candidate_manifest.json"
    query_manifest_path = bundle_dir / "query_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("protocol_id") != V3_PROTOCOL_ID:
        raise ValueError("Candidate bundle is not the frozen v3 bundle")
    if manifest.get("source_commit") != protocol["parent_lineage"][
        "v3_implementation_commit"
    ]:
        raise ValueError("Candidate bundle v3 commit changed")
    if manifest.get("role_id") != ROLE_TO_V3[role_id]:
        raise ValueError("Candidate bundle role differs from the v4 phase")
    query_record = manifest.get("query_manifest")
    if not isinstance(query_record, dict):
        raise ValueError("Candidate manifest lacks query identity")
    _verify_record(query_manifest_path, query_record, "candidate query manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Candidate manifest lacks files")
    mapping = {
        "query_vectors.float32.npy": ("queries", np.float32),
        "candidate_doc_ids.int64.npy": ("document_ids", np.int64),
        "ann_scores.float32.npy": ("base_scores", np.float32),
        "ann_residual_rows.int64.npy": ("residual_lookup", np.int64),
        "candidate_residuals.float32.npy": ("residuals", np.float32),
        "parent_role_indices.int64.npy": ("parent_role_indices", np.int64),
    }
    arrays: dict[str, Any] = {}
    for filename, (name, dtype) in mapping.items():
        record = files.get(filename)
        if not isinstance(record, dict):
            raise ValueError(f"Candidate manifest lacks {filename}")
        path = bundle_dir / filename
        _verify_record(path, record, f"candidate {filename}")
        value = np.load(path, mmap_mode="r")
        if value.dtype != np.dtype(dtype):
            raise ValueError(f"Candidate {filename} dtype changed")
        arrays[name] = value
    query_manifest = read_json(query_manifest_path)
    qids = [str(value) for value in query_manifest.get("query_ids", [])]
    expected_query_count = int(protocol["data_policy"]["roles"][role_id]["query_count"])
    if len(qids) != expected_query_count or len(qids) != len(set(qids)):
        raise ValueError("Candidate role query identity changed")
    expected_shape = (expected_query_count, int(manifest.get("candidate_count", -1)))
    for name in ("document_ids", "base_scores", "residual_lookup"):
        if arrays[name].shape != expected_shape:
            raise ValueError(f"Candidate {name} shape changed")
    if arrays["queries"].shape[0] != expected_query_count:
        raise ValueError("Candidate query vectors changed")
    if arrays["residuals"].ndim != 2 or arrays["residuals"].shape[1] != arrays[
        "queries"
    ].shape[1]:
        raise ValueError("Candidate residual dimension changed")
    lookup = np.asarray(arrays["residual_lookup"])
    if np.any(lookup < 0) or np.any(lookup >= len(arrays["residuals"])):
        raise ValueError("Candidate residual lookup is out of range")
    if not np.all(np.isfinite(np.asarray(arrays["base_scores"]))) or not np.all(
        np.isfinite(np.asarray(arrays["queries"]))
    ):
        raise ValueError("Candidate score/query arrays are non-finite")
    arrays.update(
        {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "query_manifest": query_manifest,
            "qids": qids,
        }
    )
    return arrays


def load_tristate_labels(
    label_manifest_path: Path,
    candidate: dict[str, Any],
    *,
    role_id: str,
    protocol: dict[str, Any],
    source_commit: str,
    expected_materializer_sha256: str,
    design_freeze_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = read_json(label_manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("role_id") != role_id:
        raise ValueError("Tri-state label manifest protocol/role changed")
    if manifest.get("source_commit") != source_commit:
        raise ValueError("Tri-state label source commit changed")
    if manifest.get("materializer_sha256") != expected_materializer_sha256:
        raise ValueError("Tri-state label materializer source hash changed")
    _verify_record(
        candidate["manifest_path"],
        manifest.get("candidate_manifest", {}),
        "tri-state candidate manifest",
    )
    if manifest.get("binary_candidate_relevance_read") is not False or manifest.get(
        "missing_rows_interpreted_as_explicit_negative"
    ) is not False:
        raise ValueError("Tri-state materializer used a prohibited binary shortcut")
    if manifest.get("future_method_holdout_accessed") is not False:
        raise ValueError("Tri-state materializer accessed the future holdout")
    if role_id == AUDIT_ROLE_ID:
        if design_freeze_path is None:
            raise ValueError("Audit tri-state labels require a design freeze")
        _verify_record(
            design_freeze_path,
            manifest.get("design_freeze", {}),
            "audit-label design freeze",
        )
        if manifest.get("materialized_after_design_go_freeze") is not True:
            raise ValueError("Audit labels were not released after design GO")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Tri-state label manifest lacks files")
    states_path = label_manifest_path.parent / "candidate_judgment_state.int8.npy"
    counts_path = label_manifest_path.parent / "relevant_counts.int32.npy"
    _verify_record(states_path, files.get(states_path.name, {}), "tri-state states")
    _verify_record(counts_path, files.get(counts_path.name, {}), "tri-state counts")
    states = np.load(states_path, mmap_mode="r")
    counts = np.load(counts_path, mmap_mode="r")
    validate_tristate_labels(
        states, counts, expected_shape=np.asarray(candidate["base_scores"]).shape
    )
    return states, counts, manifest


def _registered_v3_array(
    v3_output_dir: Path,
    v3_complete: dict[str, Any],
    filename: str,
    *,
    dtype: np.dtype[Any],
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    record = v3_complete.get("outputs", {}).get(filename)
    if not isinstance(record, dict):
        raise ValueError(f"V3 complete run does not register {filename}")
    path = v3_output_dir / filename
    _verify_record(path, record, f"v3 array {filename}")
    value = np.load(path, mmap_mode="r")
    if value.dtype != np.dtype(dtype) or (
        expected_shape is not None and value.shape != expected_shape
    ):
        raise ValueError(f"V3 array contract changed: {filename}")
    return value


def _output_records(output_dir: Path, paths: list[Path]) -> dict[str, Any]:
    root = output_dir.resolve()
    records: dict[str, Any] = {}
    for path in paths:
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError("Output record escaped the v4 output directory")
        relative = str(resolved.relative_to(root))
        records[relative] = file_record(resolved)
    return records


def _verify_registered_outputs(output_dir: Path, records: dict[str, Any]) -> None:
    if not isinstance(records, dict) or not records:
        raise ValueError("Complete v4 output lacks registered outputs")
    for relative_name, record in records.items():
        relative = _safe_relative_path(str(relative_name))
        _verify_record(output_dir / relative, record, f"v4 output {relative_name}")


def _reuse_if_complete(
    args: argparse.Namespace,
    *,
    role_id: str,
    expected_fingerprint: str,
) -> dict[str, Any] | None:
    summary_name = "design_summary.json" if role_id == DESIGN_ROLE_ID else "phase0_summary.json"
    summary_path = args.output_dir / summary_name
    marker_path = (
        args.output_dir / "design_freeze.json"
        if role_id == DESIGN_ROLE_ID
        else args.output_dir / "phase0_complete.json"
    )
    if not summary_path.exists() and not marker_path.exists():
        return None
    if not args.reuse_complete or not summary_path.is_file() or not marker_path.is_file():
        raise ValueError("Refusing to reuse partial/existing v4 Phase-0 output")
    summary = read_json(summary_path)
    marker = read_json(marker_path)
    if summary.get("source_commit") != args.source_commit or marker.get(
        "source_commit"
    ) != args.source_commit:
        raise ValueError("Reused v4 output source commit changed")
    if summary.get("run_fingerprint") != marker.get("run_fingerprint"):
        raise ValueError("Reused v4 output fingerprint changed")
    if summary.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("Reused v4 output no longer matches the exact inputs")
    _verify_registered_outputs(args.output_dir, marker.get("registered_outputs"))
    return summary


def _phase_role(phase: str) -> str:
    if phase == "design":
        return DESIGN_ROLE_ID
    if phase == "audit":
        return AUDIT_ROLE_ID
    raise ValueError("Unsupported v4 phase")


def run(args: argparse.Namespace) -> dict[str, Any]:
    role_id = _phase_role(args.phase)
    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_hashes = validate_protocol_and_sources(
        repo_root, args.protocol, args.source_commit
    )
    v3_parent = verify_v3_complete_run(args.v3_output_dir, protocol)
    candidate = load_candidate_bundle(
        args.candidate_bundle, role_id=role_id, protocol=protocol
    )
    design_freeze_path = args.output_dir / "design_freeze.json"
    if role_id == AUDIT_ROLE_ID:
        if not design_freeze_path.is_file():
            raise ValueError("Audit evaluation requires a durable v4 design freeze")
        design_freeze = read_json(design_freeze_path)
        if design_freeze.get("status") != "V4_DESIGN_GO_TO_DIAGNOSTIC_AUDIT":
            raise ValueError("V4 design freeze did not authorize diagnostic audit")
        if design_freeze.get("source_commit") != args.source_commit:
            raise ValueError("V4 design freeze source commit changed")
    else:
        if design_freeze_path.exists() and not args.reuse_complete:
            raise ValueError("Refusing to overwrite an existing v4 design freeze")
        design_freeze = None

    states, relevant_counts, label_manifest = load_tristate_labels(
        args.label_manifest,
        candidate,
        role_id=role_id,
        protocol=protocol,
        source_commit=args.source_commit,
        expected_materializer_sha256=source_hashes[
            "materialize_rars_v4_tristate_labels.py_sha256"
        ],
        design_freeze_path=design_freeze_path if role_id == AUDIT_ROLE_ID else None,
    )
    final_k = int(protocol["frozen_retrieval"]["final_k"])
    correction_depth = int(protocol["frozen_retrieval"]["correction_depth"])
    bootstrap_contract = protocol["metric_contract"]["bootstrap"]
    support = label_support_diagnostics(
        candidate["base_scores"],
        candidate["document_ids"],
        states,
        relevant_counts,
        final_k=final_k,
        correction_depth=correction_depth,
    )
    label_bootstrap = label_swap_bootstrap(
        support.label_swap_gain,
        replicates=int(bootstrap_contract["replicates"]),
        seed=int(bootstrap_contract["seed"]),
        confidence=float(bootstrap_contract["confidence"]),
    )
    explicit_semantics = bool(
        label_manifest.get("source_schema", {}).get(
            "explicit_negative_semantics_preserved", False
        )
    )
    pre_action = pre_action_decision(
        role_id=role_id,
        explicit_negative_semantics_preserved=explicit_semantics,
        support_summary=support.summary,
        label_bootstrap=label_bootstrap,
        thresholds=protocol["phase0_gate"],
    )

    fingerprint_payload = {
        "protocol_id": PROTOCOL_ID,
        "phase": args.phase,
        "role_id": role_id,
        "source_commit": args.source_commit,
        "source_hashes": source_hashes,
        "protocol_sha256": sha256_file(args.protocol),
        "candidate_manifest": file_record(candidate["manifest_path"]),
        "label_manifest": file_record(args.label_manifest),
        "v3_complete": v3_parent["complete_record"],
        "v3_run_fingerprint": v3_parent["run_fingerprint"],
        "frozen_action_spaces": protocol["frozen_action_spaces"],
        "phase0_gate": protocol["phase0_gate"],
        "bootstrap": bootstrap_contract,
    }
    run_fingerprint = canonical_sha256(fingerprint_payload)
    reused = _reuse_if_complete(
        args, role_id=role_id, expected_fingerprint=run_fingerprint
    )
    if reused is not None:
        return reused
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_slug = "design" if role_id == DESIGN_ROLE_ID else "audit"
    paths: list[Path] = []

    def save_array(filename: str, value: np.ndarray) -> Path:
        path = args.output_dir / filename
        atomic_save(path, np.asarray(value))
        paths.append(path)
        return path

    save_array(f"{phase_slug}_label_swap_gain.float64.npy", support.label_swap_gain)
    save_array(
        f"{phase_slug}_base_topk_membership.bool.npy",
        support.base_topk_membership.astype(np.bool_),
    )

    exact_summary: dict[str, Any] | None = None
    progressive_summary: dict[str, Any] | None = None
    action_decision: dict[str, Any] | None = None
    if pre_action["action_oracle_authorized"]:
        v3_complete = v3_parent["complete"]
        basis = _registered_v3_array(
            args.v3_output_dir,
            v3_complete,
            "progressive_svd_rank32.float32.npy",
            dtype=np.float32,
        )
        scales = _registered_v3_array(
            args.v3_output_dir,
            v3_complete,
            "progressive_svd_rank32_scales.float32.npy",
            dtype=np.float32,
        )
        representation = protocol["frozen_action_spaces"]["progressive_primary"]
        tier_costs = tuple(int(value) for value in representation["tiers_code_bytes"])
        progressive_scores = progressive_tier_scores(
            candidate["queries"],
            candidate["base_scores"],
            candidate["residual_lookup"],
            candidate["residuals"],
            basis,
            scales,
            alpha=float(representation["alpha"]),
            top_b=correction_depth,
        )
        exact_scores = exact_residual_scores(
            candidate["queries"],
            candidate["base_scores"],
            candidate["residual_lookup"],
            candidate["residuals"],
            top_b=correction_depth,
        )
        selective_exact_scores = np.stack(
            [np.asarray(candidate["base_scores"], dtype=np.float32), exact_scores],
            axis=1,
        )
        progressive_reachability = action_reachability_diagnostics(
            progressive_scores,
            candidate["document_ids"],
            support,
            correction_depth=correction_depth,
        )
        exact_reachability = action_reachability_diagnostics(
            selective_exact_scores,
            candidate["document_ids"],
            support,
            correction_depth=correction_depth,
        )
        selected_name = str(v3_parent["selected_primary_comparator"])
        comparator_filename = (
            f"design_baseline_{selected_name}_scores.float32.npy"
            if role_id == DESIGN_ROLE_ID
            else f"audit_baseline_{selected_name}_scores.float32.npy"
        )
        comparator_scores = _registered_v3_array(
            args.v3_output_dir,
            v3_complete,
            comparator_filename,
            dtype=np.float32,
            expected_shape=np.asarray(candidate["base_scores"]).shape,
        )
        exact_oracle = exact_triage_action_oracle(
            selective_exact_scores,
            (0, 1),
            states,
            candidate["document_ids"],
            relevant_counts,
            final_k=final_k,
            correction_depth=correction_depth,
            budget=int(
                protocol["frozen_action_spaces"]["selective_exact_diagnostic"][
                    "maximum_exact_actions_per_query"
                ]
            ),
        )
        progressive_oracle = exact_triage_action_oracle(
            progressive_scores,
            tier_costs,
            states,
            candidate["document_ids"],
            relevant_counts,
            final_k=final_k,
            correction_depth=correction_depth,
            budget=int(representation["budget_bytes_per_query"]),
        )
        exact_summary = {
            "reachability": exact_reachability.summary,
            "oracle": oracle_diagnostics(
                exact_oracle,
                comparator_scores,
                candidate["document_ids"],
                states,
                relevant_counts,
                support.label_swap_gain,
                exact_reachability.joint_swap_reachable_queries,
                qids=candidate["qids"],
                final_k=final_k,
                replicates=int(bootstrap_contract["replicates"]),
                seed=int(bootstrap_contract["seed"]),
                confidence=float(bootstrap_contract["confidence"]),
            ),
            "cost_unit": "exact-action count, not bytes",
        }
        progressive_summary = {
            "reachability": progressive_reachability.summary,
            "oracle": oracle_diagnostics(
                progressive_oracle,
                comparator_scores,
                candidate["document_ids"],
                states,
                relevant_counts,
                support.label_swap_gain,
                progressive_reachability.joint_swap_reachable_queries,
                qids=candidate["qids"],
                final_k=final_k,
                replicates=int(bootstrap_contract["replicates"]),
                seed=int(bootstrap_contract["seed"]),
                confidence=float(bootstrap_contract["confidence"]),
            ),
            "cost_unit": "accessed progressive code bytes",
        }
        action_decision = final_action_decision(
            role_id=role_id,
            progressive_diagnostics=progressive_summary["oracle"],
            thresholds=protocol["phase0_gate"],
        )
        save_array(
            f"{phase_slug}_progressive_oracle_recall.float64.npy",
            progressive_oracle.recall_at_k,
        )
        save_array(
            f"{phase_slug}_progressive_oracle_rates.int16.npy",
            progressive_oracle.rate_assignments,
        )
        save_array(
            f"{phase_slug}_progressive_oracle_accessed_bytes.int32.npy",
            progressive_oracle.action_cost,
        )
        save_array(
            f"{phase_slug}_progressive_oracle_topk_membership.bool.npy",
            progressive_oracle.topk_membership.astype(np.bool_),
        )
        save_array(
            f"{phase_slug}_exact_oracle_recall.float64.npy",
            exact_oracle.recall_at_k,
        )

    formal_decision = (
        action_decision["decision"]
        if action_decision is not None
        else pre_action["decision"]
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": (
            "PHASE0_DESIGN_COMPLETE"
            if role_id == DESIGN_ROLE_ID
            else "PHASE0_DIAGNOSTIC_AUDIT_COMPLETE"
        ),
        "phase": args.phase,
        "role_id": role_id,
        "source_commit": args.source_commit,
        "run_fingerprint": run_fingerprint,
        "formal_decision": formal_decision,
        "evidence_status": protocol["data_policy"]["evidence_status"],
        "explicit_negative_semantics_preserved": explicit_semantics,
        "label_source_schema": label_manifest["source_schema"],
        "label_support": support.summary,
        "label_swap_bootstrap": label_bootstrap,
        "pre_action_gate": pre_action,
        "selective_exact_diagnostic": exact_summary,
        "progressive_action_space": progressive_summary,
        "action_gate": action_decision,
        "v3_parent": {
            "run_fingerprint": v3_parent["run_fingerprint"],
            "formal_decision": v3_parent["summary"]["formal_decision"],
            "selected_primary_comparator": v3_parent[
                "selected_primary_comparator"
            ],
            "complete": v3_parent["complete_record"],
            "summary": v3_parent["summary_record"],
            "design_freeze": v3_parent["design_freeze_record"],
        },
        "future_method_holdout_accessed": False,
        "training_allowed": False,
        "qat_allowed": False,
        "external_evaluation_allowed": False,
        "go_is_method_success": False,
    }
    summary_path = args.output_dir / (
        "design_summary.json" if role_id == DESIGN_ROLE_ID else "phase0_summary.json"
    )
    atomic_json(summary_path, summary)
    paths.append(summary_path)
    registered_outputs = _output_records(args.output_dir, paths)

    if role_id == DESIGN_ROLE_ID:
        design_go = formal_decision == "DESIGN_GO_TO_DIAGNOSTIC_AUDIT"
        freeze = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": (
                "V4_DESIGN_GO_TO_DIAGNOSTIC_AUDIT"
                if design_go
                else "V4_DESIGN_STOPPED_BEFORE_DIAGNOSTIC_AUDIT"
            ),
            "source_commit": args.source_commit,
            "run_fingerprint": run_fingerprint,
            "formal_decision": formal_decision,
            "all_required_checks_passed": design_go,
            "audit_tristate_labels_materialized_before_this_freeze": False,
            "future_method_holdout_accessed": False,
            "fingerprint_payload": fingerprint_payload,
            "source_hashes": source_hashes,
            "registered_outputs": registered_outputs,
            "design_summary": file_record(summary_path),
            "expected_audit_role": AUDIT_ROLE_ID,
            "expected_audit_query_count": int(
                protocol["data_policy"]["roles"][AUDIT_ROLE_ID]["query_count"]
            ),
        }
        atomic_json(design_freeze_path, freeze)
        return summary

    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PHASE0_COMPLETE",
        "source_commit": args.source_commit,
        "run_fingerprint": run_fingerprint,
        "formal_decision": formal_decision,
        "design_freeze": file_record(design_freeze_path),
        "registered_outputs": registered_outputs,
        "future_method_holdout_accessed": False,
    }
    atomic_json(args.output_dir / "phase0_complete.json", complete)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("design", "audit"))
    parser.add_argument("--candidate-bundle", required=True, type=Path)
    parser.add_argument("--label-manifest", required=True, type=Path)
    parser.add_argument("--v3-output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reuse-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
