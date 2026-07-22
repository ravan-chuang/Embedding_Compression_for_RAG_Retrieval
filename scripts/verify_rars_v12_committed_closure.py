#!/usr/bin/env python3
"""Verify the portable committed RARS-v12 closure without the 16 MB payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v11_rank_rate_core import paired_inference  # noqa: E402
from rars_v12_ca_rpq_core import (  # noqa: E402
    PROTOCOL_ID,
    ca_rpq_decision,
    deterministic_fold_ids,
)
from rars_v8_cutoff_sidecar_core import candidate_gap_recovery  # noqa: E402


EXTERNAL_PAYLOAD = "full_corpus_ca_rpq_codes.uint8.memmap"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 changed")


def inference_kwargs(protocol: dict[str, Any], comparison: str) -> dict[str, Any]:
    inference = protocol["inference"]
    seeds = inference[comparison]
    return {
        "bootstrap_replicates": int(inference["bootstrap_replicates"]),
        "bootstrap_seed": int(seeds["bootstrap_seed"]),
        "randomization_replicates": int(inference["randomization_replicates"]),
        "randomization_seed": int(seeds["randomization_seed"]),
        "confidence": float(inference["confidence"]),
    }


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-15):
            raise ValueError(f"{label} changed: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"{label} changed: {actual!r} != {expected!r}")


def _verify_lineage(root: Path) -> None:
    lineage = root / "lineage"
    freeze = read_json(lineage / "fresh_query_freeze.json")
    bundle = read_json(lineage / "fresh_bundle_manifest.json")
    bundle_complete = read_json(lineage / "fresh_bundle_complete.json")
    started = read_json(root / "development/development_started.json")
    verify_record(
        lineage / "fresh_query_manifest.json",
        freeze["outputs"]["fresh_query_manifest.json"],
        "frozen query manifest",
    )
    verify_record(
        lineage / "fresh_qrels.json",
        freeze["outputs"]["fresh_qrels.json"],
        "frozen fresh qrels",
    )
    verify_record(
        lineage / "fresh_query_freeze.json",
        bundle["inputs"]["fresh_query_freeze"],
        "bundle input query freeze",
    )
    verify_record(
        lineage / "fresh_query_manifest.json",
        bundle["inputs"]["fresh_query_manifest"],
        "bundle input query manifest",
    )
    verify_record(
        lineage / "fresh_bundle_manifest.json",
        bundle_complete["manifest"],
        "completed bundle manifest",
    )
    verify_record(
        lineage / "fresh_bundle_manifest.json",
        started["inputs"]["fresh_bundle_manifest"],
        "training input bundle manifest",
    )
    verify_record(
        lineage / "fresh_bundle_complete.json",
        started["inputs"]["fresh_bundle_complete"],
        "training input bundle completion",
    )
    if freeze["selection"]["candidate_retrieval_performed"] is not False:
        raise ValueError("Fresh query freeze occurred after candidate retrieval")
    if bundle["metrics_computed"] is not False:
        raise ValueError("Fresh bundle contains pre-training metrics")
    if bundle["old_rars_holdout_opened"] is not False:
        raise ValueError("Fresh bundle opened an old holdout")


def _verify_notebook(root: Path, repo_root: Path, audit: dict[str, Any]) -> None:
    path = (
        root
        / "executed_notebook/MSMARCO_RARS_v12_Anchored_Cutoff_RPQ_Development.ipynb"
    )
    verify_record(path, audit["executed_notebook"], "executed V12 notebook")
    executed = read_json(path)
    source_commit = audit["source_commit"]
    clean_bytes = subprocess.check_output(
        [
            "git",
            "show",
            f"{source_commit}:notebooks/MSMARCO_RARS_v12_Anchored_Cutoff_RPQ_Development.ipynb",
        ],
        cwd=repo_root,
    )
    clean = json.loads(clean_bytes)
    if len(executed["cells"]) != len(clean["cells"]):
        raise ValueError("Executed V12 notebook cell count changed")
    for index, (actual, expected) in enumerate(
        zip(executed["cells"], clean["cells"])
    ):
        if actual["cell_type"] != expected["cell_type"] or actual["source"] != expected["source"]:
            raise ValueError(f"Executed V12 notebook source changed in cell {index}")
    code_cells = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    if counts != audit["executed_notebook"]["code_execution_counts"]:
        raise ValueError("Executed V12 notebook order changed")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if len(errors) != int(audit["executed_notebook"]["error_output_count"]):
        raise ValueError("Executed V12 notebook error count changed")


def verify_closure(root: Path, repo_root: Path) -> dict[str, Any]:
    protocol = read_json(repo_root / "protocols/rars_v12_anchored_cutoff_rpq_v1.json")
    audit = read_json(root / "artifact_audit.json")
    development = root / "development"
    complete = read_json(development / "development_complete.json")
    result = read_json(development / "development_result.json")
    method_freeze = read_json(development / "method_freeze.json")
    started = read_json(development / "development_started.json")
    source_commit = audit["source_commit"]
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V12 protocol identity changed")
    if audit.get("status") != "RARS_V12_COMMITTED_CLOSURE_AUDITED":
        raise ValueError("V12 committed closure audit is incomplete")
    if complete.get("status") != "RARS_V12_FRESH_DEVELOPMENT_COMPLETE":
        raise ValueError("V12 completion marker is invalid")
    if result.get("status") != "RARS_V12_FRESH_DEVELOPMENT_COMPLETE":
        raise ValueError("V12 result marker is invalid")
    if method_freeze.get("status") != "RARS_V12_METHOD_AND_DECISION_FROZEN":
        raise ValueError("V12 method freeze is invalid")
    for payload in (complete, result, method_freeze, started):
        if payload.get("source_commit") != source_commit:
            raise ValueError("V12 source commit changed across closure files")
    verify_record(
        development / "development_started.json",
        complete["started"],
        "V12 start marker",
    )
    external = audit["external_full_corpus_payload"]
    for name, record in complete["outputs"].items():
        if name == EXTERNAL_PAYLOAD:
            if record["bytes"] != external["bytes"] or record["sha256"] != external["sha256"]:
                raise ValueError("External V12 payload record changed")
            continue
        verify_record(development / name, record, f"V12 output {name}")
    for relative, record in started["source_blobs"].items():
        verify_record(repo_root / relative, record, f"V12 source blob {relative}")

    qids = (development / "query_ids.utf8.txt").read_text(encoding="utf-8").splitlines()
    folds = np.load(development / "fold_ids.int64.npy", allow_pickle=False)
    target = int(protocol["fresh_query_freeze"]["target_query_count"])
    if len(qids) != target or len(set(qids)) != target:
        raise ValueError("V12 committed qids changed")
    if folds.dtype != np.int64 or not np.array_equal(folds, deterministic_fold_ids(qids)):
        raise ValueError("V12 committed folds changed")
    prior: set[str] = set()
    for relative in protocol["fresh_query_freeze"]["prior_qid_sources"]:
        prior.update(str(value) for value in read_json(repo_root / relative))
    if prior.intersection(qids):
        raise ValueError("V12 closure contains a historical query id")

    def load(prefix: str, metric: str) -> np.ndarray:
        path = development / f"per_query_{prefix}_{metric}_at_10.float64.npy"
        value = np.load(path, allow_pickle=False)
        if value.shape != (target,) or value.dtype != np.float64:
            raise ValueError(f"V12 per-query contract changed: {path.name}")
        if np.any(~np.isfinite(value)) or np.any((value < 0) | (value > 1)):
            raise ValueError(f"V12 per-query values are invalid: {path.name}")
        return value

    metric_names = ("recall", "mrr", "ndcg")
    seeds = [int(value) for value in protocol["rpq_training"]["seeds"]]
    primary_seed = int(protocol["rpq_training"]["primary_seed"])
    base = {name: load("base", name) for name in metric_names}
    exact = {name: load("same_candidate_exact", name) for name in metric_names}
    unsupervised = {
        seed: {name: load(f"unsupervised_seed{seed}", name) for name in metric_names}
        for seed in seeds
    }
    challenger = {
        seed: {name: load(f"ca_rpq_seed{seed}", name) for name in metric_names}
        for seed in seeds
    }
    primary = paired_inference(
        challenger[primary_seed]["recall"],
        unsupervised[primary_seed]["recall"],
        **inference_kwargs(protocol, "primary_vs_unsupervised"),
    )
    versus_base = paired_inference(
        challenger[primary_seed]["recall"],
        base["recall"],
        **inference_kwargs(protocol, "primary_vs_base"),
    )
    seed_gains = [
        float(np.mean(challenger[seed]["recall"] - unsupervised[seed]["recall"]))
        for seed in seeds
    ]
    fold_gains = [
        float(
            np.mean(
                challenger[primary_seed]["recall"][folds == fold]
                - unsupervised[primary_seed]["recall"][folds == fold]
            )
        )
        for fold in range(5)
    ]
    gap = candidate_gap_recovery(
        challenger[primary_seed]["recall"], base["recall"], exact["recall"]
    )
    diagnostics = read_json(development / "fold_seed_diagnostics.json")
    if len(diagnostics) != 5 * len(seeds):
        raise ValueError("V12 fold/seed diagnostic count changed")
    all_nonincreasing = all(
        row["update_summary"]["fixed_assignment_objective_after"]
        <= row["update_summary"]["fixed_assignment_objective_before"] + 1e-8
        for row in diagnostics
    ) and (
        result["final_fit"]["update_summary"]["fixed_assignment_objective_after"]
        <= result["final_fit"]["update_summary"]["fixed_assignment_objective_before"]
        + 1e-8
    )
    maximum_drift = max(
        [float(row["update_summary"]["maximum_centroid_drift_fraction"]) for row in diagnostics]
        + [float(result["final_fit"]["update_summary"]["maximum_centroid_drift_fraction"])]
    )
    recomputed = ca_rpq_decision(
        primary_vs_unsupervised=primary,
        primary_vs_base=versus_base,
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        candidate_gap_recovery=gap,
        unsupervised_mrr=float(np.mean(unsupervised[primary_seed]["mrr"])),
        ca_mrr=float(np.mean(challenger[primary_seed]["mrr"])),
        unsupervised_ndcg=float(np.mean(unsupervised[primary_seed]["ndcg"])),
        ca_ndcg=float(np.mean(challenger[primary_seed]["ndcg"])),
        payload_bytes_per_document=int(external["bytes_per_document"]),
        full_corpus_codes_materialized=bool(external["verified_before_repository_import"]),
        all_objectives_nonincreasing=all_nonincreasing,
        maximum_centroid_drift_fraction=maximum_drift,
        thresholds=protocol["development_gate"],
    )
    if recomputed != result["decision"]:
        raise ValueError("V12 committed decision does not recompute")
    if not (
        complete["formal_decision"]
        == result["formal_decision"]
        == method_freeze["formal_decision"]
        == audit["formal_decision"]
        == recomputed["decision"]
    ):
        raise ValueError("V12 closure decisions disagree")
    if not np.allclose(seed_gains, result["seed_gains"], rtol=0.0, atol=1e-15):
        raise ValueError("V12 committed seed gains changed")
    if not np.allclose(fold_gains, result["fold_gains"], rtol=0.0, atol=1e-15):
        raise ValueError("V12 committed fold gains changed")
    _assert_close(gap, result["candidate_gap_recovery_fraction"], "V12 gap recovery")
    full_codes = result["final_fit"]["full_corpus_codes"]
    if full_codes["record"]["sha256"] != external["sha256"]:
        raise ValueError("V12 result payload hash differs from the audit")
    histograms = np.asarray(full_codes["code_histograms"], dtype=np.int64)
    if histograms.shape != (16, 256) or not np.all(histograms.sum(axis=1) == 1_000_000):
        raise ValueError("V12 registered full-code histograms are invalid")
    if int(np.min(np.sum(histograms > 0, axis=1))) != int(
        external["minimum_occupied_centroids_per_block"]
    ):
        raise ValueError("V12 registered occupancy differs from the audit")

    _verify_lineage(root)
    _verify_notebook(root, repo_root, audit)
    return {
        "status": "RARS_V12_COMMITTED_CLOSURE_VERIFIED",
        "source_commit": source_commit,
        "formal_decision": recomputed["decision"],
        "query_count": target,
        "primary_gain": primary["mean_difference"],
        "primary_ci95": [primary["lower"], primary["upper"]],
        "primary_randomization_p_value": primary[
            "randomization_p_value_one_sided"
        ],
        "seed_gains": seed_gains,
        "fold_gains": fold_gains,
        "external_payload_sha256": external["sha256"],
        "verified_committed_outputs": len(complete["outputs"]) - 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/rars_v12_ca_rpq",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            verify_closure(args.closure_root, args.repo_root),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

