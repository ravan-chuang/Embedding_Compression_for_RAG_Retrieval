#!/usr/bin/env python3
"""Verify the portable committed RARS-v13 closure without its 16 MB payload."""

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
from rars_v13_signed_score_core import (  # noqa: E402
    PROTOCOL_ID,
    deterministic_fold_ids,
    signed_score_decision,
)
from rars_v8_cutoff_sidecar_core import candidate_gap_recovery  # noqa: E402


EXTERNAL_PAYLOAD = "full_corpus_signed_score_assignments.uint8.memmap"
CLEAN_NOTEBOOK = Path(
    "notebooks/MSMARCO_RARS_v13_Signed_Score_RPQ_Development.ipynb"
)


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


def _inference_kwargs(protocol: dict[str, Any], comparison: str) -> dict[str, Any]:
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
    checks = (
        (
            lineage / "fresh_query_manifest.json",
            freeze["outputs"]["fresh_query_manifest.json"],
            "frozen query manifest",
        ),
        (
            lineage / "fresh_qrels.json",
            freeze["outputs"]["fresh_qrels.json"],
            "frozen qrels",
        ),
        (
            lineage / "fresh_query_freeze.json",
            bundle["inputs"]["fresh_query_freeze"],
            "bundle query freeze",
        ),
        (
            lineage / "fresh_query_manifest.json",
            bundle["inputs"]["fresh_query_manifest"],
            "bundle query manifest",
        ),
        (
            lineage / "fresh_bundle_manifest.json",
            bundle_complete["manifest"],
            "completed bundle manifest",
        ),
        (
            lineage / "fresh_bundle_manifest.json",
            started["inputs"]["fresh_bundle_manifest"],
            "training bundle manifest",
        ),
        (
            lineage / "fresh_bundle_complete.json",
            started["inputs"]["fresh_bundle_complete"],
            "training bundle completion",
        ),
    )
    for path, record, label in checks:
        verify_record(path, record, label)
    if freeze["selection"]["candidate_retrieval_performed"] is not False:
        raise ValueError("V13 query freeze occurred after candidate retrieval")
    if bundle["metrics_computed"] is not False:
        raise ValueError("V13 bundle contains pre-training metrics")
    if bundle["old_rars_holdout_opened"] is not False:
        raise ValueError("V13 bundle opened an old holdout")


def _load_clean_notebook(repo_root: Path, source_commit: str) -> dict[str, Any]:
    """Prefer the checkout, with historical Git fallback for local deletions."""

    checkout = repo_root / CLEAN_NOTEBOOK
    if checkout.is_file():
        return read_json(checkout)
    try:
        value = subprocess.check_output(
            ["git", "show", f"{source_commit}:{CLEAN_NOTEBOOK.as_posix()}"],
            cwd=repo_root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "Clean V13 notebook is absent from the checkout and source commit"
        ) from error
    return json.loads(value)


def _verify_notebook(root: Path, repo_root: Path, audit: dict[str, Any]) -> None:
    path = root / "executed_notebook" / CLEAN_NOTEBOOK.name
    verify_record(path, audit["executed_notebook"], "executed V13 notebook")
    executed = read_json(path)
    clean = _load_clean_notebook(repo_root, audit["source_commit"])
    if len(executed["cells"]) != len(clean["cells"]):
        raise ValueError("Executed V13 notebook cell count changed")
    for index, (actual, expected) in enumerate(zip(executed["cells"], clean["cells"])):
        if (
            actual["cell_type"] != expected["cell_type"]
            or actual["source"] != expected["source"]
        ):
            raise ValueError(f"Executed V13 notebook source changed in cell {index}")
    code_cells = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    if counts != audit["executed_notebook"]["code_execution_counts"]:
        raise ValueError("Executed V13 notebook order changed")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if len(errors) != int(audit["executed_notebook"]["error_output_count"]):
        raise ValueError("Executed V13 notebook error count changed")


def verify_closure(root: Path, repo_root: Path) -> dict[str, Any]:
    protocol = read_json(
        repo_root / "protocols/rars_v13_signed_score_distilled_rpq_v1.json"
    )
    audit = read_json(root / "artifact_audit.json")
    development = root / "development"
    complete = read_json(development / "development_complete.json")
    result = read_json(development / "development_result.json")
    freeze = read_json(development / "method_freeze.json")
    started = read_json(development / "development_started.json")
    source_commit = str(audit["source_commit"])
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V13 protocol identity changed")
    if audit.get("status") != "RARS_V13_COMMITTED_CLOSURE_AUDITED":
        raise ValueError("V13 closure audit is incomplete")
    expected_statuses = (
        (complete, "RARS_V13_FRESH_DEVELOPMENT_COMPLETE"),
        (result, "RARS_V13_FRESH_DEVELOPMENT_COMPLETE"),
        (freeze, "RARS_V13_METHOD_AND_DECISION_FROZEN"),
        (started, "RARS_V13_FRESH_DEVELOPMENT_STARTED"),
    )
    for payload, status in expected_statuses:
        if payload.get("status") != status or payload.get("source_commit") != source_commit:
            raise ValueError("V13 status or source lineage changed")
    verify_record(
        development / "development_started.json", complete["started"], "V13 start marker"
    )
    external = audit["external_full_corpus_payload"]
    for name, record in complete["outputs"].items():
        if name == EXTERNAL_PAYLOAD:
            if int(record["bytes"]) != int(external["bytes"]) or record[
                "sha256"
            ] != external["sha256"]:
                raise ValueError("External V13 payload record changed")
        else:
            verify_record(development / name, record, f"V13 output {name}")
    for relative, record in started["source_blobs"].items():
        verify_record(repo_root / relative, record, f"V13 source blob {relative}")

    qids = (development / "query_ids.utf8.txt").read_text(encoding="utf-8").splitlines()
    folds = np.load(development / "fold_ids.int64.npy", allow_pickle=False)
    target = int(protocol["fresh_query_freeze"]["target_query_count"])
    if len(qids) != target or len(set(qids)) != target:
        raise ValueError("V13 committed qids changed")
    if folds.dtype != np.int64 or not np.array_equal(folds, deterministic_fold_ids(qids)):
        raise ValueError("V13 committed folds changed")
    prior: set[str] = set()
    for relative in protocol["fresh_query_freeze"]["prior_qid_sources"]:
        path = repo_root / relative
        if path.suffix == ".txt":
            values = path.read_text(encoding="utf-8").splitlines()
        else:
            value = read_json(path)
            values = value.get("query_ids") if isinstance(value, dict) else value
        prior.update(str(item) for item in values)
    if len(prior) != int(protocol["fresh_query_freeze"]["expected_unique_excluded_qids"]):
        raise ValueError("V13 exclusion registry count changed")
    if prior.intersection(qids):
        raise ValueError("V13 closure contains a historical qid")

    def load(prefix: str, metric: str) -> np.ndarray:
        path = development / f"per_query_{prefix}_{metric}_at_10.float64.npy"
        value = np.load(path, allow_pickle=False)
        if value.shape != (target,) or value.dtype != np.float64:
            raise ValueError(f"V13 per-query contract changed: {path.name}")
        if np.any(~np.isfinite(value)) or np.any((value < 0) | (value > 1)):
            raise ValueError(f"V13 per-query values are invalid: {path.name}")
        return value

    metrics = ("recall", "mrr", "ndcg")
    seeds = [int(value) for value in protocol["rpq_training"]["seeds"]]
    primary_seed = int(protocol["rpq_training"]["primary_seed"])
    base = {name: load("base", name) for name in metrics}
    exact = {name: load("same_candidate_exact", name) for name in metrics}
    pca16 = {name: load("pca16", name) for name in metrics}
    unsupervised = {
        seed: {name: load(f"unsupervised_seed{seed}", name) for name in metrics}
        for seed in seeds
    }
    challenger = {
        seed: {name: load(f"signed_score_seed{seed}", name) for name in metrics}
        for seed in seeds
    }
    primary = paired_inference(
        challenger[primary_seed]["recall"],
        unsupervised[primary_seed]["recall"],
        **_inference_kwargs(protocol, "primary_vs_unsupervised"),
    )
    versus_pca = paired_inference(
        challenger[primary_seed]["recall"],
        pca16["recall"],
        **_inference_kwargs(protocol, "primary_vs_pca16"),
    )
    versus_base = paired_inference(
        challenger[primary_seed]["recall"],
        base["recall"],
        **_inference_kwargs(protocol, "primary_vs_base"),
    )
    seed_gains = [
        float(np.mean(challenger[seed]["recall"] - unsupervised[seed]["recall"]))
        for seed in seeds
    ]
    changed_sets = [
        set(
            np.flatnonzero(
                challenger[seed]["recall"] - unsupervised[seed]["recall"]
            ).tolist()
        )
        for seed in seeds
    ]
    stability = {
        "changed_counts": [len(value) for value in changed_sets],
        "union_count": len(set.union(*changed_sets)),
        "pairwise_overlap_counts": [
            len(changed_sets[0] & changed_sets[1]),
            len(changed_sets[0] & changed_sets[2]),
            len(changed_sets[1] & changed_sets[2]),
        ],
        "three_way_overlap_count": len(set.intersection(*changed_sets)),
    }
    if stability != audit["packet_verification"]["seed_changed_query_stability"]:
        raise ValueError("V13 seed-level changed-query stability changed")
    fold_gains = [
        float(
            np.mean(
                challenger[primary_seed]["recall"][folds == fold]
                - unsupervised[primary_seed]["recall"][folds == fold]
            )
        )
        for fold in range(int(protocol["cross_validation"]["fold_count"]))
    ]
    gap = candidate_gap_recovery(
        challenger[primary_seed]["recall"], base["recall"], exact["recall"]
    )
    diagnostics = read_json(development / "fold_seed_diagnostics.json")
    signed = diagnostics["signed_score"]
    if len(signed) != 5 * len(seeds):
        raise ValueError("V13 fold/seed diagnostic count changed")
    all_nonincreasing = all(row["update_summary"]["objective_nonincreasing"] for row in signed)
    all_nonincreasing &= bool(result["final_fit"]["update_summary"]["objective_nonincreasing"])
    maximum_drift = max(
        [float(row["update_summary"]["maximum_centroid_drift_fraction"]) for row in signed]
        + [float(result["final_fit"]["update_summary"]["maximum_centroid_drift_fraction"])]
    )
    assignment_changes = sum(
        int(row["update_summary"]["assignment_changes"]) for row in signed
    ) + int(result["final_fit"]["update_summary"]["assignment_changes"])
    recomputed = signed_score_decision(
        primary_vs_unsupervised=primary,
        primary_vs_pca16=versus_pca,
        primary_vs_base=versus_base,
        seed_gains=seed_gains,
        fold_gains=fold_gains,
        candidate_gap_recovery=gap,
        unsupervised_mrr=float(np.mean(unsupervised[primary_seed]["mrr"])),
        challenger_mrr=float(np.mean(challenger[primary_seed]["mrr"])),
        unsupervised_ndcg=float(np.mean(unsupervised[primary_seed]["ndcg"])),
        challenger_ndcg=float(np.mean(challenger[primary_seed]["ndcg"])),
        payload_bytes_per_document=int(external["bytes_per_document"]),
        full_corpus_codes_materialized=bool(external["verified_before_repository_import"]),
        all_objectives_nonincreasing=all_nonincreasing,
        maximum_centroid_drift_fraction=maximum_drift,
        assignment_changes=assignment_changes,
        thresholds=protocol["development_gate"],
    )
    if recomputed != result["decision"]:
        raise ValueError("V13 committed decision does not recompute")
    decisions = {
        complete["formal_decision"],
        result["formal_decision"],
        freeze["formal_decision"],
        audit["formal_decision"],
        recomputed["decision"],
    }
    if len(decisions) != 1:
        raise ValueError("V13 closure decisions disagree")
    for key, observed in (("seed_gains", seed_gains), ("fold_gains", fold_gains)):
        if not np.allclose(observed, result[key], rtol=0.0, atol=1e-15):
            raise ValueError(f"V13 {key} changed")
    _assert_close(gap, result["candidate_gap_recovery_fraction"], "V13 gap recovery")
    for method, arrays in (
        ("base", base),
        ("same_candidate_exact", exact),
        ("pca16", pca16),
        ("unsupervised_primary", unsupervised[primary_seed]),
        ("signed_score_primary", challenger[primary_seed]),
    ):
        for metric, values in arrays.items():
            _assert_close(float(np.mean(values)), result["metrics"][method][metric], f"{method} {metric}")
    full_codes = result["final_fit"]["full_corpus_codes"]
    if full_codes["record"]["sha256"] != external["sha256"]:
        raise ValueError("V13 result payload hash differs from the audit")
    histograms = np.asarray(full_codes["code_histograms"], dtype=np.int64)
    if histograms.shape != (16, 256) or not np.all(histograms.sum(axis=1) == 1_000_000):
        raise ValueError("V13 registered full-code histograms are invalid")
    occupied = np.sum(histograms > 0, axis=1)
    if int(occupied.min()) != int(external["minimum_occupied_centroids_per_block"]):
        raise ValueError("V13 minimum occupancy differs from the audit")
    if int(occupied.max()) != int(external["maximum_occupied_centroids_per_block"]):
        raise ValueError("V13 maximum occupancy differs from the audit")

    _verify_lineage(root)
    _verify_notebook(root, repo_root, audit)
    return {
        "status": "RARS_V13_COMMITTED_CLOSURE_VERIFIED",
        "source_commit": source_commit,
        "formal_decision": recomputed["decision"],
        "query_count": target,
        "primary_gain": primary["mean_difference"],
        "primary_ci95": [primary["lower"], primary["upper"]],
        "primary_randomization_p_value": primary["randomization_p_value_one_sided"],
        "seed_gains": seed_gains,
        "seed_changed_query_stability": stability,
        "fold_gains": fold_gains,
        "external_payload_sha256": external["sha256"],
        "verified_committed_outputs": len(complete["outputs"]) - 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/rars_v13_signed_score_rpq",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(verify_closure(args.closure_root, args.repo_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
