#!/usr/bin/env python3
"""Diagnose why a frozen RARS-v2.1 model does or does not change Top-k recall.

This is a development-only decomposition.  It deliberately accepts only the
``inner_validation`` bundle, never the outer validation split or closed tests.
It performs no fitting, epoch selection, parameter sweep, or retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_boundary_loss_sidecar import (
    PROTOCOL_ID,
    corrected_candidate_scores,
    load_bundle,
    quantize_coefficients,
    read_json,
)


DIAGNOSTIC_ID = "rars_v2_1_inner_validation_intervention_diagnostic_v1"


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        raise ValueError("Cannot summarize an empty array")
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or not 0 < k < values.shape[1]:
        raise ValueError("Require 0 < k < candidate count")
    return np.argsort(-values, axis=1, kind="stable")[:, :k]


def per_query_diagnostics(
    base_scores: np.ndarray,
    corrected_scores: np.ndarray,
    ungated_scores: np.ndarray,
    labels: np.ndarray,
    relevant_counts: np.ndarray,
    gate: np.ndarray,
    raw_correction: np.ndarray,
    bounded_correction: np.ndarray,
    effective_correction: np.ndarray,
    *,
    final_k: int,
    top_b: int,
    max_correction: float,
) -> list[dict[str, Any]]:
    base = np.asarray(base_scores, dtype=np.float32)
    corrected = np.asarray(corrected_scores, dtype=np.float32)
    ungated = np.asarray(ungated_scores, dtype=np.float32)
    relevance = np.asarray(labels)
    counts = np.asarray(relevant_counts)
    if not (base.shape == corrected.shape == ungated.shape == relevance.shape):
        raise ValueError("Score and relevance matrices must have matching shapes")
    if counts.shape != (len(base),) or np.any(counts <= 0):
        raise ValueError("Relevant counts must be positive and match query count")
    if not final_k < top_b <= base.shape[1]:
        raise ValueError("Require final_k < top_b <= candidate count")
    expected_correction_shape = (len(base), top_b)
    for name, value in (
        ("raw", raw_correction),
        ("bounded", bounded_correction),
        ("effective", effective_correction),
    ):
        if np.asarray(value).shape != expected_correction_shape:
            raise ValueError(f"{name} correction shape must be {expected_correction_shape}")
    if np.asarray(gate).shape != (len(base),):
        raise ValueError("Gate must have one value per query")

    base_order = np.argsort(-base, axis=1, kind="stable")
    corrected_order = np.argsort(-corrected, axis=1, kind="stable")
    ungated_order = np.argsort(-ungated, axis=1, kind="stable")
    rows: list[dict[str, Any]] = []
    for query_index in range(len(base)):
        base_top = base_order[query_index, :final_k]
        corrected_top = corrected_order[query_index, :final_k]
        ungated_top = ungated_order[query_index, :final_k]
        # The deployed scorer corrects candidate-array positions ``:top_b``.
        correctable = np.arange(top_b)
        base_hits = int(relevance[query_index, base_top].sum())
        corrected_hits = int(relevance[query_index, corrected_top].sum())
        ungated_hits = int(relevance[query_index, ungated_top].sum())
        oracle_hits = min(final_k, int(relevance[query_index, correctable].sum()))
        base_margin = float(
            base[query_index, base_order[query_index, final_k - 1]]
            - base[query_index, base_order[query_index, final_k]]
        )
        corrected_margin = float(
            corrected[query_index, corrected_order[query_index, final_k - 1]]
            - corrected[query_index, corrected_order[query_index, final_k]]
        )
        changed = final_k - len(set(base_top.tolist()) & set(corrected_top.tolist()))
        saturation = np.abs(bounded_correction[query_index]) >= 0.95 * max_correction
        rows.append({
            "query_index": query_index,
            "relevant_count": int(counts[query_index]),
            "base_hits_at_k": base_hits,
            "corrected_hits_at_k": corrected_hits,
            "ungated_hits_at_k": ungated_hits,
            "oracle_top_b_hits_at_k": oracle_hits,
            "base_recall_at_k": base_hits / float(counts[query_index]),
            "corrected_recall_at_k": corrected_hits / float(counts[query_index]),
            "ungated_recall_at_k": ungated_hits / float(counts[query_index]),
            "oracle_top_b_recall_at_k": oracle_hits / float(counts[query_index]),
            "corrected_minus_base_recall": (
                corrected_hits - base_hits
            ) / float(counts[query_index]),
            "ungated_minus_base_recall": (
                ungated_hits - base_hits
            ) / float(counts[query_index]),
            "oracle_minus_base_recall": (
                oracle_hits - base_hits
            ) / float(counts[query_index]),
            "oracle_headroom": bool(oracle_hits > base_hits),
            "gate": float(gate[query_index]),
            "base_rank_k_margin": base_margin,
            "corrected_rank_k_margin": corrected_margin,
            "top_k_membership_changes": int(changed),
            "max_abs_raw_correction": float(
                np.max(np.abs(raw_correction[query_index]))
            ),
            "max_abs_bounded_correction": float(
                np.max(np.abs(bounded_correction[query_index]))
            ),
            "max_abs_effective_correction": float(
                np.max(np.abs(effective_correction[query_index]))
            ),
            "saturated_candidate_fraction": float(np.mean(saturation)),
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No query diagnostics were produced")
    base = np.asarray([row["base_recall_at_k"] for row in rows])
    corrected = np.asarray([row["corrected_recall_at_k"] for row in rows])
    ungated = np.asarray([row["ungated_recall_at_k"] for row in rows])
    oracle = np.asarray([row["oracle_top_b_recall_at_k"] for row in rows])
    delta = corrected - base
    ungated_delta = ungated - base
    oracle_delta = oracle - base
    changed = np.asarray([row["top_k_membership_changes"] for row in rows])
    oracle_gain = float(np.mean(oracle_delta))
    actual_gain = float(np.mean(delta))
    return {
        "query_count": len(rows),
        "recall_at_10": {
            "base": float(np.mean(base)),
            "trained_gate": float(np.mean(corrected)),
            "gate_one_counterfactual": float(np.mean(ungated)),
            "top_b_relevance_oracle": float(np.mean(oracle)),
            "trained_gate_gain_over_base": actual_gain,
            "gate_one_gain_over_base": float(np.mean(ungated_delta)),
            "oracle_gain_over_base": oracle_gain,
            "fraction_of_oracle_gain_recovered": (
                None if oracle_gain <= 0 else actual_gain / oracle_gain
            ),
        },
        "query_outcomes": {
            "improved": int(np.sum(delta > 0)),
            "harmed": int(np.sum(delta < 0)),
            "unchanged": int(np.sum(delta == 0)),
            "oracle_headroom_queries": int(np.sum(oracle_delta > 0)),
            "queries_with_top_k_membership_change": int(np.sum(changed > 0)),
            "gate_one_improved": int(np.sum(ungated_delta > 0)),
            "gate_one_harmed": int(np.sum(ungated_delta < 0)),
        },
        "intervention": {
            "gate": quantiles(np.asarray([row["gate"] for row in rows])),
            "base_rank_10_margin": quantiles(np.asarray([
                row["base_rank_k_margin"] for row in rows
            ])),
            "max_abs_raw_correction": quantiles(np.asarray([
                row["max_abs_raw_correction"] for row in rows
            ])),
            "max_abs_bounded_correction": quantiles(np.asarray([
                row["max_abs_bounded_correction"] for row in rows
            ])),
            "max_abs_effective_correction": quantiles(np.asarray([
                row["max_abs_effective_correction"] for row in rows
            ])),
            "mean_saturated_candidate_fraction": float(np.mean([
                row["saturated_candidate_fraction"] for row in rows
            ])),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(bundle_dir: Path, model_dir: Path, output_dir: Path, final_k: int) -> dict[str, Any]:
    if bundle_dir.name != "inner_validation":
        raise ValueError(
            "v2.2 method diagnosis is restricted to the inner_validation bundle"
        )
    arrays = load_bundle(
        bundle_dir, expected_role="validation", require_relevant_counts=True
    )
    training = read_json(model_dir / "training_summary.json")
    if training.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Model protocol does not match the development bundle")
    if training.get("test_qrels_accessed") is not False:
        raise ValueError("Closed-test model artifacts are forbidden")
    top_b = int(training["top_b"])
    max_correction = float(training["max_correction"])
    query_projection = np.load(model_dir / "query_projection.float32.npy")
    document_projection = np.load(model_dir / "document_projection.float32.npy")
    scales = np.load(model_dir / "document_scales.float32.npy")
    gate_weight = np.load(model_dir / "query_gate_weight.float32.npy")
    gate_bias = float(np.load(model_dir / "query_gate_bias.float32.npy")[0])

    depth = min(top_b, arrays["ann_rows"].shape[1])
    residual_rows = np.asarray(arrays["residual_lookup"][:, :depth], dtype=np.int64)
    residual = np.asarray(arrays["residuals"][residual_rows], dtype=np.float32)
    coefficients = np.einsum("qcd,dr->qcr", residual, document_projection)
    coefficients = quantize_coefficients(coefficients, scales).astype(np.float32) * scales
    query_coefficients = np.asarray(arrays["queries"], dtype=np.float32) @ query_projection
    raw = np.einsum("qr,qcr->qc", query_coefficients, coefficients)
    bounded = max_correction * np.tanh(raw / max_correction)
    gate = 1.0 / (1.0 + np.exp(-(
        np.asarray(arrays["queries"], dtype=np.float32) @ gate_weight + gate_bias
    )))
    effective = gate[:, None] * bounded

    corrected = corrected_candidate_scores(
        arrays, query_projection, document_projection, gate_weight, gate_bias,
        top_b=top_b, scales=scales, max_correction=max_correction,
    )
    ungated = np.asarray(arrays["ann_scores"], dtype=np.float32).copy()
    ungated[:, :depth] += bounded
    rows = per_query_diagnostics(
        arrays["ann_scores"], corrected, ungated, arrays["labels"],
        arrays["relevant_counts"], gate, raw, bounded, effective,
        final_k=final_k, top_b=depth, max_correction=max_correction,
    )
    result = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "development_only_intervention_diagnostic_complete",
        "source_bundle_role": "inner_validation",
        "fitting_performed": False,
        "selection_performed": False,
        "parameter_sweep_performed": False,
        "retrieval_performed": False,
        "outer_validation_accessed": False,
        "test_qrels_accessed": False,
        "nq_test_retuning_authorized": False,
        "selected_epoch": training.get("selected_epoch"),
        "top_b": depth,
        "final_k": final_k,
        **summarize(rows),
    }
    write_csv(output_dir / "per_query_intervention_diagnostics.csv", rows)
    atomic_json(output_dir / "diagnostic_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--final-k", default=10, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(
        args.bundle_dir, args.model_dir, args.output_dir, args.final_k
    ), indent=2))


if __name__ == "__main__":
    main()
