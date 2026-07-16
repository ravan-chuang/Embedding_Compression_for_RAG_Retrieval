#!/usr/bin/env python3
"""Train the frozen RARS-v2.2 FP32 representation gate.

Stage A is deliberately narrow: PCA parameter warm-start, no query gate, no
quantization, dynamic boundary mining, and checkpoint selection on the fixed
inner validation split.  This trainer has no outer-validation argument.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v2_2_core import (  # noqa: E402
    FIT_ROLE_ID,
    PROMOTION,
    PROTECTION,
    PROTOCOL_ID,
    SELECTION_ROLE_ID,
    build_run_fingerprint,
    file_record,
    load_pca_warm_start,
    mine_dynamic_boundary_pairs,
    pca_fp32_scores,
    read_json,
    score_candidates_fp32,
    sha256_file,
    validate_bundle_manifest,
)
from train_boundary_loss_sidecar import (  # noqa: E402
    atomic_save_npy,
    atomic_write_json,
    recall_at_k_per_query,
)


def _verify_record(bundle_dir: Path, filename: str, record: dict[str, Any]) -> None:
    path = bundle_dir / filename
    if not path.exists():
        raise ValueError(f"Frozen bundle file is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Frozen bundle byte count changed: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Frozen bundle hash changed: {path}")


def load_bundle(bundle_dir: Path, *, expected_role_id: str) -> dict[str, Any]:
    manifest_path = bundle_dir / "v2_2_manifest.json"
    manifest = read_json(manifest_path)
    validate_bundle_manifest(manifest, expected_role_id=expected_role_id)
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise ValueError("v2.2 bundle manifest has no file records")
    required = {
        "query_vectors.float32.npy": "queries",
        "ann_rows.int64.npy": "ann_rows",
        "ann_scores.float32.npy": "ann_scores",
        "candidate_relevance.uint8.npy": "labels",
        "relevant_counts.int32.npy": "relevant_counts",
        "ann_residual_rows.int64.npy": "residual_lookup",
    }
    arrays: dict[str, Any] = {}
    for filename, key in required.items():
        if filename not in records:
            raise ValueError(f"Frozen bundle lacks {filename}")
        _verify_record(bundle_dir, filename, records[filename])
        arrays[key] = np.load(bundle_dir / filename, mmap_mode="r")
    residual_filename = next(
        (
            value
            for value in (
                "candidate_residuals.float32.npy",
                "document_residuals.float32.npy",
            )
            if value in records
        ),
        None,
    )
    if residual_filename is None:
        raise ValueError("Frozen bundle has no residual matrix")
    _verify_record(bundle_dir, residual_filename, records[residual_filename])
    arrays["residuals"] = np.load(bundle_dir / residual_filename, mmap_mode="r")
    arrays["residual_scope"] = (
        "candidate_union"
        if residual_filename.startswith("candidate_")
        else "full_corpus"
    )
    arrays["manifest"] = manifest
    arrays["manifest_path"] = manifest_path

    query_count, candidate_count = arrays["ann_scores"].shape
    if arrays["queries"].shape[0] != query_count:
        raise ValueError("Query and ANN arrays disagree")
    for key in ("ann_rows", "labels", "residual_lookup"):
        if arrays[key].shape != (query_count, candidate_count):
            raise ValueError(f"{key} does not match the ANN candidate shape")
    if arrays["relevant_counts"].shape != (query_count,):
        raise ValueError("Relevant-count denominator does not match queries")
    if np.any(np.asarray(arrays["relevant_counts"]) <= 0):
        raise ValueError("Every development query must have a positive denominator")
    if int(manifest["query_count"]) != query_count:
        raise ValueError("Manifest query count does not match arrays")
    return arrays


def exact_repo_state(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    if len(expected_commit) != 40 or any(
        value not in "0123456789abcdef" for value in expected_commit
    ):
        raise ValueError("--source-commit must be exact lowercase 40-hex")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError(f"Exact HEAD mismatch: expected {expected_commit}, got {head}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("v2.2 training requires a clean checkout")
    return {"head": head, "clean": True}


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rank": args.rank,
        "top_b": args.top_b,
        "final_k": args.final_k,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "score_batch_size": args.score_batch_size,
        "max_negatives_per_positive": args.max_negatives_per_positive,
        "promotion_mix": args.promotion_mix,
        "minimum_margin": args.minimum_margin,
        "margin_multiplier": args.margin_multiplier,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "correction_l2": args.correction_l2,
        "max_correction": args.max_correction,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "minimum_gain_over_base": args.minimum_gain_over_base,
        "minimum_gain_over_pca": args.minimum_gain_over_pca,
        "device": args.device,
    }


def _output_records(output_dir: Path, filenames: list[str]) -> dict[str, Any]:
    return {filename: file_record(output_dir / filename) for filename in filenames}


def _reuse_complete_run(output_dir: Path, run_id: str) -> dict[str, Any] | None:
    complete_path = output_dir / "training_complete.json"
    if not complete_path.exists():
        return None
    complete = read_json(complete_path)
    if complete.get("run_id") != run_id:
        raise ValueError("Existing completed run has a different fingerprint")
    for filename, record in complete.get("outputs", {}).items():
        _verify_record(output_dir, filename, record)
    return read_json(output_dir / "training_summary.json")


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    repo_root = Path(__file__).resolve().parents[1]
    repo_state = exact_repo_state(repo_root, args.source_commit)
    train_bundle = load_bundle(args.bundle_dir, expected_role_id=FIT_ROLE_ID)
    selection_bundle = load_bundle(
        args.selection_bundle_dir, expected_role_id=SELECTION_ROLE_ID
    )
    if (
        train_bundle["manifest"]["split_audit_sha256"]
        != selection_bundle["manifest"]["split_audit_sha256"]
    ):
        raise ValueError("Train and selection bundles use different split audits")
    if (
        train_bundle["manifest"]["query_ids_sha256"]
        == selection_bundle["manifest"]["query_ids_sha256"]
    ):
        raise ValueError("Train and selection query identities are identical")

    dimension = int(train_bundle["queries"].shape[1])
    query_init, document_init, pca_alpha = load_pca_warm_start(
        args.pca_basis,
        args.pca_config,
        dimension=dimension,
        rank=args.rank,
        top_b=args.top_b,
    )
    trainer_path = Path(__file__).resolve()
    fingerprint_payload = {
        "protocol_id": PROTOCOL_ID,
        "source_commit": args.source_commit,
        "trainer_sha256": sha256_file(trainer_path),
        "core_sha256": sha256_file(SCRIPT_DIR / "rars_v2_2_core.py"),
        "train_bundle_manifest_sha256": sha256_file(
            train_bundle["manifest_path"]
        ),
        "selection_bundle_manifest_sha256": sha256_file(
            selection_bundle["manifest_path"]
        ),
        "pca_basis_sha256": sha256_file(args.pca_basis),
        "pca_config_sha256": sha256_file(args.pca_config),
        "configuration": _configuration(args),
    }
    run_id = build_run_fingerprint(fingerprint_payload)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if args.reuse_complete:
            reused = _reuse_complete_run(args.output_dir, run_id)
            if reused is not None:
                return reused
        raise ValueError(
            "Output directory is non-empty; partial or mismatched runs are never reused"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no GPU is available")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass

    started = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "TRAINING_STARTED",
        "run_id": run_id,
        "fingerprint_payload": fingerprint_payload,
        "repo": repo_state,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "data_access": {
            "inner_train": True,
            "inner_validation": True,
            "outer_validation": False,
            "closed_test": False,
        },
    }
    atomic_write_json(args.output_dir / "training_started.json", started)

    query_projection = torch.nn.Linear(
        dimension, args.rank, bias=False, device=device
    )
    document_projection = torch.nn.Linear(
        dimension, args.rank, bias=False, device=device
    )
    with torch.no_grad():
        query_projection.weight.copy_(torch.from_numpy(query_init.T).to(device))
        document_projection.weight.copy_(
            torch.from_numpy(document_init.T).to(device)
        )
    optimizer = torch.optim.AdamW(
        [*query_projection.parameters(), *document_projection.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def model_arrays() -> tuple[np.ndarray, np.ndarray]:
        return (
            query_projection.weight.detach().cpu().numpy().T.astype(np.float32),
            document_projection.weight.detach().cpu().numpy().T.astype(np.float32),
        )

    def recall(bundle: dict[str, Any], scores: np.ndarray) -> float:
        return float(np.mean(recall_at_k_per_query(
            scores,
            bundle["labels"],
            bundle["relevant_counts"],
            k=args.final_k,
        )))

    base_selection_scores = np.asarray(
        selection_bundle["ann_scores"], dtype=np.float32
    )
    base_selection_recall = recall(selection_bundle, base_selection_scores)
    pca_basis = np.asarray(np.load(args.pca_basis), dtype=np.float32)
    pca_selection_scores = pca_fp32_scores(
        selection_bundle,
        pca_basis,
        alpha=pca_alpha,
        top_b=args.top_b,
        batch_size=args.score_batch_size,
    )
    pca_selection_recall = recall(selection_bundle, pca_selection_scores)
    initial_scores = score_candidates_fp32(
        selection_bundle,
        query_init,
        document_init,
        top_b=args.top_b,
        max_correction=args.max_correction,
        batch_size=args.score_batch_size,
    )
    initial_recall = recall(selection_bundle, initial_scores)
    best_recall = initial_recall
    selected_epoch = 0
    best_state = {
        "query_projection": {
            key: value.detach().cpu().clone()
            for key, value in query_projection.state_dict().items()
        },
        "document_projection": {
            key: value.detach().cpu().clone()
            for key, value in document_projection.state_dict().items()
        },
    }
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "stage": "pca_parameter_warm_start",
        "selection_fp32_recall_at_10": initial_recall,
        "query_projection_norm": float(np.linalg.norm(query_init)),
        "document_projection_norm": float(np.linalg.norm(document_init)),
    }]
    rng = np.random.default_rng(args.seed)
    stop_reason = "MAXIMUM_EPOCHS_REACHED"

    for epoch in range(1, args.epochs + 1):
        query_matrix, document_matrix = model_arrays()
        current_train_scores = score_candidates_fp32(
            train_bundle,
            query_matrix,
            document_matrix,
            top_b=args.top_b,
            max_correction=args.max_correction,
            batch_size=args.score_batch_size,
        )
        pairs = mine_dynamic_boundary_pairs(
            train_bundle["labels"],
            current_train_scores,
            final_k=args.final_k,
            top_b=args.top_b,
            max_negatives_per_positive=args.max_negatives_per_positive,
            promotion_mix=args.promotion_mix,
            minimum_margin=args.minimum_margin,
            margin_multiplier=args.margin_multiplier,
        )
        if pairs.promotion_count == 0:
            if epoch == 1:
                raise ValueError("No promotion pairs exist at the PCA warm start")
            stop_reason = "NO_REMAINING_PROMOTION_PAIRS"
            break
        permutation = rng.permutation(len(pairs))
        losses: list[float] = []
        saturation_numerator = 0
        saturation_denominator = 0
        for start in range(0, len(permutation), args.batch_size):
            take = permutation[start:start + args.batch_size]
            qi = pairs.query[take]
            pi = pairs.positive[take]
            ni = pairs.negative[take]
            positive_rows = np.asarray(
                train_bundle["residual_lookup"][qi, pi], dtype=np.int64
            )
            negative_rows = np.asarray(
                train_bundle["residual_lookup"][qi, ni], dtype=np.int64
            )
            if np.any(positive_rows < 0) or np.any(negative_rows < 0):
                raise ValueError("Mined pair contains an invalid residual row")
            q = torch.as_tensor(
                np.asarray(train_bundle["queries"][qi], dtype=np.float32),
                device=device,
            )
            positive_residual = torch.as_tensor(
                np.asarray(
                    train_bundle["residuals"][positive_rows], dtype=np.float32
                ),
                device=device,
            )
            negative_residual = torch.as_tensor(
                np.asarray(
                    train_bundle["residuals"][negative_rows], dtype=np.float32
                ),
                device=device,
            )
            zq = query_projection(q)
            raw_positive = (zq * document_projection(positive_residual)).sum(dim=1)
            raw_negative = (zq * document_projection(negative_residual)).sum(dim=1)
            positive_correction = args.max_correction * torch.tanh(
                raw_positive / args.max_correction
            )
            negative_correction = args.max_correction * torch.tanh(
                raw_negative / args.max_correction
            )
            positive_correction = positive_correction * torch.as_tensor(
                pi < args.top_b, dtype=torch.float32, device=device
            )
            negative_correction = negative_correction * torch.as_tensor(
                ni < args.top_b, dtype=torch.float32, device=device
            )
            base_margin = torch.as_tensor(
                np.asarray(
                    train_bundle["ann_scores"][qi, pi]
                    - train_bundle["ann_scores"][qi, ni],
                    dtype=np.float32,
                ),
                device=device,
            )
            corrected_margin = (
                base_margin + positive_correction - negative_correction
            )
            target_margin = torch.as_tensor(
                pairs.target_margin[take], dtype=torch.float32, device=device
            )
            pair_weight = torch.as_tensor(
                pairs.weight[take], dtype=torch.float32, device=device
            )
            loss_vector = functional.softplus(target_margin - corrected_margin)
            loss_vector = loss_vector + args.correction_l2 * (
                positive_correction.square() + negative_correction.square()
            )
            loss = torch.mean(pair_weight * loss_vector)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*query_projection.parameters(), *document_projection.parameters()],
                args.max_grad_norm,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            saturation_numerator += int(torch.sum(
                torch.abs(torch.cat([raw_positive, raw_negative]))
                >= 2.0 * args.max_correction
            ).detach().cpu())
            saturation_denominator += 2 * len(take)

        query_matrix, document_matrix = model_arrays()
        selection_scores = score_candidates_fp32(
            selection_bundle,
            query_matrix,
            document_matrix,
            top_b=args.top_b,
            max_correction=args.max_correction,
            batch_size=args.score_batch_size,
        )
        selection_recall = recall(selection_bundle, selection_scores)
        promotion_mask = pairs.kind == PROMOTION
        protection_mask = pairs.kind == PROTECTION
        record = {
            "epoch": epoch,
            "stage": "dynamic_boundary_fp32",
            "loss": float(np.mean(losses)),
            "selection_fp32_recall_at_10": selection_recall,
            "promotion_pairs": pairs.promotion_count,
            "protection_pairs": pairs.protection_count,
            "promotion_queries": int(len(np.unique(pairs.query[promotion_mask]))),
            "protection_queries": int(len(np.unique(pairs.query[protection_mask]))),
            "promotion_weight_fraction": float(
                np.sum(pairs.weight[promotion_mask]) / np.sum(pairs.weight)
            ),
            "protection_weight_fraction": float(
                np.sum(pairs.weight[protection_mask]) / np.sum(pairs.weight)
            ),
            "query_projection_norm": float(np.linalg.norm(query_matrix)),
            "document_projection_norm": float(np.linalg.norm(document_matrix)),
            "raw_correction_saturation_fraction": (
                saturation_numerator / saturation_denominator
            ),
        }
        history.append(record)
        print(
            f"epoch {epoch}/{args.epochs}: loss={record['loss']:.6f} "
            f"selection={selection_recall:.6f} "
            f"promotion={pairs.promotion_count} protection={pairs.protection_count}"
        )
        if selection_recall > best_recall + 1e-12:
            best_recall = selection_recall
            selected_epoch = epoch
            best_state = {
                "query_projection": {
                    key: value.detach().cpu().clone()
                    for key, value in query_projection.state_dict().items()
                },
                "document_projection": {
                    key: value.detach().cpu().clone()
                    for key, value in document_projection.state_dict().items()
                },
            }

    query_projection.load_state_dict(best_state["query_projection"])
    document_projection.load_state_dict(best_state["document_projection"])
    query_matrix, document_matrix = model_arrays()
    selected_scores = score_candidates_fp32(
        selection_bundle,
        query_matrix,
        document_matrix,
        top_b=args.top_b,
        max_correction=args.max_correction,
        batch_size=args.score_batch_size,
    )
    selected_per_query = recall_at_k_per_query(
        selected_scores,
        selection_bundle["labels"],
        selection_bundle["relevant_counts"],
        k=args.final_k,
    )
    base_per_query = recall_at_k_per_query(
        base_selection_scores,
        selection_bundle["labels"],
        selection_bundle["relevant_counts"],
        k=args.final_k,
    )
    selected_recall = float(np.mean(selected_per_query))
    gain_over_base = selected_recall - base_selection_recall
    gain_over_pca = selected_recall - pca_selection_recall
    passes_base = gain_over_base >= args.minimum_gain_over_base
    passes_pca = gain_over_pca >= args.minimum_gain_over_pca
    decision = (
        "GO_TO_THREE_SEED_FP32_REPLICATION"
        if passes_base and passes_pca
        else "STOP_RANK16_LEARNED_SIDECAR"
    )

    atomic_save_npy(args.output_dir / "query_projection.float32.npy", query_matrix)
    atomic_save_npy(
        args.output_dir / "document_projection.float32.npy", document_matrix
    )
    atomic_save_npy(
        args.output_dir / "selection_base_per_query.float64.npy", base_per_query
    )
    atomic_save_npy(
        args.output_dir / "selection_v2_2_per_query.float64.npy", selected_per_query
    )
    atomic_write_json(args.output_dir / "training_history.json", history)
    metrics = {
        "query_count": int(len(selected_per_query)),
        "base_recall_at_10": base_selection_recall,
        "pca_fp32_recall_at_10": pca_selection_recall,
        "pca_parameter_warm_start_bounded_recall_at_10": initial_recall,
        "v2_2_fp32_recall_at_10": selected_recall,
        "gain_over_base": gain_over_base,
        "gain_over_pca_fp32": gain_over_pca,
        "improved_queries_vs_base": int(np.sum(selected_per_query > base_per_query)),
        "harmed_queries_vs_base": int(np.sum(selected_per_query < base_per_query)),
        "unchanged_queries_vs_base": int(np.sum(selected_per_query == base_per_query)),
        "minimum_required_gain_over_base": args.minimum_gain_over_base,
        "minimum_required_gain_over_pca": args.minimum_gain_over_pca,
        "passes_base_gain_gate": passes_base,
        "passes_pca_gain_gate": passes_pca,
        "decision": decision,
    }
    atomic_write_json(args.output_dir / "selection_metrics.json", metrics)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "FP32_STAGE_A_COMPLETE",
        "decision": decision,
        "run_id": run_id,
        "source_commit": args.source_commit,
        "training_stage": "fp32_representation_gate",
        "method": (
            "PCA-warm-started no-gate rank-16 bilinear residual scorer with "
            "dynamic macro-query boundary loss"
        ),
        "rank": args.rank,
        "top_b": args.top_b,
        "selected_epoch": selected_epoch,
        "completed_epochs": max(item["epoch"] for item in history),
        "stop_reason": stop_reason,
        "quantization": "none",
        "document_codes_written": False,
        "query_gate_present": False,
        "selection": metrics,
        "configuration": _configuration(args),
        "fingerprint_payload": fingerprint_payload,
        "data_access": started["data_access"],
        "source_bundle_access_disclosure": train_bundle["manifest"]["data_access"],
    }
    atomic_write_json(args.output_dir / "training_summary.json", summary)
    output_names = [
        "training_started.json",
        "query_projection.float32.npy",
        "document_projection.float32.npy",
        "selection_base_per_query.float64.npy",
        "selection_v2_2_per_query.float64.npy",
        "training_history.json",
        "selection_metrics.json",
        "training_summary.json",
    ]
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "TRAINING_COMPLETE",
        "run_id": run_id,
        "outputs": _output_records(args.output_dir, output_names),
    }
    atomic_write_json(args.output_dir / "training_complete.json", complete)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--selection-bundle-dir", required=True, type=Path)
    parser.add_argument("--pca-basis", required=True, type=Path)
    parser.add_argument("--pca-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--top-b", type=int, default=40)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--promotion-mix", type=float, default=0.8)
    parser.add_argument("--minimum-margin", type=float, default=1e-4)
    parser.add_argument("--margin-multiplier", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correction-l2", type=float, default=1e-3)
    parser.add_argument("--max-correction", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-gain-over-base", type=float, default=0.01135)
    parser.add_argument("--minimum-gain-over-pca", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2))


if __name__ == "__main__":
    main()
