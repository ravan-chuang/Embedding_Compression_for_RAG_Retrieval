#!/usr/bin/env python3
"""Train the RARS-v5 low-rank adapter through fixed hard residual-PQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rars_v5_pq_aware_core import (
        PROTOCOL_ID,
        build_pool_boundary_pairs,
        hard_residual_pq_ste,
        known_positive_recall_at_k,
        paired_bootstrap_mean_difference,
        recovery_fraction,
    )
except ModuleNotFoundError:  # Allows import as ``scripts.<module>`` in tests.
    from scripts.rars_v5_pq_aware_core import (
        PROTOCOL_ID,
        build_pool_boundary_pairs,
        hard_residual_pq_ste,
        known_positive_recall_at_k,
        paired_bootstrap_mean_difference,
        recovery_fraction,
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_record(path: Path, record: dict[str, Any], description: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{description} byte count changed")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{description} hash changed")


def load_role(bundle_root: Path, role_name: str) -> dict[str, Any]:
    role_dir = bundle_root / role_name
    manifest_path = role_dir / "pilot_role_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing {role_name} role manifest")
    manifest = read_json(manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected pilot role protocol")
    arrays: dict[str, np.ndarray] = {}
    names = {
        "queries": "query_vectors.float32.npy",
        "candidate_rows": "candidate_local_rows.int64.npy",
        "valid": "candidate_valid.bool.npy",
        "labels": "candidate_relevance.uint8.npy",
        "counts": "known_relevant_counts.int32.npy",
        "teacher_scores": "teacher_exact_scores.float32.npy",
        "pq_scores": "base_pq_scores.float32.npy",
        "positive_rows": "known_positive_local_rows.int64.npy",
        "positive_valid": "known_positive_valid.bool.npy",
        "base_top_rows": "base_pq_top_rows.int64.npy",
        "teacher_top_rows": "teacher_exact_top_rows.int64.npy",
        "base_r10": "base_pq_recall_at_10.float64.npy",
        "base_r100": "base_pq_recall_at_100.float64.npy",
        "teacher_r100": "teacher_exact_recall_at_100.float64.npy",
    }
    for key, filename in names.items():
        path = role_dir / filename
        verify_record(path, manifest["files"][filename], f"{role_name}/{filename}")
        arrays[key] = np.load(path, mmap_mode="r")
    shape = arrays["candidate_rows"].shape
    for key in ("valid", "labels", "teacher_scores", "pq_scores"):
        if arrays[key].shape != shape:
            raise ValueError(f"{role_name} candidate matrix shape changed: {key}")
    if arrays["queries"].shape[0] != shape[0] or arrays["counts"].shape != (shape[0],):
        raise ValueError(f"{role_name} query shape changed")
    if arrays["positive_rows"].shape != arrays["positive_valid"].shape:
        raise ValueError(f"{role_name} positive matrix shape changed")
    query_count = shape[0]
    for key in ("base_top_rows", "teacher_top_rows"):
        if arrays[key].shape[0] != query_count:
            raise ValueError(f"{role_name} retrieval row count changed: {key}")
    for key in ("base_r10", "base_r100", "teacher_r100"):
        if arrays[key].shape != (query_count,):
            raise ValueError(f"{role_name} metric vector shape changed: {key}")
    if np.any(np.asarray(arrays["candidate_rows"])[np.asarray(arrays["valid"])] < 0):
        raise ValueError(f"{role_name} contains an invalid candidate row")
    arrays["manifest"] = manifest  # type: ignore[assignment]
    return arrays


def _adapter_arrays(module: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        module.down.weight.detach().cpu().numpy().T.astype(np.float32),
        module.up.weight.detach().cpu().numpy().T.astype(np.float32),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    protocol = read_json(args.protocol)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected v5 protocol")
    if len(args.source_commit) != 40:
        raise ValueError("source_commit must be a full 40-character Git commit")
    int(args.source_commit, 16)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty adapter output directory")
    frozen = protocol["frozen_training_configuration"]
    gate = protocol["stage_a_gate"]
    expected = {
        "seed": int(frozen["seed"]),
        "epochs": int(frozen["epochs"]),
        "batch_size": int(frozen["batch_size"]),
        "adapter_scale": float(frozen["adapter_scale"]),
        "score_batch_size": int(frozen["score_batch_size"]),
        "retrieval_query_batch_size": int(frozen["retrieval_query_batch_size"]),
        "learning_rate": float(frozen["learning_rate"]),
        "weight_decay": float(frozen["weight_decay"]),
        "negative_window": int(frozen["negative_window"]),
        "negatives_per_positive": int(frozen["negatives_per_positive"]),
        "mining_margin_temperature": float(frozen["mining_margin_temperature"]),
        "damage_scale": float(frozen["damage_scale"]),
        "flip_bonus": float(frozen["flip_bonus"]),
        "rank_margin": float(frozen["rank_margin"]),
        "loss_temperature": float(frozen["loss_temperature"]),
        "margin_loss_weight": float(frozen["margin_loss_weight"]),
        "reconstruction_loss_weight": float(frozen["reconstruction_loss_weight"]),
        "drift_loss_weight": float(frozen["drift_loss_weight"]),
        "max_grad_norm": float(frozen["maximum_gradient_norm"]),
        "bootstrap_replicates": int(gate["bootstrap_replicates"]),
        "bootstrap_seed": int(gate["bootstrap_seed"]),
        "minimum_recall_gain": float(gate["minimum_hard_pq_recall_at_100_gain"]),
        "minimum_gap_recovery": float(gate["minimum_teacher_gap_recovery_fraction"]),
        "minimum_improved_queries": int(gate["minimum_improved_queries"]),
        "minimum_improved_fraction": float(gate["minimum_improved_query_fraction"]),
        "maximum_recall_at_10_drop": float(gate["maximum_hard_pq_recall_at_10_drop"]),
        "maximum_fp32_recall_drop": float(gate["maximum_adapted_fp32_recall_at_100_drop"]),
        "device": str(frozen["device"]),
        "n_docs": int(
            protocol["data_policy"]["pilot_corpus"][
                "source_corpus_document_count"
            ]
        ),
    }
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise ValueError(f"Trainer arguments violate the frozen protocol: {actual}")
    index_contract = protocol["pilot_index"]
    if args.pool_k != int(index_contract["candidate_pool_metric_k"]):
        raise ValueError("pool_k violates the frozen protocol")
    if args.final_k != int(index_contract["final_metric_k"]):
        raise ValueError("final_k violates the frozen protocol")
    if args.adapter_rank != int(protocol["adapter"]["rank"]):
        raise ValueError("adapter_rank violates the frozen protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.bundle_root / "pilot_bundle_summary.json"
    summary = read_json(summary_path)
    if summary.get("status") != "PILOT_BUNDLE_COMPLETE":
        raise ValueError("Pilot bundle is incomplete")
    if summary.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Pilot bundle protocol changed")
    if summary["data_access"].get("future_method_holdout_opened") is not False:
        raise ValueError("Future holdout access is forbidden")
    if summary["data_access"].get("external_collection_opened") is not False:
        raise ValueError("External collection access is forbidden")
    train_role = load_role(args.bundle_root, "train")
    selection = load_role(args.bundle_root, "selection")

    dimension = int(summary["dimension"])
    pilot_docs = int(summary["pilot_docs"])
    global_rows_path = args.bundle_root / "pilot_global_doc_rows.int64.npy"
    assignments_path = args.bundle_root / "pilot_coarse_assignments.int64.npy"
    coarse_path = args.bundle_root / "coarse_centroids.float32.npy"
    pq_path = args.bundle_root / "pq_centroids.float32.npy"
    for path in (global_rows_path, assignments_path, coarse_path, pq_path):
        verify_record(path, summary["shared_files"][path.name], path.name)
    global_rows = np.load(global_rows_path, mmap_mode="r")
    coarse_assignments = np.load(assignments_path, mmap_mode="r")
    coarse_centroids_np = np.load(coarse_path)
    pq_centroids_np = np.load(pq_path)
    if global_rows.shape != (pilot_docs,) or coarse_assignments.shape != (pilot_docs,):
        raise ValueError("Pilot corpus mapping shape changed")
    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(args.n_docs, dimension),
    )

    pairs = build_pool_boundary_pairs(
        train_role["teacher_scores"],
        train_role["pq_scores"],
        train_role["labels"],
        train_role["valid"],
        pool_k=args.pool_k,
        negative_window=args.negative_window,
        negatives_per_positive=args.negatives_per_positive,
        margin_temperature=args.mining_margin_temperature,
        damage_scale=args.damage_scale,
        flip_bonus=args.flip_bonus,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no GPU is available")
    expected_workspace = protocol["execution_environment_contract"][
        "cublas_workspace_config"
    ]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_workspace:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must equal {expected_workspace!r}"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    np_rng = np.random.default_rng(args.seed)

    class ResidualAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = torch.nn.Linear(dimension, args.adapter_rank, bias=False)
            self.up = torch.nn.Linear(args.adapter_rank, dimension, bias=False)
            torch.nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.up.weight)

        def forward(self, values: Any) -> Any:
            adapted = values + args.adapter_scale * self.up(self.down(values))
            return functional.normalize(adapted, dim=1)

    query_adapter = ResidualAdapter().to(device)
    document_adapter = ResidualAdapter().to(device)
    optimizer = torch.optim.AdamW(
        [*query_adapter.parameters(), *document_adapter.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    coarse_centroids = torch.as_tensor(coarse_centroids_np, device=device)
    pq_centroids = torch.as_tensor(pq_centroids_np, device=device)

    configuration = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "adapter_rank": args.adapter_rank,
        "adapter_scale": args.adapter_scale,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pool_k": args.pool_k,
        "final_k": args.final_k,
        "negative_window": args.negative_window,
        "negatives_per_positive": args.negatives_per_positive,
        "mining_margin_temperature": args.mining_margin_temperature,
        "damage_scale": args.damage_scale,
        "flip_bonus": args.flip_bonus,
        "rank_margin": args.rank_margin,
        "loss_temperature": args.loss_temperature,
        "margin_loss_weight": args.margin_loss_weight,
        "reconstruction_loss_weight": args.reconstruction_loss_weight,
        "drift_loss_weight": args.drift_loss_weight,
        "max_grad_norm": args.max_grad_norm,
        "retrieval_query_batch_size": args.retrieval_query_batch_size,
        "nprobe": int(summary["ivfpq"]["nprobe"]),
        "device": str(device),
    }
    run_payload = {
        "protocol_sha256": sha256_file(args.protocol),
        "bundle_summary_sha256": sha256_file(summary_path),
        "source_commit": args.source_commit,
        "configuration": configuration,
    }
    run_id = canonical_sha256(run_payload)
    started_path = args.output_dir / "training_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "TRAINING_STARTED",
            "run_id": run_id,
            "run_payload": run_payload,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "pair_count": len(pairs),
            "pq_induced_flip_pairs": int(np.sum(pairs.pq_flip)),
            "zero_label_semantics": "unjudged mined hard negative",
        },
    )

    def document_batch(local_rows: np.ndarray) -> tuple[Any, Any]:
        local = np.asarray(local_rows, dtype=np.int64)
        global_doc_rows = np.asarray(global_rows[local], dtype=np.int64)
        raw = torch.as_tensor(
            np.asarray(embeddings[global_doc_rows], dtype=np.float32), device=device
        )
        raw = functional.normalize(raw, dim=1)
        coarse_ids = np.asarray(coarse_assignments[local], dtype=np.int64)
        coarse = coarse_centroids[
            torch.as_tensor(coarse_ids, dtype=torch.long, device=device)
        ]
        return raw, coarse

    assignment_tensor = torch.as_tensor(
        np.asarray(coarse_assignments, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    nprobe = int(summary["ivfpq"]["nprobe"])

    def retrieve_role(role: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Run end-to-end hard IVF-PQ and adapted-FP32 retrieval on 100K."""

        query_adapter.eval()
        document_adapter.eval()
        with torch.no_grad():
            hard_corpus = torch.empty(
                (pilot_docs, dimension), dtype=torch.float32, device=device
            )
            fp32_corpus = torch.empty_like(hard_corpus)
            for start in range(0, pilot_docs, args.score_batch_size):
                end = min(start + args.score_batch_size, pilot_docs)
                local = np.arange(start, end, dtype=np.int64)
                raw_docs, coarse = document_batch(local)
                adapted_docs = document_adapter(raw_docs)
                _, hard_docs, _ = hard_residual_pq_ste(
                    adapted_docs, coarse, pq_centroids
                )
                hard_corpus[start:end] = hard_docs
                fp32_corpus[start:end] = adapted_docs
            raw_queries = torch.as_tensor(
                np.asarray(role["queries"], dtype=np.float32), device=device
            )
            adapted_queries = query_adapter(raw_queries)
            probe_rows = torch.topk(
                adapted_queries @ coarse_centroids.T,
                k=nprobe,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            hard_top_rows = np.empty(
                (len(adapted_queries), args.pool_k), dtype=np.int64
            )
            fp32_top_rows = np.empty_like(hard_top_rows)
            for start in range(
                0, len(adapted_queries), args.retrieval_query_batch_size
            ):
                end = min(
                    start + args.retrieval_query_batch_size, len(adapted_queries)
                )
                query_values = adapted_queries[start:end]
                hard_scores = query_values @ hard_corpus.T
                eligible = torch.any(
                    assignment_tensor[None, None, :]
                    == probe_rows[start:end, :, None],
                    dim=1,
                )
                hard_scores.masked_fill_(~eligible, -torch.inf)
                hard_top_rows[start:end] = torch.topk(
                    hard_scores,
                    k=args.pool_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices.cpu().numpy()
                fp32_top_rows[start:end] = torch.topk(
                    query_values @ fp32_corpus.T,
                    k=args.pool_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices.cpu().numpy()
        query_adapter.train()
        document_adapter.train()
        return hard_top_rows, fp32_top_rows

    def recall(rows: np.ndarray, role: dict[str, Any], k: int) -> np.ndarray:
        return known_positive_recall_at_k(
            rows,
            role["positive_rows"],
            role["positive_valid"],
            k=k,
        )

    def metric_record(epoch: int, mean_loss: float | None) -> dict[str, Any]:
        hard_rows, fp32_rows = retrieve_role(selection)
        hard_r10 = recall(hard_rows, selection, args.final_k)
        hard_r100 = recall(hard_rows, selection, args.pool_k)
        fp32_r100 = recall(fp32_rows, selection, args.pool_k)
        if epoch == 0:
            base_r10 = np.asarray(selection["base_r10"], dtype=np.float64)
            base_r100 = np.asarray(selection["base_r100"], dtype=np.float64)
            if not (
                np.array_equal(hard_r10, base_r10)
                and np.array_equal(hard_r100, base_r100)
            ):
                raise AssertionError(
                    "Epoch-zero hard retrieval does not reproduce the frozen IVF-PQ baseline"
                )
        return {
            "epoch": epoch,
            "loss": mean_loss,
            "hard_pq_recall_at_10": float(np.mean(hard_r10)),
            "hard_pq_recall_at_100": float(np.mean(hard_r100)),
            "adapted_fp32_recall_at_100": float(np.mean(fp32_r100)),
        }

    history: list[dict[str, Any]] = [metric_record(0, None)]
    best_epoch = 0
    best_state = {
        "query": {key: value.detach().cpu().clone() for key, value in query_adapter.state_dict().items()},
        "document": {key: value.detach().cpu().clone() for key, value in document_adapter.state_dict().items()},
    }
    pair_order = np.arange(len(pairs), dtype=np.int64)
    for epoch in range(1, args.epochs + 1):
        np_rng.shuffle(pair_order)
        epoch_losses: list[float] = []
        for start in range(0, len(pair_order), args.batch_size):
            chosen = pair_order[start : start + args.batch_size]
            qi = pairs.query[chosen]
            pp = pairs.positive[chosen]
            np_ = pairs.negative[chosen]
            positive_local = np.asarray(
                train_role["candidate_rows"][qi, pp], dtype=np.int64
            )
            negative_local = np.asarray(
                train_role["candidate_rows"][qi, np_], dtype=np.int64
            )
            raw_q = torch.as_tensor(
                np.asarray(train_role["queries"][qi], dtype=np.float32), device=device
            )
            raw_p, coarse_p = document_batch(positive_local)
            raw_n, coarse_n = document_batch(negative_local)
            q = query_adapter(raw_q)
            p = document_adapter(raw_p)
            n = document_adapter(raw_n)
            p_ste, p_hard, _ = hard_residual_pq_ste(p, coarse_p, pq_centroids)
            n_ste, n_hard, _ = hard_residual_pq_ste(n, coarse_n, pq_centroids)
            student_margin = torch.sum(q * (p_ste - n_ste), dim=1)
            teacher_margin = torch.as_tensor(
                pairs.teacher_margin[chosen], device=device
            )
            weight = torch.as_tensor(pairs.weight[chosen], device=device)
            rank_loss = torch.mean(
                weight
                * functional.softplus(
                    (args.rank_margin - student_margin) / args.loss_temperature
                )
            )
            margin_loss = torch.mean(
                weight * functional.smooth_l1_loss(
                    student_margin, teacher_margin, reduction="none"
                )
            )
            reconstruction_loss = 0.5 * (
                functional.mse_loss(p, p_hard.detach())
                + functional.mse_loss(n, n_hard.detach())
            )
            drift_loss = (
                (1.0 - torch.sum(q * raw_q, dim=1)).mean()
                + (1.0 - torch.sum(p * raw_p, dim=1)).mean()
                + (1.0 - torch.sum(n * raw_n, dim=1)).mean()
            ) / 3.0
            loss = (
                rank_loss
                + args.margin_loss_weight * margin_loss
                + args.reconstruction_loss_weight * reconstruction_loss
                + args.drift_loss_weight * drift_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*query_adapter.parameters(), *document_adapter.parameters()],
                args.max_grad_norm,
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        record = metric_record(epoch, float(np.mean(epoch_losses)))
        history.append(record)
        incumbent = history[best_epoch]
        score = (
            record["hard_pq_recall_at_100"],
            record["hard_pq_recall_at_10"],
            record["adapted_fp32_recall_at_100"],
        )
        incumbent_score = (
            incumbent["hard_pq_recall_at_100"],
            incumbent["hard_pq_recall_at_10"],
            incumbent["adapted_fp32_recall_at_100"],
        )
        if score > incumbent_score:
            best_epoch = epoch
            best_state = {
                "query": {
                    key: value.detach().cpu().clone()
                    for key, value in query_adapter.state_dict().items()
                },
                "document": {
                    key: value.detach().cpu().clone()
                    for key, value in document_adapter.state_dict().items()
                },
            }
        print(json.dumps(record, allow_nan=False))

    query_adapter.load_state_dict(best_state["query"])
    document_adapter.load_state_dict(best_state["document"])
    selected_hard_rows, selected_fp32_rows = retrieve_role(selection)
    base_r10 = np.asarray(selection["base_r10"], dtype=np.float64)
    base_r100 = np.asarray(selection["base_r100"], dtype=np.float64)
    teacher_r100 = np.asarray(selection["teacher_r100"], dtype=np.float64)
    selected_r10 = recall(selected_hard_rows, selection, args.final_k)
    selected_r100 = recall(selected_hard_rows, selection, args.pool_k)
    adapted_fp32_r100 = recall(selected_fp32_rows, selection, args.pool_k)
    bootstrap = paired_bootstrap_mean_difference(
        selected_r100,
        base_r100,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    improved = int(np.sum(selected_r100 > base_r100))
    harmed = int(np.sum(selected_r100 < base_r100))
    mean_base_r100 = float(np.mean(base_r100))
    mean_selected_r100 = float(np.mean(selected_r100))
    mean_teacher_r100 = float(np.mean(teacher_r100))
    recovered = recovery_fraction(
        base=mean_base_r100,
        treatment=mean_selected_r100,
        teacher=mean_teacher_r100,
    )
    required_support = max(
        args.minimum_improved_queries,
        int(np.ceil(args.minimum_improved_fraction * len(base_r100))),
    )
    gates = {
        "minimum_recall_at_100_gain": (
            mean_selected_r100 - mean_base_r100 >= args.minimum_recall_gain
        ),
        "bootstrap_lower_above_zero": bootstrap["lower"] > 0.0,
        "minimum_gap_recovery": recovered >= args.minimum_gap_recovery,
        "minimum_improved_query_support": improved >= required_support,
        "recall_at_10_guardrail": (
            float(np.mean(selected_r10))
            >= float(np.mean(base_r10)) - args.maximum_recall_at_10_drop
        ),
        "adapted_fp32_guardrail": (
            float(np.mean(adapted_fp32_r100))
            >= mean_teacher_r100 - args.maximum_fp32_recall_drop
        ),
    }
    formal_decision = (
        "GO_TO_THREE_SEED_100K_REPLICATION"
        if all(gates.values())
        else "STOP_PQ_AWARE_100K_PILOT"
    )

    query_down, query_up = _adapter_arrays(query_adapter)
    document_down, document_up = _adapter_arrays(document_adapter)
    outputs = {
        "query_adapter_down.float32.npy": query_down,
        "query_adapter_up.float32.npy": query_up,
        "document_adapter_down.float32.npy": document_down,
        "document_adapter_up.float32.npy": document_up,
        "selection_hard_pq_top_rows.int64.npy": selected_hard_rows,
        "selection_adapted_fp32_top_rows.int64.npy": selected_fp32_rows,
        "selection_base_recall_at_100.float64.npy": base_r100,
        "selection_adapter_recall_at_100.float64.npy": selected_r100,
    }
    output_paths: dict[str, Path] = {}
    for filename, value in outputs.items():
        path = args.output_dir / filename
        atomic_save(path, value)
        output_paths[filename] = path
    history_path = args.output_dir / "training_history.json"
    atomic_json(history_path, history)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "TRAINING_COMPLETE",
        "run_id": run_id,
        "formal_decision": formal_decision,
        "selected_epoch": best_epoch,
        "pair_count": len(pairs),
        "pq_induced_flip_pairs": int(np.sum(pairs.pq_flip)),
        "selection": {
            "query_count": len(base_r100),
            "base_pq_recall_at_10": float(np.mean(base_r10)),
            "adapter_hard_pq_recall_at_10": float(np.mean(selected_r10)),
            "base_pq_recall_at_100": mean_base_r100,
            "adapter_hard_pq_recall_at_100": mean_selected_r100,
            "teacher_exact_recall_at_100": mean_teacher_r100,
            "adapted_fp32_recall_at_100": float(np.mean(adapted_fp32_r100)),
            "recall_at_100_gain": mean_selected_r100 - mean_base_r100,
            "teacher_gap_recovery_fraction": recovered,
            "improved_queries": improved,
            "harmed_queries": harmed,
            "required_improved_queries": required_support,
            "paired_bootstrap": bootstrap,
        },
        "gates": gates,
        "interpretation": (
            "development-only end-to-end 100K known-positive pilot; not "
            "official MS MARCO Recall and not external confirmation"
        ),
        "configuration": configuration,
    }
    result_path = args.output_dir / "pilot_result.json"
    atomic_json(result_path, result)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "TRAINING_COMPLETE",
        "run_id": run_id,
        "source_commit": args.source_commit,
        "formal_decision": formal_decision,
        "started": file_record(started_path),
        "history": file_record(history_path),
        "result": file_record(result_path),
        "outputs": {name: file_record(path) for name, path in output_paths.items()},
    }
    atomic_json(args.output_dir / "training_complete.json", complete)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "protocols/rars_v5_pq_aware_100k_pilot_v1.json",
    )
    parser.add_argument("--n-docs", type=int, default=1_000_000)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--score-batch-size", type=int, default=8192)
    parser.add_argument("--retrieval-query-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-window", type=int, default=16)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--mining-margin-temperature", type=float, default=0.05)
    parser.add_argument("--damage-scale", type=float, default=8.0)
    parser.add_argument("--flip-bonus", type=float, default=2.0)
    parser.add_argument("--rank-margin", type=float, default=0.001)
    parser.add_argument("--loss-temperature", type=float, default=0.05)
    parser.add_argument("--margin-loss-weight", type=float, default=0.5)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=0.01)
    parser.add_argument("--drift-loss-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--minimum-recall-gain", type=float, default=0.005)
    parser.add_argument("--minimum-gap-recovery", type=float, default=0.15)
    parser.add_argument("--minimum-improved-queries", type=int, default=20)
    parser.add_argument("--minimum-improved-fraction", type=float, default=0.02)
    parser.add_argument("--maximum-recall-at-10-drop", type=float, default=0.002)
    parser.add_argument("--maximum-fp32-recall-drop", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2))


if __name__ == "__main__":
    main()
