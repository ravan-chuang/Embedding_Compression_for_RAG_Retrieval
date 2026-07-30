#!/usr/bin/env python3
"""Train the frozen-index RARS-v7 query-side PQ adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
    from scripts import evaluate_rars_v6_1m_headroom as v6_eval
    from scripts.rars_v6_headroom_core import (
        known_positive_recall_at_k,
        map_qrels_doc_ids_to_corpus_rows,
        mine_pq_induced_flip_triplets,
    )
    from scripts.rars_v7_query_adapter_core import (
        PROMOTION,
        PROTECTION,
        concatenate_pairs,
        deterministic_query_split,
        mine_top10_protection_pairs,
        newline_sha256,
        paired_bootstrap_mean_difference,
        pilot_gate_decision,
        promotion_pairs_from_v6,
        select_checkpoint,
        subset_pairs,
        summarize_pairs,
    )
    from scripts.verify_rars_v6_1m_headroom_packet import (
        sha256_file,
        verify_packet as verify_v6_packet,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import evaluate_rars_v6_1m_headroom as v6_eval
    from rars_v6_headroom_core import (
        known_positive_recall_at_k,
        map_qrels_doc_ids_to_corpus_rows,
        mine_pq_induced_flip_triplets,
    )
    from rars_v7_query_adapter_core import (
        PROMOTION,
        PROTECTION,
        concatenate_pairs,
        deterministic_query_split,
        mine_top10_protection_pairs,
        newline_sha256,
        paired_bootstrap_mean_difference,
        pilot_gate_decision,
        promotion_pairs_from_v6,
        select_checkpoint,
        subset_pairs,
        summarize_pairs,
    )
    from verify_rars_v6_1m_headroom_packet import (
        sha256_file,
        verify_packet as verify_v6_packet,
    )


PROTOCOL_ID = "rars_v7_query_adapter_pilot_v1"
CANONICAL_PROTOCOL = Path("protocols/rars_v7_query_adapter_pilot_v1.json")
CANONICAL_SOURCES = (
    Path("scripts/rars_v7_query_adapter_core.py"),
    Path("scripts/train_rars_v7_query_adapter.py"),
    Path("scripts/verify_rars_v6_1m_headroom_packet.py"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


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
        np.save(handle, np.asarray(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} SHA-256 changed")


def _validate_exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be an exact lowercase 40-hex commit")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_exact_commit(source_commit)
    if protocol_path.resolve() != (repo_root / CANONICAL_PROTOCOL).resolve():
        raise ValueError("--protocol must be the canonical v7 protocol path")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or dirty:
        raise ValueError("V7 training requires an exact clean pinned Git commit")
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected v7 protocol identity")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_V7_TRAINING_RUN":
        raise ValueError("V7 protocol is not frozen before training")
    if protocol.get("method_revision_allowed") is not False or protocol.get(
        "outcome_informed_revision_allowed"
    ) is not False:
        raise ValueError("V7 protocol permits an outcome-informed revision")

    records: dict[str, Any] = {}
    for relative in (CANONICAL_PROTOCOL, *CANONICAL_SOURCES):
        local = (repo_root / relative).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative.as_posix()}"],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
        )
        if local != committed:
            raise ValueError(f"Canonical source differs from Git: {relative}")
        blob = subprocess.check_output(
            ["git", "rev-parse", f"{source_commit}:{relative.as_posix()}"],
            cwd=repo_root,
            text=True,
        ).strip()
        records[relative.as_posix()] = {
            "sha256": hashlib.sha256(local).hexdigest(),
            "git_blob_oid": blob,
        }
    return protocol, records


def validate_runtime(
    protocol: dict[str, Any], args: argparse.Namespace, torch: Any, faiss: Any
) -> dict[str, Any]:
    environment = protocol["execution_environment_contract"]
    observed = {
        "python": sys.version,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "faiss": str(getattr(faiss, "__version__", "UNKNOWN")),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    expected = {
        "python_version": environment["python_version"],
        "numpy": environment["numpy_version"],
        "torch": environment["torch_version"],
        "torch_cuda": environment["torch_cuda_version"],
        "faiss": "1.12.0",
        "cublas_workspace_config": environment["cublas_workspace_config"],
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"Runtime {key}={observed[key]!r}; expected {value!r}")
    if not torch.cuda.is_available() or environment["gpu_name_must_contain"] not in str(
        observed["gpu"]
    ):
        raise ValueError("The canonical v7 pilot requires the registered T4 GPU")
    if faiss.get_num_gpus() <= 0:
        raise ValueError("The canonical Faiss package lacks GPU support")
    frozen = protocol["training"]
    expected_args = {
        "seed": frozen["seed"],
        "epochs": frozen["maximum_epochs"],
        "batch_size": frozen["pair_batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "max_grad_norm": frozen["maximum_gradient_norm"],
        "corpus_load_batch_size": frozen["corpus_load_batch_size"],
        "candidate_batch_size": frozen["candidate_batch_size"],
        "bootstrap_replicates": frozen["bootstrap_replicates"],
        "bootstrap_seed": frozen["bootstrap_seed"],
    }
    for name, expected_value in expected_args.items():
        if getattr(args, name) != expected_value:
            raise ValueError(f"--{name.replace('_', '-')} violates the frozen protocol")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Deterministic PyTorch algorithms are not enabled")
    observed["deterministic_algorithms"] = True
    observed["cudnn_benchmark"] = False
    return observed


def prepare_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("Refusing to reuse a non-empty v7 output directory")
    path.mkdir(parents=True, exist_ok=True)


def load_pq_corpus_tensor(
    index: Any,
    *,
    document_count: int,
    dimension: int,
    device: Any,
    batch_size: int,
    torch: Any,
) -> Any:
    tensor = torch.empty(
        (document_count, dimension), dtype=torch.float32, device=device
    )
    for start in range(0, document_count, batch_size):
        end = min(start + batch_size, document_count)
        rows = np.arange(start, end, dtype=np.int64)
        reconstructed = np.asarray(index.reconstruct_batch(rows), dtype=np.float32)
        if reconstructed.shape != (end - start, dimension) or np.any(
            ~np.isfinite(reconstructed)
        ):
            raise ValueError("Frozen PQ reconstruction is invalid")
        tensor[start:end].copy_(torch.from_numpy(reconstructed), non_blocking=False)
    return tensor


def train(args: argparse.Namespace) -> dict[str, Any]:
    import faiss
    import torch
    import torch.nn.functional as functional

    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = validate_source(
        repo_root, args.protocol, args.source_commit
    )
    environment = validate_runtime(protocol, args, torch, faiss)
    v6_verification = verify_v6_packet(args.v6_packet_root)
    precondition = protocol["precondition"]
    if (
        v6_verification["source_commit"] != precondition["v6_source_commit"]
        or v6_verification["formal_decision"] != precondition["required_v6_decision"]
    ):
        raise ValueError("The durable v6 packet does not satisfy the v7 precondition")
    v6_expected = {
        "pq_specific_recall_at_100_gap": precondition[
            "observed_pq_specific_recall_at_100_gap"
        ],
        "uncapped_flip_triplets": precondition["observed_uncapped_flip_triplets"],
        "distinct_flip_queries": precondition["observed_distinct_flip_queries"],
        "flip_effective_sample_size": precondition[
            "observed_flip_effective_sample_size"
        ],
    }
    for key, expected in v6_expected.items():
        observed = v6_verification[key]
        if isinstance(expected, float):
            matches = np.isclose(observed, expected, rtol=0.0, atol=1e-12)
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(f"Verified v6 {key}={observed}; expected {expected}")
    prepare_empty_output(args.output_dir)
    started_wall = time.perf_counter()
    index_record_before = file_record(args.index)
    frozen_index = protocol["frozen_index_contract"]
    if index_record_before["bytes"] != frozen_index["index_bytes"] or index_record_before[
        "sha256"
    ] != frozen_index["index_sha256"]:
        raise ValueError("Frozen IVF-PQ index changed before v7 training")
    doc_ids_record = file_record(args.doc_ids)
    if doc_ids_record["bytes"] != frozen_index["doc_ids_bytes"] or doc_ids_record[
        "sha256"
    ] != frozen_index["doc_ids_sha256"]:
        raise ValueError("Frozen corpus document IDs changed")

    v6_protocol = read_json(repo_root / "protocols/rars_v6_1m_headroom_v1.json")
    query_ids, query_vectors_memmap, design_records = v6_eval.load_design_role(
        args.design_role_dir, v6_protocol
    )
    queries = np.array(query_vectors_memmap, dtype=np.float32, copy=True)
    qrels = v6_eval.load_positive_qrels(args.qrels)
    qrel_ids, qrel_valid = v6_eval.pad_qrels_for_queries(query_ids, qrels)
    doc_ids = np.memmap(
        args.doc_ids, dtype=np.int64, mode="r", shape=(frozen_index["document_count"],)
    )
    mapping = map_qrels_doc_ids_to_corpus_rows(doc_ids, qrel_ids, qrel_valid)
    positive_rows, positive_valid, coverage = v6_eval.mapping_arrays(mapping)
    split_contract = protocol["data_policy"]["split"]
    split = deterministic_query_split(
        query_ids,
        selection_count=int(split_contract["selection_query_count"]),
        salt=split_contract["salt"],
    )
    if len(split.training) != int(split_contract["training_query_count"]):
        raise ValueError("Deterministic v7 training split has the wrong size")
    split_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method": split_contract["method"],
        "salt": split_contract["salt"],
        "source_query_count": len(query_ids),
        "training_query_count": len(split.training),
        "selection_query_count": len(split.selection),
        "training_source_indices": split.training.tolist(),
        "selection_source_indices": split.selection.tolist(),
        "training_source_order_qid_sha256": newline_sha256(split.training_qids),
        "selection_source_order_qid_sha256": newline_sha256(split.selection_qids),
        "query_disjoint": True,
        "label_blind": True,
    }
    split_path = args.output_dir / "split_manifest.json"
    atomic_json(split_path, split_manifest)

    cpu_index = faiss.read_index(str(args.index))
    ivf, index_contract = v6_eval.validate_faiss_index(cpu_index, faiss)
    ivf.nprobe = int(frozen_index["nprobe"])
    inverted_lists, inverted_summary = v6_eval.inverted_lists_as_rows(ivf, faiss)
    candidates, probed_lists = v6_eval.probed_candidate_rows(
        queries, ivf, inverted_lists, nprobe=int(frozen_index["nprobe"])
    )
    registered_probes = np.load(
        args.v6_packet_root / "probed_ivf_lists.int64.npy", allow_pickle=False
    )
    if not np.array_equal(probed_lists, registered_probes):
        raise ValueError("Original-query IVF routing differs from the verified v6 packet")
    # Faiss IndexIVFPQ reconstruction by corpus row requires a DirectMap.  The
    # serialized frozen index deliberately does not contain one.  Match the
    # verified v6 evaluator by building this lookup only on the in-memory
    # downcast IVF object; the index file is never written and its before/after
    # byte hash remains guarded below.
    ivf.make_direct_map()
    base_rows = np.load(
        args.v6_packet_root / "base_pq_top_rows.int64.npy", allow_pickle=False
    )
    ivf_exact_rows = np.load(
        args.v6_packet_root / "ivf_exact_top_rows.int64.npy", allow_pickle=False
    )
    candidate_union, routed_positive_valid, union_summary = (
        v6_eval.build_flip_candidate_union(
            base_rows,
            ivf_exact_rows,
            positive_rows,
            positive_valid,
            candidates,
        )
    )

    device = torch.device(protocol["training"]["device"])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np_rng = np.random.default_rng(args.seed)
    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(frozen_index["document_count"], frozen_index["embedding_dimension"]),
    )
    fp32_corpus = v6_eval.load_corpus_tensor_torch(
        embeddings,
        device=str(device),
        load_batch_size=args.corpus_load_batch_size,
    )
    exact_union_scores, pq_union_scores = v6_eval.score_flip_candidate_union(
        queries, candidate_union, fp32_corpus, cpu_index
    )
    promotion_result = mine_pq_induced_flip_triplets(
        candidate_union,
        exact_union_scores,
        pq_union_scores,
        positive_rows,
        routed_positive_valid,
        pool_k=int(protocol["pair_mining"]["promotion_top_100"]["pool_k"]),
        negative_window=int(
            protocol["pair_mining"]["promotion_top_100"]["negative_window"]
        ),
        max_unjudged_per_positive=int(
            protocol["pair_mining"]["promotion_top_100"][
                "maximum_challengers_per_query_positive"
            ]
        ),
        margin_temperature=0.05,
        damage_scale=8.0,
        flip_bonus=2.0,
    )
    v6_result = read_json(args.v6_packet_root / "headroom_result.json")
    if promotion_result.support != v6_result["flip_support"]:
        raise ValueError("V7 cannot reproduce the verified v6 flip population")
    promotion_pairs = promotion_pairs_from_v6(promotion_result.capped)
    protection_pairs = mine_top10_protection_pairs(
        base_rows,
        candidate_union,
        exact_union_scores,
        pq_union_scores,
        positive_rows,
        routed_positive_valid,
        negative_window=int(
            protocol["pair_mining"]["protection_top_10"]["negative_window"]
        ),
        max_challengers_per_positive=int(
            protocol["pair_mining"]["protection_top_10"][
                "maximum_challengers_per_query_positive"
            ]
        ),
        margin_temperature=0.05,
        damage_scale=8.0,
    )
    all_pairs = concatenate_pairs(promotion_pairs, protection_pairs)
    training_pairs = subset_pairs(all_pairs, split.training)
    selection_pairs = subset_pairs(all_pairs, split.selection)
    training_support = summarize_pairs(training_pairs)
    selection_support = summarize_pairs(selection_pairs)
    if training_support["promotion"]["queries"] < int(
        protocol["pair_mining"]["minimum_training_promotion_queries"]
    ):
        raise ValueError("V7 training split lacks preregistered promotion-query support")
    if selection_support["promotion"]["queries"] < int(
        protocol["pair_mining"]["minimum_selection_promotion_queries"]
    ):
        raise ValueError("V7 selection split lacks preregistered promotion-query support")

    pq_corpus = load_pq_corpus_tensor(
        cpu_index,
        document_count=int(frozen_index["document_count"]),
        dimension=int(frozen_index["embedding_dimension"]),
        device=device,
        batch_size=args.corpus_load_batch_size,
        torch=torch,
    )

    adapter_contract = protocol["adapter"]
    dimension = int(frozen_index["embedding_dimension"])

    class ResidualQueryAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = torch.nn.Linear(
                dimension, int(adapter_contract["rank"]), bias=False
            )
            self.up = torch.nn.Linear(
                int(adapter_contract["rank"]), dimension, bias=False
            )
            torch.nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.up.weight)

        def forward(self, values: Any) -> Any:
            adapted = values + 0.1 * self.up(self.down(values))
            return functional.normalize(adapted, dim=1)

    adapter = ResidualQueryAdapter().to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    selection_indices = np.asarray(split.selection, dtype=np.int64)
    selection_queries = queries[selection_indices]
    selection_candidates = [candidates[int(index)] for index in selection_indices]
    selection_positive_rows = positive_rows[selection_indices]
    selection_positive_valid = positive_valid[selection_indices]
    base_r10 = np.load(
        args.v6_packet_root / "base_pq_recall_at_10.float64.npy", allow_pickle=False
    )[selection_indices]
    base_r100 = np.load(
        args.v6_packet_root / "base_pq_recall_at_100.float64.npy", allow_pickle=False
    )[selection_indices]
    teacher_r100 = np.load(
        args.v6_packet_root / "ivf_exact_recall_at_100.float64.npy", allow_pickle=False
    )[selection_indices]

    def recall(rows: np.ndarray, k: int) -> np.ndarray:
        return known_positive_recall_at_k(
            rows, selection_positive_rows, selection_positive_valid, k=k
        )

    def retrieve_selection() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        adapter.eval()
        with torch.no_grad():
            raw = torch.as_tensor(selection_queries, device=device)
            adapted = adapter(raw)
            cosine = torch.sum(raw * adapted, dim=1).cpu().numpy().astype(np.float64)
            adapted_np = adapted.cpu().numpy().astype(np.float32, copy=False)
            _, hard_rows = v6_eval.within_ivf_exact_topk_torch(
                adapted_np,
                pq_corpus,
                selection_candidates,
                k=100,
                candidate_batch_size=args.candidate_batch_size,
            )
            _, adapted_fp32_rows = v6_eval.within_ivf_exact_topk_torch(
                adapted_np,
                fp32_corpus,
                selection_candidates,
                k=100,
                candidate_batch_size=args.candidate_batch_size,
            )
        adapter.train()
        return hard_rows, adapted_fp32_rows, cosine

    def metric_record(epoch: int, mean_loss: float | None) -> dict[str, Any]:
        hard_rows, adapted_fp32_rows, cosine = retrieve_selection()
        hard_r10 = recall(hard_rows, 10)
        hard_r100 = recall(hard_rows, 100)
        fp32_r100 = recall(adapted_fp32_rows, 100)
        if epoch == 0:
            if not np.array_equal(hard_r10, base_r10) or not np.array_equal(
                hard_r100, base_r100
            ):
                raise AssertionError(
                    "Epoch-zero reconstructed-PQ retrieval does not reproduce v6 Recall"
                )
            if not np.array_equal(fp32_r100, teacher_r100):
                raise AssertionError(
                    "Epoch-zero same-route FP32 retrieval does not reproduce v6 Recall"
                )
        return {
            "epoch": int(epoch),
            "loss": mean_loss,
            "hard_pq_recall_at_10": float(np.mean(hard_r10)),
            "hard_pq_recall_at_100": float(np.mean(hard_r100)),
            "adapted_same_ivf_fp32_recall_at_100": float(np.mean(fp32_r100)),
            "mean_query_cosine": float(np.mean(cosine)),
        }

    configuration = {
        "seed": args.seed,
        "maximum_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "adapter_rank": int(adapter_contract["rank"]),
        "adapter_scale": 0.1,
        "objective": protocol["objective"],
        "routing": frozen_index["routing_query"],
        "scoring": frozen_index["scoring_query"],
    }
    run_payload = {
        "protocol_sha256": sha256_file(args.protocol),
        "v6_result_sha256": v6_verification["result"]["sha256"],
        "source_commit": args.source_commit,
        "split_manifest_sha256": sha256_file(split_path),
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
            "environment": environment,
            "v6_verification": v6_verification,
            "index_before": index_record_before,
            "qrels_mapping": coverage,
            "candidate_union": union_summary,
            "training_pair_support": training_support,
            "selection_pair_support_diagnostic_only": selection_support,
            "zero_label_semantics": "unjudged mined challenger",
            "future_or_audit_role_opened": False,
        },
    )

    thresholds = protocol["pilot_gate"]
    base_r10_mean = float(np.mean(base_r10))
    teacher_r100_mean = float(np.mean(teacher_r100))
    history: list[dict[str, Any]] = [metric_record(0, None)]
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()
    }
    pair_order = np.arange(len(training_pairs), dtype=np.int64)
    epochs_without_improvement = 0
    objective = protocol["objective"]
    training_contract = protocol["training"]

    for epoch in range(1, args.epochs + 1):
        np_rng.shuffle(pair_order)
        losses: list[float] = []
        for start in range(0, len(pair_order), args.batch_size):
            chosen = pair_order[start : start + args.batch_size]
            query_indices = training_pairs.query[chosen]
            positive_rows_batch = training_pairs.positive_row[chosen]
            challenger_rows_batch = training_pairs.challenger_row[chosen]
            raw_query = torch.as_tensor(queries[query_indices], device=device)
            adapted_query = adapter(raw_query)
            positive = pq_corpus[
                torch.as_tensor(positive_rows_batch, dtype=torch.long, device=device)
            ]
            challenger = pq_corpus[
                torch.as_tensor(challenger_rows_batch, dtype=torch.long, device=device)
            ]
            student_margin = torch.sum(adapted_query * (positive - challenger), dim=1)
            teacher_margin = torch.as_tensor(
                training_pairs.teacher_margin[chosen], device=device
            )
            balanced_weight = torch.as_tensor(
                training_pairs.balanced_weight[chosen], device=device
            )
            kinds = torch.as_tensor(training_pairs.kind[chosen], device=device)
            pair_losses = functional.softplus(
                (float(objective["rank_margin"]) - student_margin)
                / float(objective["loss_temperature"])
            )

            def weighted_component(values: Any, mask: Any) -> Any:
                weights = balanced_weight[mask]
                if int(weights.numel()) == 0:
                    return torch.zeros((), dtype=values.dtype, device=device)
                return torch.sum(weights * values[mask]) / torch.sum(weights)

            promotion_loss = weighted_component(pair_losses, kinds == PROMOTION)
            protection_loss = weighted_component(pair_losses, kinds == PROTECTION)
            distillation = functional.smooth_l1_loss(
                student_margin, teacher_margin, reduction="none"
            )
            distillation_loss = torch.sum(balanced_weight * distillation) / torch.sum(
                balanced_weight
            )
            unique_queries = np.unique(query_indices)
            raw_unique = torch.as_tensor(queries[unique_queries], device=device)
            adapted_unique = adapter(raw_unique)
            drift_loss = (1.0 - torch.sum(raw_unique * adapted_unique, dim=1)).mean()
            loss = (
                float(objective["promotion_weight"]) * promotion_loss
                + float(objective["protection_weight"]) * protection_loss
                + float(objective["margin_distillation_weight"]) * distillation_loss
                + float(objective["query_drift_weight"]) * drift_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        record = metric_record(epoch, float(np.mean(losses)))
        history.append(record)
        selected_so_far = select_checkpoint(
            history,
            base_r10=base_r10_mean,
            teacher_r100=teacher_r100_mean,
            maximum_r10_drop=float(thresholds["maximum_hard_pq_recall_at_10_drop"]),
            maximum_teacher_drop=float(
                thresholds["maximum_adapted_same_route_fp32_recall_at_100_drop"]
            ),
        )
        if selected_so_far == epoch:
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(json.dumps(record, allow_nan=False))
        if (
            epoch >= int(training_contract["minimum_epochs_before_early_stop"])
            and epochs_without_improvement
            >= int(training_contract["early_stopping_patience"])
        ):
            break

    selected_epoch = select_checkpoint(
        history,
        base_r10=base_r10_mean,
        teacher_r100=teacher_r100_mean,
        maximum_r10_drop=float(thresholds["maximum_hard_pq_recall_at_10_drop"]),
        maximum_teacher_drop=float(
            thresholds["maximum_adapted_same_route_fp32_recall_at_100_drop"]
        ),
    )
    if selected_epoch != best_epoch:
        raise AssertionError("Saved v7 checkpoint differs from frozen selection rule")
    adapter.load_state_dict(best_state)
    selected_hard_rows, selected_fp32_rows, selected_cosine = retrieve_selection()
    selected_r10 = recall(selected_hard_rows, 10)
    selected_r100 = recall(selected_hard_rows, 100)
    selected_fp32_r100 = recall(selected_fp32_rows, 100)
    bootstrap = paired_bootstrap_mean_difference(
        selected_r100,
        base_r100,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    improved = int(np.sum(selected_r100 > base_r100))
    harmed = int(np.sum(selected_r100 < base_r100))
    decision = pilot_gate_decision(
        selected_epoch=selected_epoch,
        base_r10=base_r10_mean,
        adapted_r10=float(np.mean(selected_r10)),
        base_r100=float(np.mean(base_r100)),
        adapted_r100=float(np.mean(selected_r100)),
        teacher_r100=teacher_r100_mean,
        adapted_teacher_r100=float(np.mean(selected_fp32_r100)),
        bootstrap_lower=float(bootstrap["lower"]),
        improved_queries=improved,
        harmed_queries=harmed,
        mean_query_cosine=float(np.mean(selected_cosine)),
        thresholds=thresholds,
    )

    down = adapter.down.weight.detach().cpu().numpy().astype(np.float32)
    up = adapter.up.weight.detach().cpu().numpy().astype(np.float32)
    output_arrays = {
        "query_adapter_down.float32.npy": down,
        "query_adapter_up.float32.npy": up,
        "selection_hard_pq_top_rows.int64.npy": selected_hard_rows,
        "selection_adapted_fp32_top_rows.int64.npy": selected_fp32_rows,
        "selection_base_recall_at_10.float64.npy": base_r10,
        "selection_adapter_recall_at_10.float64.npy": selected_r10,
        "selection_base_recall_at_100.float64.npy": base_r100,
        "selection_adapter_recall_at_100.float64.npy": selected_r100,
        "selection_teacher_recall_at_100.float64.npy": teacher_r100,
        "selection_adapted_fp32_recall_at_100.float64.npy": selected_fp32_r100,
    }
    for filename, value in output_arrays.items():
        atomic_save(args.output_dir / filename, value)
    history_path = args.output_dir / "training_history.json"
    atomic_json(history_path, history)
    index_record_after = file_record(args.index)
    if index_record_after != index_record_before:
        raise ValueError("Frozen IVF-PQ index changed during v7 training")
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "V7_QUERY_ADAPTER_PILOT_COMPLETE",
        "run_id": run_id,
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "selected_epoch": selected_epoch,
        "epochs_executed": len(history) - 1,
        "selection": {
            "query_count": len(selection_indices),
            "base_pq_recall_at_10": base_r10_mean,
            "adapter_hard_pq_recall_at_10": float(np.mean(selected_r10)),
            "base_pq_recall_at_100": float(np.mean(base_r100)),
            "adapter_hard_pq_recall_at_100": float(np.mean(selected_r100)),
            "same_ivf_exact_teacher_recall_at_100": teacher_r100_mean,
            "adapted_same_ivf_fp32_recall_at_100": float(
                np.mean(selected_fp32_r100)
            ),
            "mean_query_cosine": float(np.mean(selected_cosine)),
            "improved_queries": improved,
            "harmed_queries": harmed,
            "paired_bootstrap": bootstrap,
        },
        "decision": decision,
        "training_pair_support": training_support,
        "selection_pair_support_diagnostic_only": selection_support,
        "index_unchanged": True,
        "in_memory_direct_map_built": True,
        "document_reencoding_performed": False,
        "rars_used": False,
        "oracle_audit_opened": False,
        "future_method_holdout_opened": False,
        "interpretation": (
            "single-seed, outcome-informed 1M development pilot; not official "
            "MS MARCO performance or independent confirmation"
        ),
        "telemetry": {
            "total_wall_seconds": float(time.perf_counter() - started_wall),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "host_max_rss_bytes": max_rss,
            "output_disk_free_bytes": int(shutil.disk_usage(args.output_dir).free),
        },
        "source_blobs": source_blobs,
        "inputs": {
            "index": index_record_before,
            "doc_ids": doc_ids_record,
            "embeddings": file_record(args.embeddings),
            "qrels": file_record(args.qrels),
            **design_records,
            "v6_result": v6_verification["result"],
        },
    }
    result_path = args.output_dir / "pilot_result.json"
    atomic_json(result_path, result)
    output_records = {
        filename: file_record(args.output_dir / filename)
        for filename in output_arrays
    }
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "V7_QUERY_ADAPTER_PILOT_COMPLETE",
        "run_id": run_id,
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "started": file_record(started_path),
        "split": file_record(split_path),
        "history": file_record(history_path),
        "result": file_record(result_path),
        "outputs": output_records,
        "index_before": index_record_before,
        "index_after": index_record_after,
        "in_memory_direct_map_built": True,
        "document_reencoding_performed": False,
        "rars_used": False,
        "oracle_audit_opened": False,
        "future_method_holdout_opened": False,
        "corpus_tensors_persisted": False,
    }
    complete_path = args.output_dir / "training_complete.json"
    atomic_json(complete_path, complete)
    missing = [
        filename
        for filename in protocol["required_outputs"]
        if not (args.output_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"Required v7 outputs were not written: {missing}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-role-dir", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--doc-ids", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--v6-packet-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--corpus-load-batch-size", type=int, default=8192)
    parser.add_argument("--candidate-batch-size", type=int, default=32768)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
