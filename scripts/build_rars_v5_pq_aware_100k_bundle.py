#!/usr/bin/env python3
"""Build the outcome-informed 100K RARS-v5 PQ-aware pilot bundle.

The builder consumes only the already-observed RARS-v3 design/audit roles.  It
never opens the identity-only ``future_method_holdout`` role or any external
collection.  The pilot corpus includes every judged-positive document already
present in those frozen Top-100 roles, then uses a deterministic random fill to
reach 100K documents.  Consequently its relevance metric is explicitly a
known-positive, development-only diagnostic rather than official MS MARCO
Recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rars_v5_pq_aware_core import (
        PROTOCOL_ID,
        known_positive_recall_at_k,
    )
except ModuleNotFoundError:  # Allows import as ``scripts.<module>`` in tests.
    from scripts.rars_v5_pq_aware_core import (
        PROTOCOL_ID,
        known_positive_recall_at_k,
    )


ALLOWED_ROLES = ("oracle_design", "oracle_audit")


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
        "bytes": path.stat().st_size,
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
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_record(path: Path, record: dict[str, Any], description: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{description} byte count changed")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{description} hash changed")


def load_observed_role(role_dir: Path, expected_role: str) -> dict[str, Any]:
    if expected_role not in ALLOWED_ROLES:
        raise ValueError("Only already-observed design/audit roles are accepted")
    candidate_manifest_path = role_dir / "v3_candidate_manifest.json"
    label_manifest_path = role_dir / "v3_role_labels_manifest.json"
    query_manifest_path = role_dir / "query_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    label_manifest = read_json(label_manifest_path)
    query_manifest = read_json(query_manifest_path)
    if candidate_manifest.get("role_id") != expected_role:
        raise ValueError("Unexpected candidate role")
    if label_manifest.get("role_id") != expected_role:
        raise ValueError("Unexpected label role")
    if label_manifest.get("status") != "ROLE_LABELS_MATERIALIZED_FROM_FROZEN_PARENT":
        raise ValueError("Role labels are incomplete")
    if query_manifest.get("role_id") != expected_role:
        raise ValueError("Unexpected query manifest role")
    verify_record(
        candidate_manifest_path,
        label_manifest["candidate_manifest"],
        f"{expected_role} candidate manifest",
    )

    required = (
        "query_vectors.float32.npy",
        "ann_rows.int64.npy",
        "candidate_relevance.uint8.npy",
        "relevant_counts.int32.npy",
    )
    paths = {name: role_dir / name for name in required}
    for name in ("query_vectors.float32.npy", "ann_rows.int64.npy"):
        verify_record(paths[name], candidate_manifest["files"][name], name)
    for name in ("candidate_relevance.uint8.npy", "relevant_counts.int32.npy"):
        verify_record(paths[name], label_manifest["files"][name], name)

    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    query_count = int(candidate_manifest["query_count"])
    candidate_count = int(candidate_manifest["candidate_count"])
    if arrays["query_vectors.float32.npy"].shape[0] != query_count:
        raise ValueError("Query-vector count changed")
    for name in ("ann_rows.int64.npy", "candidate_relevance.uint8.npy"):
        if arrays[name].shape != (query_count, candidate_count):
            raise ValueError(f"Unexpected role array shape: {name}")
    if arrays["relevant_counts.int32.npy"].shape != (query_count,):
        raise ValueError("Relevant-count shape changed")
    qids = [str(value) for value in query_manifest["query_ids"]]
    if len(qids) != query_count or len(set(qids)) != query_count:
        raise ValueError("Invalid role query IDs")
    return {
        "role_id": expected_role,
        "role_dir": role_dir,
        "candidate_manifest": candidate_manifest,
        "label_manifest": label_manifest,
        "query_manifest": query_manifest,
        "qids": qids,
        "queries": arrays["query_vectors.float32.npy"],
        "source_rows": arrays["ann_rows.int64.npy"],
        "source_labels": arrays["candidate_relevance.uint8.npy"],
        "source_relevant_counts": arrays["relevant_counts.int32.npy"],
    }


def positive_rows_by_query(role: dict[str, Any]) -> list[np.ndarray]:
    rows = np.asarray(role["source_rows"], dtype=np.int64)
    labels = np.asarray(role["source_labels"], dtype=np.uint8)
    values: list[np.ndarray] = []
    for index in range(len(rows)):
        positive = np.unique(rows[index, labels[index] > 0]).astype(np.int64)
        values.append(positive)
    return values


def select_pilot_rows(
    positive_rows: list[np.ndarray],
    *,
    n_docs: int,
    pilot_docs: int,
    seed: int,
) -> np.ndarray:
    if not 0 < pilot_docs <= n_docs:
        raise ValueError("pilot_docs must be within the full corpus")
    nonempty = [np.asarray(value, dtype=np.int64) for value in positive_rows if len(value)]
    if not nonempty:
        raise ValueError("No observed positive documents are available")
    required = np.unique(np.concatenate(nonempty))
    if np.any(required < 0) or np.any(required >= n_docs):
        raise ValueError("Observed positive row is outside the corpus")
    if len(required) > pilot_docs:
        raise ValueError("Pilot corpus is smaller than the required positive set")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_docs)
    is_required = np.zeros(n_docs, dtype=bool)
    is_required[required] = True
    fill = permutation[~is_required[permutation]][: pilot_docs - len(required)]
    selected = np.concatenate([required, fill]).astype(np.int64)
    if len(selected) != pilot_docs or len(np.unique(selected)) != pilot_docs:
        raise AssertionError("Pilot row selection lost uniqueness")
    return selected


def train_index(
    vectors: np.ndarray,
    *,
    nlist: int,
    subquantizers: int,
    nbits: int,
    nprobe: int,
    use_gpu: bool,
) -> Any:
    import faiss

    dimension = int(vectors.shape[1])
    if dimension % subquantizers:
        raise ValueError("Embedding dimension must be divisible by PQ subquantizers")
    quantizer = faiss.IndexFlatIP(dimension)
    cpu_index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        subquantizers,
        nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    cpu_index.nprobe = nprobe
    if use_gpu and faiss.get_num_gpus() > 0:
        resource = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(resource, 0, cpu_index)
        gpu_index.train(vectors)
        gpu_index.add(vectors)
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        cpu_index.train(vectors)
        cpu_index.add(vectors)
    cpu_index.nprobe = nprobe
    cpu_index.make_direct_map()
    return cpu_index


def _candidate_union(
    retrieved: np.ndarray,
    positives: list[np.ndarray],
    *,
    append_missing_positives: bool,
) -> tuple[np.ndarray, np.ndarray]:
    extra = max(len(value) for value in positives) if append_missing_positives else 0
    width = retrieved.shape[1] + extra
    output = np.full((len(retrieved), width), -1, dtype=np.int64)
    valid = np.zeros((len(retrieved), width), dtype=bool)
    for index in range(len(retrieved)):
        ordered = [int(value) for value in retrieved[index] if int(value) >= 0]
        present = set(ordered)
        if append_missing_positives:
            ordered.extend(
                int(value) for value in positives[index] if int(value) not in present
            )
        output[index, : len(ordered)] = ordered
        valid[index, : len(ordered)] = True
    return output, valid


def write_role_bundle(
    role: dict[str, Any],
    positive_global_rows: list[np.ndarray],
    *,
    pilot_rows: np.ndarray,
    reconstructed: np.ndarray,
    index: Any,
    exact_index: Any,
    output_dir: Path,
    candidate_k: int,
    pool_k: int,
    append_missing_positives: bool,
) -> dict[str, Any]:
    queries_all = np.asarray(role["queries"], dtype=np.float32)
    keep = np.asarray([len(value) > 0 for value in positive_global_rows], dtype=bool)
    if not np.any(keep):
        raise ValueError(f"{role['role_id']} has no observed positive query")
    qids = [qid for qid, selected in zip(role["qids"], keep) if selected]
    queries = queries_all[keep].copy()
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if np.any(~np.isfinite(query_norms)) or np.any(query_norms <= 0):
        raise ValueError("Role contains an invalid query vector")
    queries /= query_norms
    selected_positive_global = [
        value for value, selected in zip(positive_global_rows, keep) if selected
    ]
    global_to_local = {int(row): index for index, row in enumerate(pilot_rows)}
    positive_local = [
        np.asarray([global_to_local[int(row)] for row in values], dtype=np.int64)
        for values in selected_positive_global
    ]
    _, retrieved = index.search(queries, candidate_k)
    _, exact_retrieved = exact_index.search(queries, candidate_k)
    retrieved = retrieved.astype(np.int64)
    exact_retrieved = exact_retrieved.astype(np.int64)
    candidate_rows, valid = _candidate_union(
        retrieved,
        positive_local,
        append_missing_positives=append_missing_positives,
    )
    width = candidate_rows.shape[1]
    exact_scores = np.full((len(queries), width), -np.inf, dtype=np.float32)
    pq_scores = np.full_like(exact_scores, -np.inf)
    relevance = np.zeros((len(queries), width), dtype=np.uint8)
    for query_index in range(len(queries)):
        selected = candidate_rows[query_index, valid[query_index]]
        original = np.asarray(
            role["_pilot_vectors"][selected], dtype=np.float32
        )
        exact_scores[query_index, : len(selected)] = original @ queries[query_index]
        pq_scores[query_index, : len(selected)] = (
            reconstructed[selected] @ queries[query_index]
        )
        relevance[query_index, : len(selected)] = np.isin(
            selected, positive_local[query_index]
        ).astype(np.uint8)
    positive_width = max(len(value) for value in positive_local)
    known_positive_rows = np.full(
        (len(queries), positive_width), -1, dtype=np.int64
    )
    known_positive_valid = np.zeros_like(known_positive_rows, dtype=bool)
    for query_index, values in enumerate(positive_local):
        known_positive_rows[query_index, : len(values)] = values
        known_positive_valid[query_index, : len(values)] = True
    relevant_counts = known_positive_valid.sum(axis=1).astype(np.int32)
    base_r10 = known_positive_recall_at_k(
        retrieved, known_positive_rows, known_positive_valid, k=10
    )
    base_r100 = known_positive_recall_at_k(
        retrieved, known_positive_rows, known_positive_valid, k=pool_k
    )
    exact_r100 = known_positive_recall_at_k(
        exact_retrieved, known_positive_rows, known_positive_valid, k=pool_k
    )

    arrays = {
        "query_vectors.float32.npy": queries,
        "candidate_local_rows.int64.npy": candidate_rows,
        "candidate_valid.bool.npy": valid,
        "candidate_relevance.uint8.npy": relevance,
        "known_relevant_counts.int32.npy": relevant_counts,
        "known_positive_local_rows.int64.npy": known_positive_rows,
        "known_positive_valid.bool.npy": known_positive_valid,
        "teacher_exact_scores.float32.npy": exact_scores,
        "base_pq_scores.float32.npy": pq_scores,
        "base_pq_top_rows.int64.npy": retrieved[:, :pool_k],
        "teacher_exact_top_rows.int64.npy": exact_retrieved[:, :pool_k],
        "base_pq_recall_at_10.float64.npy": base_r10,
        "base_pq_recall_at_100.float64.npy": base_r100,
        "teacher_exact_recall_at_100.float64.npy": exact_r100,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    for filename, value in arrays.items():
        path = output_dir / filename
        atomic_save(path, value)
        paths[filename] = path
    query_manifest_path = output_dir / "query_manifest.json"
    atomic_json(
        query_manifest_path,
        {
            "role_id": role["role_id"],
            "query_ids": qids,
            "dropped_no_observed_positive_queries": int(np.sum(~keep)),
        },
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "role_id": role["role_id"],
        "evidence_status": "OBSERVED_DEVELOPMENT_ONLY",
        "query_count": len(queries),
        "candidate_width": width,
        "retrieved_candidate_k": candidate_k,
        "pool_k": pool_k,
        "missing_positives_appended_to_pair_mining_candidates": (
            append_missing_positives
        ),
        "evaluation_candidates_are_label_independent": (
            not append_missing_positives
        ),
        "known_positive_semantics": (
            "positive documents observed in the frozen source-role Top-100 only"
        ),
        "zero_label_semantics": "unjudged mined hard negative, not explicit non-relevant",
        "base_pq_recall_at_10": float(np.mean(base_r10)),
        "base_pq_recall_at_100": float(np.mean(base_r100)),
        "teacher_exact_recall_at_100": float(np.mean(exact_r100)),
        "query_manifest": file_record(query_manifest_path),
        "files": {name: file_record(path) for name, path in paths.items()},
        "source_candidate_manifest": file_record(
            role["role_dir"] / "v3_candidate_manifest.json"
        ),
        "source_label_manifest": file_record(
            role["role_dir"] / "v3_role_labels_manifest.json"
        ),
    }
    atomic_json(output_dir / "pilot_role_manifest.json", manifest)
    return manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    protocol = read_json(args.protocol)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected v5 protocol")
    if len(args.source_commit) != 40:
        raise ValueError("source_commit must be a full 40-character Git commit")
    int(args.source_commit, 16)
    index_contract = protocol["pilot_index"]
    corpus_contract = protocol["data_policy"]["pilot_corpus"]
    frozen = {
        "n_docs": int(corpus_contract["source_corpus_document_count"]),
        "pilot_docs": int(corpus_contract["document_count"]),
        "corpus_seed": int(corpus_contract["seed"]),
        "dimension": int(index_contract["embedding_dimension"]),
        "nlist": int(index_contract["nlist"]),
        "nprobe": int(index_contract["nprobe"]),
        "subquantizers": int(index_contract["subquantizers"]),
        "nbits": int(index_contract["bits_per_subquantizer"]),
        "candidate_k": int(index_contract["candidate_k"]),
        "pool_k": int(index_contract["candidate_pool_metric_k"]),
    }
    actual = {key: getattr(args, key) for key in frozen}
    if actual != frozen:
        raise ValueError(f"Builder arguments violate the frozen protocol: {actual}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("Refusing to reuse a non-empty pilot output root")
    args.output_root.mkdir(parents=True, exist_ok=True)

    design = load_observed_role(args.design_role_dir, "oracle_design")
    audit = load_observed_role(args.audit_role_dir, "oracle_audit")
    role_positives = {
        "oracle_design": positive_rows_by_query(design),
        "oracle_audit": positive_rows_by_query(audit),
    }
    pilot_rows = select_pilot_rows(
        role_positives["oracle_design"] + role_positives["oracle_audit"],
        n_docs=args.n_docs,
        pilot_docs=args.pilot_docs,
        seed=args.corpus_seed,
    )
    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(args.n_docs, args.dimension),
    )
    pilot_vectors = np.asarray(embeddings[pilot_rows], dtype=np.float32)
    norms = np.linalg.norm(pilot_vectors, axis=1)
    if not np.all(np.isfinite(norms)) or float(np.median(norms)) < 0.9:
        raise ValueError("Pilot embeddings are not finite normalized vectors")
    pilot_vectors /= norms[:, None]
    index = train_index(
        pilot_vectors,
        nlist=args.nlist,
        subquantizers=args.subquantizers,
        nbits=args.nbits,
        nprobe=args.nprobe,
        use_gpu=args.use_gpu,
    )
    index_path = args.output_root / "pilot_ivfpq.index"
    faiss.write_index(index, str(index_path))
    reconstructed = index.reconstruct_n(0, args.pilot_docs).astype(np.float32)
    _, coarse_assignments = index.quantizer.search(pilot_vectors, 1)
    coarse_centroids = index.quantizer.reconstruct_n(0, args.nlist).astype(np.float32)
    pq_centroids = faiss.vector_to_array(index.pq.centroids).reshape(
        args.subquantizers, 1 << args.nbits, args.dimension // args.subquantizers
    ).astype(np.float32)
    exact_index_cpu = faiss.IndexFlatIP(args.dimension)
    exact_index_cpu.add(pilot_vectors)
    exact_index = exact_index_cpu
    if args.use_gpu and faiss.get_num_gpus() > 0:
        exact_resource = faiss.StandardGpuResources()
        exact_index = faiss.index_cpu_to_gpu(exact_resource, 0, exact_index_cpu)

    shared_arrays = {
        "pilot_global_doc_rows.int64.npy": pilot_rows,
        "pilot_coarse_assignments.int64.npy": coarse_assignments.reshape(-1).astype(np.int64),
        "coarse_centroids.float32.npy": coarse_centroids,
        "pq_centroids.float32.npy": pq_centroids,
    }
    shared_paths: dict[str, Path] = {}
    for filename, value in shared_arrays.items():
        path = args.output_root / filename
        atomic_save(path, value)
        shared_paths[filename] = path

    design["_pilot_vectors"] = pilot_vectors
    audit["_pilot_vectors"] = pilot_vectors
    role_manifests = {
        "oracle_design": write_role_bundle(
            design,
            role_positives["oracle_design"],
            pilot_rows=pilot_rows,
            reconstructed=reconstructed,
            index=index,
            exact_index=exact_index,
            output_dir=args.output_root / "train",
            candidate_k=args.candidate_k,
            pool_k=args.pool_k,
            append_missing_positives=True,
        ),
        "oracle_audit": write_role_bundle(
            audit,
            role_positives["oracle_audit"],
            pilot_rows=pilot_rows,
            reconstructed=reconstructed,
            index=index,
            exact_index=exact_index,
            output_dir=args.output_root / "selection",
            candidate_k=args.candidate_k,
            pool_k=args.pool_k,
            append_missing_positives=False,
        ),
    }
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PILOT_BUNDLE_COMPLETE",
        "source_commit": args.source_commit,
        "protocol_sha256": sha256_file(args.protocol),
        "pilot_docs": args.pilot_docs,
        "dimension": args.dimension,
        "ivfpq": {
            "metric": "inner_product",
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "subquantizers": args.subquantizers,
            "nbits": args.nbits,
            "fixed_coarse_assignments_during_adapter_training": True,
            "fixed_codebooks_during_adapter_training": True,
        },
        "data_access": {
            "observed_v3_design_used": True,
            "observed_v3_audit_used_for_selection": True,
            "future_method_holdout_opened": False,
            "external_collection_opened": False,
            "raw_qrels_opened_by_v5_builder": False,
        },
        "pilot_index": file_record(index_path),
        "shared_files": {name: file_record(path) for name, path in shared_paths.items()},
        "roles": role_manifests,
    }
    atomic_json(args.output_root / "pilot_bundle_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-role-dir", required=True, type=Path)
    parser.add_argument("--audit-role-dir", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "protocols/rars_v5_pq_aware_100k_pilot_v1.json",
    )
    parser.add_argument("--n-docs", type=int, default=1_000_000)
    parser.add_argument("--pilot-docs", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--subquantizers", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--corpus-seed", type=int, default=20260720)
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
