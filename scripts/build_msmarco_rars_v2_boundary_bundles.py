#!/usr/bin/env python3
"""Build corpus-aligned MS MARCO RARS-v2 development bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v2_boundary_loss_feasibility_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_qrels(path: Path) -> dict[str, set[int]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("MS MARCO qrels_subset.json must be an object")
    qrels: dict[str, set[int]] = {}
    for qid, values in payload.items():
        if isinstance(values, dict):
            positive = {int(docid) for docid, rel in values.items() if float(rel) > 0}
        elif isinstance(values, list):
            positive = {int(value) for value in values}
        else:
            raise ValueError(f"Unsupported qrels entry for {qid}")
        if positive:
            qrels[str(qid)] = positive
    if not qrels:
        raise ValueError("No positive MS MARCO qrels")
    return qrels


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    payload = read_json(path)
    qids = [str(value) for value in payload["query_ids"]]
    rows = np.asarray(payload["query_rows"], dtype=np.int64)
    if len(qids) != len(rows) or len(set(qids)) != len(qids):
        raise ValueError(f"Invalid split: {path}")
    return qids, rows


def labels_and_counts(
    qids: list[str], ann_rows: np.ndarray, doc_ids: np.ndarray,
    qrels: dict[str, set[int]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(ann_rows, dtype=np.int64)
    if rows.ndim != 2 or np.any(rows < 0) or np.any(rows >= len(doc_ids)):
        raise ValueError("Invalid ANN rows")
    labels = np.zeros(rows.shape, dtype=np.uint8)
    counts = np.empty(len(qids), dtype=np.int32)
    for index, qid in enumerate(qids):
        relevant = qrels.get(qid)
        if not relevant:
            raise ValueError(f"No qrels for development query {qid}")
        candidate_ids = np.asarray(doc_ids[rows[index]], dtype=np.int64)
        labels[index] = np.isin(candidate_ids, list(relevant)).astype(np.uint8)
        counts[index] = len(relevant)
    return labels, counts


def search_or_load(
    index: Any, queries: np.ndarray, cache_dir: Path, *, candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows_path = cache_dir / "ann_rows.npy"
    scores_path = cache_dir / "ann_scores.npy"
    if rows_path.exists() and scores_path.exists():
        return np.load(rows_path, mmap_mode="r"), np.load(scores_path, mmap_mode="r")
    scores, rows = index.search(np.asarray(queries, dtype=np.float32), candidate_k)
    return rows.astype(np.int64), scores.astype(np.float32)


def sidecar_scores(
    queries: np.ndarray, rows: np.ndarray, ann: np.ndarray, basis: np.ndarray,
    codes: np.ndarray, scales: np.ndarray, *, alpha: float, top_b: int,
) -> np.ndarray:
    result = np.asarray(ann, dtype=np.float32).copy()
    depth = min(top_b, rows.shape[1])
    q_projection = np.asarray(queries, dtype=np.float32) @ basis
    coefficients = codes[rows[:, :depth]].astype(np.float32) * scales
    result[:, :depth] += float(alpha) * np.einsum(
        "qr,qcr->qc", q_projection, coefficients
    )
    return result


def candidate_residuals(
    rows: np.ndarray, embeddings: np.memmap, index: Any, output_dir: Path,
    *, batch_size: int, cached_full_residuals: Path | None,
) -> tuple[np.ndarray, np.ndarray, Path]:
    unique = np.unique(np.asarray(rows, dtype=np.int64).reshape(-1))
    lookup = np.searchsorted(unique, rows).astype(np.int64)
    residual_path = output_dir / "candidate_residuals.float32.npy"
    temporary = residual_path.with_name(residual_path.name + ".part")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(len(unique), embeddings.shape[1]),
    )
    full = None
    if cached_full_residuals is not None and cached_full_residuals.exists():
        full = np.memmap(
            cached_full_residuals, dtype=np.float32, mode="r",
            shape=embeddings.shape,
        )
    for start in range(0, len(unique), batch_size):
        end = min(start + batch_size, len(unique))
        selected = unique[start:end]
        if full is not None:
            output[start:end] = np.asarray(full[selected], dtype=np.float32)
        else:
            reconstructed = index.reconstruct_batch(selected).astype(np.float32)
            output[start:end] = np.asarray(embeddings[selected], dtype=np.float32) - reconstructed
        output.flush()
        if end % 100_000 == 0 or end == len(unique):
            print(f"candidate residuals {end:,}/{len(unique):,}")
    del output
    temporary.replace(residual_path)
    return unique, lookup, residual_path


def build(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    qrels = load_qrels(args.qrels)
    query_vectors = np.load(args.query_vectors, mmap_mode="r")
    if query_vectors.shape != (6980, args.dim):
        raise ValueError(f"Unexpected query matrix {query_vectors.shape}")
    embeddings = np.memmap(
        args.embeddings, dtype=np.float16, mode="r", shape=(args.n_docs, args.dim)
    )
    doc_ids = np.memmap(
        args.doc_ids, dtype=np.int64, mode="r", shape=(args.n_docs,)
    )
    index = faiss.read_index(str(args.index))
    index.nprobe = args.nprobe
    if hasattr(index, "make_direct_map"):
        index.make_direct_map()

    pca_config, rars_config = read_json(args.pca_config), read_json(args.rars_config)
    pca_basis, rars_basis = np.load(args.pca_basis), np.load(args.rars_basis)
    pca_scales, rars_scales = np.load(args.pca_scales), np.load(args.rars_scales)
    pca_codes = np.memmap(
        args.pca_codes, dtype=np.int8, mode="r", shape=(args.n_docs, args.rank)
    )
    rars_codes = np.memmap(
        args.rars_codes, dtype=np.int8, mode="r", shape=(args.n_docs, args.rank)
    )

    roles = {
        "train": (args.train_split, args.cache_root / "train"),
        "validation": (args.validation_split, args.cache_root / "validation"),
    }
    summaries: dict[str, Any] = {}
    for role, (split_path, cache_dir) in roles.items():
        qids, query_rows = load_split(split_path)
        queries = np.asarray(query_vectors[query_rows], dtype=np.float32)
        ann_rows, ann_scores = search_or_load(
            index, queries, cache_dir, candidate_k=args.candidate_k
        )
        if ann_rows.shape != (len(qids), args.candidate_k):
            raise ValueError(f"Unexpected {role} candidate shape {ann_rows.shape}")
        labels, counts = labels_and_counts(qids, ann_rows, doc_ids, qrels)
        role_dir = args.output_root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        unique, lookup, residual_path = candidate_residuals(
            ann_rows, embeddings, index, role_dir, batch_size=args.residual_batch_size,
            cached_full_residuals=args.cached_full_residuals,
        )
        outputs = {
            "query_vectors.float32.npy": queries,
            "ann_rows.int64.npy": np.asarray(ann_rows, dtype=np.int64),
            "ann_scores.float32.npy": np.asarray(ann_scores, dtype=np.float32),
            "candidate_relevance.uint8.npy": labels,
            "relevant_counts.int32.npy": counts,
            "candidate_doc_rows.int64.npy": unique,
            "ann_residual_rows.int64.npy": lookup,
            "pca_scores.float32.npy": sidecar_scores(
                queries, ann_rows, ann_scores, pca_basis, pca_codes, pca_scales,
                alpha=float(pca_config["alpha"]), top_b=int(pca_config["top_b"]),
            ),
            "rars_scores.float32.npy": sidecar_scores(
                queries, ann_rows, ann_scores, rars_basis, rars_codes, rars_scales,
                alpha=float(rars_config["alpha"]), top_b=int(rars_config["top_b"]),
            ),
        }
        paths: dict[str, Path] = {}
        for filename, value in outputs.items():
            path = role_dir / filename
            atomic_save(path, value)
            paths[filename] = path
        paths[residual_path.name] = residual_path
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "split_role": role,
            "source": "MS MARCO 1M clean deterministic development split",
            "query_count": len(qids),
            "candidate_count": args.candidate_k,
            "residual_scope": "candidate_union",
            "unique_candidate_document_count": len(unique),
            "files": {name: file_record(path) for name, path in paths.items()},
            "development_qrels_used": True,
            "test_qrels_accessed": False,
            "nq_test_retuning_authorized": False,
        }
        atomic_json(role_dir / "manifest.json", manifest)
        summaries[role] = {
            "query_count": len(qids),
            "unique_candidate_document_count": len(unique),
        }
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "msmarco_development_bundles_complete",
        "roles": summaries,
        "test_qrels_accessed": False,
        "nq_test_retuning_authorized": False,
    }
    atomic_json(args.output_root / "bundle_build_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in [
        "embeddings", "doc-ids", "query-vectors", "index", "qrels",
        "train-split", "validation-split", "cache-root", "pca-config",
        "pca-basis", "pca-scales", "pca-codes", "rars-config", "rars-basis",
        "rars-scales", "rars-codes", "output-root",
    ]:
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--cached-full-residuals", type=Path)
    p.add_argument("--n-docs", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--residual-batch-size", type=int, default=20_000)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
