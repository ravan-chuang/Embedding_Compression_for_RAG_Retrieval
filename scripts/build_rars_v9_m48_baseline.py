#!/usr/bin/env python3
"""Build the qrels-free M48 limitation baseline for V9 confirmation.

This is not a new RARS method.  It answers the reviewer-facing limitation:
when rebuilding the index is allowed, how strong is a uniform 48-byte PQ code
at the same per-document code budget as M32 plus a 16-byte sidecar?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v9_locked_confirmation_v1"
CANONICAL_PROTOCOL = Path("protocols/rars_v9_locked_confirmation_v1.json")


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


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_source(protocol_path: Path, source_commit: str) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must be canonical: {canonical}")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("--source-commit must be lowercase 40-hex")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if head != source_commit:
        raise ValueError(f"Git HEAD {head} does not match {source_commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("M48 build requires a clean exact checkout")
    protocol = json.loads(canonical.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected protocol")
    return protocol


def run(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    protocol = validate_source(args.protocol, args.source_commit)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty M48 output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = protocol["immutable_inputs"]
    embedding_record = file_record(args.embeddings)
    if (
        embedding_record["bytes"] != int(contract["embeddings_bytes"])
        or embedding_record["sha256"] != contract["embeddings_sha256"]
    ):
        raise ValueError("Embedding artifact differs from the V9 contract")
    count = int(contract["document_count"])
    dimension = int(contract["embedding_dimension"])
    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(count, dimension),
    )
    if not 0 < args.training_rows <= count or args.batch_size <= 0:
        raise ValueError("Invalid training rows or batch size")
    rng = np.random.default_rng(args.training_seed)
    training_rows = np.sort(
        rng.choice(count, size=args.training_rows, replace=False).astype(np.int64)
    )
    training = np.asarray(embeddings[training_rows], dtype=np.float32)
    norms = np.linalg.norm(training, axis=1)
    if not np.all(np.isfinite(training)) or not np.allclose(
        norms, 1.0, rtol=0.0, atol=0.005
    ):
        raise ValueError("Training embeddings are not finite normalized vectors")
    row_path = args.output_dir / "training_rows.int64.npy"
    np.save(row_path, training_rows, allow_pickle=False)

    limitation = protocol["locked_limitation_baselines"][
        "m48_rebuild_nlist512_nprobe16"
    ]
    quantizer = faiss.IndexFlatIP(dimension)
    cpu_index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        int(limitation["nlist"]),
        int(limitation["subquantizers"]),
        int(limitation["bits_per_subquantizer"]),
        faiss.METRIC_INNER_PRODUCT,
    )
    cpu_index.nprobe = int(limitation["nprobe"])
    active_index = cpu_index
    resources = None
    if args.use_gpu:
        resources = faiss.StandardGpuResources()
        active_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    started = time.perf_counter()
    active_index.train(training)
    del training
    for start in range(0, count, args.batch_size):
        end = min(count, start + args.batch_size)
        block = np.asarray(embeddings[start:end], dtype=np.float32)
        if not np.all(np.isfinite(block)):
            raise ValueError("Corpus embeddings contain non-finite values")
        active_index.add(block)
    if args.use_gpu:
        cpu_index = faiss.index_gpu_to_cpu(active_index)
    if not cpu_index.is_trained or int(cpu_index.ntotal) != count:
        raise ValueError("M48 index build did not complete")
    cpu_index.nprobe = int(limitation["nprobe"])
    index_path = args.output_dir / "ivfpq_m48_nlist512_nprobe16.index"
    faiss.write_index(cpu_index, str(index_path))
    index_record = file_record(index_path)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V9_QRELS_FREE_M48_BASELINE_COMPLETE",
        "source_commit": args.source_commit,
        "index_contract": {
            "dimension": dimension,
            "ntotal": count,
            "metric": "inner_product",
            "nlist": int(limitation["nlist"]),
            "subquantizers": int(limitation["subquantizers"]),
            "bits_per_subquantizer": int(limitation["bits_per_subquantizer"]),
            "nprobe": int(limitation["nprobe"]),
        },
        "inputs": {"embeddings": embedding_record},
        "training": {
            "sample_rows": int(args.training_rows),
            "seed": int(args.training_seed),
            "rows": file_record(row_path),
        },
        "index": index_record,
        "wall_seconds": float(time.perf_counter() - started),
        "gpu_used": bool(args.use_gpu),
        "qrels_argument_accepted": False,
        "qrels_opened": False,
        "outcome_metric_computed": False,
    }
    result_path = args.output_dir / "m48_build_result.json"
    atomic_json(result_path, result)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V9_QRELS_FREE_M48_BASELINE_COMPLETE",
        "source_commit": args.source_commit,
        "result": file_record(result_path),
        "index": index_record,
        "qrels_opened": False,
        "outcome_metric_computed": False,
    }
    atomic_json(args.output_dir / "m48_build_complete.json", complete)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--training-rows", type=int, default=200000)
    parser.add_argument("--training-seed", type=int, default=20260723)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
