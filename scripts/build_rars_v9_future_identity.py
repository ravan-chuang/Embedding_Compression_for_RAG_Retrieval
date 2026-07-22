#!/usr/bin/env python3
"""Materialize only V9 future identities and query vectors without qrels.

The historical V3 builder depends on a V2 candidate bundle whose creation
parsed qrels.  Reusing that path immediately before confirmation would make
the outcome boundary needlessly ambiguous.  This builder instead reproduces
the registered V2.1 inner partition and V3 hash split from identity-only
files, then copies only the corresponding global query vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v9_locked_confirmation_v1"
CANONICAL_PROTOCOL = Path("protocols/rars_v9_locked_confirmation_v1.json")
V3_SPLIT_SALT = b"rars_v3_split_v1\0"


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


def newline_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("utf-8")
    ).hexdigest()


def inner_train_indices(qids: list[str]) -> np.ndarray:
    """Exact v2.1 complement of SHA-256 modulo-5 selection queries."""

    selection = np.asarray(
        [
            int(
                hashlib.sha256(f"rars-v2.1-inner:{qid}".encode()).hexdigest()[:16],
                16,
            )
            % 5
            == 0
            for qid in qids
        ],
        dtype=bool,
    )
    return np.flatnonzero(~selection).astype(np.int64)


def future_indices(inner_qids: list[str]) -> np.ndarray:
    buckets = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(V3_SPLIT_SALT + qid.encode("utf-8")).digest()[:8],
                "big",
                signed=False,
            )
            % 10
            for qid in inner_qids
        ],
        dtype=np.uint8,
    )
    return np.flatnonzero(buckets >= 8).astype(np.int64)


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
        raise ValueError("Future identity build requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected protocol")
    return protocol


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = validate_source(args.protocol, args.source_commit)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty future identity directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = read_json(args.train_split)
    all_qids = [str(value) for value in split["query_ids"]]
    all_rows = np.asarray(split["query_rows"], dtype=np.int64)
    if (
        len(all_qids) != len(all_rows)
        or len(set(all_qids)) != len(all_qids)
        or len(np.unique(all_rows)) != len(all_rows)
        or np.any(all_rows < 0)
    ):
        raise ValueError("Frozen training split has invalid identities")
    inner = inner_train_indices(all_qids)
    inner_qids = [all_qids[index] for index in inner]
    inner_rows = all_rows[inner]
    future = future_indices(inner_qids)
    qids = [inner_qids[index] for index in future]
    rows = inner_rows[future]
    registered = protocol["data_policy"]["confirmation_role"]
    if len(qids) != int(registered["query_count"]):
        raise ValueError("Rebuilt future role query count changed")
    if newline_sha256(qids) != registered["source_order_newline_qid_sha256"]:
        raise ValueError("Rebuilt future role source-order identity changed")
    if newline_sha256(sorted(qids, key=int)) != registered[
        "numeric_sorted_newline_qid_sha256"
    ]:
        raise ValueError("Rebuilt future role sorted identity changed")
    global_vectors = np.load(args.query_vectors, mmap_mode="r")
    dimension = int(protocol["immutable_inputs"]["embedding_dimension"])
    if global_vectors.ndim != 2 or global_vectors.shape[1] != dimension:
        raise ValueError("Global query-vector matrix has an unexpected shape")
    if np.any(rows >= len(global_vectors)):
        raise ValueError("Future query rows exceed the global vector matrix")
    vectors = np.asarray(global_vectors[rows], dtype=np.float32)
    if not np.all(np.isfinite(vectors)) or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, rtol=0.0, atol=0.005
    ):
        raise ValueError("Future query vectors are not finite normalized vectors")
    vector_path = args.output_dir / "query_vectors.float32.npy"
    np.save(vector_path, vectors, allow_pickle=False)
    query_manifest_path = args.output_dir / "query_manifest.json"
    atomic_json(
        query_manifest_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "role_id": "future_method_holdout",
            "query_ids": qids,
            "query_rows": rows.tolist(),
            "parent_inner_train_indices": future.tolist(),
            "candidate_arrays_created": False,
            "labels_materialized": False,
            "metrics_computed": False,
            "qrels_opened": False,
        },
    )
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V9_QRELS_FREE_FUTURE_IDENTITY_COMPLETE",
        "source_commit": args.source_commit,
        "role_id": "future_method_holdout",
        "query_count": len(qids),
        "query_ids_source_order_newline_sha256": newline_sha256(qids),
        "query_ids_numeric_sorted_newline_sha256": newline_sha256(
            sorted(qids, key=int)
        ),
        "inputs": {
            "train_split": file_record(args.train_split),
            "global_query_vectors": file_record(args.query_vectors),
        },
        "outputs": {
            "query_manifest.json": file_record(query_manifest_path),
            "query_vectors.float32.npy": file_record(vector_path),
        },
        "candidate_arrays_created": False,
        "labels_materialized": False,
        "metrics_computed": False,
        "qrels_argument_accepted": False,
        "qrels_opened": False,
    }
    identity_path = args.output_dir / "v9_identity_complete.json"
    atomic_json(identity_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", required=True, type=Path)
    parser.add_argument("--query-vectors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
