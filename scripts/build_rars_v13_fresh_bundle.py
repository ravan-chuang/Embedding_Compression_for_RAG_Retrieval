#!/usr/bin/env python3
"""Build the frozen V13 Top-100 candidate/residual development bundle."""

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

from evaluate_rars_v6_1m_headroom import validate_faiss_index  # noqa: E402
from rars_v13_signed_score_core import PROTOCOL_ID, deterministic_fold_ids  # noqa: E402
from train_rars_v8_cutoff_sidecar import (  # noqa: E402
    atomic_json,
    atomic_save,
    file_record,
    read_json,
    validate_runtime,
)


CANONICAL_PROTOCOL = Path(
    "protocols/rars_v13_signed_score_distilled_rpq_v1.json"
)
SOURCE_FILES = (
    CANONICAL_PROTOCOL,
    Path("scripts/rars_v13_signed_score_core.py"),
    Path("scripts/freeze_rars_v13_fresh_queries.py"),
    Path("scripts/build_rars_v13_fresh_bundle.py"),
    Path("scripts/evaluate_rars_v6_1m_headroom.py"),
    Path("scripts/train_rars_v8_cutoff_sidecar.py"),
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Registered {label} hash changed")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("--source-commit must be exact lowercase 40-hex")
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must use canonical path: {canonical}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or status:
        raise ValueError("V13 bundle build requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected V13 protocol")
    return protocol, {
        str(relative): file_record((repo_root / relative).resolve(strict=True))
        for relative in SOURCE_FILES
    }


def _load_query_freeze(
    root: Path,
    protocol: dict[str, Any],
    repo_root: Path,
    source_commit: str,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, list[int]], dict[str, Any]]:
    freeze_path = root / "fresh_query_freeze.json"
    manifest_path = root / "fresh_query_manifest.json"
    freeze = read_json(freeze_path)
    manifest = read_json(manifest_path)
    if freeze.get("status") != "RARS_V13_FRESH_QUERY_FREEZE_COMPLETE":
        raise ValueError("Fresh query freeze is incomplete")
    if manifest.get("status") != "RARS_V13_FRESH_QUERIES_FROZEN_BEFORE_CANDIDATES":
        raise ValueError("Fresh query manifest has the wrong status")
    if freeze.get("source_commit") != source_commit or manifest.get(
        "source_commit"
    ) != source_commit:
        raise ValueError("Fresh query freeze source commit changed")
    if freeze.get("selection", {}).get("candidate_retrieval_performed") is not False:
        raise ValueError("Fresh query freeze was not pre-candidate")
    for relative, record in freeze.get("source_blobs", {}).items():
        _verify_record(
            repo_root / relative,
            record,
            f"fresh-query source blob {relative}",
        )
    for name, record in freeze.get("outputs", {}).items():
        _verify_record(root / name, record, f"fresh-query output {name}")
    query_ids = [str(value) for value in manifest.get("query_ids", [])]
    target = int(protocol["fresh_query_freeze"]["target_query_count"])
    if len(query_ids) != target or len(set(query_ids)) != target:
        raise ValueError("Fresh query count or uniqueness changed")
    vectors = np.load(root / "query_vectors.float32.npy", allow_pickle=False)
    folds = np.load(root / "fold_ids.int64.npy", allow_pickle=False)
    qrels_raw = read_json(root / "fresh_qrels.json")
    qrels = {str(key): [int(value) for value in values] for key, values in qrels_raw.items()}
    if vectors.shape != (target, 384) or vectors.dtype != np.float32:
        raise ValueError("Fresh query-vector contract changed")
    frozen_qids = (root / "query_ids.utf8.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if frozen_qids != query_ids:
        raise ValueError("Fresh query-id file differs from its manifest")
    prior: set[str] = set()
    for relative in protocol["fresh_query_freeze"]["prior_qid_sources"]:
        path = repo_root / relative
        if path.suffix == ".txt":
            values = path.read_text(encoding="utf-8").splitlines()
        else:
            value = read_json(path)
            values = value.get("query_ids") if isinstance(value, dict) else value
        if not isinstance(values, list):
            raise ValueError(f"Prior qid registry is malformed: {path}")
        prior.update(str(item) for item in values)
    if len(prior) != int(
        protocol["fresh_query_freeze"]["expected_unique_excluded_qids"]
    ):
        raise ValueError("Fresh query exclusion registry count changed")
    if prior.intersection(query_ids):
        raise ValueError("Fresh query freeze contains a historical qid")
    if not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, rtol=0.0, atol=2e-5
    ):
        raise ValueError("Fresh query vectors are not L2 normalized")
    if folds.shape != (target,) or folds.dtype != np.int64:
        raise ValueError("Fresh fold-vector contract changed")
    if not np.array_equal(folds, deterministic_fold_ids(query_ids)):
        raise ValueError("Fresh fold assignments changed")
    if set(qrels) != set(query_ids) or any(not values for values in qrels.values()):
        raise ValueError("Fresh qrels do not cover every selected query")
    return query_ids, vectors, folds, qrels, {
        "fresh_query_freeze": file_record(freeze_path),
        "fresh_query_manifest": file_record(manifest_path),
    }


def _reconstruct_rows(index: Any, rows: np.ndarray, dimension: int) -> np.ndarray:
    output = np.empty((len(rows), dimension), dtype=np.float32)
    for start in range(0, len(rows), 20000):
        end = min(start + 20000, len(rows))
        output[start:end] = np.asarray(
            index.reconstruct_batch(np.ascontiguousarray(rows[start:end], dtype=np.int64)),
            dtype=np.float32,
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Frozen-index reconstructions contain non-finite values")
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = validate_source(
        repo_root, args.protocol, args.source_commit
    )
    environment = validate_runtime(protocol)
    expected_faiss = protocol["execution_environment_contract"]["faiss_version"]
    if environment["faiss_version"] != expected_faiss:
        raise ValueError(
            f"faiss={environment['faiss_version']}; expected {expected_faiss}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to overwrite a non-empty V13 fresh bundle")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_ids, queries, folds, qrels, input_records = _load_query_freeze(
        args.query_freeze_root, protocol, repo_root, args.source_commit
    )
    frozen = protocol["frozen_index_contract"]
    if args.index.stat().st_size != int(frozen["index_bytes"]):
        raise ValueError("Frozen index byte count changed")
    if sha256_file(args.index) != frozen["index_sha256"]:
        raise ValueError("Frozen index hash changed")
    n_docs = int(frozen["document_count"])
    dimension = int(frozen["embedding_dimension"])
    if args.doc_ids.stat().st_size != n_docs * 8:
        raise ValueError("Frozen doc-id byte count changed")
    if args.embeddings.stat().st_size != n_docs * dimension * 2:
        raise ValueError("Frozen embedding byte count changed")
    doc_ids = np.memmap(args.doc_ids, dtype=np.int64, mode="r", shape=(n_docs,))
    doc_to_row = {int(doc_id): row for row, doc_id in enumerate(doc_ids)}
    if len(doc_to_row) != n_docs:
        raise ValueError("Frozen document ids are not unique")
    for query_id in query_ids:
        if any(document_id not in doc_to_row for document_id in qrels[query_id]):
            raise ValueError("Fresh qrels escaped the frozen corpus")

    index = faiss.read_index(str(args.index))
    ivf, index_contract = validate_faiss_index(index, faiss)
    ivf.nprobe = int(frozen["nprobe"])
    scores, rows = index.search(
        np.ascontiguousarray(queries, dtype=np.float32),
        int(frozen["candidate_pool"]),
    )
    scores = np.asarray(scores, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64)
    expected_shape = (len(query_ids), int(frozen["candidate_pool"]))
    if rows.shape != expected_shape or scores.shape != expected_shape or np.any(rows < 0):
        raise ValueError("Frozen index did not return a complete Top-100")

    candidate_rows = np.unique(rows.reshape(-1)).astype(np.int64)
    lookup = np.searchsorted(candidate_rows, rows).astype(np.int64)
    if not np.array_equal(candidate_rows[lookup], rows):
        raise AssertionError("Candidate residual lookup construction failed")
    ivf.make_direct_map()
    reconstructed = _reconstruct_rows(index, candidate_rows, dimension)
    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(n_docs, dimension),
    )
    residuals = (
        np.asarray(embeddings[candidate_rows], dtype=np.float32) - reconstructed
    ).astype(np.float32)
    labels = np.zeros(expected_shape, dtype=np.uint8)
    relevant_counts = np.empty(len(query_ids), dtype=np.int32)
    for query_index, query_id in enumerate(query_ids):
        positive_ids = set(qrels[query_id])
        relevant_counts[query_index] = len(positive_ids)
        candidate_ids = np.asarray(doc_ids[rows[query_index]], dtype=np.int64)
        labels[query_index] = np.isin(
            candidate_ids, np.asarray(sorted(positive_ids), dtype=np.int64)
        ).astype(np.uint8)
    if np.any(relevant_counts <= 0):
        raise ValueError("Every fresh query must retain a positive denominator")

    arrays = {
        "query_vectors.float32.npy": queries.astype(np.float32),
        "fold_ids.int64.npy": folds.astype(np.int64),
        "ann_rows.int64.npy": rows,
        "ann_scores.float32.npy": scores,
        "ann_residual_rows.int64.npy": lookup,
        "candidate_doc_rows.int64.npy": candidate_rows,
        "candidate_residuals.float32.npy": residuals,
        "candidate_relevance.uint8.npy": labels,
        "relevant_counts.int32.npy": relevant_counts,
    }
    for filename, values in arrays.items():
        atomic_save(args.output_dir / filename, values)
    qids_path = args.output_dir / "query_ids.utf8.txt"
    qids_path.write_text("\n".join(query_ids) + "\n", encoding="utf-8")
    output_records = {
        name: file_record(args.output_dir / name)
        for name in (*arrays.keys(), "query_ids.utf8.txt")
    }
    manifest_path = args.output_dir / "fresh_bundle_manifest.json"
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V13_FRESH_DEVELOPMENT_BUNDLE_FROZEN",
        "source_commit": args.source_commit,
        "role_id": "fresh_train_development",
        "query_count": len(query_ids),
        "candidate_count": int(frozen["candidate_pool"]),
        "candidate_residual_count": len(candidate_rows),
        "fold_counts": np.bincount(folds, minlength=5).tolist(),
        "environment": environment,
        "index_contract": index_contract,
        "restricted_corpus": True,
        "positive_qrels_in_corpus": int(relevant_counts.sum()),
        "positive_candidate_hits": int(labels.sum()),
        "source_blobs": source_blobs,
        "inputs": {
            **input_records,
            "embeddings": file_record(args.embeddings),
            "doc_ids": file_record(args.doc_ids),
            "index": file_record(args.index),
        },
        "files": output_records,
        "candidate_retrieval_performed_after_query_freeze": True,
        "metrics_computed": False,
        "old_rars_holdout_opened": False,
    }
    atomic_json(manifest_path, manifest)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V13_FRESH_BUNDLE_COMPLETE",
        "source_commit": args.source_commit,
        "manifest": file_record(manifest_path),
        "outputs": output_records,
        "metrics_computed": False,
        "old_rars_holdout_opened": False,
    }
    atomic_json(args.output_dir / "fresh_bundle_complete.json", complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-freeze-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--doc-ids", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
