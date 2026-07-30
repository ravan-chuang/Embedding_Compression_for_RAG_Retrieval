#!/usr/bin/env python3
"""Freeze fresh MS MARCO train queries before V12 candidate evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v12_ca_rpq_core import (  # noqa: E402
    PROTOCOL_ID,
    deterministic_fold_ids,
    deterministic_query_priority,
)
from train_rars_v8_cutoff_sidecar import (  # noqa: E402
    atomic_json,
    atomic_save,
    file_record,
    read_json,
    validate_runtime,
)


CANONICAL_PROTOCOL = Path("protocols/rars_v12_anchored_cutoff_rpq_v1.json")
SOURCE_FILES = (
    CANONICAL_PROTOCOL,
    Path("scripts/rars_v12_ca_rpq_core.py"),
    Path("scripts/freeze_rars_v12_fresh_queries.py"),
    Path("scripts/train_rars_v8_cutoff_sidecar.py"),
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_commit(source_commit)
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
        raise ValueError("V12 fresh-query freeze requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != (
        "FROZEN_BEFORE_FIRST_V12_FRESH_DEVELOPMENT_RUN"
    ):
        raise ValueError("Unexpected V12 protocol identity or status")
    records = {
        str(relative): file_record((repo_root / relative).resolve(strict=True))
        for relative in SOURCE_FILES
    }
    return protocol, records


def _load_prior_qids(paths: list[Path]) -> tuple[set[str], dict[str, Any]]:
    output: set[str] = set()
    records: dict[str, Any] = {}
    for path in paths:
        value = read_json(path)
        if isinstance(value, dict):
            values = value.get("query_ids")
        else:
            values = value
        if not isinstance(values, list):
            raise ValueError(f"Prior qid file is not a list/query_ids object: {path}")
        local = {str(item) for item in values}
        if len(local) != len(values):
            raise ValueError(f"Prior qid file has duplicates: {path}")
        output.update(local)
        records[str(path)] = {**file_record(path), "query_count": len(local)}
    return output, records


def _load_corpus_doc_ids(path: Path, expected_count: int) -> set[int]:
    if path.stat().st_size != expected_count * np.dtype(np.int64).itemsize:
        raise ValueError("Frozen doc-id file byte count changed")
    values = np.memmap(path, dtype=np.int64, mode="r", shape=(expected_count,))
    if len(np.unique(values)) != expected_count:
        raise ValueError("Frozen corpus document ids are not unique")
    return {int(value) for value in values}


def _parse_covered_qrels(
    path: Path, corpus_doc_ids: set[int]
) -> tuple[dict[str, set[int]], dict[str, int]]:
    covered: dict[str, set[int]] = {}
    line_count = 0
    positive_count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 4:
            raise ValueError("MS MARCO train qrels must have four columns")
        query_id, _, document_id, relevance = fields
        line_count += 1
        if int(relevance) <= 0:
            continue
        positive_count += 1
        doc_id = int(document_id)
        if doc_id in corpus_doc_ids:
            covered.setdefault(str(query_id), set()).add(doc_id)
    return covered, {
        "line_count": line_count,
        "positive_line_count": positive_count,
        "covered_query_count": len(covered),
        "covered_positive_count": int(sum(len(value) for value in covered.values())),
    }


def _load_query_texts(path: Path, eligible: set[str]) -> tuple[dict[str, str], int]:
    texts: dict[str, str] = {}
    line_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            query_id, separator, text = raw.rstrip("\n").partition("\t")
            if not separator:
                raise ValueError("MS MARCO query row is missing its tab separator")
            line_count += 1
            query_id = str(query_id)
            if query_id in eligible:
                if query_id in texts:
                    raise ValueError(f"Duplicate MS MARCO train query id: {query_id}")
                texts[query_id] = text
    return texts, line_count


def _newline_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
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
        raise ValueError("Refusing to overwrite a non-empty V12 fresh-query freeze")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fresh = protocol["fresh_query_freeze"]
    target = int(fresh["target_query_count"])
    if args.target_query_count != target:
        raise ValueError(f"Target query count must remain frozen at {target}")

    prior_qids, prior_records = _load_prior_qids(args.prior_qids)
    expected_prior_count = 6980
    if len(prior_qids) != expected_prior_count:
        raise ValueError(
            f"Historical query registry has {len(prior_qids)} unique qids; "
            f"expected {expected_prior_count}"
        )
    corpus_ids = _load_corpus_doc_ids(
        args.doc_ids, int(protocol["frozen_index_contract"]["document_count"])
    )
    covered, qrels_summary = _parse_covered_qrels(args.qrels_train, corpus_ids)
    overlap = sorted(set(covered).intersection(prior_qids), key=int)
    eligible_ids = set(covered).difference(prior_qids)
    texts, query_line_count = _load_query_texts(args.queries_train, eligible_ids)
    eligible = [query_id for query_id in eligible_ids if query_id in texts]
    eligible.sort(key=lambda query_id: (deterministic_query_priority(query_id), int(query_id)))
    if len(eligible) < target:
        raise ValueError(
            f"Only {len(eligible)} fresh corpus-covered train queries; need {target}"
        )
    selected = eligible[:target]
    folds = deterministic_fold_ids(
        selected, fold_count=int(fresh["fold_count"])
    )
    fold_counts = np.bincount(
        folds, minlength=int(fresh["fold_count"])
    ).tolist()
    if min(fold_counts) < int(fresh["minimum_queries_per_fold"]):
        raise ValueError(f"Frozen hash folds are too small: {fold_counts}")

    expected_st = protocol["execution_environment_contract"][
        "sentence_transformers_version"
    ]
    actual_st = importlib.metadata.version("sentence-transformers")
    if actual_st != expected_st:
        raise ValueError(
            f"sentence-transformers={actual_st}; expected {expected_st}"
        )
    from sentence_transformers import SentenceTransformer

    embedding = fresh["embedding"]
    model = SentenceTransformer(
        embedding["model"], revision=embedding["model_revision"], device=args.device
    )
    selected_texts = [texts[query_id] for query_id in selected]
    vectors = model.encode(
        selected_texts,
        batch_size=int(embedding["batch_size"]),
        show_progress_bar=True,
        normalize_embeddings=bool(embedding["normalize_embeddings"]),
        convert_to_numpy=True,
    ).astype(np.float32)
    if vectors.shape != (target, int(embedding["dimension"])):
        raise ValueError(f"Unexpected fresh-query vector shape: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("Fresh query embeddings are not L2 normalized")

    qrels_payload = {
        query_id: sorted(int(value) for value in covered[query_id])
        for query_id in selected
    }
    qrels_path = args.output_dir / "fresh_qrels.json"
    atomic_json(qrels_path, qrels_payload)
    vectors_path = args.output_dir / "query_vectors.float32.npy"
    folds_path = args.output_dir / "fold_ids.int64.npy"
    atomic_save(vectors_path, vectors)
    atomic_save(folds_path, folds)
    qids_path = args.output_dir / "query_ids.utf8.txt"
    qids_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    text_digest = hashlib.sha256(
        "".join(f"{qid}\t{texts[qid]}\n" for qid in selected).encode("utf-8")
    ).hexdigest()
    manifest_path = args.output_dir / "fresh_query_manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V12_FRESH_QUERIES_FROZEN_BEFORE_CANDIDATES",
            "source_commit": args.source_commit,
            "query_count": target,
            "query_ids": selected,
            "fold_ids": folds.tolist(),
            "fold_counts": fold_counts,
            "source_order_newline_qid_sha256": _newline_hash(selected),
            "numeric_sorted_newline_qid_sha256": _newline_hash(
                sorted(selected, key=int)
            ),
            "selected_query_text_payload_sha256": text_digest,
            "embedding": {
                **embedding,
                "sentence_transformers_version": actual_st,
                "minimum_norm": float(norms.min()),
                "maximum_norm": float(norms.max()),
            },
            "restricted_corpus": True,
            "positive_qrels_in_frozen_corpus": int(
                sum(len(value) for value in qrels_payload.values())
            ),
            "historical_qid_overlap": [],
        },
    )
    outputs = {
        name: file_record(args.output_dir / name)
        for name in (
            "fresh_qrels.json",
            "query_vectors.float32.npy",
            "fold_ids.int64.npy",
            "query_ids.utf8.txt",
            "fresh_query_manifest.json",
        )
    }
    freeze_path = args.output_dir / "fresh_query_freeze.json"
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V12_FRESH_QUERY_FREEZE_COMPLETE",
        "source_commit": args.source_commit,
        "environment": environment,
        "source_blobs": source_blobs,
        "inputs": {
            "queries_train": {
                **file_record(args.queries_train),
                "line_count": query_line_count,
            },
            "qrels_train": {**file_record(args.qrels_train), **qrels_summary},
            "doc_ids": file_record(args.doc_ids),
            "prior_qid_sources": prior_records,
        },
        "selection": {
            "target_query_count": target,
            "eligible_fresh_corpus_covered_queries": len(eligible),
            "covered_qids_overlapping_historical_registry": len(overlap),
            "selected_qid_hash": _newline_hash(selected),
            "fold_counts": fold_counts,
            "candidate_retrieval_performed": False,
            "metric_computation_performed": False,
        },
        "outputs": outputs,
    }
    atomic_json(freeze_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-train", type=Path, required=True)
    parser.add_argument("--qrels-train", type=Path, required=True)
    parser.add_argument("--doc-ids", type=Path, required=True)
    parser.add_argument("--prior-qids", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--target-query-count", type=int, default=2500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
