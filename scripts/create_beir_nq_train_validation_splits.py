#!/usr/bin/env python3
"""Create deterministic BEIR NQ fit/validation manifests from train IDs only.

The command accepts BEIR's ``qrels/train.tsv`` solely as a source of official
train query IDs. It never parses relevance values and refuses a file named
``test.tsv``. It does not load test query membership or test qrels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "beir_nq_rars_pca_confirmation_v1"
SPLIT_PREFIX = f"{PROTOCOL_ID}:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--train-qrels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_bucket(qid: str) -> int:
    digest = hashlib.sha256(f"{SPLIT_PREFIX}{qid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 10


def load_queries(
    path: Path,
    *,
    allowed_qids: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Retain query text only for requested train-membership IDs.

    BEIR stores multiple query splits in one JSONL file. The splitter scans
    that source to locate registered train IDs, but it must not retain or emit
    any other query text in the pre-qrels artifact package.
    """
    texts: dict[str, str] = {}
    rows: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_index, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item.get("_id", item.get("qid", item.get("query_id")))
            text = item.get("text", item.get("query"))
            if qid is None or text is None:
                raise ValueError("Each query row requires an ID and text")
            qid = str(qid).strip()
            text = str(text).strip()
            if not qid or not text:
                raise ValueError("Query IDs and texts must be non-empty")
            if allowed_qids is not None and qid not in allowed_qids:
                continue
            if qid in texts:
                raise ValueError(f"Duplicate query ID: {qid}")
            texts[qid] = text
            rows[qid] = row_index
    if not texts and allowed_qids is None:
        raise ValueError("No requested query rows found")
    return texts, rows


def load_train_qids(path: Path) -> list[str]:
    if path.name.casefold() != "train.tsv":
        raise ValueError(
            "Only the official BEIR qrels/train.tsv membership file is allowed"
        )

    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split("\t", 1)[0].split(maxsplit=1)[0].strip()
            if line_number == 1 and first.casefold() in {
                "query-id",
                "query_id",
                "qid",
            }:
                continue
            if not first:
                raise ValueError(f"Empty train query ID at line {line_number}")
            qids.add(first)
    if not qids:
        raise ValueError("No BEIR NQ train query IDs found")
    return sorted(qids)


def partition_train_qids(qids: list[str]) -> tuple[list[str], list[str]]:
    fit = [qid for qid in qids if split_bucket(qid) != 0]
    validation = [qid for qid in qids if split_bucket(qid) == 0]
    if not fit or not validation:
        raise ValueError("Hash split produced an empty fit or validation partition")
    return fit, validation


def build_query_manifest(
    name: str,
    qids: list[str],
    *,
    texts: dict[str, str],
    rows: dict[str, int],
) -> dict[str, Any]:
    missing = [qid for qid in qids if qid not in texts]
    if missing:
        raise ValueError(
            f"{len(missing)} train query IDs are absent from queries.jsonl; "
            f"examples={missing[:5]}"
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "partition": name,
        "query_count": len(qids),
        "query_ids": qids,
        "query_rows_in_source_jsonl": [rows[qid] for qid in qids],
        "query_texts": [texts[qid] for qid in qids],
        "split_prefix": SPLIT_PREFIX,
        "split_rule": (
            "validation iff uint64(first_16_hex(sha256(prefix + qid))) "
            "modulo 10 equals 0"
        ),
        "qrels_relevance_values_used": False,
        "test_qrels_accessed": False,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_splits(
    queries_path: Path,
    train_qrels_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    train_qids = load_train_qids(train_qrels_path)
    texts, rows = load_queries(
        queries_path,
        allowed_qids=set(train_qids),
    )
    missing_train_texts = [qid for qid in train_qids if qid not in texts]
    if missing_train_texts:
        raise ValueError(
            f"{len(missing_train_texts)} train query IDs are absent from "
            f"queries.jsonl; examples={missing_train_texts[:5]}"
        )
    fit_qids, validation_qids = partition_train_qids(train_qids)

    fit_manifest = build_query_manifest(
        "fit",
        fit_qids,
        texts=texts,
        rows=rows,
    )
    validation_manifest = build_query_manifest(
        "validation",
        validation_qids,
        texts=texts,
        rows=rows,
    )
    fit_path = output_dir / "train_query_manifest.json"
    validation_path = output_dir / "validation_query_manifest.json"
    write_json(fit_path, fit_manifest)
    write_json(validation_path, validation_manifest)

    split_manifest = {
        "protocol_id": PROTOCOL_ID,
        "source_queries": {
            "path": str(queries_path),
            "bytes": queries_path.stat().st_size,
            "sha256": sha256_file(queries_path),
        },
        "train_membership_source": {
            "path": str(train_qrels_path),
            "bytes": train_qrels_path.stat().st_size,
            "sha256": sha256_file(train_qrels_path),
            "relevance_values_parsed": False,
        },
        "official_train_query_count": len(train_qids),
        "fit_query_count": len(fit_qids),
        "validation_query_count": len(validation_qids),
        "fit_validation_overlap_count": 0,
        "test_qrels_accessed": False,
        "files": {
            "train_query_manifest": {
                "path": str(fit_path),
                "bytes": fit_path.stat().st_size,
                "sha256": sha256_file(fit_path),
            },
            "validation_query_manifest": {
                "path": str(validation_path),
                "bytes": validation_path.stat().st_size,
                "sha256": sha256_file(validation_path),
            },
        },
    }
    write_json(output_dir / "train_validation_split_manifest.json", split_manifest)
    return split_manifest


def main() -> None:
    args = parse_args()
    result = create_splits(
        args.queries,
        args.train_qrels,
        args.output_dir,
    )
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "official_train_query_count": result["official_train_query_count"],
        "fit_query_count": result["fit_query_count"],
        "validation_query_count": result["validation_query_count"],
        "test_qrels_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
