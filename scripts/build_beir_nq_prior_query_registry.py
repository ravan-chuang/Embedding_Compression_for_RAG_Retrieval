#!/usr/bin/env python3
"""Freeze the committed project-query registry used by the NQ identity audit.

The registry is built before NQ test qrels access.  Dataset-local numeric IDs
are compared only inside their declared namespace; normalized query text is
compared across namespaces when a committed source contains text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    ("msmarco", ROOT / "splits" / "msmarco_rars_train_split.json"),
    ("msmarco", ROOT / "splits" / "msmarco_rars_validation_split.json"),
    ("msmarco", ROOT / "splits" / "msmarco_rars_test_split.json"),
    (
        "trec_dl_2019",
        ROOT
        / "results"
        / "external_confirmation"
        / "trec_dl_2019_msmarco_1m_restricted"
        / "query_manifest.json",
    ),
)


def normalize_query_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_repo_path(path: Path, repo: Path) -> str:
    return "repo://" + str(path.resolve().relative_to(repo.resolve())).replace("\\", "/")


def load_source(namespace: str, path: Path, repo: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    qids = [str(value) for value in payload.get("query_ids", [])]
    texts_raw = payload.get("query_texts")
    if not qids or len(qids) != len(set(qids)):
        raise ValueError(f"Invalid query_ids in {path}")
    texts: list[str | None]
    if texts_raw is None:
        texts = [None] * len(qids)
    else:
        texts = [str(value) for value in texts_raw]
        if len(texts) != len(qids):
            raise ValueError(f"query_ids/query_texts mismatch in {path}")
    for text in texts:
        if text is not None and not normalize_query_text(text):
            raise ValueError(f"Empty normalized query text in {path}")
    source = {
        "dataset_namespace": namespace,
        "path": portable_repo_path(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "query_count": len(qids),
        "query_text_count": sum(text is not None for text in texts),
        "query_ids_field": "query_ids",
        "query_texts_field": "query_texts" if texts_raw is not None else None,
    }
    return source


def build_registry(
    sources: list[tuple[str, Path]],
    output: Path,
    repo: Path,
) -> dict[str, Any]:
    source_records = []
    for namespace, path in sources:
        source_records.append(load_source(namespace, path, repo))
    payload = {
        "registry_id": "beir_nq_prior_query_registry_v1",
        "status": "frozen_before_nq_test_qrels_access",
        "normalization": "Unicode NFKC, casefold, collapse whitespace, strip",
        "same_namespace_id_overlap_only": True,
        "sources": source_records,
        "query_count_with_source_duplicates": sum(
            source["query_count"] for source in source_records
        ),
        "query_text_count_with_source_duplicates": sum(
            source["query_text_count"] for source in source_records
        ),
        "nq_test_qrels_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def parse_source(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("source must be NAMESPACE:PATH")
    namespace, path = value.split(":", 1)
    if not namespace.strip() or not path.strip():
        raise argparse.ArgumentTypeError("source must be NAMESPACE:PATH")
    return namespace.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "protocols" / "beir_nq_prior_query_registry_v1.json",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        help="Override defaults with repeatable NAMESPACE:PATH inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.source if args.source else list(DEFAULT_SOURCES)
    result = build_registry(sources, args.output, args.repo)
    print(json.dumps({
        "registry_id": result["registry_id"],
        "source_count": len(result["sources"]),
        "query_count_with_source_duplicates": result[
            "query_count_with_source_duplicates"
        ],
        "query_text_count_with_source_duplicates": result[
            "query_text_count_with_source_duplicates"
        ],
        "nq_test_qrels_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
