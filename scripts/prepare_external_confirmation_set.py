#!/usr/bin/env python3
"""Prepare and audit an external query set before one-shot RARS/PCA evaluation.

This script does not run retrieval and does not compute Base/PCA/RARS metrics.
It prepares:

- normalized external query manifest;
- query embedding matrix;
- prior-query ID/text overlap audit;
- qrels coverage audit against the indexed document IDs;
- file checksums;
- a draft external confirmation manifest.

The default qrels policy is strict: every positively judged document must exist
in the indexed corpus. For a deterministic MS MARCO 1M subset, TREC DL qrels
from the full passage corpus may fail this requirement. In that case, either
rebuild an index over the compatible full corpus or explicitly preregister a
corpus-restricted qrels policy before evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", required=True, type=Path)
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--doc-ids", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--n-docs", default=1_000_000, type=int)
    p.add_argument(
        "--prior-query-manifest",
        action="append",
        default=[],
        type=Path,
        help="Repeat for each prior query manifest containing query_ids and optionally query_texts.",
    )
    p.add_argument(
        "--qrels-policy",
        choices=("require_all_positive_in_corpus", "restrict_to_indexed_corpus"),
        default="require_all_positive_in_corpus",
    )
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--evaluator", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--pca-config", required=True, type=Path)
    p.add_argument("--pca-basis", required=True, type=Path)
    p.add_argument("--pca-scales", required=True, type=Path)
    p.add_argument("--pca-codes", required=True, type=Path)
    p.add_argument("--rars-config", required=True, type=Path)
    p.add_argument("--rars-basis", required=True, type=Path)
    p.add_argument("--rars-scales", required=True, type=Path)
    p.add_argument("--rars-codes", required=True, type=Path)
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def load_topics(path: Path) -> tuple[list[str], list[str]]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            rows = [
                (str(qid).strip(), str(text).strip())
                for qid, text in payload.items()
            ]
        elif isinstance(payload, list):
            rows = []
            for item in payload:
                qid = item.get("qid", item.get("query_id"))
                text = item.get("query", item.get("text", item.get("title")))
                if qid is None or text is None:
                    raise ValueError("JSON topic record requires qid and query text")
                rows.append((str(qid).strip(), str(text).strip()))
        else:
            raise ValueError("Unsupported JSON topics structure")
    else:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    qid, text = line.split("\t", 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        raise ValueError(f"Unsupported topics line {line_no}")
                    qid, text = parts
                if line_no == 1 and qid.casefold() in {"qid", "query_id"}:
                    continue
                rows.append((qid.strip(), text.strip()))

    qids = [qid for qid, _ in rows]
    texts = [text for _, text in rows]
    if not qids:
        raise ValueError("No external topics found")
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate external query IDs")
    if any(not text for text in texts):
        raise ValueError("Empty external query text")
    return qids, texts


def load_positive_qrels(path: Path) -> dict[str, set[int]]:
    positives: dict[str, set[int]] = {}
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            for qid, docs in payload.items():
                if not isinstance(docs, dict):
                    raise ValueError("JSON qrels mapping values must be mappings")
                for docid, rel in docs.items():
                    if float(rel) > 0:
                        positives.setdefault(str(qid).strip(), set()).add(int(docid))
            return positives
        if isinstance(payload, list):
            for item in payload:
                qid = item.get("qid", item.get("query_id"))
                docid = item.get("docid", item.get("doc_id", item.get("pid")))
                rel = item.get("relevance", item.get("score", 1))
                if float(rel) > 0:
                    positives.setdefault(str(qid).strip(), set()).add(int(docid))
            return positives
        raise ValueError("Unsupported JSON qrels structure")

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if line_no == 1 and any(
                value.casefold() in {"qid", "query_id"} for value in parts
            ):
                continue
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
            else:
                raise ValueError(f"Unsupported qrels line {line_no}")
            if float(rel) > 0:
                positives.setdefault(str(qid).strip(), set()).add(int(docid))
    return positives


def load_prior_queries(
    paths: list[Path],
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    prior_ids: set[str] = set()
    prior_texts: set[str] = set()
    sources: list[dict[str, Any]] = []

    for path in paths:
        payload = read_json(path)
        qids = [str(x).strip() for x in payload.get("query_ids", [])]
        texts = payload.get("query_texts", payload.get("queries", []))
        normalized_texts: list[str] = []

        if isinstance(texts, dict):
            normalized_texts = [
                normalize_query_text(value) for value in texts.values()
            ]
        elif isinstance(texts, list):
            normalized_texts = [normalize_query_text(value) for value in texts]

        prior_ids.update(qids)
        prior_texts.update(value for value in normalized_texts if value)
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "query_id_count": len(qids),
            "query_text_count": len(normalized_texts),
        })

    return prior_ids, prior_texts, sources


def audit_overlap(
    qids: list[str],
    texts: list[str],
    prior_ids: set[str],
    prior_texts: set[str],
) -> dict[str, Any]:
    id_overlap = [qid for qid in qids if qid in prior_ids]
    normalized = [normalize_query_text(text) for text in texts]
    text_overlap = [
        {"qid": qid, "normalized_query": norm}
        for qid, norm in zip(qids, normalized)
        if norm in prior_texts
    ]
    excluded = sorted(
        set(id_overlap) | {item["qid"] for item in text_overlap}
    )
    return {
        "external_query_count": len(qids),
        "prior_id_overlap_count": len(id_overlap),
        "prior_text_overlap_count": len(text_overlap),
        "id_overlap_qids": id_overlap,
        "text_overlap": text_overlap,
        "excluded_qids": excluded,
        "eligible_qids": [qid for qid in qids if qid not in set(excluded)],
        "audit_complete": True,
        "outcome_values_used": False,
    }


def audit_qrels_coverage(
    qids: list[str],
    positives: dict[str, set[int]],
    indexed_docids: set[int],
) -> dict[str, Any]:
    missing_query_qrels = [qid for qid in qids if qid not in positives]
    by_query: list[dict[str, Any]] = []
    total_positive = 0
    total_in_corpus = 0

    for qid in qids:
        docs = positives.get(qid, set())
        in_corpus = docs & indexed_docids
        total_positive += len(docs)
        total_in_corpus += len(in_corpus)
        by_query.append({
            "qid": qid,
            "positive_qrels": len(docs),
            "positive_in_indexed_corpus": len(in_corpus),
            "positive_missing_from_indexed_corpus": len(docs - indexed_docids),
        })

    return {
        "query_count": len(qids),
        "queries_missing_qrels": missing_query_qrels,
        "total_positive_qrels": total_positive,
        "total_positive_in_indexed_corpus": total_in_corpus,
        "total_positive_missing_from_indexed_corpus": (
            total_positive - total_in_corpus
        ),
        "positive_qrels_corpus_coverage": (
            total_in_corpus / total_positive if total_positive else 0.0
        ),
        "by_query": by_query,
    }


def create_restricted_qrels(
    source: Path,
    destination: Path,
    indexed_docids: set[int],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) == 3:
                _, docid, _ = parts
            elif len(parts) >= 4:
                _, _, docid, _ = parts[:4]
            else:
                continue
            if int(docid) in indexed_docids:
                dst.write(raw if raw.endswith("\n") else raw + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qids, texts = load_topics(args.topics)
    prior_ids, prior_texts, prior_sources = load_prior_queries(
        args.prior_query_manifest
    )
    overlap = audit_overlap(qids, texts, prior_ids, prior_texts)

    doc_ids = np.memmap(
        args.doc_ids,
        dtype=np.int64,
        mode="r",
        shape=(args.n_docs,),
    )
    indexed_docids = set(np.asarray(doc_ids, dtype=np.int64).tolist())
    positives = load_positive_qrels(args.qrels)
    coverage = audit_qrels_coverage(qids, positives, indexed_docids)

    if coverage["queries_missing_qrels"]:
        raise ValueError(
            f"{len(coverage['queries_missing_qrels'])} external queries have no positive qrels"
        )

    qrels_for_evaluation = args.qrels
    if args.qrels_policy == "require_all_positive_in_corpus":
        if coverage["total_positive_missing_from_indexed_corpus"] != 0:
            write_json(args.output_dir / "qrels_coverage_audit.json", coverage)
            raise ValueError(
                "Positive TREC qrels are missing from the indexed corpus. "
                "The current 1M subset is not fully compatible with this external set. "
                "Rebuild a compatible full-corpus index or preregister "
                "--qrels-policy restrict_to_indexed_corpus."
            )
    else:
        if args.qrels.suffix.lower() == ".json":
            raise ValueError(
                "Corpus-restricted qrels generation currently requires text/TREC qrels"
            )
        qrels_for_evaluation = args.output_dir / "qrels.indexed_corpus.txt"
        create_restricted_qrels(
            args.qrels,
            qrels_for_evaluation,
            indexed_docids,
        )

    eligible = set(overlap["eligible_qids"])
    filtered_qids = [qid for qid in qids if qid in eligible]
    filtered_texts = [
        text for qid, text in zip(qids, texts) if qid in eligible
    ]
    if not filtered_qids:
        raise ValueError("No external queries remain after prior-overlap exclusion")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install sentence-transformers to encode queries") from exc

    model = SentenceTransformer(args.model)
    vectors = model.encode(
        filtered_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    query_vectors_path = args.output_dir / "query_vectors.fp32.npy"
    np.save(query_vectors_path, vectors)

    query_manifest = {
        "dataset_name": args.dataset_name,
        "query_ids": filtered_qids,
        "query_rows": list(range(len(filtered_qids))),
        "query_texts": filtered_texts,
        "normalized_query_texts": [
            normalize_query_text(text) for text in filtered_texts
        ],
        "embedding_model": args.model,
        "embedding_normalized": True,
        "dimension": int(vectors.shape[1]),
        "excluded_prior_overlap_qids": overlap["excluded_qids"],
    }
    query_manifest_path = args.output_dir / "query_manifest.json"
    write_json(query_manifest_path, query_manifest)
    write_json(args.output_dir / "prior_query_overlap_audit.json", {
        **overlap,
        "prior_sources": prior_sources,
    })
    write_json(args.output_dir / "qrels_coverage_audit.json", {
        **coverage,
        "qrels_policy": args.qrels_policy,
        "qrels_for_evaluation": str(qrels_for_evaluation),
    })

    frozen_paths = {
        "query_vectors": query_vectors_path,
        "query_manifest": query_manifest_path,
        "qrels": qrels_for_evaluation,
        "doc_ids": args.doc_ids,
        "index": args.index,
        "pca_config": args.pca_config,
        "pca_basis": args.pca_basis,
        "pca_scales": args.pca_scales,
        "pca_codes": args.pca_codes,
        "rars_config": args.rars_config,
        "rars_basis": args.rars_basis,
        "rars_scales": args.rars_scales,
        "rars_codes": args.rars_codes,
        "evaluator": args.evaluator,
        "protocol": args.protocol,
    }

    manifest = {
        "manifest_id": f"{args.dataset_name}_external_confirmation_v1",
        "status": "DRAFT_READY_FOR_REVIEW",
        "protocol_id": "rars_pca_comparator_v1",
        "dataset_name": args.dataset_name,
        "query_count": len(filtered_qids),
        "primary_metric": "recall@10",
        "primary_contrast": "rars_r16_int8_minus_pca_r16_int8",
        "prior_overlap_audit_complete": True,
        "outcome_inspection_before_freeze": False,
        "qrels_policy": args.qrels_policy,
        "positive_qrels_corpus_coverage": coverage[
            "positive_qrels_corpus_coverage"
        ],
        "files": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in frozen_paths.items()
        },
    }
    write_json(args.output_dir / "external_confirmation_manifest.draft.json", manifest)

    print(json.dumps({
        "dataset_name": args.dataset_name,
        "eligible_query_count": len(filtered_qids),
        "excluded_prior_overlap_count": len(overlap["excluded_qids"]),
        "positive_qrels_corpus_coverage": coverage[
            "positive_qrels_corpus_coverage"
        ],
        "qrels_policy": args.qrels_policy,
        "manifest_status": manifest["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
