#!/usr/bin/env python3
"""Prepare same-encoder FiQA and SciFact inputs for the V16 bundle builder.

This script downloads the public BEIR archives, deterministically splits the
union of positive-qrels queries into fit/evaluation roles, encodes both
corpora with one pinned BGE revision, and builds the same IVF-PQ recipe.
It does not compute any retrieval metric or fit a residual sidecar basis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import urllib.request
import zipfile
from typing import Any

import numpy as np


MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"
DIMENSION = 384
DATASETS = {
    "fiqa_bge_same_encoder": {
        "archive": "fiqa.zip",
        "directory": "fiqa",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    },
    "scifact_bge_same_encoder": {
        "archive": "scifact.zip",
        "directory": "scifact",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    },
}


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
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_positive_qrels(directory: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    paths = sorted((directory / "qrels").glob("*.tsv"))
    if not paths:
        raise ValueError(f"No qrels TSV files found under {directory}")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if int(row["score"]) > 0:
                    output.setdefault(str(row["query-id"]), set()).add(
                        str(row["corpus-id"])
                    )
    return output


def deterministic_role(query_id: str, domain_id: str) -> str:
    digest = hashlib.sha256(
        f"rars-v16-role-v1\0{domain_id}\0{query_id}".encode("utf-8")
    ).digest()
    return "fit" if int.from_bytes(digest[:8], "big") % 5 < 3 else "evaluation"


def _download_and_extract(
    cache_root: Path, domain_id: str
) -> tuple[Path, dict[str, Any]]:
    spec = DATASETS[domain_id]
    archive = cache_root / spec["archive"]
    directory = cache_root / spec["directory"]
    cache_root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(str(spec["url"]), archive)
    if not directory.exists():
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(cache_root)
    required = (
        directory / "corpus.jsonl",
        directory / "queries.jsonl",
        directory / "qrels",
    )
    if any(not path.exists() for path in required):
        raise ValueError(f"Incomplete BEIR archive extraction for {domain_id}")
    return directory, {
        "url": spec["url"],
        "archive": file_record(archive),
        "corpus": file_record(directory / "corpus.jsonl"),
        "queries": file_record(directory / "queries.jsonl"),
        "qrels": {
            path.name: file_record(path)
            for path in sorted((directory / "qrels").glob("*.tsv"))
        },
    }


def _encode(
    texts: list[str],
    *,
    model: Any,
    batch_size: int,
) -> np.ndarray:
    values = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(texts), DIMENSION) or not np.all(np.isfinite(values)):
        raise ValueError("Encoder returned an unexpected shape or non-finite values")
    return values


def _build_index(
    embeddings: np.ndarray,
    *,
    nlist: int,
    nprobe: int,
    subquantizers: int,
    nbits: int,
    seed: int,
) -> Any:
    import faiss

    if embeddings.shape[1] % subquantizers:
        raise ValueError("Embedding dimension must be divisible by PQ M")
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFPQ(
        quantizer,
        embeddings.shape[1],
        nlist,
        subquantizers,
        nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.nprobe = nprobe
    rng = np.random.default_rng(seed)
    train_count = min(len(embeddings), 50_000)
    selected = (
        np.arange(len(embeddings))
        if train_count == len(embeddings)
        else np.sort(rng.choice(len(embeddings), train_count, replace=False))
    )
    index.train(
        np.ascontiguousarray(embeddings[selected], dtype=np.float32)
    )
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    if int(index.ntotal) != len(embeddings):
        raise AssertionError("Faiss index did not add every document")
    return index


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("Refusing to reuse a non-empty V16 preparation root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    domains: list[dict[str, Any]] = []
    for domain_position, domain_id in enumerate(DATASETS):
        source_dir, sources = _download_and_extract(args.cache_root, domain_id)
        corpus = _read_jsonl(source_dir / "corpus.jsonl")
        queries = {
            str(row["_id"]): str(row["text"])
            for row in _read_jsonl(source_dir / "queries.jsonl")
        }
        qrels = _read_positive_qrels(source_dir)
        doc_ids = [str(row["_id"]) for row in corpus]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError(f"{domain_id} has duplicate document IDs")
        doc_to_row = {doc_id: row for row, doc_id in enumerate(doc_ids)}
        eligible = sorted(
            query_id
            for query_id, positives in qrels.items()
            if query_id in queries
            and positives
            and all(doc_id in doc_to_row for doc_id in positives)
        )
        roles = {
            role: [
                query_id
                for query_id in eligible
                if deterministic_role(query_id, domain_id) == role
            ]
            for role in ("fit", "evaluation")
        }
        if len(roles["fit"]) < 150 or len(roles["evaluation"]) < 200:
            raise ValueError(
                f"{domain_id} deterministic split is undersized: "
                f"{ {role: len(ids) for role, ids in roles.items()} }"
            )
        doc_texts = [
            ((row.get("title") or "") + "\n" + (row.get("text") or "")).strip()
            for row in corpus
        ]
        doc_embeddings = _encode(
            doc_texts, model=model, batch_size=args.document_batch_size
        )
        domain_dir = args.output_root / domain_id
        domain_dir.mkdir()
        embeddings_path = domain_dir / "embeddings.float16.memmap"
        memmap = np.memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float16,
            shape=doc_embeddings.shape,
        )
        memmap[:] = doc_embeddings.astype(np.float16)
        memmap.flush()
        del memmap
        index = _build_index(
            doc_embeddings,
            nlist=args.nlist,
            nprobe=args.nprobe,
            subquantizers=args.subquantizers,
            nbits=args.nbits,
            seed=args.seed + domain_position,
        )
        index_path = domain_dir / "frozen_ivfpq.index"
        faiss.write_index(index, str(index_path))
        role_records: dict[str, Any] = {}
        for role, query_ids in roles.items():
            role_dir = domain_dir / role
            role_dir.mkdir()
            query_vectors = _encode(
                [queries[query_id] for query_id in query_ids],
                model=model,
                batch_size=args.query_batch_size,
            )
            query_ids_path = role_dir / "query_ids.utf8.txt"
            query_ids_path.write_text(
                "\n".join(query_ids) + "\n", encoding="utf-8"
            )
            query_vectors_path = role_dir / "query_vectors.float32.npy"
            atomic_save(query_vectors_path, query_vectors)
            qrels_rows_path = role_dir / "qrels_rows.json"
            atomic_json(
                qrels_rows_path,
                {
                    query_id: sorted(
                        doc_to_row[doc_id] for doc_id in qrels[query_id]
                    )
                    for query_id in query_ids
                },
            )
            role_records[role] = {
                "query_count": len(query_ids),
                "query_ids": file_record(query_ids_path),
                "query_vectors": file_record(query_vectors_path),
                "qrels_rows": file_record(qrels_rows_path),
            }
        metadata = {
            "schema_version": 1,
            "status": "RARS_V16_BEIR_DOMAIN_INPUTS_PREPARED",
            "domain_id": domain_id,
            "source": sources,
            "encoder": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "dimension": DIMENSION,
                "normalize_embeddings": True,
            },
            "document_count": len(doc_ids),
            "eligible_query_count": len(eligible),
            "roles": role_records,
            "embeddings": file_record(embeddings_path),
            "index": file_record(index_path),
            "index_contract": {
                "nlist": args.nlist,
                "nprobe": args.nprobe,
                "subquantizers": args.subquantizers,
                "nbits": args.nbits,
                "metric": "inner_product",
            },
            "metrics_computed": False,
            "sidecar_basis_fitted": False,
        }
        metadata_path = domain_dir / "prepared_domain.json"
        atomic_json(metadata_path, metadata)
        domains.append(
            {
                "domain_id": domain_id,
                "prepared_domain": file_record(metadata_path),
            }
        )
    del model
    complete = {
        "schema_version": 1,
        "status": "RARS_V16_SAME_ENCODER_DOMAIN_INPUTS_PREPARED",
        "domains": domains,
        "encoder_id": MODEL_ID,
        "encoder_revision": MODEL_REVISION,
        "index_recipe_identical": True,
        "metrics_computed": False,
        "sidecar_basis_fitted": False,
    }
    atomic_json(args.output_root / "preparation_complete.json", complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261600)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--subquantizers", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
