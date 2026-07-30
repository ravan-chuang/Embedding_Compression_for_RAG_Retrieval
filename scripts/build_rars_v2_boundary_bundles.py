#!/usr/bin/env python3
"""Build leakage-guarded RARS-v2 train/validation candidate bundles.

Only BEIR NQ ``qrels/train.tsv`` is parsed.  The builder reuses the frozen
Stage-1 query vectors and Stage-2 ANN caches, labels candidates, records the
full-qrels relevant denominator, and stores residuals only for the union of
candidate document rows.  This avoids materializing a multi-gigabyte full
residual matrix on Google Drive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v2_boundary_loss_feasibility_v1"
SOURCE_PROTOCOL_ID = "beir_nq_rars_pca_confirmation_v1"
FORBIDDEN_PATH_MARKERS = (
    "qrels/test.tsv",
    "stage3/evaluation",
    "stage3/posthoc",
    "posthoc_diagnosis",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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


def reject_closed_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    marker = next(
        (value for value in FORBIDDEN_PATH_MARKERS if value in normalized), None
    )
    if marker is not None:
        raise ValueError(f"Closed-test/post-hoc path is forbidden: {marker}")


def load_train_qrels(path: Path) -> dict[str, set[str]]:
    reject_closed_path(path)
    if path.name.casefold() != "train.tsv":
        raise ValueError("Only qrels/train.tsv may supply development labels")
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                fields = line.split()
            if line_number == 1 and fields[0].casefold() in {
                "query-id", "query_id", "qid"
            }:
                continue
            if len(fields) < 2:
                raise ValueError(f"Malformed train qrels line {line_number}")
            qid, docid = fields[0].strip(), fields[1].strip()
            relevance = float(fields[2]) if len(fields) >= 3 else 1.0
            if relevance > 0:
                qrels.setdefault(qid, set()).add(docid)
    if not qrels:
        raise ValueError("No positive train qrels were found")
    return qrels


def decode_docids(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def candidate_labels(
    qids: list[str],
    ann_rows: np.ndarray,
    doc_ids: np.ndarray,
    qrels: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(ann_rows, dtype=np.int64)
    if rows.ndim != 2 or np.any(rows < 0) or np.any(rows >= len(doc_ids)):
        raise ValueError("ANN candidate rows are invalid")
    labels = np.zeros(rows.shape, dtype=np.uint8)
    counts = np.empty(len(qids), dtype=np.int32)
    for query_index, qid in enumerate(qids):
        relevant = qrels.get(str(qid), set())
        if not relevant:
            raise ValueError(f"Development query {qid} has no positive train qrels")
        candidate_ids = decode_docids(doc_ids[rows[query_index]])
        labels[query_index] = np.asarray(
            [docid in relevant for docid in candidate_ids], dtype=np.uint8
        )
        counts[query_index] = len(relevant)
    return labels, counts


def sidecar_candidate_scores(
    queries: np.ndarray,
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    basis: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    """Reproduce a frozen rank-16 PCA/RARS validation score matrix."""

    rows = np.asarray(ann_rows, dtype=np.int64)
    result = np.asarray(ann_scores, dtype=np.float32).copy()
    depth = min(int(top_b), rows.shape[1])
    selected = rows[:, :depth]
    if np.any(selected < 0):
        raise ValueError("Frozen baseline candidates contain invalid rows")
    query_projection = np.asarray(queries, dtype=np.float32) @ np.asarray(
        basis, dtype=np.float32
    )
    coefficients = codes[selected].astype(np.float32) * np.asarray(
        scales, dtype=np.float32
    )
    correction = np.einsum("qr,qcr->qc", query_projection, coefficients)
    result[:, :depth] += float(alpha) * correction
    return result


def verify_relevant_documents_exist(
    doc_ids: np.ndarray, qrels: dict[str, set[str]]
) -> None:
    required = set().union(*qrels.values())
    remaining = set(required)
    batch_size = 250_000
    for start in range(0, len(doc_ids), batch_size):
        end = min(start + batch_size, len(doc_ids))
        remaining.difference_update(decode_docids(doc_ids[start:end]))
        if not remaining:
            break
    if remaining:
        raise ValueError(
            f"{len(remaining)} positive train-qrels documents are absent from the "
            f"frozen corpus; examples={sorted(remaining)[:5]}"
        )


def candidate_residual_table(
    index: Any,
    embeddings: np.memmap,
    ann_rows: np.ndarray,
    output_dir: Path,
    *,
    batch_size: int,
) -> tuple[Path, Path, Path]:
    rows = np.asarray(ann_rows, dtype=np.int64)
    unique_rows = np.unique(rows.reshape(-1))
    if len(unique_rows) == 0 or unique_rows[0] < 0:
        raise ValueError("Cannot build residuals from invalid candidate rows")
    local = np.searchsorted(unique_rows, rows).astype(np.int64)
    if not np.array_equal(unique_rows[local], rows):
        raise AssertionError("Candidate-to-residual mapping failed")
    unique_path = output_dir / "candidate_doc_rows.int64.npy"
    lookup_path = output_dir / "ann_residual_rows.int64.npy"
    residual_path = output_dir / "candidate_residuals.float32.npy"
    atomic_save_npy(unique_path, unique_rows.astype(np.int64))
    atomic_save_npy(lookup_path, local)
    temporary = residual_path.with_name(residual_path.name + ".part")
    residuals = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(unique_rows), embeddings.shape[1]),
    )
    for start in range(0, len(unique_rows), batch_size):
        end = min(start + batch_size, len(unique_rows))
        doc_rows = unique_rows[start:end]
        reconstructed = index.reconstruct_batch(doc_rows).astype(np.float32)
        original = np.asarray(embeddings[doc_rows], dtype=np.float32)
        residuals[start:end] = original - reconstructed
        residuals.flush()
        if end % 100_000 == 0 or end == len(unique_rows):
            print(f"candidate residuals {end:,}/{len(unique_rows):,}")
    del residuals
    temporary.replace(residual_path)
    return unique_path, lookup_path, residual_path


def validate_source_manifest(payload: dict[str, Any], label: str) -> None:
    if payload.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise ValueError(f"Unexpected {label} source protocol")
    if payload.get("test_qrels_accessed") is not False:
        raise ValueError(f"Unsafe {label} source test flag")


def build_bundles(
    artifact_root: Path,
    output_root: Path,
    *,
    residual_batch_size: int,
) -> dict[str, Any]:
    reject_closed_path(artifact_root)
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_manifest = read_json(
        artifact_root / "stage1/corpus/corpus_artifacts_manifest.json"
    )
    index_manifest = read_json(artifact_root / "stage1/index/index_manifest.json")
    vector_manifest = read_json(
        artifact_root / "stage1/queries/train_validation_query_vector_manifest.json"
    )
    for payload, label in [
        (corpus_manifest, "corpus"),
        (index_manifest, "index"),
        (vector_manifest, "query-vector"),
    ]:
        validate_source_manifest(payload, label)

    qrels_path = artifact_root / "stage1/data/nq/qrels/train.tsv"
    qrels = load_train_qrels(qrels_path)
    n_docs = int(corpus_manifest["document_count"])
    dimension = int(corpus_manifest["dimension"])
    doc_ids = np.memmap(
        Path(corpus_manifest["doc_ids"]["path"]),
        dtype=f"S{int(corpus_manifest['doc_id_width_bytes'])}",
        mode="r",
        shape=(n_docs,),
    )
    verify_relevant_documents_exist(doc_ids, qrels)
    embeddings = np.memmap(
        Path(corpus_manifest["document_embeddings"]["path"]),
        dtype=np.float16,
        mode="r",
        shape=(n_docs, dimension),
    )
    all_queries = np.load(Path(vector_manifest["vectors"]["path"]), mmap_mode="r")

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - Colab dependency
        raise RuntimeError("Faiss is required to reconstruct candidate residuals") from exc
    index = faiss.read_index(str(Path(index_manifest["index"]["path"])))
    if hasattr(index, "make_direct_map"):
        index.make_direct_map()

    sidecar_manifest = read_json(
        artifact_root / "stage2/sidecars/sidecar_training_manifest.json"
    )
    if sidecar_manifest.get("test_qrels_accessed") is not False:
        raise ValueError("Unsafe frozen sidecar manifest test flag")
    rank = 16
    frozen_baselines: dict[str, dict[str, Any]] = {}
    for name in ("pca", "rars"):
        config = read_json(
            artifact_root / "stage2/sidecars" / name / "selected_config.json"
        )
        basis = np.load(Path(sidecar_manifest["files"][f"{name}_basis"]["path"]))
        scales = np.load(Path(sidecar_manifest["files"][f"{name}_scales"]["path"]))
        codes = np.memmap(
            Path(sidecar_manifest["files"][f"{name}_codes"]["path"]),
            dtype=np.int8,
            mode="r",
            shape=(n_docs, rank),
        )
        frozen_baselines[name] = {
            "config": config,
            "basis": basis,
            "scales": scales,
            "codes": codes,
        }

    role_specs = {
        "train": ("fit", "train_query_manifest.json"),
        "validation": ("validation", "validation_query_manifest.json"),
    }
    role_manifests: dict[str, Any] = {}
    for role, (cache_name, query_manifest_name) in role_specs.items():
        split_manifest = read_json(
            artifact_root / "stage1/query_splits" / query_manifest_name
        )
        validate_source_manifest(split_manifest, role)
        qids = [str(value) for value in split_manifest["query_ids"]]
        block = vector_manifest["blocks"][cache_name]
        start, stop = int(block["start"]), int(block["stop"])
        queries = np.asarray(all_queries[start:stop], dtype=np.float32)
        if len(queries) != len(qids):
            raise ValueError(f"{role} query vector count mismatch")
        cache = artifact_root / "stage2/sidecars/candidate_cache" / cache_name
        rows_path = cache / "ann_rows.int64.npy"
        scores_path = cache / "ann_scores.float32.npy"
        rows = np.load(rows_path, mmap_mode="r")
        scores = np.load(scores_path, mmap_mode="r")
        if rows.shape != scores.shape or rows.shape[0] != len(qids):
            raise ValueError(f"{role} candidate cache shape mismatch")
        labels, counts = candidate_labels(qids, rows, doc_ids, qrels)

        role_dir = output_root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        query_path = role_dir / "query_vectors.float32.npy"
        output_rows_path = role_dir / "ann_rows.int64.npy"
        output_scores_path = role_dir / "ann_scores.float32.npy"
        labels_path = role_dir / "candidate_relevance.uint8.npy"
        counts_path = role_dir / "relevant_counts.int32.npy"
        atomic_save_npy(query_path, queries)
        atomic_save_npy(output_rows_path, np.asarray(rows, dtype=np.int64))
        atomic_save_npy(output_scores_path, np.asarray(scores, dtype=np.float32))
        atomic_save_npy(labels_path, labels)
        atomic_save_npy(counts_path, counts)
        baseline_paths: dict[str, Path] = {}
        for name, baseline in frozen_baselines.items():
            config = baseline["config"]
            baseline_scores = sidecar_candidate_scores(
                queries,
                rows,
                scores,
                baseline["basis"],
                baseline["codes"],
                baseline["scales"],
                alpha=float(config["alpha"]),
                top_b=int(config["top_b"]),
            )
            baseline_path = role_dir / f"{name}_scores.float32.npy"
            atomic_save_npy(baseline_path, baseline_scores)
            baseline_paths[name] = baseline_path
        unique_path, lookup_path, residual_path = candidate_residual_table(
            index,
            embeddings,
            rows,
            role_dir,
            batch_size=residual_batch_size,
        )
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "split_role": role,
            "source": "BEIR NQ train archive deterministic development split",
            "query_count": len(qids),
            "candidate_count": int(rows.shape[1]),
            "dimension": dimension,
            "residual_scope": "candidate_union",
            "unique_candidate_document_count": int(len(np.load(unique_path, mmap_mode="r"))),
            "qrels_source": file_record(qrels_path),
            "files": {
                "queries": file_record(query_path),
                "ann_rows": file_record(output_rows_path),
                "ann_scores": file_record(output_scores_path),
                "candidate_relevance": file_record(labels_path),
                "relevant_counts": file_record(counts_path),
                "candidate_doc_rows": file_record(unique_path),
                "ann_residual_rows": file_record(lookup_path),
                "candidate_residuals": file_record(residual_path),
                "pca_scores": file_record(baseline_paths["pca"]),
                "rars_scores": file_record(baseline_paths["rars"]),
            },
            "development_qrels_used": True,
            "test_qrels_accessed": False,
            "nq_test_retuning_authorized": False,
        }
        atomic_write_json(role_dir / "manifest.json", manifest)
        role_manifests[role] = manifest

    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "development_bundles_complete",
        "roles": {
            role: {
                "query_count": value["query_count"],
                "unique_candidate_document_count": value[
                    "unique_candidate_document_count"
                ],
            }
            for role, value in role_manifests.items()
        },
        "train_validation_overlap_allowed": False,
        "test_qrels_accessed": False,
        "nq_test_retuning_authorized": False,
    }
    atomic_write_json(output_root / "bundle_build_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--residual-batch-size", default=20_000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_bundles(
        args.artifact_root,
        args.output_root,
        residual_batch_size=args.residual_batch_size,
    ), indent=2))


if __name__ == "__main__":
    main()
