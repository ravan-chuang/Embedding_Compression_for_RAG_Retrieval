#!/usr/bin/env python3
"""Build one candidate/residual bundle for a frozen RARS-v17 setting.

The builder deliberately stops before any retrieval metric or learned basis is
computed.  It accepts query vectors and positive qrels already mapped to the
row order of a frozen Faiss index, retrieves one deterministic candidate pool,
and materializes only the residual rows reached by those candidates.

This generic row-based contract avoids assuming that document identifiers are
numeric or shared across corpora.  Dataset preparation must freeze the mapping
from public document identifiers to index rows before this script is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v17_million_scale_setting_transfer_v1"
CANONICAL_PROTOCOL = Path(
    "protocols/rars_v17_million_scale_setting_transfer_v1.json"
)
FORBIDDEN_PATH_TOKENS = (
    "clean_test",
    "future_method_holdout",
    "oracle_audit",
    "trec_dl",
)
NQ_SETTING = "beir_nq_2_68m_bge_opened_test_diagnostic"
MSMARCO_SETTING = "msmarco_1m_bge_opened_development"


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
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def _reject_forbidden_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"V17 refuses forbidden {label} path: {path}")


def _load_protocol(
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
        raise ValueError("V17 bundle build requires a clean exact checkout")
    protocol = json.loads(canonical.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status")
        != "FROZEN_BEFORE_FIRST_V17_MILLION_SCALE_DIAGNOSTIC_RUN"
    ):
        raise ValueError("Unexpected V17 protocol identity or status")
    source_paths = (
        CANONICAL_PROTOCOL,
        Path("scripts/build_rars_v17_setting_bundle.py"),
        Path("scripts/freeze_rars_v17_setting_manifest.py"),
        Path("scripts/prepare_rars_v17_nq_roles.py"),
        Path("scripts/adapt_rars_v17_msmarco_bundle.py"),
        Path("scripts/rars_v17_million_scale_core.py"),
        Path("scripts/evaluate_rars_v17_million_scale.py"),
    )
    return protocol, {
        str(relative): file_record((repo_root / relative).resolve(strict=True))
        for relative in source_paths
    }


def _load_matrix(
    path: Path,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, int],
) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"{path} has shape/dtype {value.shape}/{value.dtype}; "
                f"expected {shape}/{dtype}"
            )
        return value
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {expected_bytes}"
        )
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def _load_qrels_rows(
    path: Path, query_ids: list[str], document_count: int
) -> dict[str, list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(map(str, raw)) != set(query_ids):
        raise ValueError("Qrels-row JSON must cover exactly the frozen query ids")
    output: dict[str, list[int]] = {}
    for query_id in query_ids:
        values = raw.get(query_id)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Query {query_id} has no positive qrels rows")
        rows = sorted({int(value) for value in values})
        if len(rows) != len(values):
            raise ValueError(f"Query {query_id} has duplicate qrels rows")
        if rows[0] < 0 or rows[-1] >= document_count:
            raise ValueError(f"Query {query_id} qrels escaped the frozen corpus")
        output[query_id] = rows
    return output


def _reconstruct_rows(index: Any, rows: np.ndarray, dimension: int) -> np.ndarray:
    output = np.empty((len(rows), dimension), dtype=np.float32)
    for start in range(0, len(rows), 20_000):
        end = min(start + 20_000, len(rows))
        contiguous = np.ascontiguousarray(rows[start:end], dtype=np.int64)
        output[start:end] = np.asarray(
            index.reconstruct_batch(contiguous), dtype=np.float32
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Frozen-index reconstructions contain non-finite values")
    return output


def _extract_ivfpq(index: Any, faiss_module: Any) -> tuple[Any, Any]:
    """Downcast Faiss's generic IVF wrapper before inspecting PQ fields."""

    try:
        extracted = faiss_module.extract_index_ivf(index)
        ivf = faiss_module.downcast_index(extracted)
    except Exception as error:
        raise ValueError("V17 requires a frozen IVF-family Faiss index") from error
    pq = getattr(ivf, "pq", None)
    if pq is None:
        raise ValueError("V17 requires a frozen IVF-PQ index")
    return ivf, pq


def _deterministic_fold(query_id: str, setting_id: str, fold_count: int) -> int:
    payload = (
        "rars_v17_setting_fold_v1\0" + setting_id + "\0" + query_id
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % fold_count


def build(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    repo_root = Path(__file__).resolve().parents[1]
    for path, label in (
        (args.query_ids, "query ids"),
        (args.query_vectors, "query vectors"),
        (args.qrels_rows, "qrels rows"),
        (args.embeddings, "embeddings"),
        (args.index, "index"),
        (args.output_dir, "output"),
    ):
        _reject_forbidden_path(path, label)
    protocol, source_blobs = _load_protocol(
        repo_root, args.protocol, args.source_commit
    )
    allowed_settings = protocol["data_policy"]["allowed_development_settings"]
    if args.setting_id not in allowed_settings:
        raise ValueError(f"Setting {args.setting_id!r} is not preregistered")
    if args.evidence_role not in ("fit", "evaluation"):
        raise ValueError("V17 domain role is not a permitted development role")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to overwrite a non-empty V17 domain bundle")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_ids = args.query_ids.read_text(encoding="utf-8").splitlines()
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("Query ids must be non-empty and unique")
    queries = np.load(args.query_vectors, mmap_mode="r", allow_pickle=False)
    if queries.shape != (len(query_ids), args.dimension) or queries.dtype != np.float32:
        raise ValueError("Query vectors differ from the V17 shape/dtype contract")
    if not np.all(np.isfinite(queries)):
        raise ValueError("Query vectors contain non-finite values")
    qrels = _load_qrels_rows(args.qrels_rows, query_ids, args.document_count)
    embeddings_dtype = np.dtype(args.embeddings_dtype)
    if embeddings_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("Embeddings must be float16 or float32")
    embeddings = _load_matrix(
        args.embeddings,
        dtype=embeddings_dtype,
        shape=(args.document_count, args.dimension),
    )

    index = faiss.read_index(str(args.index))
    if int(index.d) != args.dimension or int(index.ntotal) != args.document_count:
        raise ValueError("Faiss index dimension/document count changed")
    ivf, pq = _extract_ivfpq(index, faiss)
    expected_index = protocol["index_policy"].get(args.setting_id)
    if not isinstance(expected_index, dict):
        raise ValueError("V17 setting lacks an index contract")
    expected_nprobe = int(expected_index["nprobe"])
    if args.nprobe is not None and int(args.nprobe) != expected_nprobe:
        raise ValueError(
            f"V17 {args.setting_id} nprobe must be {expected_nprobe}, "
            f"not {args.nprobe}"
        )
    ivf.nprobe = expected_nprobe
    common_index = protocol["index_policy"]["common"]
    expected_metric = str(common_index["metric"])
    metric_by_name = {
        "inner_product": int(faiss.METRIC_INNER_PRODUCT),
        "l2": int(faiss.METRIC_L2),
    }
    if expected_metric not in metric_by_name:
        raise ValueError(f"Unsupported protocol metric: {expected_metric}")
    observed_index = {
        "dimension": int(index.d),
        "nlist": int(ivf.nlist),
        "nprobe": int(ivf.nprobe),
        "subquantizers": int(pq.M),
        "bits_per_subquantizer": int(pq.nbits),
        "metric_type": int(index.metric_type),
    }
    expected_values = {
        "dimension": int(common_index["dimension"]),
        "nlist": int(expected_index["nlist"]),
        "nprobe": int(expected_index["nprobe"]),
        "subquantizers": int(common_index["subquantizers"]),
        "bits_per_subquantizer": int(common_index["bits_per_subquantizer"]),
        "metric_type": metric_by_name[expected_metric],
    }
    if observed_index != expected_values:
        raise ValueError(
            f"V17 index recipe changed: {observed_index} != {expected_values}"
        )
    minimum_documents = max(
        int(protocol["data_policy"]["minimum_document_count_per_setting"]),
        int(expected_index["minimum_documents"]),
    )
    if args.document_count < minimum_documents:
        raise ValueError(
            f"V17 requires at least {minimum_documents} documents for "
            f"{args.setting_id}"
        )
    if args.candidate_pool != int(common_index["candidate_pool"]):
        raise ValueError("Candidate pool differs from the V17 protocol")
    minimum_query_key = (
        "minimum_fit_queries_per_domain"
        if args.evidence_role == "fit"
        else "minimum_evaluation_queries_per_domain"
    )
    minimum_queries = int(protocol["data_policy"][minimum_query_key])
    if len(query_ids) < minimum_queries:
        raise ValueError(
            f"V17 {args.setting_id} {args.evidence_role} requires at least "
            f"{minimum_queries} queries"
        )
    scores, rows = index.search(
        np.ascontiguousarray(queries, dtype=np.float32), args.candidate_pool
    )
    scores = np.asarray(scores, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64)
    expected_shape = (len(query_ids), args.candidate_pool)
    if (
        scores.shape != expected_shape
        or rows.shape != expected_shape
        or np.any(rows < 0)
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("Frozen index did not return a complete finite candidate pool")

    candidate_rows = np.unique(rows.reshape(-1)).astype(np.int64)
    lookup = np.searchsorted(candidate_rows, rows).astype(np.int64)
    if not np.array_equal(candidate_rows[lookup], rows):
        raise AssertionError("Candidate residual lookup construction failed")
    ivf.make_direct_map()
    reconstructed = _reconstruct_rows(index, candidate_rows, args.dimension)
    residuals = (
        np.asarray(embeddings[candidate_rows], dtype=np.float32) - reconstructed
    ).astype(np.float32)

    labels = np.zeros(expected_shape, dtype=np.uint8)
    relevant_counts = np.empty(len(query_ids), dtype=np.int32)
    for query_index, query_id in enumerate(query_ids):
        positive_rows = np.asarray(qrels[query_id], dtype=np.int64)
        relevant_counts[query_index] = len(positive_rows)
        labels[query_index] = np.isin(
            rows[query_index], positive_rows, assume_unique=False
        ).astype(np.uint8)
    folds = np.asarray(
        [
            _deterministic_fold(query_id, args.setting_id, args.fold_count)
            for query_id in query_ids
        ],
        dtype=np.int64,
    )
    if len(np.unique(folds)) != args.fold_count:
        raise ValueError("Hash split left at least one V17 fold empty")

    arrays = {
        "query_vectors.float32.npy": np.asarray(queries, dtype=np.float32),
        "fold_ids.int64.npy": folds,
        "ann_rows.int64.npy": rows,
        "ann_scores.float32.npy": scores,
        "ann_residual_rows.int64.npy": lookup,
        "candidate_doc_rows.int64.npy": candidate_rows,
        "candidate_residuals.float32.npy": residuals,
        "candidate_relevance.uint8.npy": labels,
        "relevant_counts.int32.npy": relevant_counts,
    }
    for filename, value in arrays.items():
        atomic_save(args.output_dir / filename, value)
    output_qids = args.output_dir / "query_ids.utf8.txt"
    output_qids.write_text("\n".join(query_ids) + "\n", encoding="utf-8")
    records = {
        filename: file_record(args.output_dir / filename)
        for filename in (*arrays, "query_ids.utf8.txt")
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V17_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS",
        "source_commit": args.source_commit,
        "domain_id": args.setting_id,
        "setting_id": args.setting_id,
        "evidence_role": args.evidence_role,
        "encoder_id": args.encoder_id,
        "encoder_revision": args.encoder_revision,
        "dimension": args.dimension,
        "encoder": {
            "id": args.encoder_id,
            "revision": args.encoder_revision,
            "dimension": args.dimension,
        },
        "query_count": len(query_ids),
        "fold_count": args.fold_count,
        "fold_counts": np.bincount(
            folds, minlength=args.fold_count
        ).astype(int).tolist(),
        "document_count": args.document_count,
        "candidate_pool": args.candidate_pool,
        "candidate_residual_count": len(candidate_rows),
        "index_contract": {
            "dimension": int(index.d),
            "document_count": int(index.ntotal),
            "nlist": int(ivf.nlist),
            "nprobe": int(ivf.nprobe),
            "subquantizers": int(pq.M),
            "bits_per_subquantizer": int(pq.nbits),
            "metric_type": int(index.metric_type),
        },
        "inputs": {
            "query_ids": file_record(args.query_ids),
            "query_vectors": file_record(args.query_vectors),
            "qrels_rows": file_record(args.qrels_rows),
            "embeddings": file_record(args.embeddings),
            "index": file_record(args.index),
        },
        "source_blobs": source_blobs,
        "opened_development_evidence": True,
        "files": records,
        "metrics_computed": False,
        "basis_fitted": False,
        "opened_nq_test_evidence": args.setting_id == NQ_SETTING,
        "prior_opened_test_artifact_reused": args.setting_id == NQ_SETTING,
        "prior_confirmation_outcomes_known": args.setting_id == NQ_SETTING,
        "independent_confirmation_claim_allowed": False,
        "closed_test_opened": args.setting_id == NQ_SETTING,
    }
    manifest_path = args.output_dir / "bundle_manifest.json"
    atomic_json(manifest_path, manifest)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V17_DOMAIN_BUNDLE_COMPLETE",
        "source_commit": args.source_commit,
        "manifest": file_record(manifest_path),
        "outputs": records,
        "metrics_computed": False,
        "basis_fitted": False,
        "opened_nq_test_evidence": args.setting_id == NQ_SETTING,
        "prior_opened_test_artifact_reused": args.setting_id == NQ_SETTING,
        "prior_confirmation_outcomes_known": args.setting_id == NQ_SETTING,
        "independent_confirmation_claim_allowed": False,
        "closed_test_opened": args.setting_id == NQ_SETTING,
    }
    atomic_json(args.output_dir / "bundle_complete.json", complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting-id",
        "--domain-id",
        dest="setting_id",
        required=True,
    )
    parser.add_argument("--encoder-id", required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--evidence-role", required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--query-vectors", type=Path, required=True)
    parser.add_argument("--qrels-rows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--embeddings-dtype", choices=("float16", "float32"), required=True
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--document-count", type=int, required=True)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--nprobe", type=int)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--fold-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
