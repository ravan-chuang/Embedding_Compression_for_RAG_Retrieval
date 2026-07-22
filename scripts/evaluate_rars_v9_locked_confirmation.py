#!/usr/bin/env python3
"""One-shot evaluator for frozen RARS-v8 within-program confirmation.

All artifact and identity checks, including the qrels-free M48 rebuild, occur
before the durable start marker.  Qrels are parsed only after that marker is
fsynced.  The future role is prospective relative to V8 but is deliberately
not described as independent evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_rars_v6_1m_headroom import (  # noqa: E402
    load_positive_qrels,
    mapping_arrays,
    pad_qrels_for_queries,
    validate_faiss_index,
)
from rars_v6_headroom_core import (  # noqa: E402
    map_qrels_doc_ids_to_corpus_rows,
)
from rars_v8_cutoff_sidecar_core import (  # noqa: E402
    score_sidecar_candidates,
    validate_orthonormal_basis,
)
from rars_v9_confirmation_core import (  # noqa: E402
    PROTOCOL_ID,
    candidate_gap_recovery,
    comparison,
    confirmation_decision,
    per_query_metrics,
    summarize_metric,
)


CANONICAL_PROTOCOL = Path("protocols/rars_v9_locked_confirmation_v1.json")
FUTURE_ROLE = "future_method_holdout"


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


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if int(path.stat().st_size) != int(record.get("bytes", -1)):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 changed")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def newline_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("utf-8")
    ).hexdigest()


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
        raise ValueError("Confirmation requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected confirmation protocol")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_V9_OUTCOME_ACCESS":
        raise ValueError("V9 protocol is not frozen")
    return protocol


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("Refusing to reuse a non-empty confirmation output")
    path.mkdir(parents=True, exist_ok=True)


def validate_future_identity(
    role_dir: Path,
    protocol: dict[str, Any],
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    expected_files = {
        "query_manifest.json",
        "query_vectors.float32.npy",
        "v9_identity_complete.json",
    }
    if {path.name for path in role_dir.iterdir()} != expected_files:
        raise ValueError("Future role is not identity-only before confirmation")
    query_path = role_dir / "query_manifest.json"
    identity_path = role_dir / "v9_identity_complete.json"
    query_vectors_path = role_dir / "query_vectors.float32.npy"
    query = read_json(query_path)
    identity = read_json(identity_path)
    for payload in (query, identity):
        if payload.get("role_id") != FUTURE_ROLE:
            raise ValueError("Unexpected future role identity")
        for flag in ("candidate_arrays_created", "labels_materialized", "metrics_computed"):
            if payload.get(flag) is not False:
                raise ValueError(f"Future role flag is not sealed: {flag}")
    qids = [str(value) for value in query["query_ids"]]
    query_rows = np.asarray(query["query_rows"], dtype=np.int64)
    parent_indices = np.asarray(query["parent_inner_train_indices"], dtype=np.int64)
    registered = protocol["data_policy"]["confirmation_role"]
    if not (
        len(qids)
        == len(query_rows)
        == len(parent_indices)
        == int(registered["query_count"])
    ):
        raise ValueError("Future role query count changed")
    if len(set(qids)) != len(qids) or len(np.unique(query_rows)) != len(query_rows):
        raise ValueError("Future query identities must be unique")
    if newline_sha256(qids) != registered["source_order_newline_qid_sha256"]:
        raise ValueError("Future source-order query identity changed")
    if newline_sha256(sorted(qids, key=int)) != registered[
        "numeric_sorted_newline_qid_sha256"
    ]:
        raise ValueError("Future sorted query identity changed")
    if (
        identity.get("status")
        != "RARS_V9_QRELS_FREE_FUTURE_IDENTITY_COMPLETE"
        or identity.get("qrels_opened") is not False
        or identity.get("qrels_argument_accepted") is not False
        or int(identity.get("query_count", -1)) != len(qids)
    ):
        raise ValueError("Future identity manifest count changed")
    verify_record(
        query_path,
        identity["outputs"]["query_manifest.json"],
        "future query manifest",
    )
    verify_record(
        query_vectors_path,
        identity["outputs"]["query_vectors.float32.npy"],
        "future query vectors",
    )
    if np.any(parent_indices < 0) or len(np.unique(parent_indices)) != len(parent_indices):
        raise ValueError("Future parent indices are invalid")
    queries = np.asarray(np.load(query_vectors_path, mmap_mode="r"), dtype=np.float32)
    dimension = int(protocol["immutable_inputs"]["embedding_dimension"])
    if queries.shape != (len(qids), dimension):
        raise ValueError("Future query vectors have an unexpected shape")
    if not np.all(np.isfinite(queries)) or not np.allclose(
        np.linalg.norm(queries, axis=1), 1.0, rtol=0.0, atol=0.005
    ):
        raise ValueError("Future query vectors are not finite normalized vectors")
    return qids, queries, {
        "query_manifest": file_record(query_path),
        "identity_manifest": file_record(identity_path),
        "query_vectors": file_record(query_vectors_path),
    }


def validate_frozen_v8_packet(
    development_packet: Path,
    sidecar_root: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    lineage = protocol["frozen_v8_lineage"]
    freeze_path = development_packet / "method_freeze.json"
    result_path = development_packet / "development_result.json"
    if sha256_file(freeze_path) != lineage["method_freeze_sha256"]:
        raise ValueError("Frozen V8 method record changed")
    if sha256_file(result_path) != lineage["development_result_sha256"]:
        raise ValueError("Frozen V8 development result changed")
    freeze = read_json(freeze_path)
    if (
        freeze.get("source_commit") != lineage["source_commit"]
        or freeze.get("formal_decision") != lineage["formal_development_decision"]
    ):
        raise ValueError("Frozen V8 lineage changed")
    loaded: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {
        "method_freeze": file_record(freeze_path),
        "development_result": file_record(result_path),
    }
    for name in ("pca", "rars"):
        directory = sidecar_root / name
        manifest_path = directory / "manifest.json"
        expected_manifest = lineage[f"{name}_manifest_sha256"]
        if sha256_file(manifest_path) != expected_manifest:
            raise ValueError(f"Frozen {name} manifest changed")
        manifest = read_json(manifest_path)
        for filename, record in manifest["files"].items():
            verify_record(directory / filename, record, f"{name} {filename}")
        codes_path = directory / "codes.int8.npy"
        if sha256_file(codes_path) != lineage[f"{name}_codes_sha256"]:
            raise ValueError(f"Frozen {name} codes changed")
        config = read_json(directory / "sidecar_config.json")
        method = protocol["frozen_methods"][
            "pca_rank16_int8" if name == "pca" else "rars_v8_rank16_int8"
        ]
        if (
            int(config["rank"]) != int(method["rank"])
            or float(config["alpha"]) != float(method["alpha"])
            or int(config["top_b"]) != int(method["top_b"])
            or config["coefficient_dtype"] != method["coefficient_dtype"]
        ):
            raise ValueError(f"Frozen {name} sidecar configuration changed")
        dimension = int(protocol["immutable_inputs"]["embedding_dimension"])
        rank = int(method["rank"])
        basis = validate_orthonormal_basis(
            np.load(directory / "basis.float32.npy", allow_pickle=False),
            dimension=dimension,
            rank=rank,
        )
        scales = np.load(directory / "scales.float32.npy", allow_pickle=False)
        codes = np.load(codes_path, mmap_mode="r")
        if scales.shape != (rank,) or codes.shape != (
            int(protocol["immutable_inputs"]["document_count"]),
            rank,
        ):
            raise ValueError(f"Frozen {name} sidecar array shape changed")
        loaded[name] = {
            "basis": basis,
            "scales": np.asarray(scales, dtype=np.float32),
            "codes": codes,
            "alpha": float(method["alpha"]),
            "top_b": int(method["top_b"]),
        }
        records[f"{name}_manifest"] = file_record(manifest_path)
        records[f"{name}_codes"] = file_record(codes_path)
    return loaded, records


def validate_m48_packet(
    index_path: Path, complete_path: Path, protocol: dict[str, Any], faiss: Any
) -> tuple[Any, dict[str, Any]]:
    complete = read_json(complete_path)
    if (
        complete.get("protocol_id") != PROTOCOL_ID
        or complete.get("status") != "RARS_V9_QRELS_FREE_M48_BASELINE_COMPLETE"
        or complete.get("qrels_opened") is not False
        or complete.get("outcome_metric_computed") is not False
    ):
        raise ValueError("M48 packet is not a sealed qrels-free build")
    verify_record(index_path, complete["index"], "M48 index")
    index = faiss.read_index(str(index_path))
    try:
        ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    except Exception as error:
        raise ValueError("M48 baseline is not an IVF index") from error
    pq = getattr(ivf, "pq", None)
    registered = protocol["locked_limitation_baselines"][
        "m48_rebuild_nlist512_nprobe16"
    ]
    observed = {
        "dimension": int(getattr(ivf, "d", -1)),
        "ntotal": int(getattr(ivf, "ntotal", -1)),
        "nlist": int(getattr(ivf, "nlist", -1)),
        "subquantizers": int(getattr(pq, "M", -1)),
        "bits_per_subquantizer": int(getattr(pq, "nbits", -1)),
        "metric_type": int(getattr(ivf, "metric_type", -1)),
        "nprobe": int(registered["nprobe"]),
    }
    expected = {
        "dimension": int(protocol["immutable_inputs"]["embedding_dimension"]),
        "ntotal": int(protocol["immutable_inputs"]["document_count"]),
        "nlist": int(registered["nlist"]),
        "subquantizers": int(registered["subquantizers"]),
        "bits_per_subquantizer": int(registered["bits_per_subquantizer"]),
        "metric_type": int(faiss.METRIC_INNER_PRODUCT),
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            raise ValueError(f"M48 {key}={observed[key]}; expected {expected_value}")
    if not bool(getattr(ivf, "is_trained", False)):
        raise ValueError("M48 baseline is not trained")
    ivf.nprobe = int(registered["nprobe"])
    return index, {"complete": file_record(complete_path), "index": file_record(index_path), "contract": observed}


def search(
    index: Any, queries: np.ndarray, *, nprobe: int, k: int, faiss: Any
) -> tuple[np.ndarray, np.ndarray, float]:
    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    ivf.nprobe = int(nprobe)
    started = time.perf_counter()
    scores, rows = index.search(np.asarray(queries, dtype=np.float32), int(k))
    elapsed = time.perf_counter() - started
    scores = np.asarray(scores, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64)
    if (
        scores.shape != (len(queries), k)
        or rows.shape != scores.shape
        or np.any(rows < 0)
        or np.any(~np.isfinite(scores))
    ):
        raise ValueError("Faiss search returned invalid candidates")
    if any(len(np.unique(row)) != k for row in rows):
        raise ValueError("Faiss search returned duplicate candidates")
    return scores, rows, float(elapsed)


def exact_candidate_scores(
    queries: np.ndarray,
    candidate_rows: np.ndarray,
    embeddings: np.memmap,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    output = np.empty(candidate_rows.shape, dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        documents = np.asarray(embeddings[candidate_rows[start:end]], dtype=np.float32)
        output[start:end] = np.einsum(
            "qd,qkd->qk", queries[start:end], documents, optimize=True
        )
    if np.any(~np.isfinite(output)):
        raise ValueError("Exact candidate scores contain non-finite values")
    return output


def metric_bundle(
    scores: np.ndarray,
    rows: np.ndarray,
    positive_rows: np.ndarray,
    positive_valid: np.ndarray,
    doc_ids: np.memmap,
) -> dict[str, np.ndarray]:
    tie_ids = np.asarray(doc_ids[rows], dtype=np.int64)
    return per_query_metrics(
        scores,
        rows,
        positive_rows,
        positive_valid,
        k=10,
        tie_identifiers=tie_ids,
    )


def metric_means(bundle: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: summarize_metric(values) for name, values in bundle.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    wall_started = time.perf_counter()
    protocol = validate_source(args.protocol, args.source_commit)
    prepare_output(args.output_dir)
    qids, queries, future_records = validate_future_identity(
        args.future_role_dir, protocol
    )
    sidecars, sidecar_records = validate_frozen_v8_packet(
        args.v8_development_packet, args.sidecar_root, protocol
    )
    immutable = protocol["immutable_inputs"]
    embedding_record = file_record(args.embeddings)
    doc_id_record = file_record(args.doc_ids)
    m32_record = file_record(args.m32_index)
    for observed, bytes_key, hash_key, label in (
        (embedding_record, "embeddings_bytes", "embeddings_sha256", "embeddings"),
        (doc_id_record, "doc_ids_bytes", "doc_ids_sha256", "document IDs"),
        (m32_record, "m32_index_bytes", "m32_index_sha256", "M32 index"),
    ):
        if observed["bytes"] != int(immutable[bytes_key]) or observed["sha256"] != immutable[hash_key]:
            raise ValueError(f"Frozen {label} differs from the V9 contract")
    document_count = int(immutable["document_count"])
    dimension = int(immutable["embedding_dimension"])
    doc_ids = np.memmap(args.doc_ids, dtype=np.int64, mode="r", shape=(document_count,))
    if len(np.unique(np.asarray(doc_ids))) != document_count:
        raise ValueError("Frozen document IDs are not unique")
    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(document_count, dimension),
    )
    m32 = faiss.read_index(str(args.m32_index))
    m32_ivf, m32_contract = validate_faiss_index(m32, faiss)
    m32_ivf.nprobe = 16
    m48, m48_records = validate_m48_packet(
        args.m48_index, args.m48_complete, protocol, faiss
    )
    input_freeze = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V9_INPUTS_FROZEN_BEFORE_OUTCOME_ACCESS",
        "source_commit": args.source_commit,
        "protocol": file_record(args.protocol),
        "future_identity": future_records,
        "frozen_v8": sidecar_records,
        "embeddings": embedding_record,
        "doc_ids": doc_id_record,
        "m32_index": m32_record,
        "m32_contract": m32_contract,
        "m48": m48_records,
        "outcome_opened": False,
        "qrels_opened": False,
        "confirmation_role": FUTURE_ROLE,
        "evidence_tier": protocol["claim_boundary"]["evidence_tier"],
    }
    input_freeze_path = args.output_dir / "input_freeze.json"
    atomic_json(input_freeze_path, input_freeze)
    started_payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V9_CONFIRMATION_STARTED",
        "source_commit": args.source_commit,
        "input_freeze": file_record(input_freeze_path),
        "outcome_opened": False,
        "qrels_opened": False,
        "method_or_threshold_tuning_authorized": False,
    }
    started_path = args.output_dir / "confirmation_started.json"
    atomic_json(started_path, started_payload)

    # FIRST OUTCOME ACCESS.  Every method, threshold, artifact, query identity,
    # and qrels-free limitation baseline is durable above this line.
    qrels = load_positive_qrels(args.qrels)
    qrel_doc_ids, qrel_valid = pad_qrels_for_queries(qids, qrels)
    mapping = map_qrels_doc_ids_to_corpus_rows(doc_ids, qrel_doc_ids, qrel_valid)
    positive_rows, positive_valid, qrels_coverage = mapping_arrays(mapping)

    base_scores, base_rows, base_seconds = search(
        m32, queries, nprobe=16, k=100, faiss=faiss
    )
    nprobe32_scores, nprobe32_rows, nprobe32_seconds = search(
        m32, queries, nprobe=32, k=100, faiss=faiss
    )
    nprobe64_scores, nprobe64_rows, nprobe64_seconds = search(
        m32, queries, nprobe=64, k=100, faiss=faiss
    )
    m48_scores, m48_rows, m48_seconds = search(
        m48, queries, nprobe=16, k=100, faiss=faiss
    )
    exact_scores = exact_candidate_scores(queries, base_rows, embeddings)
    corrected: dict[str, np.ndarray] = {}
    correction_seconds: dict[str, float] = {}
    for name, artifact in sidecars.items():
        correction_started = time.perf_counter()
        corrected[name] = score_sidecar_candidates(
            queries,
            base_rows,
            base_rows,
            base_scores,
            artifact["basis"],
            artifact["codes"],
            artifact["scales"],
            alpha=artifact["alpha"],
            top_b=artifact["top_b"],
        )
        correction_seconds[name] = float(time.perf_counter() - correction_started)

    method_scores = {
        "base_m32_nprobe16": (base_scores, base_rows),
        "pca_rank16_int8": (corrected["pca"], base_rows),
        "rars_v8_rank16_int8": (corrected["rars"], base_rows),
        "same_candidate_exact": (exact_scores, base_rows),
        "m32_nprobe32": (nprobe32_scores, nprobe32_rows),
        "m32_nprobe64": (nprobe64_scores, nprobe64_rows),
        "m48_rebuild_nlist512_nprobe16": (m48_scores, m48_rows),
    }
    bundles = {
        name: metric_bundle(scores, rows, positive_rows, positive_valid, doc_ids)
        for name, (scores, rows) in method_scores.items()
    }
    bootstrap = int(protocol["metrics"]["bootstrap"]["paired_query_replicates"])
    randomization = int(
        protocol["metrics"]["primary_randomization_test"]["paired_sign_replicates"]
    )
    inference_seed = int(protocol["metrics"]["bootstrap"]["seed"])
    recall = {name: bundle["recall"] for name, bundle in bundles.items()}
    comparisons = {
        "rars_vs_base_recall_at_10": comparison(
            recall["rars_v8_rank16_int8"],
            recall["base_m32_nprobe16"],
            bootstrap_replicates=bootstrap,
            randomization_replicates=randomization,
            seed=inference_seed,
        ),
        "pca_vs_base_recall_at_10": comparison(
            recall["pca_rank16_int8"],
            recall["base_m32_nprobe16"],
            bootstrap_replicates=bootstrap,
            randomization_replicates=randomization,
            seed=inference_seed + 10,
        ),
        "rars_vs_pca_recall_at_10": comparison(
            recall["rars_v8_rank16_int8"],
            recall["pca_rank16_int8"],
            bootstrap_replicates=bootstrap,
            randomization_replicates=randomization,
            seed=inference_seed + 20,
        ),
    }
    gap_recovery = candidate_gap_recovery(
        recall["rars_v8_rank16_int8"],
        recall["base_m32_nprobe16"],
        recall["same_candidate_exact"],
    )
    decision = confirmation_decision(
        rars_vs_base=comparisons["rars_vs_base_recall_at_10"],
        pca_vs_base=comparisons["pca_vs_base_recall_at_10"],
        rars_vs_pca=comparisons["rars_vs_pca_recall_at_10"],
        gap_recovery=gap_recovery,
        thresholds=protocol["confirmation_gate"],
    )
    arrays: dict[str, np.ndarray] = {
        "query_ids": np.asarray(qids),
        "positive_rows": positive_rows,
        "positive_valid": positive_valid,
    }
    for method_name, bundle in bundles.items():
        for metric_name, values in bundle.items():
            arrays[f"{method_name}__{metric_name}_at_10"] = values
    per_query_path = args.output_dir / "per_query_metrics.npz"
    atomic_npz(per_query_path, arrays)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V9_LOCKED_CONFIRMATION_COMPLETE",
        "source_commit": args.source_commit,
        "evidence_tier": protocol["claim_boundary"]["evidence_tier"],
        "independent_confirmation_claim_allowed": False,
        "query_count": len(qids),
        "qrels_mapping": qrels_coverage,
        "metrics": {name: metric_means(bundle) for name, bundle in bundles.items()},
        "comparisons": comparisons,
        "candidate_gap_recovery_fraction": gap_recovery,
        "decision": decision,
        "timing": {
            "hardware_specific": True,
            "python_prototype_not_fused_kernel": True,
            "query_count": len(qids),
            "base_m32_nprobe16_seconds": base_seconds,
            "m32_nprobe32_seconds": nprobe32_seconds,
            "m32_nprobe64_seconds": nprobe64_seconds,
            "m48_nprobe16_seconds": m48_seconds,
            "pca_correction_seconds": correction_seconds["pca"],
            "rars_correction_seconds": correction_seconds["rars"],
        },
        "inputs": {
            "confirmation_started": file_record(started_path),
            "input_freeze": file_record(input_freeze_path),
            "qrels_after_outcome_open": file_record(args.qrels),
        },
        "per_query_metrics": file_record(per_query_path),
        "opened_roles": [FUTURE_ROLE],
        "forbidden_roles_opened": [],
        "method_or_threshold_tuning_authorized": False,
        "telemetry": {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        },
    }
    result_path = args.output_dir / "confirmation_result.json"
    atomic_json(result_path, result)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V9_LOCKED_CONFIRMATION_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "evidence_tier": protocol["claim_boundary"]["evidence_tier"],
        "independent_confirmation_claim_allowed": False,
        "confirmation_started": file_record(started_path),
        "input_freeze": file_record(input_freeze_path),
        "per_query_metrics": file_record(per_query_path),
        "confirmation_result": file_record(result_path),
        "future_method_holdout_opened_once": True,
        "method_or_threshold_tuning_authorized": False,
    }
    atomic_json(args.output_dir / "confirmation_complete.json", complete)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--future-role-dir", required=True, type=Path)
    parser.add_argument("--v8-development-packet", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--doc-ids", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--m32-index", required=True, type=Path)
    parser.add_argument("--m48-index", required=True, type=Path)
    parser.add_argument("--m48-complete", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
