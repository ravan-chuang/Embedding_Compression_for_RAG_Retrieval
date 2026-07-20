#!/usr/bin/env python3
"""Measure 1M IVF routing and PQ-specific recall headroom without training.

This evaluator is deliberately diagnostic-only.  It accepts only the already
observed v3 ``oracle_design`` query role, keeps the frozen IVF-PQ index
unchanged, and never constructs an adapter or a RARS sidecar.  Heavy Faiss and
Torch imports are local so the validation and reporting helpers remain
unit-testable in a CPU-only environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rars_v6_headroom_core import (  # noqa: E402
    GATE_THRESHOLDS,
    decompose_recall_gaps,
    diagnostic_gate_decision,
    known_positive_recall_at_k,
    map_qrels_doc_ids_to_corpus_rows,
    mine_pq_induced_flip_triplets,
)


PROTOCOL_ID = "rars_v6_1m_headroom_v1"
DESIGN_ROLE_ID = "oracle_design"
CANONICAL_PROTOCOL = Path("protocols/rars_v6_1m_headroom_v1.json")
CANONICAL_SOURCES = (
    Path("scripts/rars_v6_headroom_core.py"),
    Path("scripts/evaluate_rars_v6_1m_headroom.py"),
)
EXPECTED_PROTOCOL_STATUSES = {"FROZEN_BEFORE_FIRST_1M_HEADROOM_RUN"}
EXPECTED_INDEX = {
    "dimension": 384,
    "nlist": 512,
    "nprobe": 16,
    "subquantizers": 32,
    "nbits": 8,
    "ntotal": 1_000_000,
}
TOP_K = 200


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("--source-commit must be an exact lowercase 40-hex commit")


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo_root, stderr=subprocess.STDOUT
    )


def validate_clean_pinned_repository(repo_root: Path, source_commit: str) -> None:
    _validate_exact_commit(source_commit)
    head = _git_bytes(repo_root, "rev-parse", "HEAD").decode().strip()
    if head != source_commit:
        raise ValueError(f"Git HEAD {head} does not match {source_commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("RARS-v6 headroom evaluation requires a clean Git worktree")


def validate_protocol_and_source_blobs(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the canonical protocol and byte-identical committed sources."""

    canonical = (repo_root / CANONICAL_PROTOCOL).resolve()
    if protocol_path.resolve() != canonical:
        raise ValueError(f"--protocol must be the canonical path {canonical}")
    validate_clean_pinned_repository(repo_root, source_commit)
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected RARS-v6 protocol ID")
    if protocol.get("status") not in EXPECTED_PROTOCOL_STATUSES:
        raise ValueError("RARS-v6 protocol is not frozen before the diagnostic")
    if "method_revision_allowed" in protocol and protocol["method_revision_allowed"] is not False:
        raise ValueError("RARS-v6 protocol permits method revision")
    if (
        "outcome_informed_revision_allowed" in protocol
        and protocol["outcome_informed_revision_allowed"] is not False
    ):
        raise ValueError("RARS-v6 protocol permits outcome-informed revision")

    records: dict[str, Any] = {}
    for relative in (CANONICAL_PROTOCOL, *CANONICAL_SOURCES):
        local_path = repo_root / relative
        if not local_path.is_file():
            raise ValueError(f"Missing canonical source: {relative}")
        try:
            committed = _git_bytes(
                repo_root, "show", f"{source_commit}:{relative.as_posix()}"
            )
            blob_oid = _git_bytes(
                repo_root, "rev-parse", f"{source_commit}:{relative.as_posix()}"
            ).decode().strip()
        except subprocess.CalledProcessError as error:
            raise ValueError(f"{relative} is absent from {source_commit}") from error
        local = local_path.read_bytes()
        if local != committed:
            raise ValueError(f"Canonical source differs from Git blob: {relative}")
        records[relative.as_posix()] = {
            "sha256": hashlib.sha256(local).hexdigest(),
            "git_blob_oid": blob_oid,
        }
    return protocol, records


def _nested(payload: dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _registered_sha256(protocol: dict[str, Any], label: str) -> str:
    """Read one required frozen-input hash from accepted protocol spellings."""

    aliases = {
        "index": (
            ("input_contract", "frozen_index", "sha256"),
            ("frozen_inputs", "index_sha256"),
            ("frozen_inputs", "frozen_index_sha256"),
            ("frozen_index", "sha256"),
            ("parent_lineage", "frozen_index_sha256"),
        ),
        "doc_ids": (
            ("input_contract", "doc_ids", "sha256"),
            ("frozen_inputs", "doc_ids_sha256"),
            ("parent_lineage", "frozen_doc_ids_sha256"),
            ("corpus", "doc_ids_sha256"),
        ),
    }
    for path in aliases[label]:
        value = _nested(protocol, path)
        if isinstance(value, str) and len(value) == 64:
            return value
    raise ValueError(f"Protocol lacks a registered {label} SHA-256")


def verify_frozen_inputs(
    protocol: dict[str, Any], *, embeddings: Path, doc_ids: Path, qrels: Path, index: Path
) -> dict[str, Any]:
    paths = {
        "embeddings": Path(embeddings),
        "doc_ids": Path(doc_ids),
        "qrels": Path(qrels),
        "index": Path(index),
    }
    records: dict[str, Any] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Missing frozen {label} input: {path}")
        record = file_record(path)
        if label in {"doc_ids", "index"} and record["sha256"] != _registered_sha256(protocol, label):
            raise ValueError(f"Frozen {label} hash changed")
        records[label] = record
    contract = protocol["input_contract"]
    expected_embedding_bytes = int(contract["document_count"]) * int(
        contract["embedding_dimension"]
    ) * np.dtype(np.float16).itemsize
    expected_doc_id_bytes = int(contract["doc_ids"]["bytes"])
    if records["embeddings"]["bytes"] != expected_embedding_bytes:
        raise ValueError("Embedding memmap is not exactly 1M x 384 float16")
    if records["doc_ids"]["bytes"] != expected_doc_id_bytes:
        raise ValueError("Document-ID memmap is not exactly 1M int64 values")
    if records["index"]["bytes"] != int(contract["frozen_index"]["bytes"]):
        raise ValueError("Frozen index byte count changed")
    return records


def _newline_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def load_design_role(
    role_dir: Path, protocol: dict[str, Any]
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    role_dir = Path(role_dir)
    if role_dir.name != DESIGN_ROLE_ID:
        raise ValueError("Only the v3 oracle_design role is allowed")
    query_manifest_path = role_dir / "query_manifest.json"
    query_vectors_path = role_dir / "query_vectors.float32.npy"
    candidate_manifest_path = role_dir / "v3_candidate_manifest.json"
    if not all(
        path.is_file()
        for path in (query_manifest_path, query_vectors_path, candidate_manifest_path)
    ):
        raise ValueError(
            "oracle_design requires its query manifest, vectors, and v3 candidate manifest"
        )
    candidate_manifest = read_json(candidate_manifest_path)
    if candidate_manifest.get("role_id") != DESIGN_ROLE_ID:
        raise ValueError("V3 candidate manifest is not oracle_design")
    if candidate_manifest.get("source_commit") != protocol["parent_lineage"][
        "v3_implementation_commit"
    ]:
        raise ValueError("V3 oracle_design source commit changed")
    if (
        candidate_manifest.get("data_access", {}).get("qrels_opened_or_parsed")
        is not False
    ):
        raise ValueError("V3 oracle_design candidate freeze read qrels")
    registered_query = candidate_manifest.get("query_manifest")
    registered_vectors = candidate_manifest.get("files", {}).get(
        "query_vectors.float32.npy"
    )
    for label, path, record in (
        ("query manifest", query_manifest_path, registered_query),
        ("query vectors", query_vectors_path, registered_vectors),
    ):
        if not isinstance(record, dict):
            raise ValueError(f"V3 candidate manifest lacks {label} registration")
        observed = file_record(path)
        if observed["bytes"] != int(record.get("bytes", -1)):
            raise ValueError(f"Registered {label} byte count changed")
        if observed["sha256"] != record.get("sha256"):
            raise ValueError(f"Registered {label} hash changed")
    manifest = read_json(query_manifest_path)
    if manifest.get("role_id") != DESIGN_ROLE_ID:
        raise ValueError("Query manifest is not the oracle_design role")
    qids = [str(value) for value in manifest.get("query_ids", [])]
    if not qids or len(qids) != len(set(qids)):
        raise ValueError("Design query IDs are empty or non-unique")
    queries = np.load(query_vectors_path, mmap_mode="r")
    if queries.dtype != np.float32 or queries.shape != (
        len(qids),
        EXPECTED_INDEX["dimension"],
    ):
        raise ValueError("Design query vectors have an unexpected shape or dtype")
    if not np.all(np.isfinite(np.asarray(queries))):
        raise ValueError("Design query vectors contain non-finite values")
    query_norms = np.linalg.norm(np.asarray(queries), axis=1)
    if not np.allclose(query_norms, 1.0, rtol=0.0, atol=0.005):
        raise ValueError("Design query vectors are not L2 normalized")
    role_contract = protocol["data_policy"]["diagnostic_role"]
    registered_count = role_contract["source_query_count"]
    if registered_count is not None and int(registered_count) != len(qids):
        raise ValueError("Design query count differs from the protocol")
    if role_contract["source_order_newline_qid_sha256"] != _newline_sha256(qids):
        raise ValueError("Design source-order query identity hash changed")
    sorted_qids = sorted(qids, key=lambda value: int(value))
    if role_contract["numeric_sorted_newline_qid_sha256"] != _newline_sha256(sorted_qids):
        raise ValueError("Design numeric-sorted query identity hash changed")
    records = {
        "v3_candidate_manifest": file_record(candidate_manifest_path),
        "query_manifest": file_record(query_manifest_path),
        "query_vectors": file_record(query_vectors_path),
    }
    return qids, queries, records


def load_positive_qrels(path: Path) -> dict[str, set[int]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("qrels_subset.json must be a JSON object")
    output: dict[str, set[int]] = {}
    for raw_qid, raw_values in payload.items():
        positives: set[int] = set()
        if isinstance(raw_values, dict):
            positives = {
                int(docid) for docid, relevance in raw_values.items() if float(relevance) > 0
            }
        elif isinstance(raw_values, list):
            for item in raw_values:
                if isinstance(item, dict):
                    docid = item.get("doc_id", item.get("docid", item.get("pid")))
                    relevance = item.get("relevance", item.get("score", item.get("rel", 1)))
                    if docid is not None and float(relevance) > 0:
                        positives.add(int(docid))
                else:
                    positives.add(int(item))
        else:
            raise ValueError(f"Unsupported qrels entry for {raw_qid}")
        if positives:
            output[str(raw_qid)] = positives
    if not output:
        raise ValueError("No positive qrels were parsed")
    return output


def pad_qrels_for_queries(
    qids: list[str], qrels: dict[str, set[int]]
) -> tuple[np.ndarray, np.ndarray]:
    missing = [qid for qid in qids if not qrels.get(qid)]
    if missing:
        raise ValueError(f"Design queries lack positive qrels: {missing[:5]}")
    width = max(len(qrels[qid]) for qid in qids)
    values = np.full((len(qids), width), -1, dtype=np.int64)
    valid = np.zeros((len(qids), width), dtype=bool)
    for query_index, qid in enumerate(qids):
        selected = np.asarray(sorted(qrels[qid]), dtype=np.int64)
        values[query_index, : len(selected)] = selected
        valid[query_index, : len(selected)] = True
    return values, valid


def mapping_arrays(mapping: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Adapt the numeric core's frozen mapping object to evaluator arrays."""

    row_names = ("rows", "positive_rows", "known_positive_rows")
    valid_names = ("qrels_valid", "positive_valid", "valid", "known_positive_valid")
    rows = next((getattr(mapping, name) for name in row_names if hasattr(mapping, name)), None)
    valid = next((getattr(mapping, name) for name in valid_names if hasattr(mapping, name)), None)
    if rows is None or valid is None:
        raise TypeError("QrelsRowMapping lacks padded positive rows and validity")
    rows_array = np.asarray(rows, dtype=np.int64)
    valid_array = np.asarray(valid, dtype=bool)
    if rows_array.shape != valid_array.shape or rows_array.ndim != 2:
        raise ValueError("Mapped qrels rows and valid mask must be matching matrices")
    summary = dict(getattr(mapping, "coverage", {}))
    coverage = summary.get("qrels_corpus_coverage")
    if coverage is None:
        raise TypeError("QrelsRowMapping lacks qrels corpus coverage")
    if float(coverage) != 1.0:
        raise ValueError("Design qrels do not have corpus coverage 1.0")
    return rows_array, valid_array, summary


def validate_faiss_index(index: Any, faiss_module: Any) -> tuple[Any, dict[str, Any]]:
    """Validate the immutable 1M M32 IVF-PQ index before any search."""

    try:
        ivf = faiss_module.extract_index_ivf(index)
    except Exception as error:  # pragma: no cover - depends on Faiss wrapper
        raise ValueError("Frozen index is not an IVF index") from error
    pq = getattr(ivf, "pq", None)
    observed = {
        "dimension": int(getattr(ivf, "d", -1)),
        "nlist": int(getattr(ivf, "nlist", -1)),
        "serialized_nprobe": int(getattr(ivf, "nprobe", -1)),
        "subquantizers": int(getattr(pq, "M", -1)),
        "nbits": int(getattr(pq, "nbits", -1)),
        "ntotal": int(getattr(ivf, "ntotal", -1)),
        "metric_type": int(getattr(ivf, "metric_type", -1)),
    }
    for key, expected in EXPECTED_INDEX.items():
        if key == "nprobe":
            continue
        if observed[key] != expected:
            raise ValueError(f"Frozen index {key}={observed[key]}; expected {expected}")
    expected_metric = int(faiss_module.METRIC_INNER_PRODUCT)
    if observed["metric_type"] != expected_metric:
        raise ValueError("Frozen index metric is not inner product")
    if not bool(getattr(ivf, "is_trained", False)):
        raise ValueError("Frozen IVF-PQ index is not trained")
    observed["metric"] = "inner_product"
    observed["runtime_nprobe"] = EXPECTED_INDEX["nprobe"]
    return ivf, observed


def inverted_lists_as_rows(ivf: Any, faiss_module: Any) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Copy row IDs from each Faiss inverted list and validate a 1:1 cover."""

    lists: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    for list_id in range(int(ivf.nlist)):
        size = int(ivf.invlists.list_size(list_id))
        if size:
            values = np.asarray(
                faiss_module.rev_swig_ptr(ivf.invlists.get_ids(list_id), size),
                dtype=np.int64,
            ).copy()
        else:
            values = np.empty(0, dtype=np.int64)
        if np.any(values < 0) or np.any(values >= int(ivf.ntotal)):
            raise ValueError(f"Inverted list {list_id} contains an invalid row")
        lists.append(values)
        all_rows.append(values)
    concatenated = np.concatenate(all_rows) if all_rows else np.empty(0, np.int64)
    if len(concatenated) != int(ivf.ntotal) or len(np.unique(concatenated)) != int(ivf.ntotal):
        raise ValueError("Inverted lists do not form a one-to-one corpus-row cover")
    if not np.array_equal(np.sort(concatenated), np.arange(int(ivf.ntotal), dtype=np.int64)):
        raise ValueError("Inverted-list row cover differs from [0, ntotal)")
    sizes = np.asarray([len(value) for value in lists], dtype=np.int64)
    return lists, {
        "list_count": len(lists),
        "row_count": int(sizes.sum()),
        "minimum_list_size": int(sizes.min()),
        "maximum_list_size": int(sizes.max()),
        "mean_list_size": float(sizes.mean()),
    }


def probed_candidate_rows(
    queries: np.ndarray, ivf: Any, inverted_lists: list[np.ndarray], *, nprobe: int
) -> tuple[list[np.ndarray], np.ndarray]:
    _, list_ids = ivf.quantizer.search(np.asarray(queries, dtype=np.float32), nprobe)
    list_ids = np.asarray(list_ids, dtype=np.int64)
    if list_ids.shape != (len(queries), nprobe) or np.any(list_ids < 0):
        raise ValueError("IVF coarse quantizer returned invalid probed lists")
    candidates: list[np.ndarray] = []
    for row in list_ids:
        values = np.concatenate([inverted_lists[int(list_id)] for list_id in row])
        if len(values) != len(np.unique(values)):
            raise ValueError("A query's probed IVF lists contain duplicate corpus rows")
        candidates.append(values.astype(np.int64, copy=False))
    return candidates, list_ids


def _merge_torch_topk(
    best_scores: Any, best_rows: Any, scores: Any, rows: Any, *, k: int, torch: Any
) -> tuple[Any, Any]:
    take = min(k, int(scores.shape[1]))
    block_scores, positions = torch.topk(scores, k=take, dim=1, largest=True, sorted=True)
    block_rows = rows[positions]
    if best_scores is None:
        return block_scores, block_rows
    combined_scores = torch.cat((best_scores, block_scores), dim=1)
    combined_rows = torch.cat((best_rows, block_rows), dim=1)
    keep = min(k, int(combined_scores.shape[1]))
    merged_scores, selected = torch.topk(
        combined_scores, k=keep, dim=1, largest=True, sorted=True
    )
    return merged_scores, torch.gather(combined_rows, 1, selected)


def load_corpus_tensor_torch(
    embeddings: np.memmap,
    *,
    device: str,
    load_batch_size: int,
    norm_tolerance: float = 0.005,
) -> Any:
    """Load the corpus to a temporary GPU tensor using bounded host batches."""

    import torch

    corpus = torch.empty(embeddings.shape, dtype=torch.float32, device=device)
    for start in range(0, len(embeddings), load_batch_size):
        end = min(start + load_batch_size, len(embeddings))
        host = np.asarray(embeddings[start:end], dtype=np.float32)
        if not np.all(np.isfinite(host)):
            raise ValueError("Corpus embeddings contain non-finite values")
        norms = np.linalg.norm(host, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=norm_tolerance):
            raise ValueError("Corpus embeddings are not L2 normalized")
        corpus[start:end].copy_(torch.from_numpy(host), non_blocking=False)
    return corpus


def full_exact_topk_torch(
    queries: np.ndarray,
    corpus_tensor: Any,
    *,
    k: int,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU full-exact inner-product search over a temporary corpus tensor."""

    import torch

    output_scores = np.empty((len(queries), k), dtype=np.float32)
    output_rows = np.empty((len(queries), k), dtype=np.int64)
    for query_start in range(0, len(queries), query_batch_size):
        query_end = min(query_start + query_batch_size, len(queries))
        query_tensor = torch.as_tensor(
            np.asarray(queries[query_start:query_end], dtype=np.float32),
            device=corpus_tensor.device,
        )
        scores = query_tensor @ corpus_tensor.T
        best_scores, best_rows = torch.topk(
            scores, k=k, dim=1, largest=True, sorted=True
        )
        output_scores[query_start:query_end] = best_scores.cpu().numpy().astype(np.float32)
        output_rows[query_start:query_end] = best_rows.cpu().numpy().astype(np.int64)
        del query_tensor, scores, best_scores, best_rows
    return output_scores, output_rows


def within_ivf_exact_topk_torch(
    queries: np.ndarray,
    corpus_tensor: Any,
    candidates: list[np.ndarray],
    *,
    k: int,
    candidate_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-score each query only on documents in its probed IVF lists."""

    import torch

    scores_out = np.full((len(queries), k), -np.inf, dtype=np.float32)
    rows_out = np.full((len(queries), k), -1, dtype=np.int64)
    for query_index, rows in enumerate(candidates):
        query_tensor = torch.as_tensor(
            np.asarray(queries[query_index], dtype=np.float32),
            device=corpus_tensor.device,
        ).reshape(1, -1)
        best_scores = best_rows = None
        for start in range(0, len(rows), candidate_batch_size):
            selected = np.asarray(rows[start : start + candidate_batch_size], dtype=np.int64)
            selected_tensor = torch.as_tensor(
                selected, dtype=torch.int64, device=corpus_tensor.device
            )
            candidate_tensor = corpus_tensor[selected_tensor]
            score = query_tensor @ candidate_tensor.T
            row_tensor = selected_tensor[None, :]
            best_scores, best_rows = _merge_torch_topk(
                best_scores, best_rows, score, row_tensor, k=k, torch=torch
            )
            del selected_tensor, candidate_tensor, score, row_tensor
        if best_scores is None:
            raise ValueError(f"Query {query_index} has no same-IVF candidates")
        take = int(best_scores.shape[1])
        scores_out[query_index, :take] = best_scores.cpu().numpy()[0]
        rows_out[query_index, :take] = best_rows.cpu().numpy()[0]
        del query_tensor, best_scores, best_rows
    return scores_out, rows_out


def build_flip_candidate_union(
    base_rows: np.ndarray,
    ivf_exact_rows: np.ndarray,
    positive_rows: np.ndarray,
    positive_valid: np.ndarray,
    probed_candidates: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a deterministic scoreable union for PQ-specific flip mining.

    Known positives outside the probed IVF lists are routing misses.  They stay
    in every recall denominator but are deliberately excluded from the flip
    miner, because no within-list score correction can retrieve them.
    """

    base = np.asarray(base_rows, dtype=np.int64)
    exact = np.asarray(ivf_exact_rows, dtype=np.int64)
    positives = np.asarray(positive_rows, dtype=np.int64)
    valid = np.asarray(positive_valid, dtype=bool)
    if base.ndim != 2 or exact.ndim != 2 or positives.ndim != 2:
        raise ValueError("Candidate and positive rows must be matrices")
    if len(base) != len(exact) or len(base) != len(positives):
        raise ValueError("Candidate and positive query counts must match")
    if positives.shape != valid.shape or len(probed_candidates) != len(base):
        raise ValueError("Positive masks and probed candidates must match queries")

    eligible = np.zeros_like(valid, dtype=bool)
    unions: list[np.ndarray] = []
    per_query_routed_positives: list[int] = []
    for query_index in range(len(base)):
        probed = np.asarray(probed_candidates[query_index], dtype=np.int64)
        selected_positives = positives[query_index, valid[query_index]]
        routed = np.isin(selected_positives, probed, assume_unique=False)
        eligible_positions = np.flatnonzero(valid[query_index])
        eligible[query_index, eligible_positions[routed]] = True
        routed_positives = selected_positives[routed]
        per_query_routed_positives.append(int(len(routed_positives)))

        values = np.concatenate(
            [
                base[query_index, base[query_index] >= 0],
                exact[query_index, exact[query_index] >= 0],
                routed_positives,
            ]
        )
        union = np.unique(values)
        if len(union) <= 100:
            raise ValueError("Flip candidate union must contain more than 100 rows")
        unions.append(union.astype(np.int64, copy=False))

    width = max(len(value) for value in unions)
    padded = np.full((len(unions), width), -1, dtype=np.int64)
    for query_index, values in enumerate(unions):
        padded[query_index, : len(values)] = values
    total_positives = int(valid.sum())
    routed_total = int(eligible.sum())
    summary = {
        "minimum_union_rows": int(min(len(value) for value in unions)),
        "maximum_union_rows": int(width),
        "mean_union_rows": float(np.mean([len(value) for value in unions])),
        "total_positive_qrels": total_positives,
        "routed_positive_qrels": routed_total,
        "routing_missed_positive_qrels": total_positives - routed_total,
        "queries_with_routed_positive": int(
            np.count_nonzero(np.asarray(per_query_routed_positives) > 0)
        ),
    }
    return padded, eligible, summary


def score_flip_candidate_union(
    queries: np.ndarray,
    candidate_rows: np.ndarray,
    corpus_tensor: Any,
    cpu_index: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every valid union row with original FP32 and reconstructed PQ."""

    import torch

    rows = np.asarray(candidate_rows, dtype=np.int64)
    if rows.ndim != 2 or len(rows) != len(queries):
        raise ValueError("Flip candidate rows must match query count")
    exact_scores = np.full(rows.shape, -np.inf, dtype=np.float32)
    pq_scores = np.full(rows.shape, -np.inf, dtype=np.float32)
    for query_index, padded in enumerate(rows):
        valid_rows = np.ascontiguousarray(padded[padded >= 0], dtype=np.int64)
        query = np.asarray(queries[query_index], dtype=np.float32)
        row_tensor = torch.as_tensor(
            valid_rows, dtype=torch.int64, device=corpus_tensor.device
        )
        query_tensor = torch.as_tensor(query, device=corpus_tensor.device)
        exact = corpus_tensor[row_tensor] @ query_tensor
        exact_scores[query_index, : len(valid_rows)] = (
            exact.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        reconstructed = np.asarray(
            cpu_index.reconstruct_batch(valid_rows), dtype=np.float32
        )
        if reconstructed.shape != (len(valid_rows), len(query)):
            raise ValueError("Faiss PQ reconstruction has an unexpected shape")
        if not np.all(np.isfinite(reconstructed)):
            raise ValueError("Faiss PQ reconstruction contains non-finite values")
        pq_scores[query_index, : len(valid_rows)] = reconstructed @ query
        del row_tensor, query_tensor, exact
    return exact_scores, pq_scores


def _stage_timer(telemetry: dict[str, float], name: str) -> float:
    now = time.perf_counter()
    prior = telemetry.pop("_stage_start", now)
    telemetry[f"{name}_wall_seconds"] = float(now - prior)
    telemetry["_stage_start"] = now
    return now


def _recall_payload(
    rows: np.ndarray, positives: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "recall_at_10": np.asarray(
            known_positive_recall_at_k(rows, positives, valid, k=10), dtype=np.float64
        ),
        "recall_at_100": np.asarray(
            known_positive_recall_at_k(rows, positives, valid, k=100), dtype=np.float64
        ),
    }


def _gate_call(
    decomposition: dict[str, Any], flip_summary: dict[str, Any]
) -> dict[str, Any]:
    uncapped = flip_summary["uncapped"]
    return diagnostic_gate_decision(
        pq_specific_r100_gap=float(decomposition["pq_specific_r100_gap"]),
        uncapped_triplets=int(uncapped["triplets"]),
        distinct_flip_queries=int(uncapped["distinct_flip_queries"]),
        effective_sample_size=float(uncapped["effective_sample_size"]),
        max_query_weight_share=float(uncapped["max_query_weight_share"]),
        qrels_corpus_coverage=float(
            decomposition["qrels_corpus_coverage"]
        ),
    )


def validate_runtime_contract(
    protocol: dict[str, Any], args: argparse.Namespace, torch: Any, faiss: Any
) -> dict[str, Any]:
    """Reject an environment or batch configuration outside the frozen contract."""

    config = protocol["diagnostic_configuration"]
    expected_args = {
        "query_batch_size": int(config["query_batch_size"]),
        "corpus_load_batch_size": int(config["corpus_load_batch_size"]),
        "candidate_batch_size": int(config["candidate_batch_size"]),
    }
    for name, expected in expected_args.items():
        if int(getattr(args, name)) != expected:
            raise ValueError(f"--{name.replace('_', '-')} must equal {expected}")
    if int(config["analysis_k"]) != TOP_K:
        raise ValueError("Protocol analysis_k differs from evaluator Top-K")

    contract = protocol["execution_environment_contract"]
    if ".".join(str(value) for value in sys.version_info[:3]) != contract["python_version"]:
        raise ValueError("Python version differs from the frozen environment")
    if np.__version__ != contract["numpy_version"]:
        raise ValueError("NumPy version differs from the frozen environment")
    if torch.__version__ != contract["torch_version"]:
        raise ValueError("Torch version differs from the frozen environment")
    if str(torch.version.cuda) != contract["torch_cuda_version"]:
        raise ValueError("Torch CUDA version differs from the frozen environment")
    if str(getattr(faiss, "__version__", "")) != "1.12.0":
        raise ValueError("Faiss version differs from the frozen environment")
    if not torch.cuda.is_available() or faiss.get_num_gpus() <= 0:
        raise ValueError("The frozen diagnostic requires Torch and Faiss CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if contract["gpu_name_must_contain"] not in gpu_name:
        raise ValueError("GPU differs from the frozen T4 environment")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", "UNSET") != contract[
        "cublas_workspace_config"
    ]:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs from the contract")
    if torch.backends.cudnn.benchmark is not contract["cudnn_benchmark"]:
        raise ValueError("cuDNN benchmark flag differs from the contract")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Deterministic Torch algorithms are not enabled")
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "faiss": str(getattr(faiss, "__version__", "")),
        "gpu": gpu_name,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def validate_signal_gate_contract(protocol: dict[str, Any]) -> None:
    registered = protocol["signal_gate"]
    expected = {
        "minimum_pq_specific_recall_at_100_gap": GATE_THRESHOLDS[
            "minimum_pq_specific_r100_gap"
        ],
        "minimum_uncapped_flip_triplets": GATE_THRESHOLDS[
            "minimum_uncapped_triplets"
        ],
        "minimum_distinct_flip_queries": GATE_THRESHOLDS[
            "minimum_distinct_flip_queries"
        ],
        "minimum_flip_weight_effective_sample_size": GATE_THRESHOLDS[
            "minimum_effective_sample_size"
        ],
        "maximum_single_query_flip_weight_share": GATE_THRESHOLDS[
            "maximum_query_weight_share"
        ],
        "minimum_qrels_in_corpus_coverage": GATE_THRESHOLDS[
            "required_qrels_corpus_coverage"
        ],
    }
    for key, value in expected.items():
        if registered.get(key) != value:
            raise ValueError(f"Protocol and executable gate disagree on {key}")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_empty_output(path: Path) -> None:
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError("--output-dir exists and is not a directory")
        if any(path.iterdir()):
            raise ValueError("Refusing to use a nonempty output directory")
    else:
        path.mkdir(parents=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import faiss
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = validate_protocol_and_source_blobs(
        repo_root, args.protocol, args.source_commit
    )
    validate_signal_gate_contract(protocol)
    if not args.use_gpu:
        raise ValueError("This canonical diagnostic requires --use-gpu")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    environment = validate_runtime_contract(protocol, args, torch, faiss)
    scratch_free_before = int(shutil.disk_usage(args.scratch_dir).free)
    minimum_free = int(protocol["resource_contract"]["minimum_local_free_bytes_before_run"])
    if scratch_free_before < minimum_free:
        raise ValueError(
            f"STOP_RESOURCE_SMOKE_FAILED: {args.scratch_dir} has "
            f"{scratch_free_before} free bytes; need {minimum_free}"
        )
    _prepare_empty_output(args.output_dir)
    started_wall = time.perf_counter()
    telemetry: dict[str, float] = {"_stage_start": started_wall}

    input_records = verify_frozen_inputs(
        protocol,
        embeddings=args.embeddings,
        doc_ids=args.doc_ids,
        qrels=args.qrels,
        index=args.index,
    )
    qids, queries, role_records = load_design_role(args.design_role_dir, protocol)
    input_records.update(role_records)
    _stage_timer(telemetry, "input_validation")

    qrels = load_positive_qrels(args.qrels)
    doc_ids = np.memmap(
        args.doc_ids, dtype=np.int64, mode="r", shape=(EXPECTED_INDEX["ntotal"],)
    )
    qrels_doc_ids, qrels_valid = pad_qrels_for_queries(qids, qrels)
    mapping = map_qrels_doc_ids_to_corpus_rows(doc_ids, qrels_doc_ids, qrels_valid)
    positive_rows, positive_valid, mapping_summary = mapping_arrays(mapping)
    if positive_rows.shape[0] != len(qids) or np.any(positive_valid.sum(axis=1) <= 0):
        raise ValueError("Every design query must have at least one corpus-covered positive")
    _stage_timer(telemetry, "qrels_mapping")

    cpu_index = faiss.read_index(str(args.index))
    ivf, index_contract = validate_faiss_index(cpu_index, faiss)
    ivf.nprobe = EXPECTED_INDEX["nprobe"]
    inverted_lists, inverted_summary = inverted_lists_as_rows(ivf, faiss)
    candidates, probed_lists = probed_candidate_rows(
        queries, ivf, inverted_lists, nprobe=EXPECTED_INDEX["nprobe"]
    )
    _stage_timer(telemetry, "index_and_inverted_list_validation")

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    base_scores, base_rows = cpu_index.search(
        np.asarray(queries, dtype=np.float32), TOP_K
    )
    base_scores = np.asarray(base_scores, dtype=np.float32)
    base_rows = np.asarray(base_rows, dtype=np.int64)
    if base_rows.shape != (len(qids), TOP_K) or np.any(base_rows < 0):
        raise ValueError("Base IVF-PQ search did not return a complete Top-200")
    _stage_timer(telemetry, "base_ivfpq_search")

    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(EXPECTED_INDEX["ntotal"], EXPECTED_INDEX["dimension"]),
    )
    corpus_tensor = load_corpus_tensor_torch(
        embeddings,
        device=device,
        load_batch_size=args.corpus_load_batch_size,
    )
    torch.cuda.synchronize()
    _stage_timer(telemetry, "corpus_gpu_load")
    full_scores, full_rows = full_exact_topk_torch(
        queries,
        corpus_tensor,
        k=TOP_K,
        query_batch_size=args.query_batch_size,
    )
    torch.cuda.synchronize()
    _stage_timer(telemetry, "full_exact_search")
    within_scores, within_rows = within_ivf_exact_topk_torch(
        queries,
        corpus_tensor,
        candidates,
        k=TOP_K,
        candidate_batch_size=args.candidate_batch_size,
    )
    torch.cuda.synchronize()
    _stage_timer(telemetry, "same_ivf_exact_search")

    recalls = {
        "base_pq": _recall_payload(base_rows, positive_rows, positive_valid),
        "same_ivf_exact": _recall_payload(within_rows, positive_rows, positive_valid),
        "full_exact": _recall_payload(full_rows, positive_rows, positive_valid),
    }
    decomposition = decompose_recall_gaps(
        recalls["full_exact"]["recall_at_100"],
        recalls["same_ivf_exact"]["recall_at_100"],
        recalls["base_pq"]["recall_at_100"],
    )
    decomposition["qrels_corpus_coverage"] = float(
        mapping_summary["qrels_corpus_coverage"]
    )
    flip_rows, flip_positive_valid, flip_union_summary = build_flip_candidate_union(
        base_rows,
        within_rows,
        positive_rows,
        positive_valid,
        candidates,
    )
    ivf.make_direct_map()
    flip_exact_scores, flip_pq_scores = score_flip_candidate_union(
        queries, flip_rows, corpus_tensor, cpu_index
    )
    torch.cuda.synchronize()
    del corpus_tensor
    torch.cuda.empty_cache()
    gc.collect()
    flips = mine_pq_induced_flip_triplets(
        flip_rows,
        flip_exact_scores,
        flip_pq_scores,
        positive_rows,
        flip_positive_valid,
        pool_k=int(protocol["diagnostic_configuration"]["pool_k"]),
        negative_window=int(
            protocol["diagnostic_configuration"]["negative_window"]
        ),
        max_unjudged_per_positive=int(
            protocol["diagnostic_configuration"]["capped_negatives_per_positive"]
        ),
        margin_temperature=float(
            protocol["diagnostic_configuration"]["margin_temperature"]
        ),
        damage_scale=float(protocol["diagnostic_configuration"]["damage_scale"]),
        flip_bonus=float(protocol["diagnostic_configuration"]["flip_bonus"]),
    )
    flip_summary = flips.support
    decision = _gate_call(decomposition, flip_summary)
    _stage_timer(telemetry, "metrics_and_gate")

    output_arrays = {
        "base_pq_top_scores.float32.npy": base_scores,
        "base_pq_top_rows.int64.npy": base_rows,
        "ivf_exact_top_scores.float32.npy": within_scores,
        "ivf_exact_top_rows.int64.npy": within_rows,
        "full_exact_top_scores.float32.npy": full_scores,
        "full_exact_top_rows.int64.npy": full_rows,
        "probed_ivf_lists.int64.npy": probed_lists,
    }
    for method, values in recalls.items():
        for metric, array in values.items():
            output_method = "ivf_exact" if method == "same_ivf_exact" else method
            output_arrays[f"{output_method}_{metric}.float64.npy"] = array
    for name, value in output_arrays.items():
        atomic_save(args.output_dir / name, value)

    telemetry.pop("_stage_start", None)
    max_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    max_rss_bytes = max_rss_raw if platform.system() == "Darwin" else max_rss_raw * 1024
    telemetry.update(
        {
            "total_wall_seconds": float(time.perf_counter() - started_wall),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "host_max_rss_bytes": max_rss_bytes,
            "scratch_disk_free_bytes_before": scratch_free_before,
            "scratch_disk_free_bytes_after": int(shutil.disk_usage(args.scratch_dir).free),
            "output_disk_free_bytes_after": int(shutil.disk_usage(args.output_dir).free),
        }
    )
    output_records = {
        name: file_record(args.output_dir / name) for name in output_arrays
    }
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V6_1M_HEADROOM_COMPLETE",
        "source_commit": args.source_commit,
        "evidence_role": DESIGN_ROLE_ID,
        "training_performed": False,
        "adapter_used": False,
        "rars_used": False,
        "future_or_audit_role_opened": False,
        "environment": environment,
        "index_contract": index_contract,
        "inverted_lists": inverted_summary,
        "qrels_mapping": mapping_summary,
        "flip_candidate_union": flip_union_summary,
        "mean_recall": {
            method: {metric: float(np.mean(array)) for metric, array in values.items()}
            for method, values in recalls.items()
        },
        "recall_gap_decomposition": _json_safe(decomposition),
        "flip_support": _json_safe(flip_summary),
        "formal_decision": decision["decision"],
        "signal_gate": _json_safe(decision),
        "telemetry": telemetry,
        "source_blobs": source_blobs,
        "inputs": input_records,
        "outputs": output_records,
    }
    summary_path = args.output_dir / "headroom_result.json"
    atomic_json(summary_path, summary)
    result_record = file_record(summary_path)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V6_1M_HEADROOM_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": decision["decision"],
        "signal_gate": _json_safe(decision),
        "result": result_record,
        "outputs": output_records,
        "all_input_byte_hashes_recorded": True,
        "registered_input_hashes_verified": [
            "doc_ids",
            "index",
            "v3_candidate_manifest",
            "query_manifest",
            "query_vectors",
        ],
        "corpus_coverage": 1.0,
        "corpus_tensor_persisted": False,
        "training_performed": False,
        "adapter_used": False,
        "rars_used": False,
        "future_or_audit_role_opened": False,
    }
    atomic_json(args.output_dir / "headroom_complete.json", complete)
    missing_outputs = [
        name
        for name in protocol["required_outputs"]
        if not (args.output_dir / name).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"Required v6 outputs were not written: {missing_outputs}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-role-dir", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--doc-ids", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / CANONICAL_PROTOCOL,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--scratch-dir", type=Path, default=Path("/content"))
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--corpus-load-batch-size", type=int, default=8192)
    parser.add_argument("--candidate-batch-size", type=int, default=32768)
    args = parser.parse_args()
    for name in (
        "query_batch_size",
        "corpus_load_batch_size",
        "candidate_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
