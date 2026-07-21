#!/usr/bin/env python3
"""Build full-corpus RARS-v8 and PCA sidecars without opening qrels.

This is the bridge between outcome-informed method development and a later
locked evaluation.  It verifies the development packet, keeps the IVF-PQ file
byte-identical, calibrates int8 scales over all one million residuals, and
writes row-aligned code arrays.  The CLI intentionally has no qrels, query, or
holdout argument.
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

from evaluate_rars_v6_1m_headroom import validate_faiss_index  # noqa: E402
from rars_v8_cutoff_sidecar_core import PROTOCOL_ID, validate_orthonormal_basis  # noqa: E402


CANONICAL_PROTOCOL = Path("protocols/rars_v8_cutoff_sidecar_v1.json")
GO_DECISIONS = {
    "GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL",
    "GO_TO_GENERIC_SIDECAR_CONFIRMATION_PROTOCOL",
}


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 changed")


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("--source-commit must be exact lowercase 40-hex")
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must use the canonical path: {canonical}")
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
        raise ValueError("Sidecar build requires a clean exact checkout")
    protocol = read_json(canonical)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected V8 protocol")
    return protocol


def verify_development_packet(
    packet_root: Path, *, source_commit: str
) -> dict[str, Any]:
    packet_root = packet_root.resolve()
    complete_path = packet_root / "development_complete.json"
    freeze_path = packet_root / "method_freeze.json"
    result_path = packet_root / "development_result.json"
    complete = read_json(complete_path)
    freeze = read_json(freeze_path)
    result = read_json(result_path)
    for payload, label in (
        (complete, "complete"),
        (freeze, "freeze"),
        (result, "result"),
    ):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"Development {label} protocol changed")
        if payload.get("source_commit") != source_commit:
            raise ValueError(f"Development {label} source commit changed")
        if payload.get("formal_decision") not in GO_DECISIONS:
            raise ValueError(f"Development {label} does not authorize artifact build")
        if payload.get("future_method_holdout_opened") is not False and label != "freeze":
            raise ValueError(f"Development {label} reports future access")
    verify_record(result_path, complete["outputs"]["development_result.json"], "result")
    verify_record(freeze_path, complete["outputs"]["method_freeze.json"], "method freeze")
    for filename, record in complete.get("outputs", {}).items():
        verify_record(packet_root / filename, record, filename)
    for key in ("pca_basis", "rars_basis"):
        record = freeze[key]
        path = packet_root / Path(record["path"]).name
        verify_record(path, record, key)
    return {
        "status": "RARS_V8_DEVELOPMENT_PACKET_VERIFIED",
        "formal_decision": result["formal_decision"],
        "complete": file_record(complete_path),
        "method_freeze": file_record(freeze_path),
        "result": file_record(result_path),
    }


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("Refusing to reuse a non-empty sidecar output root")
    path.mkdir(parents=True, exist_ok=True)


def _load_basis(path: Path, *, dimension: int, rank: int) -> np.ndarray:
    return validate_orthonormal_basis(
        np.load(path, allow_pickle=False), dimension=dimension, rank=rank
    )


def _reconstruct_residual_block(
    embeddings: np.memmap, index: Any, start: int, end: int
) -> np.ndarray:
    rows = np.arange(start, end, dtype=np.int64)
    reconstructed = np.asarray(index.reconstruct_batch(rows), dtype=np.float32)
    original = np.asarray(embeddings[start:end], dtype=np.float32)
    residual = original - reconstructed
    if not np.all(np.isfinite(residual)):
        raise ValueError("Full-corpus residual block contains non-finite values")
    return residual


def build_one_sidecar(
    *,
    name: str,
    basis: np.ndarray,
    embeddings: np.memmap,
    index: Any,
    output_dir: Path,
    alpha: float,
    top_b: int,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rank = basis.shape[1]
    maximum = np.zeros(rank, dtype=np.float32)
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        coefficients = _reconstruct_residual_block(
            embeddings, index, start, end
        ) @ basis
        maximum = np.maximum(maximum, np.max(np.abs(coefficients), axis=0))
    scales = np.maximum(
        maximum / 127.0, np.finfo(np.float32).tiny
    ).astype(np.float32)
    basis_path = output_dir / "basis.float32.npy"
    scales_path = output_dir / "scales.float32.npy"
    codes_path = output_dir / "codes.int8.npy"
    atomic_save(basis_path, basis.astype(np.float32))
    atomic_save(scales_path, scales)
    codes = np.lib.format.open_memmap(
        codes_path,
        mode="w+",
        dtype=np.int8,
        shape=(len(embeddings), rank),
    )
    saturated = 0
    coefficient_count = 0
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        coefficients = _reconstruct_residual_block(
            embeddings, index, start, end
        ) @ basis
        rounded = np.rint(coefficients / scales[None, :])
        saturated += int(np.sum(np.abs(rounded) > 127))
        coefficient_count += int(rounded.size)
        codes[start:end] = np.clip(rounded, -127, 127).astype(np.int8)
        codes.flush()
    del codes
    config = {
        "schema_version": 1,
        "artifact_type": "row_aligned_frozen_ivfpq_residual_sidecar",
        "method": name,
        "protocol_id": PROTOCOL_ID,
        "dimension": int(basis.shape[0]),
        "rank": int(rank),
        "coefficient_dtype": "int8",
        "alpha": float(alpha),
        "top_b": int(top_b),
        "row_alignment": (
            "code row equals the zero-based corpus row returned by the frozen index; "
            "external document IDs are not duplicated in the representation payload"
        ),
        "basis_file": basis_path.name,
        "scales_file": scales_path.name,
        "codes_file": codes_path.name,
        "score_formula": "s_pq + alpha * dot(q @ B, int8_code[row] * scales)",
    }
    config_path = output_dir / "sidecar_config.json"
    atomic_json(config_path, config)
    files = {
        path.name: file_record(path)
        for path in (basis_path, scales_path, codes_path, config_path)
    }
    raw_representation_bytes = sum(
        files[name]["bytes"]
        for name in (basis_path.name, scales_path.name, codes_path.name)
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method": name,
        "files": files,
        "quantization": {
            "coefficient_count": coefficient_count,
            "saturated_coefficients": saturated,
            "saturation_fraction": float(saturated / coefficient_count),
        },
        "storage": {
            "code_payload_bytes_per_document": int(rank),
            "representation_bytes": int(raw_representation_bytes),
            "representation_bytes_per_document": float(
                raw_representation_bytes / len(embeddings)
            ),
            "external_document_id_bytes_duplicated": 0,
            "configuration_bytes_excluded_from_representation": int(
                config_path.stat().st_size
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": file_record(manifest_path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import faiss

    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    protocol = validate_source(repo_root, args.protocol, args.source_commit)
    development = verify_development_packet(
        args.development_packet, source_commit=args.source_commit
    )
    _prepare_output(args.output_root)
    frozen = protocol["frozen_index_contract"]
    index_before = file_record(args.index)
    doc_ids_record = file_record(args.doc_ids)
    if index_before["bytes"] != int(frozen["index_bytes"]) or index_before[
        "sha256"
    ] != frozen["index_sha256"]:
        raise ValueError("Frozen IVF-PQ index differs from the V8 contract")
    if doc_ids_record["bytes"] != int(frozen["doc_ids_bytes"]) or doc_ids_record[
        "sha256"
    ] != frozen["doc_ids_sha256"]:
        raise ValueError("Frozen document IDs differ from the V8 contract")
    doc_ids = np.memmap(
        args.doc_ids,
        dtype=np.int64,
        mode="r",
        shape=(int(frozen["document_count"]),),
    )
    if np.unique(np.asarray(doc_ids)).size != len(doc_ids):
        raise ValueError("Frozen document IDs are not unique")
    embeddings = np.memmap(
        args.embeddings,
        dtype=np.float16,
        mode="r",
        shape=(int(frozen["document_count"]), int(frozen["embedding_dimension"])),
    )
    cpu_index = faiss.read_index(str(args.index))
    ivf, index_contract = validate_faiss_index(cpu_index, faiss)
    ivf.nprobe = int(frozen["nprobe"])
    ivf.make_direct_map()
    basis_paths = {
        "pca": args.development_packet / "pca_basis_rank16.float32.npy",
        "rars": args.development_packet / "rars_basis_rank16.float32.npy",
    }
    bases = {
        name: _load_basis(
            path,
            dimension=int(frozen["embedding_dimension"]),
            rank=int(protocol["method"]["rank"]),
        )
        for name, path in basis_paths.items()
    }
    manifests = {
        name: build_one_sidecar(
            name=name,
            basis=basis,
            embeddings=embeddings,
            # The direct map belongs to the down-cast IVF object. Passing the
            # IVF explicitly avoids wrapper-dependent reconstruct_batch
            # failures (the root cause of an earlier V7 execution failure).
            index=ivf,
            output_dir=args.output_root / name,
            alpha=float(protocol["method"]["alpha"]),
            top_b=int(protocol["method"]["top_b"]),
            batch_size=args.batch_size,
        )
        for name, basis in bases.items()
    }
    index_after = file_record(args.index)
    if index_after != index_before:
        raise ValueError("Frozen IVF-PQ index changed during sidecar construction")
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V8_FULL_CORPUS_SIDECARS_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": development["formal_decision"],
        "development": development,
        "index_contract": index_contract,
        "index_before": index_before,
        "index_after": index_after,
        "index_unchanged": True,
        "inputs": {
            "embeddings": file_record(args.embeddings),
            "doc_ids": doc_ids_record,
        },
        "sidecars": manifests,
        "qrels_argument_accepted": False,
        "query_argument_accepted": False,
        "future_method_holdout_opened": False,
        "telemetry": {
            "wall_seconds": float(time.perf_counter() - started),
            "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
        },
    }
    result_path = args.output_root / "sidecars_result.json"
    atomic_json(result_path, result)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V8_FULL_CORPUS_SIDECARS_COMPLETE",
        "source_commit": args.source_commit,
        "formal_decision": development["formal_decision"],
        "result": file_record(result_path),
        "sidecar_manifests": {
            name: manifest["manifest"] for name, manifest in manifests.items()
        },
        "index_before": index_before,
        "index_after": index_after,
        "qrels_opened": False,
        "future_method_holdout_opened": False,
    }
    atomic_json(args.output_root / "sidecars_complete.json", complete)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-packet", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--doc-ids", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
