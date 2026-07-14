#!/usr/bin/env python3
"""Prepare qrels-free BEIR NQ artifacts on Colab T4 + Google Drive.

The commands in this module are deliberately staged and resumable.  Before
any network or dataset operation, ``init`` verifies a clean, user-supplied Git
design-freeze commit. Extraction uses the official split archives separately:
``nq.zip`` supplies only the test-corpus ``corpus.jsonl`` during Stage 1, while
``nq-train.zip`` supplies only train ``queries.jsonl`` and ``qrels/train.tsv``.
Test queries and ``qrels/test.tsv`` remain unopened in ``nq.zip`` until the
separate method-artifact freeze is committed.

Typical order::

    python scripts/prepare_beir_nq_colab.py init ...
    python scripts/prepare_beir_nq_colab.py download ...
    python scripts/prepare_beir_nq_colab.py extract-train-only ...
    python scripts/prepare_beir_nq_colab.py scan-corpus ...
    python scripts/create_beir_nq_train_validation_splits.py ...
    python scripts/prepare_beir_nq_colab.py encode-corpus ...
    python scripts/prepare_beir_nq_colab.py encode-train-queries ...
    python scripts/prepare_beir_nq_colab.py build-index ...

Operational batch sizes may be changed for memory safety.  Scientific values
are loaded from and checked against the frozen protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols" / "beir_nq_rars_pca_confirmation_v1.json"
PROTOCOL_ID = "beir_nq_rars_pca_confirmation_v1"
ARCHIVES = {
    "test": {
        "name": "nq.zip",
        "md5": "d4d3d2e48787a744b6f6e691ff534307",
        "sha256": "2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752",
        "bytes": 498_307_926,
    },
    "train": {
        "name": "nq-train.zip",
        "md5": "966435435932347d5513f56fed19161c",
        "sha256": "3aa8eec8d67174d85c055fce6971fa3127e830335842696756373f491bf391c9",
        "bytes": 1_405_702_846,
    },
}
REQUIRED_FREEZE_PATHS = (
    "protocols/beir_nq_rars_pca_confirmation_v1.json",
    "protocols/beir_nq_pre_qrels_manifest.template.json",
    "docs/beir_nq_confirmation_protocol.md",
    "scripts/prepare_beir_nq_colab.py",
    "scripts/create_beir_nq_train_validation_splits.py",
    "scripts/train_select_beir_nq_sidecars.py",
    "scripts/build_beir_nq_pre_qrels_manifest.py",
    "scripts/build_beir_nq_prior_query_registry.py",
    "scripts/evaluate_beir_nq_frozen.py",
    "scripts/validate_nq_pre_qrels_freeze.py",
    "protocols/beir_nq_prior_query_registry_v1.json",
    "tests/test_prepare_beir_nq_colab.py",
    "tests/test_train_select_beir_nq_sidecars.py",
    "tests/test_create_beir_nq_splits.py",
    "tests/test_nq_pre_qrels_freeze.py",
    "tests/test_beir_nq_freeze_helpers.py",
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    try:
        digest = hashlib.md5(usedforsecurity=False)
    except TypeError:  # pragma: no cover - older Python builds
        digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, include_md5: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if include_md5:
        record["md5"] = md5_file(path)
    return record


def run_git(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", "")
        raise RuntimeError(f"Git verification failed: {detail}") from exc


def verify_design_freeze_commit(repo: Path, expected_commit: str) -> None:
    expected_commit = expected_commit.strip().casefold()
    if not COMMIT_RE.fullmatch(expected_commit):
        raise ValueError("--design-freeze-commit must be a full 40-hex Git commit")
    head = run_git(repo, "rev-parse", "HEAD").casefold()
    if head != expected_commit:
        raise ValueError(f"HEAD {head} does not equal design freeze {expected_commit}")
    if run_git(repo, "status", "--porcelain"):
        raise ValueError("The design-freeze checkout must be clean before Stage 1")
    for relative in REQUIRED_FREEZE_PATHS:
        try:
            run_git(repo, "cat-file", "-e", f"{expected_commit}:{relative}")
        except RuntimeError as exc:
            raise ValueError(
                f"Design freeze commit does not contain required path: {relative}"
            ) from exc


def protocol_values(protocol_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected protocol ID")
    embedding = protocol.get("embedding", {})
    base = protocol.get("base_index", {})
    expected = {
        "model": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "document_storage_dtype": "float16",
    }
    for key, value in expected.items():
        if embedding.get(key) != value:
            raise ValueError(f"Protocol embedding.{key} drifted")
    dataset = protocol.get("dataset", {})
    for name, expected_archive in ARCHIVES.items():
        archive = dataset.get(f"{name}_archive", {})
        for key in ("md5", "sha256", "bytes"):
            if archive.get(key) != expected_archive[key]:
                raise ValueError(f"Protocol dataset.{name}_archive.{key} drifted")
    if dataset.get("test_archive", {}).get("stage1_members_allowed") != [
        "corpus.jsonl"
    ]:
        raise ValueError("Protocol test-archive Stage-1 access rule drifted")
    if dataset.get("test_archive", {}).get("stage3_members_allowed") != [
        "queries.jsonl",
        "qrels/test.tsv",
    ]:
        raise ValueError("Protocol test-archive Stage-3 access rule drifted")
    train_archive = dataset.get("train_archive", {})
    if train_archive.get("stage1_members_allowed") != [
        "queries.jsonl",
        "qrels/train.tsv",
    ]:
        raise ValueError("Protocol train-archive Stage-1 access rule drifted")
    if train_archive.get("corpus_member_prohibited") is not True:
        raise ValueError("Protocol must prohibit the train-archive corpus")
    for key, value in {
        "m": 32,
        "nbits": 8,
        "nlist": 2048,
        "nprobe": 32,
        "training_document_sample_max": 300_000,
        "training_seed": 42,
        "coarse_kmeans_iterations": 25,
        "pq_kmeans_iterations": 25,
        "build_backend": "single_faiss_gpu_nvidia_t4",
        "gpu_float16_lookup_tables": False,
    }.items():
        if base.get(key) != value:
            raise ValueError(f"Protocol base_index.{key} drifted")
    return protocol


def gate_path(artifact_root: Path) -> Path:
    return artifact_root / "stage0" / "design_freeze_gate.json"


def initialize_gate(
    artifact_root: Path,
    repo: Path,
    protocol_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    protocol_values(protocol_path)
    verify_design_freeze_commit(repo, expected_commit)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "design_frozen_before_dataset_access",
        "design_freeze_commit": expected_commit.casefold(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "created_utc": utc_now(),
        "test_qrels_accessed": False,
        "test_retrieval_performed": False,
        "test_outcomes_observed": False,
    }
    existing_path = gate_path(artifact_root)
    if existing_path.exists():
        existing = read_json(existing_path)
        immutable = ("protocol_id", "design_freeze_commit", "protocol_sha256")
        if any(existing.get(key) != payload.get(key) for key in immutable):
            raise ValueError("Existing design-freeze gate conflicts with this checkout")
        return existing
    atomic_write_json(existing_path, payload)
    return payload


def verify_gate(
    artifact_root: Path,
    repo: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    path = gate_path(artifact_root)
    if not path.is_file():
        raise FileNotFoundError("Run the init stage before dataset operations")
    gate = read_json(path)
    if gate.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Design-freeze gate has the wrong protocol")
    for key in (
        "test_qrels_accessed",
        "test_retrieval_performed",
        "test_outcomes_observed",
    ):
        if gate.get(key) is not False:
            raise ValueError(f"Unsafe design-freeze gate flag: {key}")
    protocol_values(protocol_path)
    if gate.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Protocol changed after the Stage-0 design freeze")
    verify_design_freeze_commit(repo, str(gate.get("design_freeze_commit", "")))
    return gate


def archive_path(artifact_root: Path, archive_name: str) -> Path:
    return artifact_root / "source" / str(ARCHIVES[archive_name]["name"])


def download_with_resume(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if md5_file(destination) == expected_md5:
            print(f"Archive already verified: {destination}")
            return
        raise ValueError(f"Existing archive has the wrong MD5: {destination}")

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - frozen URL
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        mode = "ab" if append else "wb"
        if offset and not append:
            offset = 0
        written = offset
        with partial.open(mode) as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                written += len(chunk)
                print(f"downloaded {written / 1e9:.3f} GB", end="\r", flush=True)
            output.flush()
            os.fsync(output.fileno())
    print()
    actual = md5_file(partial)
    if actual != expected_md5:
        raise ValueError(f"Downloaded archive MD5 {actual} != {expected_md5}")
    partial.replace(destination)


def download_archives(
    artifact_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = protocol_values(protocol_path)
    records: dict[str, Any] = {}
    for name in ("test", "train"):
        registered = protocol["dataset"][f"{name}_archive"]
        target = archive_path(artifact_root, name)
        download_with_resume(registered["url"], target, registered["md5"])
        record = file_record(target, include_md5=True)
        if record["bytes"] != registered["bytes"]:
            raise ValueError(f"{name} archive byte count does not match protocol")
        if record["md5"] != registered["md5"]:
            raise ValueError(f"{name} archive MD5 does not match protocol")
        if record["sha256"] != registered["sha256"]:
            raise ValueError(f"{name} archive SHA-256 does not match protocol")
        records[f"{name}_dataset_archive"] = record
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "archives": records,
        "download_completed_utc": utc_now(),
        "test_qrels_opened": False,
        "test_queries_opened": False,
    }
    atomic_write_json(artifact_root / "source" / "archive_manifest.json", manifest)
    return manifest


def safe_member_name(value: str) -> PurePosixPath:
    normalized = PurePosixPath(value.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe ZIP member path: {value}")
    return normalized


def choose_members(
    archive: zipfile.ZipFile,
    suffixes: dict[str, tuple[str, ...]],
) -> dict[str, zipfile.ZipInfo]:
    selected: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        path = safe_member_name(info.filename)
        for label, suffix in suffixes.items():
            if tuple(path.parts[-len(suffix):]) == suffix:
                if label in selected:
                    raise ValueError(f"Archive has duplicate {label} members")
                selected[label] = info
    missing = sorted(set(suffixes) - set(selected))
    if missing:
        raise FileNotFoundError(f"Archive is missing required inputs: {missing}")
    return selected


def extract_train_only(artifact_root: Path) -> dict[str, Any]:
    sources = {
        name: archive_path(artifact_root, name)
        for name in ("test", "train")
    }
    for name, source in sources.items():
        if not source.is_file() or md5_file(source) != ARCHIVES[name]["md5"]:
            raise ValueError(f"Verified NQ {name} archive is unavailable")
    target_root = artifact_root / "stage1" / "data" / "nq"
    outputs = {
        "corpus.jsonl": target_root / "corpus.jsonl",
        "queries.jsonl": target_root / "train" / "queries.jsonl",
        "qrels/train.tsv": target_root / "qrels" / "train.tsv",
    }
    archive_specs = {
        "test": {"corpus.jsonl": ("corpus.jsonl",)},
        "train": {
            "queries.jsonl": ("queries.jsonl",),
            "qrels/train.tsv": ("qrels", "train.tsv"),
        },
    }
    selected_sources: dict[str, str] = {}
    for archive_name, suffixes in archive_specs.items():
        with zipfile.ZipFile(sources[archive_name]) as archive:
            selected = choose_members(archive, suffixes)
            for label, info in selected.items():
                selected_sources[label] = archive_name
                destination = outputs[label]
                if destination.exists() and destination.stat().st_size == info.file_size:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".part")
                with archive.open(info, "r") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                if temporary.stat().st_size != info.file_size:
                    raise IOError(f"Incomplete extraction for {label}")
                temporary.replace(destination)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "extraction_policy": "test_corpus_plus_train_queries_and_train_qrels_only",
        "member_sources": selected_sources,
        "files": {label: file_record(path) for label, path in outputs.items()},
        "test_queries_extracted": False,
        "test_qrels_extracted": False,
        "test_qrels_opened": False,
        "completed_utc": utc_now(),
    }
    atomic_write_json(
        artifact_root / "stage1" / "data" / "train_only_extraction_manifest.json",
        manifest,
    )
    return manifest


def corpus_text(item: dict[str, Any]) -> str:
    title = str(item.get("title", "") or "").strip()
    text = str(item.get("text", "") or "").strip()
    if not text:
        raise ValueError("Corpus row has empty text")
    return f"{title}\n{text}" if title else text


def corpus_id(item: dict[str, Any]) -> str:
    value = item.get("_id", item.get("id", item.get("doc_id")))
    if value is None:
        raise ValueError("Corpus row has no document ID")
    result = str(value).strip()
    if not result:
        raise ValueError("Corpus row has an empty document ID")
    return result


def scan_corpus(corpus_path: Path, output_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    max_id_bytes = 0
    seen: set[str] = set()
    with corpus_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            item = json.loads(raw)
            doc_id = corpus_id(item)
            corpus_text(item)
            if doc_id in seen:
                raise ValueError(f"Duplicate corpus ID at line {line_number}: {doc_id}")
            seen.add(doc_id)
            max_id_bytes = max(max_id_bytes, len(doc_id.encode("utf-8")))
            count += 1
            if count % 100_000 == 0:
                print(f"scanned {count:,} documents")
    if count == 0:
        raise ValueError("Corpus is empty")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "corpus_path": str(corpus_path),
        "corpus_bytes": corpus_path.stat().st_size,
        "corpus_sha256": digest.hexdigest(),
        "document_count": count,
        "max_doc_id_utf8_bytes": max_id_bytes,
        "duplicate_document_ids": 0,
        "test_qrels_accessed": False,
        "completed_utc": utc_now(),
    }
    atomic_write_json(output_path, manifest)
    return manifest


def snapshot_manifest(snapshot_dir: Path, output_path: Path) -> dict[str, Any]:
    files = []
    for path in sorted(snapshot_dir.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(snapshot_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    if not files:
        raise ValueError("Embedding-model snapshot is empty")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "model": "BAAI/bge-small-en-v1.5",
        "snapshot_directory": str(snapshot_dir),
        "files": files,
        "test_qrels_accessed": False,
    }
    atomic_write_json(output_path, payload)
    return payload


def verify_snapshot_files(manifest_path: Path) -> Path:
    manifest = read_json(manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Embedding-model snapshot manifest has the wrong protocol")
    snapshot_dir = Path(manifest["snapshot_directory"])
    for item in manifest.get("files", []):
        path = snapshot_dir / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Embedding-model snapshot file changed: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Embedding-model snapshot hash changed: {path}")
    return snapshot_dir


def load_or_create_model(
    artifact_root: Path,
    model_name: str,
    device: str,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install sentence-transformers in the Colab runtime") from exc

    snapshot_dir = artifact_root / "stage1" / "model" / "bge-small-en-v1.5"
    manifest_path = artifact_root / "stage1" / "model" / "snapshot_manifest.json"
    if snapshot_dir.is_dir() and manifest_path.is_file():
        verified_snapshot = verify_snapshot_files(manifest_path)
        return SentenceTransformer(str(verified_snapshot), device=device)

    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    temporary = snapshot_dir.with_name(snapshot_dir.name + ".part")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save(str(temporary))
    temporary.replace(snapshot_dir)
    snapshot_manifest(snapshot_dir, manifest_path)
    return model


def assert_t4() -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for the registered T4 run") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU runtime before continuing")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name.upper():
        raise RuntimeError(f"Frozen protocol requires an NVIDIA T4; found {name}")


def load_scan_manifest(artifact_root: Path) -> tuple[Path, dict[str, Any]]:
    path = artifact_root / "stage1" / "corpus" / "corpus_scan_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Run scan-corpus before encoding")
    return path, read_json(path)


def encode_corpus(
    artifact_root: Path,
    protocol_path: Path,
    *,
    batch_size: int,
    checkpoint_rows: int,
    device: str,
) -> dict[str, Any]:
    if device == "cuda":
        assert_t4()
    protocol = protocol_values(protocol_path)
    _, scan = load_scan_manifest(artifact_root)
    corpus_path = Path(scan["corpus_path"])
    n_docs = int(scan["document_count"])
    dim = int(protocol["embedding"]["dimension"])
    id_width = int(scan["max_doc_id_utf8_bytes"])
    output_dir = artifact_root / "stage1" / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.float16.memmap"
    doc_ids_path = output_dir / "doc_ids.utf8.memmap"
    embeddings_part = embeddings_path.with_name(embeddings_path.name + ".part")
    doc_ids_part = doc_ids_path.with_name(doc_ids_path.name + ".part")
    progress_path = output_dir / "embedding_progress.json"
    complete_path = output_dir / "corpus_artifacts_manifest.json"
    if complete_path.is_file() and embeddings_path.is_file() and doc_ids_path.is_file():
        return read_json(complete_path)

    model = load_or_create_model(
        artifact_root,
        protocol["embedding"]["model"],
        device,
    )
    if progress_path.exists():
        progress = read_json(progress_path)
        written = int(progress["rows_written"])
        byte_offset = int(progress["corpus_byte_offset"])
        mode = "r+"
    else:
        written = 0
        byte_offset = 0
        mode = "w+"
    embeddings = np.memmap(
        embeddings_part,
        dtype=np.float16,
        mode=mode,
        shape=(n_docs, dim),
    )
    doc_ids = np.memmap(
        doc_ids_part,
        dtype=f"S{id_width}",
        mode=mode,
        shape=(n_docs,),
    )

    last_checkpoint = written
    texts: list[str] = []
    ids: list[bytes] = []
    with corpus_path.open("rb") as handle:
        handle.seek(byte_offset)

        def flush() -> None:
            nonlocal written, last_checkpoint
            if not texts:
                return
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(texts), dim) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid document embedding batch: {vectors.shape}")
            end = written + len(texts)
            embeddings[written:end] = vectors.astype(np.float16)
            doc_ids[written:end] = np.asarray(ids, dtype=f"S{id_width}")
            written = end
            texts.clear()
            ids.clear()
            if written - last_checkpoint >= checkpoint_rows or written == n_docs:
                embeddings.flush()
                doc_ids.flush()
                atomic_write_json(progress_path, {
                    "protocol_id": PROTOCOL_ID,
                    "rows_written": written,
                    "corpus_byte_offset": handle.tell(),
                    "document_count": n_docs,
                    "dimension": dim,
                    "doc_id_width_bytes": id_width,
                    "test_qrels_accessed": False,
                })
                last_checkpoint = written
                print(f"encoded {written:,}/{n_docs:,} documents")

        for raw in handle:
            if not raw.strip():
                continue
            item = json.loads(raw)
            texts.append(corpus_text(item))
            ids.append(corpus_id(item).encode("utf-8"))
            if len(texts) >= batch_size:
                flush()
        flush()

    embeddings.flush()
    doc_ids.flush()
    del embeddings, doc_ids
    if written != n_docs:
        raise ValueError(f"Encoded {written} documents, expected {n_docs}")
    embeddings_part.replace(embeddings_path)
    doc_ids_part.replace(doc_ids_path)
    progress_path.unlink(missing_ok=True)
    model_manifest_path = artifact_root / "stage1" / "model" / "snapshot_manifest.json"
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "document_count": n_docs,
        "dimension": dim,
        "embedding_dtype": "float16",
        "faiss_and_residual_compute_dtype": "float32",
        "doc_id_dtype": f"S{id_width}",
        "doc_id_width_bytes": id_width,
        "document_text_rule": protocol["embedding"]["document_text"],
        "corpus_source": file_record(corpus_path),
        "document_embeddings": file_record(embeddings_path),
        "doc_ids": file_record(doc_ids_path),
        "embedding_model_snapshot_manifest": file_record(model_manifest_path),
        "test_qrels_accessed": False,
        "completed_utc": utc_now(),
    }
    atomic_write_json(complete_path, manifest)
    return manifest


def qid_digest(qids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for qid in qids:
        digest.update(str(qid).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_query_partition(payload: dict[str, Any], name: str) -> tuple[list[str], list[str]]:
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("partition") != name:
        raise ValueError(f"Unexpected {name} query manifest")
    if payload.get("qrels_relevance_values_used") is not False:
        raise ValueError("Train relevance values may not be used")
    if payload.get("test_qrels_accessed") is not False:
        raise ValueError("Test qrels access flag is unsafe")
    qids = [str(value) for value in payload.get("query_ids", [])]
    texts = [str(value) for value in payload.get("query_texts", [])]
    if not qids or len(qids) != len(texts) or len(qids) != len(set(qids)):
        raise ValueError(f"Invalid {name} query IDs/texts")
    return qids, texts


def encode_train_queries(
    artifact_root: Path,
    protocol_path: Path,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    if device == "cuda":
        assert_t4()
    protocol = protocol_values(protocol_path)
    split_dir = artifact_root / "stage1" / "query_splits"
    fit_path = split_dir / "train_query_manifest.json"
    validation_path = split_dir / "validation_query_manifest.json"
    fit = read_json(fit_path)
    validation = read_json(validation_path)
    fit_qids, fit_texts = validate_query_partition(fit, "fit")
    val_qids, val_texts = validate_query_partition(validation, "validation")
    if not set(fit_qids).isdisjoint(val_qids):
        raise ValueError("Fit and validation query IDs overlap")
    qids = fit_qids + val_qids
    texts = fit_texts + val_texts
    dim = int(protocol["embedding"]["dimension"])
    output_dir = artifact_root / "stage1" / "queries"
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "train_validation_query_vectors.float32.npy"
    manifest_path = output_dir / "train_validation_query_vector_manifest.json"
    if vectors_path.is_file() and manifest_path.is_file():
        return read_json(manifest_path)

    model = load_or_create_model(
        artifact_root,
        protocol["embedding"]["model"],
        device,
    )
    temporary = vectors_path.with_name(vectors_path.name + ".part")
    vectors = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(texts), dim),
    )
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = model.encode(
            texts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        batch = np.asarray(batch, dtype=np.float32)
        if batch.shape != (end - start, dim) or not np.isfinite(batch).all():
            raise ValueError(f"Invalid query embedding batch: {batch.shape}")
        vectors[start:end] = batch
        vectors.flush()
        print(f"encoded {end:,}/{len(texts):,} train/validation queries")
    del vectors
    temporary.replace(vectors_path)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "vector_order": ["fit", "validation"],
        "dtype": "float32",
        "dimension": dim,
        "query_count": len(qids),
        "blocks": {
            "fit": {
                "start": 0,
                "stop": len(fit_qids),
                "count": len(fit_qids),
                "query_ids_sha256": qid_digest(fit_qids),
            },
            "validation": {
                "start": len(fit_qids),
                "stop": len(qids),
                "count": len(val_qids),
                "query_ids_sha256": qid_digest(val_qids),
            },
        },
        "vectors": file_record(vectors_path),
        "fit_manifest": file_record(fit_path),
        "validation_manifest": file_record(validation_path),
        "embedding_model_snapshot_manifest": file_record(
            artifact_root / "stage1" / "model" / "snapshot_manifest.json"
        ),
        "train_qrels_relevance_values_used": False,
        "test_qrels_accessed": False,
        "completed_utc": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install faiss-gpu-cu12 in the Colab runtime") from exc
    return faiss


def clone_to_gpu(faiss: Any, cpu_index: Any) -> tuple[Any, Any]:
    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
        raise RuntimeError("The registered index build requires Faiss GPU support")
    resources = faiss.StandardGpuResources()
    if hasattr(resources, "setTempMemory"):
        resources.setTempMemory(512 * 1024 * 1024)
    options = faiss.GpuClonerOptions()
    options.useFloat16LookupTables = False
    options.useFloat16CoarseQuantizer = False
    return resources, faiss.index_cpu_to_gpu(resources, 0, cpu_index, options)


def build_index(
    artifact_root: Path,
    protocol_path: Path,
    *,
    add_batch_size: int,
    checkpoint_rows: int,
    backend: str,
) -> dict[str, Any]:
    protocol = protocol_values(protocol_path)
    if backend != "gpu":
        raise ValueError("The frozen full run requires --backend gpu")
    assert_t4()
    faiss = import_faiss()
    corpus_manifest_path = artifact_root / "stage1" / "corpus" / "corpus_artifacts_manifest.json"
    corpus_manifest = read_json(corpus_manifest_path)
    n_docs = int(corpus_manifest["document_count"])
    dim = int(corpus_manifest["dimension"])
    embeddings_path = Path(corpus_manifest["document_embeddings"]["path"])
    embeddings = np.memmap(
        embeddings_path,
        dtype=np.float16,
        mode="r",
        shape=(n_docs, dim),
    )
    base = protocol["base_index"]
    output_dir = artifact_root / "stage1" / "index"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "frozen_ivfpq_m32_nlist2048.index"
    manifest_path = output_dir / "index_manifest.json"
    if final_path.is_file() and manifest_path.is_file():
        return read_json(manifest_path)

    sample_path = output_dir / "training_sample_rows.int64.npy"
    if sample_path.exists():
        sample_rows = np.load(sample_path).astype(np.int64)
    else:
        rng = np.random.default_rng(int(base["training_seed"]))
        sample_count = min(int(base["training_document_sample_max"]), n_docs)
        sample_rows = np.sort(
            rng.choice(n_docs, size=sample_count, replace=False).astype(np.int64)
        )
        temporary_sample = sample_path.with_name(sample_path.name + ".tmp")
        with temporary_sample.open("wb") as handle:
            np.save(handle, sample_rows)
        temporary_sample.replace(sample_path)

    checkpoint_path = output_dir / "index_add_checkpoint.index"
    progress_path = output_dir / "index_progress.json"
    if checkpoint_path.exists() and progress_path.exists():
        cpu_index = faiss.read_index(str(checkpoint_path))
        progress = read_json(progress_path)
        if int(progress["rows_added"]) != int(cpu_index.ntotal):
            raise ValueError("Index checkpoint row count mismatch")
    else:
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            int(base["nlist"]),
            int(base["m"]),
            int(base["nbits"]),
            faiss.METRIC_INNER_PRODUCT,
        )
        cpu_index.cp.seed = int(base["training_seed"])
        cpu_index.cp.niter = int(base["coarse_kmeans_iterations"])
        cpu_index.pq.cp.seed = int(base["training_seed"])
        cpu_index.pq.cp.niter = int(base["pq_kmeans_iterations"])
        resources, gpu_index = clone_to_gpu(faiss, cpu_index)
        training_vectors = np.ascontiguousarray(
            embeddings[sample_rows].astype(np.float32)
        )
        gpu_index.train(training_vectors)
        del training_vectors
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        del gpu_index, resources
        temporary_index = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        faiss.write_index(cpu_index, str(temporary_index))
        temporary_index.replace(checkpoint_path)
        atomic_write_json(progress_path, {
            "protocol_id": PROTOCOL_ID,
            "rows_added": 0,
            "training_complete": True,
            "test_qrels_accessed": False,
        })

    resources, gpu_index = clone_to_gpu(faiss, cpu_index)
    start = int(cpu_index.ntotal)
    last_checkpoint = start
    for row_start in range(start, n_docs, add_batch_size):
        row_end = min(row_start + add_batch_size, n_docs)
        batch = np.ascontiguousarray(
            embeddings[row_start:row_end].astype(np.float32)
        )
        gpu_index.add(batch)
        if row_end - last_checkpoint >= checkpoint_rows or row_end == n_docs:
            cpu_checkpoint = faiss.index_gpu_to_cpu(gpu_index)
            temporary_index = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
            faiss.write_index(cpu_checkpoint, str(temporary_index))
            temporary_index.replace(checkpoint_path)
            atomic_write_json(progress_path, {
                "protocol_id": PROTOCOL_ID,
                "rows_added": int(cpu_checkpoint.ntotal),
                "training_complete": True,
                "test_qrels_accessed": False,
            })
            last_checkpoint = row_end
            print(f"indexed {row_end:,}/{n_docs:,} documents")
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    del gpu_index, resources
    if int(cpu_index.ntotal) != n_docs:
        raise ValueError(f"Index ntotal {cpu_index.ntotal} != {n_docs}")
    cpu_index.nprobe = int(base["nprobe"])
    if hasattr(cpu_index, "make_direct_map"):
        cpu_index.make_direct_map()
    temporary_final = final_path.with_name(final_path.name + ".tmp")
    faiss.write_index(cpu_index, str(temporary_final))
    temporary_final.replace(final_path)
    progress_path.unlink(missing_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "index": file_record(final_path),
        "training_sample_rows": file_record(sample_path),
        "document_embeddings": file_record(embeddings_path),
        "document_count": n_docs,
        "dimension": dim,
        "type": "IndexIVFPQ",
        "metric": "inner_product",
        "m": int(base["m"]),
        "nbits": int(base["nbits"]),
        "nlist": int(base["nlist"]),
        "nprobe": int(base["nprobe"]),
        "training_seed": int(base["training_seed"]),
        "training_document_sample_count": len(sample_rows),
        "coarse_kmeans_iterations": int(base["coarse_kmeans_iterations"]),
        "pq_kmeans_iterations": int(base["pq_kmeans_iterations"]),
        "build_backend": "single_faiss_gpu_nvidia_t4",
        "gpu_float16_lookup_tables": False,
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "test_qrels_accessed": False,
        "completed_utc": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    add_common_arguments(init)
    init.add_argument("--design-freeze-commit", required=True)

    for name in ("download", "extract-train-only", "scan-corpus"):
        sub = subparsers.add_parser(name)
        add_common_arguments(sub)

    encode = subparsers.add_parser("encode-corpus")
    add_common_arguments(encode)
    encode.add_argument("--batch-size", type=int, default=256)
    encode.add_argument("--checkpoint-rows", type=int, default=10_000)
    encode.add_argument("--device", choices=("cuda",), default="cuda")

    queries = subparsers.add_parser("encode-train-queries")
    add_common_arguments(queries)
    queries.add_argument("--batch-size", type=int, default=512)
    queries.add_argument("--device", choices=("cuda",), default="cuda")

    index = subparsers.add_parser("build-index")
    add_common_arguments(index)
    index.add_argument("--add-batch-size", type=int, default=50_000)
    index.add_argument("--checkpoint-rows", type=int, default=250_000)
    index.add_argument("--backend", choices=("gpu",), default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        result = initialize_gate(
            args.artifact_root,
            args.repo,
            args.protocol,
            args.design_freeze_commit,
        )
    else:
        verify_gate(args.artifact_root, args.repo, args.protocol)
        if args.command == "download":
            result = download_archives(args.artifact_root, args.protocol)
        elif args.command == "extract-train-only":
            result = extract_train_only(args.artifact_root)
        elif args.command == "scan-corpus":
            corpus = args.artifact_root / "stage1" / "data" / "nq" / "corpus.jsonl"
            output = args.artifact_root / "stage1" / "corpus" / "corpus_scan_manifest.json"
            result = scan_corpus(corpus, output)
        elif args.command == "encode-corpus":
            result = encode_corpus(
                args.artifact_root,
                args.protocol,
                batch_size=args.batch_size,
                checkpoint_rows=args.checkpoint_rows,
                device=args.device,
            )
        elif args.command == "encode-train-queries":
            result = encode_train_queries(
                args.artifact_root,
                args.protocol,
                batch_size=args.batch_size,
                device=args.device,
            )
        elif args.command == "build-index":
            result = build_index(
                args.artifact_root,
                args.protocol,
                add_batch_size=args.add_batch_size,
                checkpoint_rows=args.checkpoint_rows,
                backend=args.backend,
            )
        else:  # pragma: no cover - argparse enforces choices
            raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
