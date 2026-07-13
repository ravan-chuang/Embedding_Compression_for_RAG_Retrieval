#!/usr/bin/env python3
"""Audit and one-shot evaluate the frozen BEIR NQ confirmation package.

This file is frozen and hashed before NQ test qrels are opened.  Its ``audit``
command verifies the Stage-2 Git commit and every pre-qrels artifact, extracts
the official test-qrels member for the first time, checks full corpus coverage,
and freezes identity-only exclusions.  Its ``evaluate`` command then runs Base,
PCA, and RARS together and performs the registered paired bootstrap once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols" / "beir_nq_rars_pca_confirmation_v1.json"
DEFAULT_MANIFEST = ROOT / "protocols" / "beir_nq_pre_qrels_manifest.json"
PROTOCOL_ID = "beir_nq_rars_pca_confirmation_v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
METRICS = ("recall_at_10", "success_at_10", "mrr_at_10", "ndcg_at_10")
SYSTEMS = ("base_m32", "pca_r16_int8", "rars_r16_int8")


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


def run_git_bytes(repo: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Git verification failed: {detail}") from exc


def verify_method_freeze(
    repo: Path,
    commit: str,
    pre_qrels_manifest: Path,
) -> None:
    commit = commit.strip().casefold()
    if not COMMIT_RE.fullmatch(commit):
        raise ValueError("--method-freeze-commit must be a full 40-hex commit")
    head = run_git_bytes(repo, "rev-parse", "HEAD").decode().strip().casefold()
    if head != commit:
        raise ValueError(f"HEAD {head} does not equal method freeze {commit}")
    if run_git_bytes(repo, "status", "--porcelain").strip():
        raise ValueError("Method-freeze checkout must be clean before test access")
    try:
        manifest_relative = pre_qrels_manifest.resolve().relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError("Pre-qrels manifest must be inside the frozen repository") from exc
    frozen_manifest = run_git_bytes(repo, "show", f"{commit}:{manifest_relative.as_posix()}")
    if hashlib.sha256(frozen_manifest).hexdigest() != sha256_file(pre_qrels_manifest):
        raise ValueError("Working pre-qrels manifest differs from the frozen commit")
    evaluator_relative = Path(__file__).resolve().relative_to(repo.resolve())
    frozen_evaluator = run_git_bytes(repo, "show", f"{commit}:{evaluator_relative.as_posix()}")
    if hashlib.sha256(frozen_evaluator).hexdigest() != sha256_file(Path(__file__)):
        raise ValueError("Working evaluator differs from the frozen commit")


def verify_audit_freeze(
    repo: Path,
    audit_freeze_commit: str,
    method_freeze_commit: str,
    pre_qrels_manifest: Path,
    audit_path: Path,
) -> None:
    audit_freeze_commit = audit_freeze_commit.strip().casefold()
    if not COMMIT_RE.fullmatch(audit_freeze_commit):
        raise ValueError("--audit-freeze-commit must be a full 40-hex commit")
    head = run_git_bytes(repo, "rev-parse", "HEAD").decode().strip().casefold()
    if head != audit_freeze_commit:
        raise ValueError(f"HEAD {head} does not equal audit freeze {audit_freeze_commit}")
    if run_git_bytes(repo, "status", "--porcelain").strip():
        raise ValueError("Audit-freeze checkout must be clean before retrieval")
    try:
        subprocess.check_call(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                method_freeze_commit,
                audit_freeze_commit,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Method freeze is not an ancestor of the audit freeze") from exc
    for path, label in [
        (pre_qrels_manifest, "pre-qrels manifest"),
        (Path(__file__), "evaluator"),
        (audit_path, "identity/coverage audit"),
    ]:
        try:
            relative = path.resolve().relative_to(repo.resolve())
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the frozen repository") from exc
        frozen = run_git_bytes(repo, "show", f"{audit_freeze_commit}:{relative.as_posix()}")
        if hashlib.sha256(frozen).hexdigest() != sha256_file(path):
            raise ValueError(f"Working {label} differs from the audit-freeze commit")


def load_validator() -> Any:
    path = ROOT / "scripts" / "validate_nq_pre_qrels_freeze.py"
    spec = importlib.util.spec_from_file_location("nq_freeze_validator", path)
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load pre-qrels validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_pre_qrels_manifest(
    manifest_path: Path,
    protocol_path: Path,
    artifact_root: Path,
    repo: Path,
) -> tuple[dict[str, Any], Any]:
    validator = load_validator()
    manifest = read_json(manifest_path)
    protocol = read_json(protocol_path)
    validator.validate_manifest(
        manifest,
        protocol=protocol,
        protocol_path=protocol_path,
        artifact_root=artifact_root,
        repo_root=repo,
        verify_files=True,
    )
    return manifest, validator


def resolve_entry(
    manifest: dict[str, Any],
    label: str,
    validator: Any,
    artifact_root: Path,
    repo: Path,
) -> Path:
    return validator.resolve_artifact_path(
        ROOT,
        manifest["files"][label]["path"],
        artifact_root=artifact_root,
        repo_root=repo,
    )


def normalize_query_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def safe_zip_member(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member: {value}")
    return path


def extract_test_qrels(archive_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    matches = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = safe_zip_member(info.filename)
            if tuple(path.parts[-2:]) == ("qrels", "test.tsv"):
                matches.append(info)
        if len(matches) != 1:
            raise ValueError(f"Expected one qrels/test.tsv member, found {len(matches)}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".part")
        with archive.open(matches[0], "r") as source, temporary.open("wb") as output:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(output_path)
    return output_path


def load_positive_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if line_number == 1 and parts[0].casefold() in {
                "query-id",
                "query_id",
                "qid",
            }:
                continue
            if len(parts) < 3:
                raise ValueError(f"Malformed qrels row {line_number}")
            qid, doc_id, relevance = parts[0], parts[1], parts[-1]
            if float(relevance) > 0:
                qrels.setdefault(str(qid), set()).add(str(doc_id))
    if not qrels:
        raise ValueError("No positive NQ test qrels found")
    return qrels


def load_query_texts(path: Path, wanted_qids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            item = json.loads(raw)
            qid = str(item.get("_id", item.get("qid", item.get("query_id", ""))))
            if qid not in wanted_qids:
                continue
            text = str(item.get("text", item.get("query", ""))).strip()
            if not text:
                raise ValueError(f"Empty test-query text for {qid}")
            if qid in result:
                raise ValueError(f"Duplicate query ID in queries.jsonl: {qid}")
            result[qid] = text
    missing = sorted(wanted_qids - set(result))
    if missing:
        raise ValueError(f"Missing {len(missing)} test query texts; examples={missing[:5]}")
    return result


def resolve_portable(value: str, artifact_root: Path, repo: Path) -> Path:
    if value.startswith("artifact://"):
        return artifact_root / value.removeprefix("artifact://")
    if value.startswith("repo://"):
        return repo / value.removeprefix("repo://")
    path = Path(value)
    return path if path.is_absolute() else repo / path


def load_prior_registry(
    path: Path,
    artifact_root: Path,
    repo: Path,
) -> tuple[set[str], set[str], dict[str, Any]]:
    registry = read_json(path)
    if registry.get("registry_id") != "beir_nq_prior_query_registry_v1":
        raise ValueError("Unexpected prior-query registry")
    if registry.get("status") != "frozen_before_nq_test_qrels_access":
        raise ValueError("Prior-query registry is not frozen")
    if registry.get("nq_test_qrels_accessed") is not False:
        raise ValueError("Prior-query registry has an unsafe test flag")
    nq_ids: set[str] = set()
    normalized_texts: set[str] = set()
    for source in registry.get("sources", []):
        source_path = resolve_portable(source["path"], artifact_root, repo)
        if source_path.stat().st_size != int(source["bytes"]):
            raise ValueError(f"Prior-query source byte mismatch: {source_path}")
        if sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Prior-query source hash mismatch: {source_path}")
        payload = read_json(source_path)
        qids = [str(value) for value in payload[source["query_ids_field"]]]
        text_field = source.get("query_texts_field")
        texts = payload[text_field] if text_field else [None] * len(qids)
        if len(qids) != len(texts):
            raise ValueError(f"Prior-query ID/text mismatch: {source_path}")
        if source["dataset_namespace"] == "beir/nq":
            nq_ids.update(qids)
        for text in texts:
            if text is not None:
                normalized_texts.add(normalize_query_text(str(text)))
    return nq_ids, normalized_texts, registry


def verify_qrel_corpus_coverage(
    positive_doc_ids: set[str],
    doc_ids_path: Path,
    *,
    n_docs: int,
    width: int,
    batch_size: int = 250_000,
) -> None:
    remaining = {value.encode("utf-8") for value in positive_doc_ids}
    doc_ids = np.memmap(
        doc_ids_path,
        dtype=f"S{width}",
        mode="r",
        shape=(n_docs,),
    )
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        if not remaining:
            break
        present = set(np.asarray(doc_ids[start:end]).tolist())
        remaining.difference_update(present)
    if remaining:
        examples = [value.decode("utf-8", errors="replace") for value in list(remaining)[:5]]
        raise ValueError(
            f"{len(remaining)} positive qrel documents are absent; examples={examples}"
        )


def run_audit(
    artifact_root: Path,
    repo: Path,
    protocol_path: Path,
    pre_qrels_manifest_path: Path,
    method_freeze_commit: str,
    output_dir: Path,
) -> dict[str, Any]:
    verify_method_freeze(repo, method_freeze_commit, pre_qrels_manifest_path)
    manifest, validator = verify_pre_qrels_manifest(
        pre_qrels_manifest_path,
        protocol_path,
        artifact_root,
        repo,
    )
    protocol = read_json(protocol_path)
    archive = resolve_entry(manifest, "dataset_archive", validator, artifact_root, repo)
    registry_path = resolve_entry(
        manifest,
        "prior_query_registry",
        validator,
        artifact_root,
        repo,
    )
    qrels_path = extract_test_qrels(archive, output_dir / "qrels" / "test.tsv")
    qrels = load_positive_qrels(qrels_path)
    query_path = artifact_root / "stage1" / "data" / "nq" / "queries.jsonl"
    query_texts = load_query_texts(query_path, set(qrels))
    prior_nq_ids, prior_texts, registry = load_prior_registry(
        registry_path,
        artifact_root,
        repo,
    )

    corpus_manifest_path = resolve_entry(
        manifest,
        "corpus_manifest",
        validator,
        artifact_root,
        repo,
    )
    corpus_manifest = read_json(corpus_manifest_path)
    doc_ids_path = resolve_entry(manifest, "doc_ids", validator, artifact_root, repo)
    positive_docs = {doc_id for values in qrels.values() for doc_id in values}
    verify_qrel_corpus_coverage(
        positive_docs,
        doc_ids_path,
        n_docs=int(corpus_manifest["document_count"]),
        width=int(corpus_manifest["doc_id_width_bytes"]),
    )

    exclusions = []
    eligible = []
    for qid in sorted(qrels):
        reasons = []
        if qid in prior_nq_ids:
            reasons.append("same_namespace_query_id_overlap")
        normalized = normalize_query_text(query_texts[qid])
        if normalized in prior_texts:
            reasons.append("normalized_query_text_overlap")
        if reasons:
            exclusions.append({"query_id": qid, "reasons": reasons})
        else:
            eligible.append(qid)
    minimum = int(protocol["query_usage"]["minimum_final_test_queries"])
    if len(eligible) < minimum:
        raise ValueError(f"Only {len(eligible)} eligible queries remain; minimum={minimum}")
    audit = {
        "protocol_id": PROTOCOL_ID,
        "status": "eligible_test_queries_frozen_before_retrieval",
        "method_freeze_commit": method_freeze_commit.casefold(),
        "pre_qrels_manifest_sha256": sha256_file(pre_qrels_manifest_path),
        "test_qrels": {
            "path": str(qrels_path),
            "bytes": qrels_path.stat().st_size,
            "sha256": sha256_file(qrels_path),
            "positive_query_count": len(qrels),
            "positive_document_count": len(positive_docs),
            "all_positive_documents_in_corpus": True,
        },
        "prior_query_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "source_count": len(registry["sources"]),
        },
        "official_positive_query_count": len(qrels),
        "excluded_query_count": len(exclusions),
        "eligible_query_count": len(eligible),
        "minimum_eligible_query_count": minimum,
        "exclusions": exclusions,
        "eligible_query_ids": eligible,
        "eligible_query_texts": [query_texts[qid] for qid in eligible],
        "test_retrieval_performed": False,
        "test_outcomes_observed_beyond_identity_and_coverage": False,
        "created_utc": utc_now(),
    }
    output_path = output_dir / "eligible_test_query_audit.json"
    atomic_write_json(output_path, audit)
    return audit


def import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install faiss-gpu-cu12 in the Colab runtime") from exc
    return faiss


def assert_t4() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU runtime")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name.upper():
        raise RuntimeError(f"Frozen protocol requires an NVIDIA T4; found {name}")


def clone_to_gpu(faiss: Any, cpu_index: Any) -> tuple[Any, Any]:
    resources = faiss.StandardGpuResources()
    if hasattr(resources, "setTempMemory"):
        resources.setTempMemory(512 * 1024 * 1024)
    options = faiss.GpuClonerOptions()
    options.useFloat16LookupTables = False
    options.useFloat16CoarseQuantizer = False
    return resources, faiss.index_cpu_to_gpu(resources, 0, cpu_index, options)


def verify_model_snapshot(manifest_path: Path) -> Path:
    manifest = read_json(manifest_path)
    root = Path(manifest["snapshot_directory"])
    for item in manifest["files"]:
        path = root / item["path"]
        if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Embedding model snapshot drift: {path}")
    return root


def encode_test_queries(model_root: Path, texts: list[str], output_path: Path) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_root), device="cuda")
    vectors = model.encode(
        texts,
        batch_size=512,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid test-query vector shape: {vectors.shape}")
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, vectors)
    temporary.replace(output_path)
    return vectors


def search_candidates(index_path: Path, queries: np.ndarray, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows_path = output_dir / "ann_rows.int64.npy"
    scores_path = output_dir / "ann_scores.float32.npy"
    if rows_path.exists() and scores_path.exists():
        return np.load(rows_path), np.load(scores_path)
    faiss = import_faiss()
    cpu_index = faiss.read_index(str(index_path))
    cpu_index.nprobe = 32
    resources, gpu_index = clone_to_gpu(faiss, cpu_index)
    gpu_index.nprobe = 32
    score_parts = []
    row_parts = []
    for start in range(0, len(queries), 256):
        end = min(start + 256, len(queries))
        scores, rows = gpu_index.search(
            np.ascontiguousarray(queries[start:end], dtype=np.float32),
            100,
        )
        score_parts.append(scores.astype(np.float32))
        row_parts.append(rows.astype(np.int64))
    del gpu_index, resources
    ann_scores = np.vstack(score_parts)
    ann_rows = np.vstack(row_parts)
    for path, value in [(rows_path, ann_rows), (scores_path, ann_scores)]:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value)
        temporary.replace(path)
    return ann_rows, ann_scores


def corrected_ranking_rows(
    queries: np.ndarray,
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.memmap,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    ids = ann_rows[:, :top_b]
    valid = ids >= 0
    safe_ids = np.where(valid, ids, 0)
    coefficients = codes[safe_ids].astype(np.float32) * scales[None, None, :]
    correction = np.einsum(
        "qbr,qr->qb",
        coefficients,
        queries @ basis,
        optimize=True,
    ).astype(np.float32)
    correction[~valid] = 0.0
    corrected = ann_scores.copy()
    corrected[:, :top_b] += float(alpha) * correction
    order = np.argsort(-corrected, axis=1, kind="stable")
    return np.take_along_axis(ann_rows, order, axis=1)


def decode_doc_id(value: np.bytes_) -> str:
    return bytes(value).rstrip(b"\x00").decode("utf-8")


def per_query_metrics(ranked_doc_ids: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    ranked = ranked_doc_ids[:k]
    hits = [1 if doc_id in relevant else 0 for doc_id in ranked]
    hit_count = sum(hits)
    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "recall_at_10": hit_count / len(relevant),
        "success_at_10": float(hit_count > 0),
        "mrr_at_10": reciprocal_rank,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
    }


def evaluate_rankings(
    rankings: dict[str, np.ndarray],
    qids: list[str],
    qrels: dict[str, set[str]],
    doc_ids: np.memmap,
) -> dict[str, dict[str, np.ndarray]]:
    result = {
        system: {metric: np.empty(len(qids), dtype=np.float64) for metric in METRICS}
        for system in rankings
    }
    for query_index, qid in enumerate(qids):
        for system, rows in rankings.items():
            ranked_ids = [
                decode_doc_id(doc_ids[int(row)])
                for row in rows[query_index, :10]
                if row >= 0
            ]
            values = per_query_metrics(ranked_ids, qrels[qid])
            for metric, value in values.items():
                result[system][metric][query_index] = value
    return result


def paired_bootstrap(
    values: dict[str, dict[str, np.ndarray]],
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 256,
) -> list[dict[str, Any]]:
    contrasts = (
        ("rars_minus_pca", "rars_r16_int8", "pca_r16_int8"),
        ("pca_minus_base", "pca_r16_int8", "base_m32"),
        ("rars_minus_base", "rars_r16_int8", "base_m32"),
    )
    differences = {
        (name, metric): values[left][metric] - values[right][metric]
        for name, left, right in contrasts
        for metric in METRICS
    }
    samples = {
        key: np.empty(replicates, dtype=np.float64) for key in differences
    }
    rng = np.random.default_rng(seed)
    n_queries = len(next(iter(differences.values())))
    written = 0
    while written < replicates:
        count = min(chunk_size, replicates - written)
        indices = rng.integers(0, n_queries, size=(count, n_queries))
        for key, difference in differences.items():
            samples[key][written:written + count] = difference[indices].mean(axis=1)
        written += count
    rows = []
    for (contrast, metric), distribution in samples.items():
        difference = differences[(contrast, metric)]
        lower, upper = np.percentile(distribution, [2.5, 97.5])
        rows.append({
            "contrast": contrast,
            "metric": metric,
            "mean_difference": float(difference.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        })
    return rows


def write_per_query_csv(
    path: Path,
    qids: list[str],
    values: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = ["query_id"] + [
        f"{system}_{metric}" for system in SYSTEMS for metric in METRICS
    ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, qid in enumerate(qids):
            row: dict[str, Any] = {"query_id": qid}
            for system in SYSTEMS:
                for metric in METRICS:
                    row[f"{system}_{metric}"] = float(values[system][metric][index])
            writer.writerow(row)
    temporary.replace(path)


def run_evaluation(
    artifact_root: Path,
    repo: Path,
    protocol_path: Path,
    pre_qrels_manifest_path: Path,
    method_freeze_commit: str,
    audit_freeze_commit: str,
    audit_path: Path,
    output_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    assert_t4()
    verify_audit_freeze(
        repo,
        audit_freeze_commit,
        method_freeze_commit,
        pre_qrels_manifest_path,
        audit_path,
    )
    manifest, validator = verify_pre_qrels_manifest(
        pre_qrels_manifest_path,
        protocol_path,
        artifact_root,
        repo,
    )
    protocol = read_json(protocol_path)
    audit = read_json(audit_path)
    if audit.get("status") != "eligible_test_queries_frozen_before_retrieval":
        raise ValueError("Identity/coverage audit is not frozen")
    if audit.get("method_freeze_commit") != method_freeze_commit.casefold():
        raise ValueError("Audit method-freeze commit mismatch")
    qrels_path = Path(audit["test_qrels"]["path"])
    if sha256_file(qrels_path) != audit["test_qrels"]["sha256"]:
        raise ValueError("Test qrels changed after identity audit")

    output_dir.mkdir(parents=True, exist_ok=True)
    started_path = output_dir / "evaluation_started.json"
    complete_path = output_dir / "evaluation_complete.json"
    input_digest = hashlib.sha256(
        (
            sha256_file(pre_qrels_manifest_path)
            + sha256_file(audit_path)
            + method_freeze_commit.casefold()
        ).encode("ascii")
    ).hexdigest()
    if complete_path.exists():
        raise ValueError("The registered one-shot evaluation is already complete")
    if started_path.exists():
        started = read_json(started_path)
        if not resume or started.get("input_digest") != input_digest:
            raise ValueError("Evaluation already started; use --resume with identical inputs")
    else:
        atomic_write_json(started_path, {
            "protocol_id": PROTOCOL_ID,
            "input_digest": input_digest,
            "method_freeze_commit": method_freeze_commit.casefold(),
            "started_utc": utc_now(),
            "one_shot": True,
        })

    qids = [str(value) for value in audit["eligible_query_ids"]]
    texts = [str(value) for value in audit["eligible_query_texts"]]
    if len(qids) < int(protocol["query_usage"]["minimum_final_test_queries"]):
        raise ValueError("Eligible query count fell below the frozen minimum")
    model_manifest_path = resolve_entry(
        manifest,
        "embedding_model_snapshot_manifest",
        validator,
        artifact_root,
        repo,
    )
    model_root = verify_model_snapshot(model_manifest_path)
    queries = encode_test_queries(
        model_root,
        texts,
        output_dir / "test_query_vectors.float32.npy",
    )
    index_path = resolve_entry(manifest, "frozen_index", validator, artifact_root, repo)
    ann_rows, ann_scores = search_candidates(index_path, queries, output_dir)

    corpus_manifest = read_json(
        resolve_entry(manifest, "corpus_manifest", validator, artifact_root, repo)
    )
    n_docs = int(corpus_manifest["document_count"])
    rank = int(protocol["shared_sidecar"]["rank"])
    rankings = {"base_m32": ann_rows}
    for short_name, method_id in [("pca", "pca_r16_int8"), ("rars", "rars_r16_int8")]:
        config = read_json(resolve_entry(
            manifest,
            f"{short_name}_config",
            validator,
            artifact_root,
            repo,
        ))
        if config.get("method_id") != method_id:
            raise ValueError(f"Unexpected {short_name} config")
        basis = np.load(resolve_entry(
            manifest, f"{short_name}_basis", validator, artifact_root, repo
        )).astype(np.float32)
        scales = np.load(resolve_entry(
            manifest, f"{short_name}_scales", validator, artifact_root, repo
        )).astype(np.float32)
        codes = np.memmap(
            resolve_entry(
                manifest, f"{short_name}_codes", validator, artifact_root, repo
            ),
            dtype=np.int8,
            mode="r",
            shape=(n_docs, rank),
        )
        rankings[method_id] = corrected_ranking_rows(
            queries,
            ann_rows,
            ann_scores,
            basis,
            scales,
            codes,
            alpha=float(config["alpha"]),
            top_b=int(config["top_b"]),
        )

    qrels_all = load_positive_qrels(qrels_path)
    qrels = {qid: qrels_all[qid] for qid in qids}
    doc_ids = np.memmap(
        resolve_entry(manifest, "doc_ids", validator, artifact_root, repo),
        dtype=f"S{int(corpus_manifest['doc_id_width_bytes'])}",
        mode="r",
        shape=(n_docs,),
    )
    values = evaluate_rankings(rankings, qids, qrels, doc_ids)
    per_query_path = output_dir / "per_query_metrics.csv"
    write_per_query_csv(per_query_path, qids, values)
    bootstrap = paired_bootstrap(
        values,
        replicates=int(protocol["evaluation"]["bootstrap_replicates"]),
        seed=int(protocol["evaluation"]["bootstrap_seed"]),
    )
    systems = {
        system: {
            metric: float(values[system][metric].mean()) for metric in METRICS
        }
        for system in SYSTEMS
    }
    primary = next(
        row for row in bootstrap
        if row["contrast"] == "rars_minus_pca" and row["metric"] == "recall_at_10"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "one_shot_evaluation_complete",
        "method_freeze_commit": method_freeze_commit.casefold(),
        "audit_freeze_commit": audit_freeze_commit.casefold(),
        "eligible_query_count": len(qids),
        "systems": systems,
        "paired_bootstrap": bootstrap,
        "primary_contrast": primary,
        "primary_supported": bool(primary["ci_lower"] > 0),
        "files": {
            "per_query_metrics": {
                "path": str(per_query_path),
                "bytes": per_query_path.stat().st_size,
                "sha256": sha256_file(per_query_path),
            },
            "test_query_vectors": {
                "path": str(output_dir / "test_query_vectors.float32.npy"),
                "bytes": (output_dir / "test_query_vectors.float32.npy").stat().st_size,
                "sha256": sha256_file(output_dir / "test_query_vectors.float32.npy"),
            },
            "ann_rows": {
                "path": str(output_dir / "ann_rows.int64.npy"),
                "bytes": (output_dir / "ann_rows.int64.npy").stat().st_size,
                "sha256": sha256_file(output_dir / "ann_rows.int64.npy"),
            },
            "ann_scores": {
                "path": str(output_dir / "ann_scores.float32.npy"),
                "bytes": (output_dir / "ann_scores.float32.npy").stat().st_size,
                "sha256": sha256_file(output_dir / "ann_scores.float32.npy"),
            },
        },
        "completed_utc": utc_now(),
        "retuning_performed": False,
    }
    atomic_write_json(output_dir / "metrics_summary.json", result)
    completion = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_stop_no_retuning",
        "input_digest": input_digest,
        "metrics_summary_sha256": sha256_file(output_dir / "metrics_summary.json"),
        "completed_utc": utc_now(),
    }
    atomic_write_json(complete_path, completion)
    return result


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--pre-qrels-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--method-freeze-commit", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    common_arguments(audit)
    audit.add_argument("--output-dir", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate")
    common_arguments(evaluate)
    evaluate.add_argument("--audit", required=True, type=Path)
    evaluate.add_argument("--audit-freeze-commit", required=True)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        result = run_audit(
            args.artifact_root,
            args.repo,
            args.protocol,
            args.pre_qrels_manifest,
            args.method_freeze_commit,
            args.output_dir,
        )
    else:
        result = run_evaluation(
            args.artifact_root,
            args.repo,
            args.protocol,
            args.pre_qrels_manifest,
            args.method_freeze_commit,
            args.audit_freeze_commit,
            args.audit,
            args.output_dir,
            resume=args.resume,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
