#!/usr/bin/env python3
"""Prepare the already-opened full BEIR NQ setting for RARS-v17.

The command verifies the frozen Stage-1 corpus/index artifacts and the opened
Stage-3 audit/evaluation lineage, then deterministically repartitions the
eligible NQ test queries into V17 fit/evaluation development roles.  It maps
public string document identifiers to frozen-index rows without computing a
retrieval metric or fitting a sidecar basis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v17_million_scale_setting_transfer_v1"
SOURCE_PROTOCOL_ID = "beir_nq_rars_pca_confirmation_v1"
SETTING_ID = "beir_nq_2_68m_bge_opened_test_diagnostic"
STATUS = "RARS_V17_NQ_ROLES_PREPARED"
CANONICAL_PROTOCOL = Path(
    "protocols/rars_v17_million_scale_setting_transfer_v1.json"
)
ROLE_NAMESPACE = b"rars-v17-nq-role-v1\0"
MINIMUM_DOCUMENT_COUNT = 2_000_000
DIMENSION = 384
PRE_QRELS_MANIFEST = Path("stage2/pre_qrels_manifest.json")
ELIGIBLE_QUERY_AUDIT = Path(
    "stage3/audit/eligible_test_query_audit.json"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


def _exact_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def verify_source(
    repo_root: Path,
    protocol_path: Path,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_commit(source_commit)
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must use canonical path: {canonical}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or dirty:
        raise ValueError("V17 NQ preparation requires a clean exact checkout")
    protocol = read_json(canonical)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status")
        != "FROZEN_BEFORE_FIRST_V17_MILLION_SCALE_DIAGNOSTIC_RUN"
        or protocol.get("method_revision_allowed") is not False
    ):
        raise ValueError("Unexpected or mutable V17 protocol")
    sources = {
        str(relative): file_record((repo_root / relative).resolve(strict=True))
        for relative in (
            CANONICAL_PROTOCOL,
            Path("scripts/prepare_rars_v17_nq_roles.py"),
        )
    }
    return protocol, sources


def _resolve_record_path(record: dict[str, Any], artifact_root: Path) -> Path:
    raw = str(record.get("path", ""))
    if raw.startswith("artifact://"):
        return (artifact_root / raw.removeprefix("artifact://")).resolve(
            strict=True
        )
    path = Path(raw)
    if not path.is_absolute():
        path = artifact_root / path
    return path.resolve(strict=True)


def verify_record(
    record: dict[str, Any],
    artifact_root: Path,
    label: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Missing registered record for {label}")
    path = _resolve_record_path(record, artifact_root)
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise ValueError(f"Registered {label} path changed: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Registered {label} hash changed")
    return path


def _verify_pre_qrels_doc_id_reconciliation(
    artifact_root: Path,
    *,
    corpus_manifest_path: Path,
    corpus_doc_ids_record: dict[str, Any],
    expected_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a known Stage-1/pre-qrels document-ID lineage conflict.

    The historical NQ packet may contain a stale document-ID digest inside
    the Stage-1 corpus manifest.  The later pre-qrels manifest independently
    hashed the exact file before test-qrels access, and the frozen query audit
    binds that manifest into the one-shot evaluation.  A fallback is safe only
    when all three links verify without changing any historical artifact.
    """

    pre_qrels_path = (
        artifact_root / PRE_QRELS_MANIFEST
    ).resolve(strict=True)
    pre_qrels = read_json(pre_qrels_path)
    if (
        pre_qrels.get("protocol_id") != SOURCE_PROTOCOL_ID
        or pre_qrels.get("status") != "frozen_before_test_qrels_access"
    ):
        raise ValueError("Unexpected NQ pre-qrels freeze manifest")
    for key in (
        "test_qrels_accessed",
        "test_retrieval_performed",
        "test_outcomes_observed",
        "train_qrels_relevance_values_used",
    ):
        if pre_qrels.get(key) is not False:
            raise ValueError(f"Unsafe NQ pre-qrels flag: {key}")

    files = pre_qrels.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("NQ pre-qrels freeze has no file registry")
    verify_record(
        files.get("corpus_manifest", {}),
        artifact_root,
        "pre-qrels corpus manifest",
        expected_path=corpus_manifest_path,
    )
    frozen_doc_ids_record = files.get("doc_ids", {})
    doc_ids = verify_record(
        frozen_doc_ids_record,
        artifact_root,
        "pre-qrels document IDs",
    )
    stage1_doc_ids = _resolve_record_path(
        corpus_doc_ids_record,
        artifact_root,
    )
    if doc_ids != stage1_doc_ids:
        raise ValueError(
            "Stage-1 and pre-qrels document-ID paths do not identify "
            "the same frozen file"
        )
    if (
        doc_ids.stat().st_size != expected_bytes
        or int(frozen_doc_ids_record.get("bytes", -1)) != expected_bytes
    ):
        raise ValueError(
            "Pre-qrels document-ID byte count violates the corpus contract"
        )

    audit_path = (
        artifact_root / ELIGIBLE_QUERY_AUDIT
    ).resolve(strict=True)
    audit = read_json(audit_path)
    if (
        audit.get("protocol_id") != SOURCE_PROTOCOL_ID
        or audit.get("status")
        != "eligible_test_queries_frozen_before_retrieval"
        or audit.get("pre_qrels_manifest_sha256")
        != sha256_file(pre_qrels_path)
    ):
        raise ValueError(
            "Opened NQ query audit does not bind the pre-qrels manifest"
        )

    return doc_ids, {
        "status": (
            "PRE_QRELS_DOC_IDS_VERIFIED_AFTER_STAGE1_HASH_MISMATCH"
        ),
        "reason": (
            "The Stage-1 corpus manifest carries an earlier document-ID "
            "digest, while the later pre-qrels freeze independently "
            "registered the exact file used by the one-shot evaluation."
        ),
        "historical_artifacts_modified": False,
        "stage1_registered_doc_ids": corpus_doc_ids_record,
        "authoritative_pre_qrels_doc_ids": frozen_doc_ids_record,
        "pre_qrels_manifest": file_record(pre_qrels_path),
        "binding_query_audit": file_record(audit_path),
    }


def deterministic_roles(query_ids: list[str]) -> dict[str, list[str]]:
    """Apply the preregistered SHA-256-ranked 60/40 NQ role split."""

    normalized = [str(value) for value in query_ids]
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("Eligible NQ query IDs must be non-empty and unique")
    ranked = sorted(
        normalized,
        key=lambda qid: (
            hashlib.sha256(ROLE_NAMESPACE + qid.encode("utf-8")).digest(),
            qid,
        ),
    )
    fit_count = 3 * len(ranked) // 5
    return {
        "fit": ranked[:fit_count],
        "evaluation": ranked[fit_count:],
    }


def load_positive_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", "\t").split()
            if line_number == 1 and fields[0].casefold() in {
                "query-id",
                "query_id",
                "qid",
            }:
                continue
            if len(fields) < 3:
                raise ValueError(f"Malformed qrels row {line_number}")
            query_id, document_id, relevance = (
                str(fields[0]),
                str(fields[1]),
                fields[-1],
            )
            if float(relevance) > 0:
                qrels.setdefault(query_id, set()).add(document_id)
    if not qrels:
        raise ValueError("No positive NQ qrels were found")
    return qrels


def map_document_rows(
    doc_ids_path: Path,
    *,
    document_count: int,
    width: int,
    required_ids: set[str],
    batch_size: int = 250_000,
) -> dict[str, int]:
    """Map required UTF-8 document IDs to frozen corpus rows in bounded RAM."""

    if width <= 0:
        raise ValueError("Document ID width must be positive")
    encoded: dict[bytes, str] = {}
    for document_id in required_ids:
        raw = document_id.encode("utf-8")
        if not raw or len(raw) > width:
            raise ValueError(
                f"Qrel document ID does not fit the frozen width: {document_id!r}"
            )
        encoded[raw] = document_id
    values = np.memmap(
        doc_ids_path,
        mode="r",
        dtype=f"S{width}",
        shape=(document_count,),
    )
    rows: dict[str, int] = {}
    targets = np.asarray(sorted(encoded), dtype=f"S{width}")
    for start in range(0, document_count, batch_size):
        end = min(start + batch_size, document_count)
        batch = np.asarray(values[start:end])
        positions = np.flatnonzero(np.isin(batch, targets))
        for position in positions:
            raw = bytes(batch[int(position)]).rstrip(b"\x00")
            document_id = encoded.get(raw)
            if document_id is None:
                continue
            row = start + int(position)
            if document_id in rows:
                raise ValueError(
                    f"Frozen corpus contains duplicate document ID {document_id!r}"
                )
            rows[document_id] = row
        if len(rows) == len(encoded):
            break
    missing = sorted(required_ids - set(rows))
    if missing:
        raise ValueError(
            f"{len(missing)} positive qrel documents are absent from the "
            f"frozen corpus; examples={missing[:5]}"
        )
    return rows


def _verify_qid_alignment(path: Path, eligible_qids: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "query_id" not in reader.fieldnames:
            raise ValueError("Stage-3 per-query artifact lacks query_id")
        observed = [str(row["query_id"]) for row in reader]
    if observed != eligible_qids:
        raise ValueError(
            "Stage-3 evaluation query order is not aligned with the audit qids"
        )


def _validate_stage1(
    artifact_root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    corpus_manifest_path = (
        artifact_root / "stage1/corpus/corpus_artifacts_manifest.json"
    ).resolve(strict=True)
    index_manifest_path = (
        artifact_root / "stage1/index/index_manifest.json"
    ).resolve(strict=True)
    corpus = read_json(corpus_manifest_path)
    index = read_json(index_manifest_path)
    for payload, label in ((corpus, "corpus"), (index, "index")):
        if payload.get("protocol_id") != SOURCE_PROTOCOL_ID:
            raise ValueError(f"Unexpected Stage-1 {label} protocol")
        if payload.get("test_qrels_accessed") is not False:
            raise ValueError(f"Unsafe Stage-1 {label} test-qrels flag")

    setting_contract = protocol["index_policy"][SETTING_ID]
    document_count = int(corpus.get("document_count", -1))
    minimum = max(
        MINIMUM_DOCUMENT_COUNT,
        int(setting_contract["minimum_documents"]),
    )
    if document_count < minimum:
        raise ValueError(
            f"Full BEIR NQ must contain at least {minimum:,} documents"
        )
    if (
        int(corpus.get("dimension", -1)) != DIMENSION
        or corpus.get("embedding_dtype") != "float16"
    ):
        raise ValueError("Unexpected Stage-1 corpus dimension or dtype")
    width = int(corpus.get("doc_id_width_bytes", -1))
    if corpus.get("doc_id_dtype") != f"S{width}":
        raise ValueError("Stage-1 document-ID dtype/width mismatch")

    embeddings = verify_record(
        corpus.get("document_embeddings", {}),
        artifact_root,
        "document embeddings",
    )
    expected_embedding_bytes = document_count * DIMENSION * np.dtype(
        np.float16
    ).itemsize
    expected_doc_id_bytes = document_count * width
    corpus_doc_ids_record = corpus.get("doc_ids", {})
    try:
        doc_ids = verify_record(
            corpus_doc_ids_record,
            artifact_root,
            "document IDs",
        )
        doc_id_lineage = {
            "status": "STAGE1_DOCUMENT_IDS_VERIFIED",
            "historical_artifacts_modified": False,
            "authoritative_stage1_doc_ids": corpus_doc_ids_record,
        }
    except ValueError as error:
        if str(error) != "Registered document IDs hash changed":
            raise
        doc_ids, doc_id_lineage = (
            _verify_pre_qrels_doc_id_reconciliation(
                artifact_root,
                corpus_manifest_path=corpus_manifest_path,
                corpus_doc_ids_record=corpus_doc_ids_record,
                expected_bytes=expected_doc_id_bytes,
            )
        )
    if embeddings.stat().st_size != expected_embedding_bytes:
        raise ValueError("Document embedding memmap size is inconsistent")
    if doc_ids.stat().st_size != expected_doc_id_bytes:
        raise ValueError("Document-ID memmap size is inconsistent")

    expected_index = {
        "document_count": document_count,
        "dimension": DIMENSION,
        "type": "IndexIVFPQ",
        "metric": "inner_product",
        "m": int(protocol["index_policy"]["common"]["subquantizers"]),
        "nbits": int(
            protocol["index_policy"]["common"]["bits_per_subquantizer"]
        ),
        "nlist": int(setting_contract["nlist"]),
        "nprobe": int(setting_contract["nprobe"]),
    }
    observed_index = {
        key: int(index.get(key, -1))
        if key not in ("type", "metric")
        else index.get(key)
        for key in expected_index
    }
    if observed_index != expected_index:
        raise ValueError(
            f"Stage-1 NQ index contract changed: "
            f"{observed_index} != {expected_index}"
        )
    index_embeddings = index.get("document_embeddings", {})
    if (
        int(index_embeddings.get("bytes", -1))
        != int(corpus["document_embeddings"]["bytes"])
        or index_embeddings.get("sha256")
        != corpus["document_embeddings"]["sha256"]
    ):
        raise ValueError("Index and corpus manifests register different embeddings")
    verify_record(
        index_embeddings,
        artifact_root,
        "index document embeddings",
        expected_path=embeddings,
    )
    index_path = verify_record(
        index.get("index", {}),
        artifact_root,
        "frozen IVF-PQ index",
    )
    return {
        "document_count": document_count,
        "dimension": DIMENSION,
        "width": width,
        "embeddings": embeddings,
        "doc_ids": doc_ids,
        "index": index_path,
        "corpus_manifest": corpus_manifest_path,
        "index_manifest": index_manifest_path,
        "corpus": corpus,
        "index_payload": index,
        "doc_id_lineage": doc_id_lineage,
    }


def _validate_stage3(
    artifact_root: Path,
) -> dict[str, Any]:
    audit_path = (
        artifact_root / "stage3/audit/eligible_test_query_audit.json"
    ).resolve(strict=True)
    evaluation_dir = artifact_root / "stage3/evaluation"
    summary_path = (evaluation_dir / "metrics_summary.json").resolve(
        strict=True
    )
    complete_path = (evaluation_dir / "evaluation_complete.json").resolve(
        strict=True
    )
    audit = read_json(audit_path)
    summary = read_json(summary_path)
    complete = read_json(complete_path)
    if (
        audit.get("protocol_id") != SOURCE_PROTOCOL_ID
        or audit.get("status")
        != "eligible_test_queries_frozen_before_retrieval"
    ):
        raise ValueError("Unexpected Stage-3 NQ identity audit")
    qids = [str(value) for value in audit.get("eligible_query_ids", [])]
    texts = [str(value) for value in audit.get("eligible_query_texts", [])]
    if (
        not qids
        or len(qids) != len(set(qids))
        or len(qids) != len(texts)
        or len(qids) != int(audit.get("eligible_query_count", -1))
    ):
        raise ValueError("Stage-3 audit query identity is inconsistent")
    qrels_path = verify_record(
        audit.get("test_qrels", {}),
        artifact_root,
        "opened official test qrels",
    )
    if (
        summary.get("protocol_id") != SOURCE_PROTOCOL_ID
        or summary.get("status") != "one_shot_evaluation_complete"
        or int(summary.get("eligible_query_count", -1)) != len(qids)
    ):
        raise ValueError("Unexpected Stage-3 NQ evaluation summary")
    if (
        complete.get("protocol_id") != SOURCE_PROTOCOL_ID
        or complete.get("status") != "complete_stop_no_retuning"
        or complete.get("metrics_summary_sha256") != sha256_file(summary_path)
    ):
        raise ValueError("Stage-3 NQ evaluation completion is not verified")
    vectors_path = verify_record(
        summary.get("files", {}).get("test_query_vectors", {}),
        artifact_root,
        "Stage-3 test query vectors",
        expected_path=evaluation_dir / "test_query_vectors.float32.npy",
    )
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if (
        vectors.shape != (len(qids), DIMENSION)
        or vectors.dtype != np.float32
        or not np.all(np.isfinite(vectors))
    ):
        raise ValueError("Stage-3 test query vectors changed shape/dtype/value")
    per_query_path = verify_record(
        summary.get("files", {}).get("per_query_metrics", {}),
        artifact_root,
        "Stage-3 per-query evaluation artifact",
    )
    _verify_qid_alignment(per_query_path, qids)
    return {
        "qids": qids,
        "vectors": vectors,
        "vectors_path": vectors_path,
        "qrels_path": qrels_path,
        "audit_path": audit_path,
        "summary_path": summary_path,
        "complete_path": complete_path,
        "per_query_path": per_query_path,
    }


def _verify_output_record(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Prepared {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Prepared {label} hash changed")
    return path


def verify_complete(
    output_root: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    complete_path = output_root / "preparation_complete.json"
    complete = read_json(complete_path)
    if (
        complete.get("protocol_id") != PROTOCOL_ID
        or complete.get("status") != STATUS
        or complete.get("source_commit") != source_commit
        or complete.get("prior_confirmation_outcomes_known") is not True
        or complete.get("metrics_computed") is not False
        or complete.get("sidecar_basis_fitted") is not False
    ):
        raise ValueError("Existing V17 NQ completion marker is not reusable")
    prepared_path = _verify_output_record(
        complete.get("prepared_domain", {}),
        "domain manifest",
    )
    prepared = read_json(prepared_path)
    if (
        prepared.get("protocol_id") != PROTOCOL_ID
        or prepared.get("status") != STATUS
        or prepared.get("source_commit") != source_commit
    ):
        raise ValueError("Existing V17 NQ prepared-domain manifest changed")
    for role in ("fit", "evaluation"):
        record = prepared.get("roles", {}).get(role, {})
        for label in ("query_ids", "query_vectors", "qrels_rows"):
            _verify_output_record(
                record.get(label, {}),
                f"{role} {label}",
            )
    for label in ("embeddings", "index"):
        _verify_output_record(prepared.get(label, {}), label)
    return complete


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = verify_source(
        repo_root,
        args.protocol,
        args.source_commit,
    )
    complete_path = args.output_root / "preparation_complete.json"
    if complete_path.is_file():
        return verify_complete(
            args.output_root,
            source_commit=args.source_commit,
        )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(
            "Refusing to resume V17 NQ preparation without a verified "
            "complete marker"
        )

    artifact_root = args.artifact_root.resolve(strict=True)
    stage1 = _validate_stage1(artifact_root, protocol)
    stage3 = _validate_stage3(artifact_root)
    roles = deterministic_roles(stage3["qids"])
    minimums = {
        "fit": int(
            protocol["data_policy"]["minimum_fit_queries_per_domain"]
        ),
        "evaluation": int(
            protocol["data_policy"]["minimum_evaluation_queries_per_domain"]
        ),
    }
    for role, minimum in minimums.items():
        if len(roles[role]) < minimum:
            raise ValueError(
                f"NQ {role} role has {len(roles[role])} queries; "
                f"minimum={minimum}"
            )
    if set(roles["fit"]) & set(roles["evaluation"]):
        raise AssertionError("V17 NQ role split is not disjoint")
    if set(roles["fit"]) | set(roles["evaluation"]) != set(stage3["qids"]):
        raise AssertionError("V17 NQ role split is not exhaustive")

    qrels = load_positive_qrels(stage3["qrels_path"])
    missing_qids = sorted(set(stage3["qids"]) - set(qrels))
    if missing_qids:
        raise ValueError(
            f"Eligible audit queries lack positive qrels: {missing_qids[:5]}"
        )
    eligible_qrels = {
        qid: qrels[qid]
        for qid in stage3["qids"]
    }
    required_doc_ids = set().union(*eligible_qrels.values())
    document_rows = map_document_rows(
        stage1["doc_ids"],
        document_count=stage1["document_count"],
        width=stage1["width"],
        required_ids=required_doc_ids,
    )

    args.output_root.mkdir(parents=True)
    setting_dir = args.output_root / SETTING_ID
    setting_dir.mkdir()
    source_position = {
        qid: position for position, qid in enumerate(stage3["qids"])
    }
    role_records: dict[str, Any] = {}
    for role in ("fit", "evaluation"):
        role_dir = setting_dir / role
        role_dir.mkdir()
        role_qids = roles[role]
        qid_path = role_dir / "query_ids.utf8.txt"
        qid_path.write_text("\n".join(role_qids) + "\n", encoding="utf-8")
        selected_rows = np.asarray(
            [source_position[qid] for qid in role_qids],
            dtype=np.int64,
        )
        vector_path = role_dir / "query_vectors.float32.npy"
        atomic_save(
            vector_path,
            np.asarray(stage3["vectors"][selected_rows], dtype=np.float32),
        )
        qrels_path = role_dir / "qrels_rows.json"
        atomic_json(
            qrels_path,
            {
                qid: sorted(
                    document_rows[doc_id]
                    for doc_id in eligible_qrels[qid]
                )
                for qid in role_qids
            },
        )
        role_records[role] = {
            "query_count": len(role_qids),
            "query_ids": file_record(qid_path),
            "query_vectors": file_record(vector_path),
            "qrels_rows": file_record(qrels_path),
        }

    corpus_manifest_record = file_record(stage1["corpus_manifest"])
    snapshot_record = stage1["corpus"].get(
        "embedding_model_snapshot_manifest",
        {},
    )
    revision = (
        "frozen-stage1-snapshot:"
        + str(snapshot_record.get("sha256", "unregistered"))
    )
    prepared = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "source_commit": args.source_commit,
        "domain_id": SETTING_ID,
        "setting_id": SETTING_ID,
        "encoder": {
            "id": "BAAI/bge-small-en-v1.5",
            "revision": revision,
            "dimension": DIMENSION,
            "exact_huggingface_revision_registered": False,
        },
        "document_count": stage1["document_count"],
        "eligible_query_count": len(stage3["qids"]),
        "roles": role_records,
        "embeddings": file_record(stage1["embeddings"]),
        "doc_ids": file_record(stage1["doc_ids"]),
        "index": file_record(stage1["index"]),
        "index_contract": {
            "dimension": DIMENSION,
            "document_count": stage1["document_count"],
            "metric": "inner_product",
            "subquantizers": 32,
            "bits_per_subquantizer": 8,
            "nlist": 2048,
            "nprobe": 32,
        },
        "sources": {
            "artifact_root": str(artifact_root),
            "corpus_manifest": corpus_manifest_record,
            "index_manifest": file_record(stage1["index_manifest"]),
            "eligible_test_query_audit": file_record(stage3["audit_path"]),
            "official_test_qrels": file_record(stage3["qrels_path"]),
            "evaluation_complete": file_record(stage3["complete_path"]),
            "metrics_summary": file_record(stage3["summary_path"]),
            "test_query_vectors": file_record(stage3["vectors_path"]),
            "per_query_evaluation_artifact": file_record(
                stage3["per_query_path"]
            ),
            "document_id_lineage": stage1["doc_id_lineage"],
        },
        "source_blobs": source_blobs,
        "evidence_boundary": {
            "opened_official_nq_test_role_reused": True,
            "prior_confirmation_outcomes_known": True,
            "independent_confirmation_claim_allowed": False,
            "retuning_old_confirmation_claim_allowed": False,
            "evaluation_used_for_v17_method_selection": False,
        },
        "retrieval_performed": False,
        "metrics_computed": False,
        "sidecar_basis_fitted": False,
    }
    prepared_path = setting_dir / "prepared_domain.json"
    atomic_json(prepared_path, prepared)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "source_commit": args.source_commit,
        "domain_id": SETTING_ID,
        "document_count": stage1["document_count"],
        "eligible_query_count": len(stage3["qids"]),
        "role_query_counts": {
            role: len(qids) for role, qids in roles.items()
        },
        "prepared_domain": file_record(prepared_path),
        "prior_confirmation_outcomes_known": True,
        "opened_official_nq_test_role_reused": True,
        "independent_confirmation_claim_allowed": False,
        "retrieval_performed": False,
        "metrics_computed": False,
        "sidecar_basis_fitted": False,
    }
    atomic_json(complete_path, complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
