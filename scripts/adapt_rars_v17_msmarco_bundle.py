#!/usr/bin/env python3
"""Adapt an opened MS MARCO 1M development bundle to the V17 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from rars_v17_million_scale_core import deterministic_fold_ids


PROTOCOL_ID = "rars_v17_million_scale_setting_transfer_v1"
STATUS = "RARS_V17_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS"
SETTING_ID = "msmarco_1m_bge_opened_development"
CANONICAL_PROTOCOL = Path(
    "protocols/rars_v17_million_scale_setting_transfer_v1.json"
)
PARENT_FILES = (
    "query_vectors.float32.npy",
    "ann_rows.int64.npy",
    "ann_scores.float32.npy",
    "ann_residual_rows.int64.npy",
    "candidate_residuals.float32.npy",
    "candidate_relevance.uint8.npy",
    "relevant_counts.int32.npy",
)


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


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def verify_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("--source-commit must be exact lowercase 40-hex")
    if protocol_path.resolve(strict=True) != (
        repo_root / CANONICAL_PROTOCOL
    ).resolve(strict=True):
        raise ValueError("V17 requires the canonical protocol path")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or dirty:
        raise ValueError("V17 adaptation requires a clean exact checkout")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status")
        != "FROZEN_BEFORE_FIRST_V17_MILLION_SCALE_DIAGNOSTIC_RUN"
    ):
        raise ValueError("Unexpected V17 protocol")
    return protocol


def copy_or_link(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def verify_parent_file(
    parent_dir: Path, parent_manifest: dict[str, Any], filename: str
) -> Path:
    path = parent_dir / filename
    record = parent_manifest.get("files", {}).get(filename)
    if not path.is_file() or not isinstance(record, dict):
        raise ValueError(f"Parent bundle lacks {filename}")
    observed = file_record(path)
    if (
        observed["bytes"] != int(record.get("bytes", -1))
        or observed["sha256"] != record.get("sha256")
    ):
        raise ValueError(f"Parent bundle changed: {filename}")
    return path


def adapt(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    protocol = verify_source(
        repo_root, args.protocol.resolve(strict=True), args.source_commit
    )
    if args.role not in ("fit", "evaluation"):
        raise ValueError("--role must be fit or evaluation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to overwrite a non-empty V17 output")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parent_dir = args.parent_bundle.resolve(strict=True)
    parent_manifest_path = parent_dir / "manifest.json"
    parent_manifest = json.loads(
        parent_manifest_path.read_text(encoding="utf-8")
    )
    expected_parent_role = (
        "inner_train" if args.role == "fit" else "inner_validation"
    )
    if (
        parent_manifest.get("role_id") != expected_parent_role
        or int(parent_manifest.get("candidate_count", -1)) != 100
        or parent_manifest.get("test_qrels_accessed") is not False
    ):
        raise ValueError("Unexpected MS MARCO parent development bundle")
    query_manifest_path = parent_dir / "query_manifest.json"
    query_manifest = json.loads(
        query_manifest_path.read_text(encoding="utf-8")
    )
    qids = [str(value) for value in query_manifest["query_ids"]]
    if (
        len(qids) != int(parent_manifest["query_count"])
        or len(qids) != len(set(qids))
    ):
        raise ValueError("MS MARCO parent query identity changed")

    records: dict[str, Any] = {}
    for filename in PARENT_FILES:
        source = verify_parent_file(parent_dir, parent_manifest, filename)
        destination = args.output_dir / filename
        copy_or_link(source, destination)
        records[filename] = file_record(destination)
    folds = deterministic_fold_ids(
        qids, fold_count=5, namespace=PROTOCOL_ID + ":" + SETTING_ID
    )
    fold_path = args.output_dir / "fold_ids.int64.npy"
    atomic_save(fold_path, folds.astype(np.int64))
    records[fold_path.name] = file_record(fold_path)
    qid_path = args.output_dir / "query_ids.utf8.txt"
    qid_path.write_text("\n".join(qids) + "\n", encoding="utf-8")
    records[qid_path.name] = file_record(qid_path)

    minimum = int(
        protocol["data_policy"][
            "minimum_fit_queries_per_domain"
            if args.role == "fit"
            else "minimum_evaluation_queries_per_domain"
        ]
    )
    if len(qids) < minimum:
        raise ValueError(f"MS MARCO {args.role} has only {len(qids)} queries")
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "source_commit": args.source_commit,
        "domain_id": SETTING_ID,
        "setting_id": SETTING_ID,
        "evidence_role": args.role,
        "encoder_id": "BAAI/bge-small-en-v1.5",
        "encoder_revision": "legacy_msmarco_bge_snapshot_revision_unregistered",
        "dimension": 384,
        "encoder": {
            "id": "BAAI/bge-small-en-v1.5",
            "revision": "legacy_msmarco_bge_snapshot_revision_unregistered",
            "dimension": 384
        },
        "encoder_provenance_warning": (
            "The legacy MS MARCO cache predates exact snapshot pinning; its "
            "BGE-small identity was previously checked by query-vector "
            "compatibility, but an exact model revision is unavailable."
        ),
        "query_count": len(qids),
        "fold_count": 5,
        "fold_counts": np.bincount(folds, minlength=5).astype(int).tolist(),
        "document_count": 1000000,
        "candidate_pool": 100,
        "candidate_residual_count": int(
            np.load(
                args.output_dir / "candidate_residuals.float32.npy",
                mmap_mode="r",
                allow_pickle=False,
            ).shape[0]
        ),
        "index_contract": {
            "dimension": 384,
            "document_count": 1000000,
            "nlist": 512,
            "nprobe": 16,
            "subquantizers": 32,
            "bits_per_subquantizer": 8,
            "metric_type": 0
        },
        "parent": {
            "manifest": file_record(parent_manifest_path),
            "query_manifest": file_record(query_manifest_path),
            "role_id": expected_parent_role
        },
        "files": records,
        "opened_development_evidence": True,
        "metrics_computed": False,
        "basis_fitted": False,
        "closed_test_opened": False
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
        "closed_test_opened": False
    }
    atomic_json(args.output_dir / "bundle_complete.json", complete)
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-bundle", type=Path, required=True)
    parser.add_argument("--role", choices=("fit", "evaluation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(adapt(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
