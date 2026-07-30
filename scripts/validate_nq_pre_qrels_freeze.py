#!/usr/bin/env python3
"""Validate the BEIR NQ method freeze without reading test qrels.

This command deliberately has no qrels argument. It validates the design
protocol and the method-artifact manifest that must be frozen before anyone
opens, parses, or summarizes the NQ test qrels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "protocols" / "beir_nq_rars_pca_confirmation_v1.json"
)

EXPECTED_FILE_LABELS = frozenset(
    {
        "test_dataset_archive",
        "train_dataset_archive",
        "corpus_manifest",
        "document_embeddings",
        "doc_ids",
        "train_query_manifest",
        "validation_query_manifest",
        "train_validation_query_vectors",
        "train_validation_query_vector_manifest",
        "embedding_model_snapshot_manifest",
        "prior_query_registry",
        "frozen_index",
        "pca_config",
        "pca_basis",
        "pca_scales",
        "pca_codes",
        "rars_config",
        "rars_basis",
        "rars_scales",
        "rars_codes",
        "index_builder",
        "query_splitter",
        "pca_trainer",
        "rars_trainer",
        "manifest_builder",
        "prior_query_registry_builder",
        "evaluator",
    }
)
EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "manifest_id",
        "status",
        "protocol_id",
        "protocol_sha256",
        "test_qrels_accessed",
        "test_retrieval_performed",
        "test_outcomes_observed",
        "train_qrels_relevance_values_used",
        "corpus_document_count",
        "train_query_count",
        "validation_query_count",
        "selected_configs",
        "files",
    }
)
FORBIDDEN_PRE_QRELS_PATH_FRAGMENTS = (
    "qrels/test",
    "test/qrels",
    "test_qrels",
    "test-query",
    "test_query",
    "per_query_metrics",
    "paired_bootstrap",
    "test_metrics",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
EXPECTED_ARCHIVES = {
    "test_dataset_archive": {
        "md5": "d4d3d2e48787a744b6f6e691ff534307",
        "sha256": "2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752",
        "bytes": 498_307_926,
    },
    "train_dataset_archive": {
        "md5": "966435435932347d5513f56fed19161c",
        "sha256": "3aa8eec8d67174d85c055fce6971fa3127e830335842696756373f491bf391c9",
        "bytes": 1_405_702_846,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Base directory for relative artifact paths.",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Validate placeholders without declaring the freeze complete.",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="Verify bytes and hashes for every artifact path.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Root used by portable artifact:// manifest paths.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Root used by portable repo:// manifest paths.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def validate_protocol(protocol: dict[str, Any]) -> None:
    require_equal(
        protocol.get("protocol_id"),
        "beir_nq_rars_pca_confirmation_v1",
        "protocol_id",
    )
    require_equal(
        protocol.get("status"),
        "ready_for_design_freeze_before_test_qrels_access",
        "protocol status",
    )

    dataset = protocol.get("dataset", {})
    require_equal(dataset.get("benchmark"), "BEIR", "dataset benchmark")
    require_equal(dataset.get("dataset_id"), "nq", "dataset ID")
    for name, label in (("test", "test_dataset_archive"), ("train", "train_dataset_archive")):
        archive = dataset.get(f"{name}_archive", {})
        expected_archive = EXPECTED_ARCHIVES[label]
        for key in ("md5", "sha256", "bytes"):
            require_equal(
                archive.get(key),
                expected_archive[key],
                f"dataset {name} archive {key}",
            )
    require_equal(
        dataset.get("test_archive", {}).get("stage1_members_allowed"),
        ["corpus.jsonl"],
        "test archive Stage-1 members",
    )
    require_equal(
        dataset.get("test_archive", {}).get("stage3_members_allowed"),
        ["queries.jsonl", "qrels/test.tsv"],
        "test archive Stage-3 members",
    )
    require_equal(
        dataset.get("train_archive", {}).get("stage1_members_allowed"),
        ["queries.jsonl", "qrels/train.tsv"],
        "train archive Stage-1 members",
    )
    require_equal(
        dataset.get("train_archive", {}).get("corpus_member_prohibited"),
        True,
        "train archive corpus prohibition",
    )
    require_equal(
        dataset.get("qrels_policy"),
        "require_all_positive_in_corpus",
        "qrels policy",
    )
    require_equal(
        dataset.get("test_qrels_access_before_method_freeze"),
        False,
        "test-qrels gate",
    )

    embedding = protocol.get("embedding", {})
    for key, expected in {
        "model": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "document_storage_dtype": "float16",
        "document_compute_dtype": "float32",
        "query_storage_and_compute_dtype": "float32",
        "query_text": "raw BEIR query text without an added instruction prefix",
    }.items():
        require_equal(embedding.get(key), expected, f"embedding.{key}")

    base = protocol.get("base_index", {})
    expected_base = {
        "type": "IndexIVFPQ",
        "metric": "inner_product",
        "m": 32,
        "nbits": 8,
        "nlist": 2048,
        "nprobe": 32,
        "candidate_k": 100,
        "final_k": 10,
        "coarse_kmeans_iterations": 25,
        "pq_kmeans_iterations": 25,
        "build_backend": "single_faiss_gpu_nvidia_t4",
        "search_backend": "single_faiss_gpu_nvidia_t4",
        "gpu_float16_lookup_tables": False,
    }
    for key, expected in expected_base.items():
        require_equal(base.get(key), expected, f"base_index.{key}")

    sidecar = protocol.get("shared_sidecar", {})
    for key, expected in {
        "rank": 16,
        "coefficient_dtype": "int8",
        "candidate_k": 100,
        "final_k": 10,
    }.items():
        require_equal(sidecar.get(key), expected, f"shared_sidecar.{key}")

    require_equal(
        protocol.get("pca", {}).get("method_id"),
        "pca_r16_int8",
        "PCA method ID",
    )
    require_equal(
        protocol.get("rars", {}).get("method_id"),
        "rars_r16_int8",
        "RARS method ID",
    )
    require_equal(
        protocol.get("rars", {}).get("method_revision_allowed"),
        False,
        "RARS revision flag",
    )
    require_equal(
        protocol.get("rars", {}).get("residual_sample_draws"),
        300_000,
        "RARS sample draws",
    )

    validation = protocol.get("validation", {})
    if not validation.get("alphas") or not validation.get("top_b"):
        raise ValueError("Validation grid must be non-empty")
    require_equal(validation.get("qrels_used"), False, "validation qrels flag")

    evaluation = protocol.get("evaluation", {})
    require_equal(
        evaluation.get("primary_metric"),
        "recall@10",
        "primary metric",
    )
    require_equal(
        evaluation.get("primary_contrast"),
        "rars_r16_int8_minus_pca_r16_int8",
        "primary contrast",
    )
    require_equal(
        evaluation.get("bootstrap_replicates"),
        20_000,
        "bootstrap replicates",
    )
    require_equal(
        evaluation.get("bootstrap_seed"),
        20_260_720,
        "bootstrap seed",
    )
    require_equal(evaluation.get("one_shot"), True, "one-shot flag")

    if int(protocol.get("query_usage", {}).get(
        "minimum_final_test_queries", 0
    )) < 3000:
        raise ValueError("Minimum final test-query count must remain at least 3000")


def is_placeholder(value: Any) -> bool:
    return value is None or value == "TO_BE_FILLED"


def validate_file_entry(
    label: str,
    entry: Any,
    *,
    allow_draft: bool,
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"File entry {label} must be an object")

    path_value = entry.get("path")
    bytes_value = entry.get("bytes")
    sha_value = entry.get("sha256")
    if allow_draft and is_placeholder(path_value) and is_placeholder(
        bytes_value
    ) and is_placeholder(sha_value):
        return

    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"File entry {label} has no path")
    normalized_path = path_value.replace("\\", "/").casefold()
    if any(
        fragment in normalized_path
        for fragment in FORBIDDEN_PRE_QRELS_PATH_FRAGMENTS
    ):
        raise ValueError(
            f"File entry {label} points to a forbidden pre-qrels path"
        )
    if allow_draft and is_placeholder(bytes_value) and is_placeholder(sha_value):
        return
    if not isinstance(bytes_value, int) or bytes_value <= 0:
        raise ValueError(f"File entry {label} has invalid byte count")
    if not isinstance(sha_value, str) or not SHA256_RE.fullmatch(sha_value):
        raise ValueError(f"File entry {label} has invalid SHA-256")

    if label in EXPECTED_ARCHIVES:
        require_equal(
            bytes_value,
            EXPECTED_ARCHIVES[label]["bytes"],
            f"{label} bytes",
        )
        require_equal(
            sha_value,
            EXPECTED_ARCHIVES[label]["sha256"],
            f"{label} SHA-256",
        )
        md5_value = entry.get("md5")
        if not isinstance(md5_value, str) or not MD5_RE.fullmatch(md5_value):
            raise ValueError("Dataset archive has invalid MD5")
        require_equal(
            md5_value,
            EXPECTED_ARCHIVES[label]["md5"],
            f"{label} MD5",
        )


def validate_selected_configs(
    selected: Any,
    *,
    protocol: dict[str, Any],
    allow_draft: bool,
) -> None:
    if not isinstance(selected, dict):
        raise ValueError("selected_configs must be an object")

    allowed_alphas = {
        float(value) for value in protocol["validation"]["alphas"]
    }
    allowed_top_b = {
        int(value) for value in protocol["validation"]["top_b"]
    }
    for name, method_id in [
        ("pca", "pca_r16_int8"),
        ("rars", "rars_r16_int8"),
    ]:
        value = selected.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Missing selected config for {name}")
        require_equal(value.get("method_id"), method_id, f"{name} method ID")
        alpha = value.get("alpha")
        top_b = value.get("top_b")
        if allow_draft and alpha is None and top_b is None:
            continue
        if not isinstance(alpha, (int, float)) or float(alpha) not in allowed_alphas:
            raise ValueError(f"Selected {name} alpha is outside the frozen grid")
        if not isinstance(top_b, int) or top_b not in allowed_top_b:
            raise ValueError(f"Selected {name} Top-B is outside the frozen grid")


def resolve_artifact_path(
    root: Path,
    value: str,
    *,
    artifact_root: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    if value.startswith("artifact://"):
        if artifact_root is None:
            raise ValueError("artifact:// path requires --artifact-root")
        return artifact_root / value.removeprefix("artifact://")
    if value.startswith("repo://"):
        if repo_root is None:
            raise ValueError("repo:// path requires --repo-root")
        return repo_root / value.removeprefix("repo://")
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_manifest(
    manifest: dict[str, Any],
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    root: Path = ROOT,
    artifact_root: Path | None = None,
    repo_root: Path | None = None,
    allow_draft: bool = False,
    verify_files: bool = False,
) -> dict[str, Any]:
    validate_protocol(protocol)
    actual_manifest_keys = set(manifest)
    if actual_manifest_keys != EXPECTED_MANIFEST_KEYS:
        missing = sorted(EXPECTED_MANIFEST_KEYS - actual_manifest_keys)
        extra = sorted(actual_manifest_keys - EXPECTED_MANIFEST_KEYS)
        raise ValueError(
            f"Pre-qrels manifest fields mismatch; missing={missing}, extra={extra}"
        )
    require_equal(
        manifest.get("manifest_id"),
        "beir_nq_rars_pca_confirmation_v1_pre_qrels",
        "manifest ID",
    )
    require_equal(
        manifest.get("protocol_id"),
        protocol["protocol_id"],
        "manifest protocol ID",
    )

    expected_status = (
        "DRAFT_NOT_FROZEN" if allow_draft else "frozen_before_test_qrels_access"
    )
    require_equal(manifest.get("status"), expected_status, "manifest status")

    for key in [
        "test_qrels_accessed",
        "test_retrieval_performed",
        "test_outcomes_observed",
        "train_qrels_relevance_values_used",
    ]:
        require_equal(manifest.get(key), False, key)

    protocol_hash = manifest.get("protocol_sha256")
    if not (allow_draft and is_placeholder(protocol_hash)):
        require_equal(
            protocol_hash,
            sha256_file(protocol_path),
            "protocol SHA-256",
        )

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Manifest files must be an object")
    actual_labels = set(files)
    if actual_labels != EXPECTED_FILE_LABELS:
        missing = sorted(EXPECTED_FILE_LABELS - actual_labels)
        extra = sorted(actual_labels - EXPECTED_FILE_LABELS)
        raise ValueError(
            f"Pre-qrels file contract mismatch; missing={missing}, extra={extra}"
        )
    for label, entry in files.items():
        validate_file_entry(label, entry, allow_draft=allow_draft)

    validate_selected_configs(
        manifest.get("selected_configs"),
        protocol=protocol,
        allow_draft=allow_draft,
    )

    count_fields = [
        "corpus_document_count",
        "train_query_count",
        "validation_query_count",
    ]
    if not allow_draft:
        for key in count_fields:
            value = manifest.get(key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if manifest["corpus_document_count"] < 2_000_000:
            raise ValueError("Unexpectedly small BEIR NQ corpus")

    if verify_files:
        if allow_draft:
            raise ValueError("Cannot verify files for a placeholder draft")
        for label, entry in files.items():
            path = resolve_artifact_path(
                root,
                entry["path"],
                artifact_root=artifact_root,
                repo_root=repo_root,
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
            require_equal(path.stat().st_size, entry["bytes"], f"{label} bytes")
            require_equal(sha256_file(path), entry["sha256"], f"{label} SHA-256")
            if label in EXPECTED_ARCHIVES:
                require_equal(md5_file(path), entry["md5"], f"{label} MD5")

    return {
        "protocol_id": protocol["protocol_id"],
        "manifest_status": manifest["status"],
        "artifact_count": len(files),
        "test_qrels_accessed": False,
        "test_outcomes_observed": False,
        "files_verified": bool(verify_files),
    }


def main() -> None:
    args = parse_args()
    protocol = read_json(args.protocol)
    manifest = read_json(args.manifest)
    summary = validate_manifest(
        manifest,
        protocol=protocol,
        protocol_path=args.protocol,
        root=args.root,
        artifact_root=args.artifact_root,
        repo_root=args.repo_root,
        allow_draft=args.allow_draft,
        verify_files=args.verify_files,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
