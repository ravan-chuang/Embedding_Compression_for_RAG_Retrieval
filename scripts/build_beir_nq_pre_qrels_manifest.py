#!/usr/bin/env python3
"""Build the portable BEIR NQ method-artifact freeze manifest.

The command copies only the two small selected configuration JSON files into
the repository, fills hashes for Drive-backed artifacts and frozen source code,
and writes the manifest that must be committed before the test-qrels audit.
It never accepts or reads a test-qrels path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols" / "beir_nq_rars_pca_confirmation_v1.json"
DEFAULT_TEMPLATE = ROOT / "protocols" / "beir_nq_pre_qrels_manifest.template.json"
DEFAULT_OUTPUT = ROOT / "protocols" / "beir_nq_pre_qrels_manifest.json"
PROTOCOL_ID = "beir_nq_rars_pca_confirmation_v1"


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
    except TypeError:  # pragma: no cover
        digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, *, artifact_root: Path, repo: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(artifact_root.resolve())
        return "artifact://" + relative.as_posix()
    except ValueError:
        pass
    try:
        relative = resolved.relative_to(repo.resolve())
        return "repo://" + relative.as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact is outside both registered roots: {path}") from exc


def file_entry(
    path: Path,
    *,
    artifact_root: Path,
    repo: Path,
    include_md5: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entry: dict[str, Any] = {
        "path": portable_path(path, artifact_root=artifact_root, repo=repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if include_md5:
        entry["md5"] = md5_file(path)
    return entry


def reject_unsafe_flags(value: Any, trail: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_folded = str(key).casefold()
            if key_folded in {
                "test_qrels_accessed",
                "test_retrieval_performed",
                "test_outcomes_observed",
                "train_qrels_relevance_values_used",
                "validation_qrels_used",
            } and child is not False:
                raise ValueError(f"Unsafe flag {trail}.{key}: {child!r}")
            reject_unsafe_flags(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_unsafe_flags(child, f"{trail}[{index}]")


def load_validator() -> Any:
    path = ROOT / "scripts" / "validate_nq_pre_qrels_freeze.py"
    spec = importlib.util.spec_from_file_location("nq_freeze_validator", path)
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load pre-qrels validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_stage0_gate(artifact_root: Path, repo: Path, protocol_path: Path) -> None:
    path = ROOT / "scripts" / "prepare_beir_nq_colab.py"
    spec = importlib.util.spec_from_file_location("nq_stage0_gate", path)
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load the Stage-0 gate verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_gate(artifact_root, repo, protocol_path)


def copy_selected_config(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return destination


def verify_model_snapshot_manifest(path: Path) -> None:
    manifest = read_json(path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Embedding-model snapshot manifest protocol mismatch")
    root = Path(manifest["snapshot_directory"])
    for item in manifest.get("files", []):
        file_path = root / item["path"]
        if not file_path.is_file() or file_path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Embedding-model snapshot changed: {file_path}")
        if sha256_file(file_path) != item["sha256"]:
            raise ValueError(f"Embedding-model snapshot hash changed: {file_path}")


def build_manifest(
    artifact_root: Path,
    repo: Path,
    protocol_path: Path,
    template_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    verify_stage0_gate(artifact_root, repo, protocol_path)
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected protocol")
    template = read_json(template_path)
    corpus_manifest_path = artifact_root / "stage1" / "corpus" / "corpus_artifacts_manifest.json"
    vector_manifest_path = (
        artifact_root
        / "stage1"
        / "queries"
        / "train_validation_query_vector_manifest.json"
    )
    index_manifest_path = artifact_root / "stage1" / "index" / "index_manifest.json"
    training_manifest_path = (
        artifact_root / "stage2" / "sidecars" / "sidecar_training_manifest.json"
    )
    corpus_manifest = read_json(corpus_manifest_path)
    vector_manifest = read_json(vector_manifest_path)
    index_manifest = read_json(index_manifest_path)
    training_manifest = read_json(training_manifest_path)
    model_manifest_path = artifact_root / "stage1" / "model" / "snapshot_manifest.json"
    verify_model_snapshot_manifest(model_manifest_path)
    fit_manifest_path = artifact_root / "stage1" / "query_splits" / "train_query_manifest.json"
    val_manifest_path = artifact_root / "stage1" / "query_splits" / "validation_query_manifest.json"
    fit_manifest = read_json(fit_manifest_path)
    val_manifest = read_json(val_manifest_path)
    for payload in (
        corpus_manifest,
        vector_manifest,
        index_manifest,
        training_manifest,
        fit_manifest,
        val_manifest,
    ):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("Input manifest protocol mismatch")
        reject_unsafe_flags(payload)

    selected_dir = repo / "results" / "beir_nq_confirmation" / "pre_qrels"
    pca_config = copy_selected_config(
        Path(training_manifest["files"]["pca_config"]["path"]),
        selected_dir / "selected_pca_config.json",
    )
    rars_config = copy_selected_config(
        Path(training_manifest["files"]["rars_config"]["path"]),
        selected_dir / "selected_rars_config.json",
    )
    pca = read_json(pca_config)
    rars = read_json(rars_config)
    reject_unsafe_flags(pca)
    reject_unsafe_flags(rars)

    paths = {
        "test_dataset_archive": artifact_root / "source" / "nq.zip",
        "train_dataset_archive": artifact_root / "source" / "nq-train.zip",
        "corpus_manifest": corpus_manifest_path,
        "document_embeddings": Path(corpus_manifest["document_embeddings"]["path"]),
        "doc_ids": Path(corpus_manifest["doc_ids"]["path"]),
        "train_query_manifest": fit_manifest_path,
        "validation_query_manifest": val_manifest_path,
        "train_validation_query_vectors": Path(vector_manifest["vectors"]["path"]),
        "train_validation_query_vector_manifest": vector_manifest_path,
        "embedding_model_snapshot_manifest": (
            model_manifest_path
        ),
        "prior_query_registry": repo / "protocols" / "beir_nq_prior_query_registry_v1.json",
        "frozen_index": Path(index_manifest["index"]["path"]),
        "pca_config": pca_config,
        "pca_basis": Path(training_manifest["files"]["pca_basis"]["path"]),
        "pca_scales": Path(training_manifest["files"]["pca_scales"]["path"]),
        "pca_codes": Path(training_manifest["files"]["pca_codes"]["path"]),
        "rars_config": rars_config,
        "rars_basis": Path(training_manifest["files"]["rars_basis"]["path"]),
        "rars_scales": Path(training_manifest["files"]["rars_scales"]["path"]),
        "rars_codes": Path(training_manifest["files"]["rars_codes"]["path"]),
        "index_builder": repo / "scripts" / "prepare_beir_nq_colab.py",
        "query_splitter": repo / "scripts" / "create_beir_nq_train_validation_splits.py",
        "pca_trainer": repo / "scripts" / "train_select_beir_nq_sidecars.py",
        "rars_trainer": repo / "scripts" / "train_select_beir_nq_sidecars.py",
        "manifest_builder": repo / "scripts" / "build_beir_nq_pre_qrels_manifest.py",
        "prior_query_registry_builder": (
            repo / "scripts" / "build_beir_nq_prior_query_registry.py"
        ),
        "evaluator": repo / "scripts" / "evaluate_beir_nq_frozen.py",
    }
    if set(paths) != set(template["files"]):
        missing = sorted(set(template["files"]) - set(paths))
        extra = sorted(set(paths) - set(template["files"]))
        raise ValueError(f"Manifest path contract mismatch; missing={missing}, extra={extra}")

    manifest = template
    manifest.update({
        "status": "frozen_before_test_qrels_access",
        "protocol_sha256": sha256_file(protocol_path),
        "test_qrels_accessed": False,
        "test_retrieval_performed": False,
        "test_outcomes_observed": False,
        "train_qrels_relevance_values_used": False,
        "corpus_document_count": int(corpus_manifest["document_count"]),
        "train_query_count": int(fit_manifest["query_count"]),
        "validation_query_count": int(val_manifest["query_count"]),
        "selected_configs": {
            "pca": {
                "method_id": "pca_r16_int8",
                "alpha": float(pca["alpha"]),
                "top_b": int(pca["top_b"]),
            },
            "rars": {
                "method_id": "rars_r16_int8",
                "alpha": float(rars["alpha"]),
                "top_b": int(rars["top_b"]),
            },
        },
    })
    manifest["files"] = {
        label: file_entry(
            path,
            artifact_root=artifact_root,
            repo=repo,
            include_md5=(label in {"test_dataset_archive", "train_dataset_archive"}),
        )
        for label, path in paths.items()
    }
    reject_unsafe_flags(manifest)
    atomic_write_json(output_path, manifest)
    artifact_copy = artifact_root / "stage2" / "pre_qrels_manifest.json"
    atomic_write_json(artifact_copy, manifest)

    validator = load_validator()
    validator.validate_manifest(
        manifest,
        protocol=protocol,
        protocol_path=protocol_path,
        artifact_root=artifact_root,
        repo_root=repo,
        verify_files=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_manifest(
        args.artifact_root,
        args.repo,
        args.protocol,
        args.template,
        args.output,
    )
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "status": result["status"],
        "corpus_document_count": result["corpus_document_count"],
        "train_query_count": result["train_query_count"],
        "validation_query_count": result["validation_query_count"],
        "selected_configs": result["selected_configs"],
        "artifact_count": len(result["files"]),
        "test_qrels_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
