#!/usr/bin/env python3
"""Run the outcome-informed RARS-v16 mechanism decomposition.

The evaluator consumes exactly two hash-registered, same-encoder prepared
domains.  It does not open a full corpus or rebuild an index.  The experiment
separates candidate headroom, FP32 rank capacity, int8 coding loss,
cutoff-aware objective value, fit-domain interaction, and pooled-fit repair.
Results are diagnostic development evidence, never an unseen confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rars_v16_causal_generalization_core import (  # noqa: E402
    PROTOCOL_ID,
    candidate_gap_decomposition,
    causal_decision,
    paired_query_inference,
    score_fp32_sidecar_candidates,
    subspace_alignment_metrics,
)
from rars_v8_cutoff_sidecar_core import (  # noqa: E402
    CutoffPairBatch,
    encode_residuals_int8,
    fit_cutoff_aware_basis,
    fit_int8_scales,
    fit_uncentered_pca_basis,
    mine_cutoff_pairs,
    per_query_metrics,
    query_role_balanced_weights,
    score_sidecar_candidates,
    summarize_pairs,
)


CANONICAL_PROTOCOL = Path(
    "protocols/rars_v16_causal_generalization_diagnostic_v1.json"
)
CANONICAL_SOURCES = (
    CANONICAL_PROTOCOL,
    Path("scripts/build_rars_v16_domain_bundle.py"),
    Path("scripts/evaluate_rars_v16_causal_generalization.py"),
    Path("scripts/freeze_rars_v16_domain_manifest.py"),
    Path("scripts/prepare_rars_v16_beir_domains.py"),
    Path("scripts/rars_v16_causal_generalization_core.py"),
    Path("scripts/rars_v8_cutoff_sidecar_core.py"),
)
STANDARD_FILES = {
    "query_ids.utf8.txt": None,
    "fold_ids.int64.npy": np.int64,
    "query_vectors.float32.npy": np.float32,
    "ann_rows.int64.npy": np.int64,
    "ann_scores.float32.npy": np.float32,
    "ann_residual_rows.int64.npy": np.int64,
    "candidate_residuals.float32.npy": np.float32,
    "candidate_relevance.uint8.npy": np.uint8,
    "relevant_counts.int32.npy": np.int32,
}
FORBIDDEN_CLOSED_PATH_TOKENS = (
    "future_method_holdout",
    "oracle_audit",
    "rars_v9",
    "rars-v9",
    "beir_nq_confirmation",
    "trec_dl_2019",
    "external_confirmation",
    "rars_clean_split/test",
)


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
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _exact_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def _reject_forbidden_closed_path(path: Path, label: str) -> None:
    normalized = str(Path(path)).replace("\\", "/").casefold()
    if any(token in normalized for token in FORBIDDEN_CLOSED_PATH_TOKENS):
        raise ValueError(f"V16 refuses forbidden closed-test {label} path: {path}")


def _prepare_empty_output(path: Path) -> None:
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError("V16 output exists and is not a directory")
        if any(path.iterdir()):
            raise ValueError("Refusing to reuse a non-empty V16 output directory")
    else:
        path.mkdir(parents=True)


def validate_source(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_commit(source_commit)
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    if protocol_path.resolve(strict=True) != canonical:
        raise ValueError(f"Protocol must use canonical path: {canonical}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or dirty:
        raise ValueError("V16 requires a clean exact source checkout")
    protocol = read_json(canonical)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status")
        != "FROZEN_BEFORE_FIRST_V16_MECHANISM_DIAGNOSTIC_RUN"
        or protocol.get("method_revision_allowed") is not False
    ):
        raise ValueError("Unexpected or mutable V16 protocol")
    records: dict[str, Any] = {}
    for relative in CANONICAL_SOURCES:
        path = (repo_root / relative).resolve(strict=True)
        local = path.read_bytes()
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{source_commit}:{relative.as_posix()}"],
                cwd=repo_root,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"Canonical source is absent from commit: {relative}"
            ) from error
        if local != committed:
            raise ValueError(f"Canonical source differs from Git blob: {relative}")
        records[relative.as_posix()] = file_record(path)
    return protocol, records


def _resolve_registered_path(manifest_path: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(strict=True)


def load_domain_manifest(
    manifest_path: Path, protocol: dict[str, Any]
) -> tuple[str, dict[str, dict[str, Path]], dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve(strict=True)
    _reject_forbidden_closed_path(manifest_path, "domain manifest")
    payload = read_json(manifest_path)
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "V16_DOMAIN_BUNDLES_FROZEN"
    ):
        raise ValueError("Domain manifest is not frozen for V16")
    source_domain = str(payload.get("source_domain_id", ""))
    entries = payload.get("domains")
    if not source_domain or not isinstance(entries, list):
        raise ValueError("V16 domain manifest lacks a source or domain list")
    expected_domains = set(
        protocol["data_policy"]["allowed_development_domains"]
    )
    if len(entries) != int(protocol["data_policy"]["domain_count"]):
        raise ValueError("Domain count differs from the protocol")
    domains: dict[str, dict[str, Path]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every domain entry must be an object")
        domain_id = str(entry.get("domain_id", ""))
        roles = entry.get("roles", entry.get("bundles"))
        role_manifests = entry.get("role_manifests")
        if not domain_id or domain_id in domains or not isinstance(roles, dict):
            raise ValueError("Domain IDs and role mappings must be unique")
        if not isinstance(roles.get("fit"), str) or not isinstance(
            roles.get("evaluation"), str
        ):
            raise ValueError(f"Domain {domain_id} lacks fit/evaluation roles")
        if not isinstance(role_manifests, dict):
            raise ValueError(f"Domain {domain_id} lacks frozen role manifests")
        domains[domain_id] = {
            "fit": _resolve_registered_path(manifest_path, roles["fit"]),
            "evaluation": _resolve_registered_path(
                manifest_path, roles["evaluation"]
            ),
        }
        for role, path in domains[domain_id].items():
            _reject_forbidden_closed_path(path, f"{domain_id} {role}")
            registered = role_manifests.get(role)
            if not isinstance(registered, dict):
                raise ValueError(f"Domain {domain_id} lacks {role} hash")
            _verify_registered_file(
                path / "bundle_manifest.json",
                registered,
                f"{domain_id} {role} bundle manifest",
            )
    if set(domains) != expected_domains or source_domain not in domains:
        raise ValueError("Domain identities differ from the frozen protocol")
    return source_domain, domains, file_record(manifest_path)


def _verify_registered_file(
    path: Path, record: dict[str, Any], description: str
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing registered {description}: {path}")
    observed = file_record(path)
    if observed["bytes"] != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {description} byte count changed")
    if observed["sha256"] != record.get("sha256"):
        raise ValueError(f"Registered {description} hash changed")
    return observed


def load_bundle(
    bundle_dir: Path,
    *,
    expected_domain: str,
    expected_role: str,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir).resolve(strict=True)
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status")
        != "RARS_V16_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS"
        or manifest.get("domain_id") != expected_domain
        or manifest.get("evidence_role") != expected_role
    ):
        raise ValueError("Bundle identity, role, protocol, or status changed")
    encoder = manifest.get("encoder", {})
    encoder_id = str(manifest.get("encoder_id", encoder.get("id", "")))
    encoder_revision = str(
        manifest.get("encoder_revision", encoder.get("revision", ""))
    )
    dimension = int(manifest.get("dimension", encoder.get("dimension", 0)))
    query_count = int(manifest.get("query_count", 0))
    if not encoder_id or not encoder_revision or dimension <= 0 or query_count <= 0:
        raise ValueError("Bundle encoder or shape identity is incomplete")
    registered = manifest.get("files")
    if not isinstance(registered, dict):
        raise ValueError("Bundle manifest lacks registered files")
    records: dict[str, Any] = {
        "bundle_manifest.json": file_record(manifest_path)
    }
    paths: dict[str, Path] = {}
    for name in STANDARD_FILES:
        record = registered.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Bundle manifest lacks {name}")
        path = bundle_dir / name
        records[name] = _verify_registered_file(path, record, name)
        paths[name] = path
    qids = paths["query_ids.utf8.txt"].read_text(encoding="utf-8").splitlines()
    if (
        len(qids) != query_count
        or len(set(qids)) != query_count
        or any(not query_id for query_id in qids)
    ):
        raise ValueError("Bundle query IDs are invalid")
    arrays = {
        name: np.load(paths[name], mmap_mode="r", allow_pickle=False)
        for name, dtype in STANDARD_FILES.items()
        if dtype is not None
    }
    for name, dtype in STANDARD_FILES.items():
        if dtype is not None and arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"{name} has an unexpected dtype")
    queries = arrays["query_vectors.float32.npy"]
    folds = arrays["fold_ids.int64.npy"]
    rows = arrays["ann_rows.int64.npy"]
    scores = arrays["ann_scores.float32.npy"]
    lookup = arrays["ann_residual_rows.int64.npy"]
    residuals = arrays["candidate_residuals.float32.npy"]
    labels = arrays["candidate_relevance.uint8.npy"]
    relevant = arrays["relevant_counts.int32.npy"]
    if queries.shape != (query_count, dimension) or folds.shape != (query_count,):
        raise ValueError("Bundle query or fold shape changed")
    if not (rows.shape == scores.shape == lookup.shape == labels.shape):
        raise ValueError("Candidate matrices must have identical shapes")
    if (
        rows.shape[0] != query_count
        or residuals.ndim != 2
        or residuals.shape[1] != dimension
        or relevant.shape != (query_count,)
    ):
        raise ValueError("Bundle residual or denominator shape changed")
    valid = rows >= 0
    if (
        np.any(lookup[valid] < 0)
        or np.any(lookup[valid] >= len(residuals))
        or np.any(lookup[~valid] != -1)
        or np.any(relevant <= 0)
        or np.any((labels != 0) & (labels != 1))
    ):
        raise ValueError("Bundle lookup, relevance, or padding contract failed")
    for name in (
        "query_vectors.float32.npy",
        "ann_scores.float32.npy",
        "candidate_residuals.float32.npy",
    ):
        if not np.all(np.isfinite(np.asarray(arrays[name]))):
            raise ValueError(f"{name} contains non-finite values")
    if protocol is not None:
        minimum_key = (
            "minimum_fit_queries_per_domain"
            if expected_role == "fit"
            else "minimum_evaluation_queries_per_domain"
        )
        if query_count < int(protocol["data_policy"][minimum_key]):
            raise ValueError(f"{expected_domain} {expected_role} is undersized")
    return {
        "manifest": manifest,
        "qids": qids,
        "arrays": arrays,
        "records": records,
        "encoder_key": (encoder_id, encoder_revision, dimension),
        "query_count": query_count,
        "dimension": dimension,
    }


def _exact_candidate_scores(bundle: dict[str, Any]) -> np.ndarray:
    arrays = bundle["arrays"]
    queries = np.asarray(arrays["query_vectors.float32.npy"], dtype=np.float32)
    rows = np.asarray(arrays["ann_rows.int64.npy"], dtype=np.int64)
    lookup = np.asarray(
        arrays["ann_residual_rows.int64.npy"], dtype=np.int64
    )
    residuals = arrays["candidate_residuals.float32.npy"]
    output = np.asarray(arrays["ann_scores.float32.npy"], dtype=np.float32).copy()
    for start in range(0, len(queries), 128):
        end = min(start + 128, len(queries))
        local_lookup = lookup[start:end]
        valid = rows[start:end] >= 0
        safe_lookup = np.where(valid, local_lookup, 0)
        correction = np.einsum(
            "qd,qcd->qc",
            queries[start:end],
            np.asarray(residuals[safe_lookup], dtype=np.float32),
        )
        output[start:end] += np.where(valid, correction, 0.0).astype(np.float32)
    return output


def _fit_pca_from_bundles(
    bundles: Iterable[dict[str, Any]], *, rank: int
) -> np.ndarray:
    bundles = list(bundles)
    if not bundles:
        raise ValueError("Cannot fit PCA from no bundles")
    dimension = bundles[0]["dimension"]
    moment = np.zeros((dimension, dimension), dtype=np.float64)
    count = 0
    for bundle in bundles:
        if bundle["dimension"] != dimension:
            raise ValueError("Pooled PCA dimensions differ")
        residuals = bundle["arrays"]["candidate_residuals.float32.npy"]
        for start in range(0, len(residuals), 8192):
            block = np.asarray(residuals[start : start + 8192], dtype=np.float64)
            moment += block.T @ block
            count += len(block)
    if not count:
        raise ValueError("Cannot fit PCA from empty residuals")
    eigenvalues, eigenvectors = np.linalg.eigh(moment / count)
    order = np.argsort(-eigenvalues, kind="stable")[:rank]
    basis = eigenvectors[:, order]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis.astype(np.float32)


def _mine_bundle_pairs(
    bundle: dict[str, Any], protocol: dict[str, Any]
) -> CutoffPairBatch:
    arrays = bundle["arrays"]
    mining = protocol["pair_mining"]
    config = protocol["diagnostic_configuration"]
    return mine_cutoff_pairs(
        arrays["ann_rows.int64.npy"],
        arrays["ann_residual_rows.int64.npy"],
        arrays["ann_scores.float32.npy"],
        _exact_candidate_scores(bundle),
        arrays["candidate_relevance.uint8.npy"],
        final_k=int(config["final_k"]),
        top_b=int(config["top_b"]),
        protection_window=int(mining["protection_window"]),
        max_challengers_per_positive=int(
            mining["maximum_challengers_per_positive"]
        ),
        margin_temperature=float(mining["margin_temperature"]),
        damage_scale=float(mining["damage_scale"]),
        promotion_mass=float(mining["promotion_total_loss_mass"]),
    )


def _combine_pairs(
    batches: list[CutoffPairBatch],
    query_offsets: list[int],
    residual_offsets: list[int],
    *,
    promotion_mass: float,
) -> CutoffPairBatch:
    if not batches or not any(len(batch) for batch in batches):
        raise ValueError("Cutoff-aware fitting found no supported pairs")
    values: dict[str, np.ndarray] = {}
    for name in CutoffPairBatch.__dataclass_fields__:
        pieces: list[np.ndarray] = []
        for batch, query_offset, residual_offset in zip(
            batches, query_offsets, residual_offsets
        ):
            value = np.asarray(getattr(batch, name)).copy()
            if name == "query":
                value += query_offset
            elif name in ("positive_residual_row", "challenger_residual_row"):
                value += residual_offset
            pieces.append(value)
        values[name] = np.concatenate(pieces)
    values["balanced_weight"] = query_role_balanced_weights(
        values["query"],
        values["kind"],
        values["raw_weight"],
        promotion_mass=promotion_mass,
    )
    return CutoffPairBatch(**values)


def _fit_cutoff_basis(
    bundles: Iterable[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    bundles = list(bundles)
    config = protocol["diagnostic_configuration"]
    optimization = protocol["basis_optimization"]
    rank = int(config["rars_rank"])
    query_offsets: list[int] = []
    residual_offsets: list[int] = []
    query_count = 0
    residual_count = 0
    pair_batches: list[CutoffPairBatch] = []
    for bundle in bundles:
        query_offsets.append(query_count)
        residual_offsets.append(residual_count)
        query_count += bundle["query_count"]
        residual_count += len(
            bundle["arrays"]["candidate_residuals.float32.npy"]
        )
        pair_batches.append(_mine_bundle_pairs(bundle, protocol))
    pairs = _combine_pairs(
        pair_batches,
        query_offsets,
        residual_offsets,
        promotion_mass=float(
            protocol["pair_mining"]["promotion_total_loss_mass"]
        ),
    )
    queries = np.concatenate(
        [
            np.asarray(
                bundle["arrays"]["query_vectors.float32.npy"], dtype=np.float32
            )
            for bundle in bundles
        ]
    )
    residuals = np.concatenate(
        [
            np.asarray(
                bundle["arrays"]["candidate_residuals.float32.npy"],
                dtype=np.float32,
            )
            for bundle in bundles
        ]
    )
    anchor = _fit_pca_from_bundles(bundles, rank=rank)
    basis, history = fit_cutoff_aware_basis(
        queries,
        residuals,
        pairs,
        anchor,
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
        anchor_weight=float(optimization["pca_anchor_weight"]),
        huber_delta=float(optimization["huber_delta"]),
        gradient_clip=float(optimization["gradient_norm_clip"]),
    )
    return basis, {
        "fit_domain_count": len(bundles),
        "fit_query_count": query_count,
        "fit_residual_count": residual_count,
        "pair_support": summarize_pairs(pairs),
        "optimization_initial_loss": float(history[0]["loss"]),
        "optimization_final_loss": float(history[-1]["loss"]),
    }


def _score_fp32(
    bundle: dict[str, Any], basis: np.ndarray, *, alpha: float, top_b: int
) -> np.ndarray:
    arrays = bundle["arrays"]
    return score_fp32_sidecar_candidates(
        arrays["query_vectors.float32.npy"],
        arrays["ann_rows.int64.npy"],
        arrays["ann_residual_rows.int64.npy"],
        arrays["ann_scores.float32.npy"],
        arrays["candidate_residuals.float32.npy"],
        basis,
        alpha=alpha,
        top_b=top_b,
    )


def _score_int8(
    bundle: dict[str, Any], basis: np.ndarray, *, alpha: float, top_b: int
) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = bundle["arrays"]
    residuals = arrays["candidate_residuals.float32.npy"]
    scales = fit_int8_scales(residuals, basis)
    codes, quantization = encode_residuals_int8(residuals, basis, scales)
    scores = score_sidecar_candidates(
        arrays["query_vectors.float32.npy"],
        arrays["ann_rows.int64.npy"],
        arrays["ann_residual_rows.int64.npy"],
        arrays["ann_scores.float32.npy"],
        basis,
        codes,
        scales,
        alpha=alpha,
        top_b=top_b,
    )
    return scores, {
        **quantization,
        "rank": int(basis.shape[1]),
        "payload_bytes_per_materialized_candidate_residual": int(
            basis.shape[1]
        ),
        "scales_fit_without_labels": True,
    }


def _metrics(
    bundle: dict[str, Any], scores: np.ndarray, *, final_k: int
) -> dict[str, np.ndarray]:
    arrays = bundle["arrays"]
    return per_query_metrics(
        scores,
        arrays["ann_rows.int64.npy"],
        arrays["candidate_relevance.uint8.npy"],
        arrays["relevant_counts.int32.npy"],
        k=final_k,
    )


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    if not normalized:
        raise ValueError("Cannot construct a safe artifact name")
    return normalized


def _inference_controls(
    protocol: dict[str, Any], label: str
) -> dict[str, Any]:
    contract = protocol["inference"]
    seed_base = int(contract["seed_base"])
    offset = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return {
        "bootstrap_replicates": int(contract["bootstrap_replicates"]),
        "bootstrap_seed": (seed_base + offset) % (2**31 - 1),
        "randomization_replicates": int(contract["randomization_replicates"]),
        "randomization_seed": (seed_base + offset + 1) % (2**31 - 1),
        "confidence": float(contract["confidence"]),
    }


def _summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(value)) for name, value in metrics.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    for path, label in (
        (args.domain_manifest, "domain manifest"),
        (args.output_dir, "output"),
        (args.protocol, "protocol"),
    ):
        _reject_forbidden_closed_path(path, label)
    protocol, source_blobs = validate_source(
        repo_root, args.protocol, args.source_commit
    )
    source_domain, domain_paths, domain_manifest_record = load_domain_manifest(
        args.domain_manifest, protocol
    )
    _prepare_empty_output(args.output_dir)

    bundles: dict[str, dict[str, dict[str, Any]]] = {}
    encoder_keys: set[tuple[str, str, int]] = set()
    index_recipes: set[tuple[int, int, int, int, int]] = set()
    input_records: dict[str, Any] = {
        "domain_manifest": domain_manifest_record,
        "domains": {},
    }
    for domain_id, roles in domain_paths.items():
        fit = load_bundle(
            roles["fit"],
            expected_domain=domain_id,
            expected_role="fit",
            protocol=protocol,
        )
        evaluation = load_bundle(
            roles["evaluation"],
            expected_domain=domain_id,
            expected_role="evaluation",
            protocol=protocol,
        )
        if fit["encoder_key"] != evaluation["encoder_key"]:
            raise ValueError(f"{domain_id} fit/evaluation encoder changed")
        fit_index = fit["manifest"].get("index_contract", {})
        evaluation_index = evaluation["manifest"].get("index_contract", {})
        recipe = tuple(
            int(fit_index.get(key, -1))
            for key in (
                "nlist",
                "nprobe",
                "subquantizers",
                "bits_per_subquantizer",
                "metric_type",
            )
        )
        evaluation_recipe = tuple(
            int(evaluation_index.get(key, -1))
            for key in (
                "nlist",
                "nprobe",
                "subquantizers",
                "bits_per_subquantizer",
                "metric_type",
            )
        )
        if recipe != evaluation_recipe or any(value < 0 for value in recipe):
            raise ValueError(f"{domain_id} fit/evaluation index recipe changed")
        index_recipes.add(recipe)
        if set(fit["qids"]) & set(evaluation["qids"]):
            raise ValueError(f"{domain_id} fit/evaluation query IDs overlap")
        encoder_keys.add(fit["encoder_key"])
        bundles[domain_id] = {"fit": fit, "evaluation": evaluation}
        input_records["domains"][domain_id] = {
            "fit": fit["records"],
            "evaluation": evaluation["records"],
        }
    if len(encoder_keys) != 1:
        raise ValueError("V16 primary diagnostic requires one encoder revision")
    if len(index_recipes) != 1:
        raise ValueError("V16 domains must use one IVF-PQ index recipe")

    config = protocol["diagnostic_configuration"]
    alpha = float(config["alpha"])
    top_b = int(config["top_b"])
    final_k = int(config["final_k"])
    fit_bundles = {
        domain_id: roles["fit"] for domain_id, roles in bundles.items()
    }
    source_basis, source_fit = _fit_cutoff_basis(
        [fit_bundles[source_domain]], protocol
    )
    pooled_basis, pooled_fit = _fit_cutoff_basis(
        fit_bundles.values(), protocol
    )

    started_path = args.output_dir / "diagnostic_started.json"
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "RARS_V16_MECHANISM_DIAGNOSTIC_STARTED",
            "source_commit": args.source_commit,
            "source_blobs": source_blobs,
            "inputs": input_records,
            "source_domain_id": source_domain,
            "domain_ids": list(bundles),
            "same_encoder_required": True,
            "evaluation_used_for_selection": False,
            "full_corpus_materialized": False,
            "closed_confirmation_role_opened": False,
        },
    )

    output_records: dict[str, Any] = {}
    domain_results: dict[str, Any] = {}
    observed_by_domain: dict[str, dict[str, float | int]] = {}
    for domain_id, roles in bundles.items():
        fit = roles["fit"]
        evaluation = roles["evaluation"]
        local_basis, local_fit = _fit_cutoff_basis([fit], protocol)
        pca16 = fit_uncentered_pca_basis(
            fit["arrays"]["candidate_residuals.float32.npy"], rank=16
        )
        pca64 = fit_uncentered_pca_basis(
            fit["arrays"]["candidate_residuals.float32.npy"], rank=64
        )
        base_scores = np.asarray(
            evaluation["arrays"]["ann_scores.float32.npy"], dtype=np.float32
        )
        exact_scores = _exact_candidate_scores(evaluation)
        pca16_int8_scores, pca16_quantization = _score_int8(
            evaluation, pca16, alpha=alpha, top_b=top_b
        )
        source_int8_scores, source_quantization = _score_int8(
            evaluation, source_basis, alpha=alpha, top_b=top_b
        )
        local_int8_scores, local_quantization = _score_int8(
            evaluation, local_basis, alpha=alpha, top_b=top_b
        )
        pooled_int8_scores, pooled_quantization = _score_int8(
            evaluation, pooled_basis, alpha=alpha, top_b=top_b
        )
        score_map = {
            "base": base_scores,
            "same_candidate_exact": exact_scores,
            "pca_local_r16_fp32": _score_fp32(
                evaluation, pca16, alpha=alpha, top_b=top_b
            ),
            "pca_local_r16_int8": pca16_int8_scores,
            "pca_local_r64_fp32": _score_fp32(
                evaluation, pca64, alpha=alpha, top_b=top_b
            ),
            "rars_source_r16_int8": source_int8_scores,
            "rars_local_r16_int8": local_int8_scores,
            "rars_pooled_r16_int8": pooled_int8_scores,
        }
        metrics = {
            method: _metrics(evaluation, scores, final_k=final_k)
            for method, scores in score_map.items()
        }
        contrast_methods = {
            "candidate_headroom": ("same_candidate_exact", "base"),
            "rank_capacity": ("pca_local_r64_fp32", "pca_local_r16_fp32"),
            "int8_coding_loss": (
                "pca_local_r16_fp32",
                "pca_local_r16_int8",
            ),
            "objective_value": (
                "rars_local_r16_int8",
                "pca_local_r16_int8",
            ),
            "fit_domain_interaction": (
                "rars_local_r16_int8",
                "rars_source_r16_int8",
            ),
            "pooled_repair": (
                "rars_pooled_r16_int8",
                "rars_source_r16_int8",
            ),
            "pooled_vs_local": (
                "rars_pooled_r16_int8",
                "rars_local_r16_int8",
            ),
            "pooled_vs_base": ("rars_pooled_r16_int8", "base"),
        }
        comparisons = {
            name: paired_query_inference(
                metrics[treatment]["recall"],
                metrics[baseline]["recall"],
                **_inference_controls(protocol, f"{domain_id}:{name}"),
            )
            for name, (treatment, baseline) in contrast_methods.items()
        }
        gap = candidate_gap_decomposition(
            metrics["rars_pooled_r16_int8"]["recall"],
            metrics["base"]["recall"],
            metrics["same_candidate_exact"]["recall"],
        )
        local_advantage = comparisons["fit_domain_interaction"][
            "mean_difference"
        ]
        pooled_advantage = comparisons["pooled_repair"]["mean_difference"]
        pooled_recovery = (
            pooled_advantage / local_advantage if local_advantage > 0 else 0.0
        )
        observed_by_domain[domain_id] = {
            "n_queries": evaluation["query_count"],
            "headroom": comparisons["candidate_headroom"]["mean_difference"],
            "capacity_gain": comparisons["rank_capacity"]["mean_difference"],
            "coding_gap": comparisons["int8_coding_loss"]["mean_difference"],
            "objective_gain": comparisons["objective_value"]["mean_difference"],
            "domain_interaction": comparisons["fit_domain_interaction"][
                "mean_difference"
            ],
            "pooled_recovery": pooled_recovery,
            "pooled_gain": pooled_advantage,
            "improved_queries": comparisons["pooled_vs_base"][
                "improved_queries"
            ],
            "harmed_queries": comparisons["pooled_vs_base"]["harmed_queries"],
            "gap_recovery": gap["gap_recovery_fraction"],
            "worst_domain_gain": comparisons["pooled_vs_base"][
                "mean_difference"
            ],
        }

        safe_domain = _safe_name(domain_id)
        for method, values in metrics.items():
            for metric, array in values.items():
                name = (
                    f"{safe_domain}__{method}__{metric}_at_"
                    f"{final_k}.float64.npy"
                )
                path = args.output_dir / name
                atomic_save(path, array)
                output_records[name] = file_record(path)
        basis_outputs = {
            f"{safe_domain}__rars_local_r16_basis.float32.npy": local_basis,
            f"{safe_domain}__pca_local_r16_basis.float32.npy": pca16,
            f"{safe_domain}__pca_local_r64_basis.float32.npy": pca64,
        }
        for name, array in basis_outputs.items():
            path = args.output_dir / name
            atomic_save(path, array)
            output_records[name] = file_record(path)
        domain_results[domain_id] = {
            "query_count": evaluation["query_count"],
            "encoder_id": evaluation["encoder_key"][0],
            "encoder_revision": evaluation["encoder_key"][1],
            "mean_metrics": {
                method: _summary(values) for method, values in metrics.items()
            },
            "comparisons": comparisons,
            "candidate_gap": gap,
            "fit_diagnostics": {
                "source": source_fit,
                "local": local_fit,
                "pooled": pooled_fit,
            },
            "quantization": {
                "pca_local_r16_int8": pca16_quantization,
                "rars_source_r16_int8": source_quantization,
                "rars_local_r16_int8": local_quantization,
                "rars_pooled_r16_int8": pooled_quantization,
            },
            "subspace_alignment": {
                "local_rars_vs_local_pca16": subspace_alignment_metrics(
                    local_basis, pca16
                ),
                "local_rars_vs_source_rars": subspace_alignment_metrics(
                    local_basis, source_basis
                ),
                "local_rars_vs_pooled_rars": subspace_alignment_metrics(
                    local_basis, pooled_basis
                ),
            },
            "observed_factor_values": observed_by_domain[domain_id],
        }

    shared_outputs = {
        "rars_source_r16_basis.float32.npy": source_basis,
        "rars_pooled_r16_basis.float32.npy": pooled_basis,
    }
    for name, array in shared_outputs.items():
        path = args.output_dir / name
        atomic_save(path, array)
        output_records[name] = file_record(path)

    equal_domain = {
        key: float(np.mean([float(values[key]) for values in observed_by_domain.values()]))
        for key in (
            "headroom",
            "capacity_gain",
            "coding_gap",
            "objective_gain",
            "domain_interaction",
            "pooled_recovery",
            "pooled_gain",
            "gap_recovery",
        )
    }
    equal_domain["n_queries"] = int(
        sum(int(values["n_queries"]) for values in observed_by_domain.values())
    )
    equal_domain["improved_queries"] = int(
        sum(
            int(values["improved_queries"])
            for values in observed_by_domain.values()
        )
    )
    equal_domain["harmed_queries"] = int(
        sum(
            int(values["harmed_queries"])
            for values in observed_by_domain.values()
        )
    )
    equal_domain["worst_domain_gain"] = float(
        min(
            float(values["worst_domain_gain"])
            for values in observed_by_domain.values()
        )
    )
    decision = causal_decision(
        equal_domain, protocol["diagnostic_thresholds"]
    )
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V16_MECHANISM_DIAGNOSTIC_COMPLETE",
        "source_commit": args.source_commit,
        "scope": "outcome-informed same-encoder prepared-bundle diagnostic",
        "source_domain_id": source_domain,
        "same_encoder_required": True,
        "domains": domain_results,
        "equal_domain_factor_summary": equal_domain,
        "diagnostic_decision": decision,
        "candidate_ceiling_evaluated": True,
        "rank_capacity_evaluated": True,
        "int8_coding_evaluated": True,
        "cutoff_objective_evaluated": True,
        "fit_domain_interaction_evaluated": True,
        "full_corpus_materialized": False,
        "closed_confirmation_role_opened": False,
        "evaluation_used_for_selection": False,
        "confirmatory_claim_allowed": False,
        "outputs": output_records,
    }
    result_path = args.output_dir / "diagnostic_result.json"
    atomic_json(result_path, result)
    complete = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RARS_V16_MECHANISM_DIAGNOSTIC_COMPLETE",
        "source_commit": args.source_commit,
        "started": file_record(started_path),
        "result": file_record(result_path),
        "outputs": output_records,
        "all_registered_bundle_hashes_verified": True,
        "full_corpus_materialized": False,
        "closed_confirmation_role_opened": False,
        "evaluation_used_for_selection": False,
    }
    atomic_json(args.output_dir / "diagnostic_complete.json", complete)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / CANONICAL_PROTOCOL,
    )
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
