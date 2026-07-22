#!/usr/bin/env python3
"""Fit and select storage-matched PCA and RARS sidecars on BEIR NQ train data.

This command consumes only the Stage-1 corpus/index artifacts and the
deterministic fit/validation query package.  It never accepts a test-qrels
argument.  Candidate caches use fit queries for RARS basis construction and
validation queries for the registered qrels-free exact-candidate-overlap
selection rule.

The implementation streams IVF-PQ residuals.  It does not materialize the
full FP32 residual matrix on Google Drive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols" / "beir_nq_rars_pca_confirmation_v1.json"
PROTOCOL_ID = "beir_nq_rars_pca_confirmation_v1"


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


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install faiss-gpu-cu12 in the Colab runtime") from exc
    return faiss


def verify_stage0_gate(artifact_root: Path, repo: Path, protocol_path: Path) -> None:
    path = ROOT / "scripts" / "prepare_beir_nq_colab.py"
    spec = importlib.util.spec_from_file_location("nq_stage0_gate", path)
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load the Stage-0 gate verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_gate(artifact_root, repo, protocol_path)


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


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected protocol ID")
    base = protocol.get("base_index", {})
    sidecar = protocol.get("shared_sidecar", {})
    pca = protocol.get("pca", {})
    rars = protocol.get("rars", {})
    expected = {
        "base_index.m": (base.get("m"), 32),
        "base_index.nbits": (base.get("nbits"), 8),
        "base_index.nlist": (base.get("nlist"), 2048),
        "base_index.nprobe": (base.get("nprobe"), 32),
        "base_index.candidate_k": (base.get("candidate_k"), 100),
        "base_index.final_k": (base.get("final_k"), 10),
        "base_index.search_backend": (
            base.get("search_backend"),
            "single_faiss_gpu_nvidia_t4",
        ),
        "shared_sidecar.rank": (sidecar.get("rank"), 16),
        "shared_sidecar.coefficient_dtype": (
            sidecar.get("coefficient_dtype"),
            "int8",
        ),
        "pca.residual_sample_max": (pca.get("residual_sample_max"), 300_000),
        "pca.sample_seed": (pca.get("sample_seed"), 42),
        "rars.residual_sample_draws": (
            rars.get("residual_sample_draws"),
            300_000,
        ),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"Protocol drift for {label}: {actual!r} != {wanted!r}")
    if pca.get("qrels_used") is not False or rars.get("qrels_used") is not False:
        raise ValueError("PCA/RARS qrels flags must remain false")
    if rars.get("method_revision_allowed") is not False:
        raise ValueError("RARS method revision flag drifted")


def validate_partition(path: Path, name: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("partition") != name:
        raise ValueError(f"Unexpected {name} query manifest")
    if payload.get("qrels_relevance_values_used") is not False:
        raise ValueError("Train relevance values may not be used")
    if payload.get("test_qrels_accessed") is not False:
        raise ValueError("Unsafe test-qrels flag")
    qids = [str(value) for value in payload.get("query_ids", [])]
    if not qids or len(qids) != len(set(qids)):
        raise ValueError(f"Invalid {name} query IDs")
    if int(payload.get("query_count", -1)) != len(qids):
        raise ValueError(f"Invalid {name} query count")
    return payload


def clone_to_gpu(faiss: Any, cpu_index: Any) -> tuple[Any, Any]:
    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
        raise RuntimeError("The frozen search backend requires Faiss GPU support")
    resources = faiss.StandardGpuResources()
    if hasattr(resources, "setTempMemory"):
        resources.setTempMemory(512 * 1024 * 1024)
    options = faiss.GpuClonerOptions()
    options.useFloat16LookupTables = False
    options.useFloat16CoarseQuantizer = False
    return resources, faiss.index_cpu_to_gpu(resources, 0, cpu_index, options)


def validate_index(index: Any, faiss: Any, *, n_docs: int, dim: int) -> None:
    if int(index.ntotal) != n_docs:
        raise ValueError(f"Index ntotal {index.ntotal} != corpus count {n_docs}")
    if int(index.d) != dim:
        raise ValueError(f"Index dimension {index.d} != {dim}")
    if int(index.nlist) != 2048:
        raise ValueError("Frozen index nlist drifted")
    if int(index.pq.M) != 32 or int(index.pq.nbits) != 8:
        raise ValueError("Frozen index PQ configuration drifted")
    if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
        raise ValueError("Frozen index metric is not inner product")


def open_npy_memmap(path: Path, shape: tuple[int, ...], dtype: Any, mode: str) -> np.memmap:
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


def build_ann_cache(
    faiss: Any,
    cpu_index: Any,
    queries: np.ndarray,
    output_dir: Path,
    *,
    top_k: int,
    nprobe: int,
    batch_size: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "ann_rows.int64.npy"
    scores_path = output_dir / "ann_scores.float32.npy"
    progress_path = output_dir / "ann_progress.json"
    shape = (len(queries), top_k)
    if rows_path.exists() and scores_path.exists() and not progress_path.exists():
        return rows_path, scores_path

    rows_part = rows_path.with_name(rows_path.name + ".part")
    scores_part = scores_path.with_name(scores_path.name + ".part")
    if progress_path.exists():
        progress = read_json(progress_path)
        completed = int(progress["queries_completed"])
        mode = "r+"
    else:
        completed = 0
        mode = "w+"
    rows = open_npy_memmap(rows_part, shape, np.int64, mode)
    scores = open_npy_memmap(scores_part, shape, np.float32, mode)
    cpu_index.nprobe = nprobe
    resources, gpu_index = clone_to_gpu(faiss, cpu_index)
    gpu_index.nprobe = nprobe
    for start in range(completed, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch = np.ascontiguousarray(queries[start:end].astype(np.float32))
        batch_scores, batch_rows = gpu_index.search(batch, top_k)
        rows[start:end] = batch_rows.astype(np.int64)
        scores[start:end] = batch_scores.astype(np.float32)
        rows.flush()
        scores.flush()
        atomic_write_json(progress_path, {
            "protocol_id": PROTOCOL_ID,
            "queries_completed": end,
            "query_count": len(queries),
            "candidate_k": top_k,
            "test_qrels_accessed": False,
        })
        if end % 1000 == 0 or end == len(queries):
            print(f"ANN searched {end:,}/{len(queries):,} queries")
    del gpu_index, resources, rows, scores
    rows_part.replace(rows_path)
    scores_part.replace(scores_path)
    progress_path.unlink(missing_ok=True)
    return rows_path, scores_path


def build_exact_score_cache(
    queries: np.ndarray,
    document_embeddings: np.memmap,
    ann_rows_path: Path,
    output_path: Path,
    *,
    batch_size: int,
) -> Path:
    ann_rows = np.load(ann_rows_path, mmap_mode="r")
    shape = tuple(int(value) for value in ann_rows.shape)
    if output_path.exists() and not output_path.with_suffix(output_path.suffix + ".progress.json").exists():
        return output_path
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    temporary = output_path.with_name(output_path.name + ".part")
    if progress_path.exists():
        completed = int(read_json(progress_path)["queries_completed"])
        mode = "r+"
    else:
        completed = 0
        mode = "w+"
    exact = open_npy_memmap(temporary, shape, np.float32, mode)
    for start in range(completed, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        ids = np.asarray(ann_rows[start:end], dtype=np.int64)
        valid = ids >= 0
        safe_ids = np.where(valid, ids, 0)
        docs = np.asarray(document_embeddings[safe_ids], dtype=np.float32)
        q = np.asarray(queries[start:end], dtype=np.float32)
        values = np.einsum("bkd,bd->bk", docs, q, optimize=True).astype(np.float32)
        values[~valid] = -np.inf
        exact[start:end] = values
        exact.flush()
        atomic_write_json(progress_path, {
            "protocol_id": PROTOCOL_ID,
            "queries_completed": end,
            "query_count": len(queries),
            "test_qrels_accessed": False,
        })
        if end % 1000 == 0 or end == len(queries):
            print(f"exact-scored {end:,}/{len(queries):,} queries")
    del exact
    temporary.replace(output_path)
    progress_path.unlink(missing_ok=True)
    return output_path


def orient_basis_deterministically(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float32).copy()
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            result[:, column_index] *= -1.0
    return result


def reconstruct_residuals(
    index: Any,
    embeddings: np.memmap,
    rows: np.ndarray,
) -> np.ndarray:
    row_ids = np.asarray(rows, dtype=np.int64)
    reconstructed = index.reconstruct_batch(row_ids).astype(np.float32)
    original = np.asarray(embeddings[row_ids], dtype=np.float32)
    return original - reconstructed


def residual_covariance_basis(
    index: Any,
    embeddings: np.memmap,
    rows: np.ndarray,
    *,
    rank: int,
    batch_size: int,
    compute_device: str,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        raise ValueError("No residual rows were supplied")
    if counts is not None:
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != rows.shape or np.any(counts <= 0):
            raise ValueError("Residual covariance counts are invalid")
    dimension = int(index.d)
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    torch = None
    if compute_device == "cuda":
        assert_t4()
        import torch as torch_module
        torch = torch_module
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        residual = reconstruct_residuals(index, embeddings, rows[start:end])
        if counts is not None:
            residual *= np.sqrt(counts[start:end, None]).astype(np.float32)
        if torch is not None:
            tensor = torch.from_numpy(np.ascontiguousarray(residual)).to("cuda")
            product = (tensor.T @ tensor).cpu().numpy()
            covariance += product.astype(np.float64)
            del tensor
        else:
            covariance += (residual.T @ residual).astype(np.float64)
        if end % 50_000 == 0 or end == len(rows):
            print(f"basis covariance {end:,}/{len(rows):,} residual rows")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")[:rank]
    return orient_basis_deterministically(eigenvectors[:, order])


def pca_sample_rows(n_docs: int, sample_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(n_docs, size=min(n_docs, sample_count), replace=False).astype(np.int64)
    )


def aggregate_score_error_weights(
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    exact_scores: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    rows = np.asarray(ann_rows).reshape(-1).astype(np.int64)
    ann = np.asarray(ann_scores).reshape(-1).astype(np.float64)
    exact = np.asarray(exact_scores).reshape(-1).astype(np.float64)
    valid = (rows >= 0) & np.isfinite(ann) & np.isfinite(exact)
    errors = np.abs(exact[valid] - ann[valid])
    weights = np.bincount(
        rows[valid],
        weights=errors,
        minlength=n_docs,
    ).astype(np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError("Fit-query score errors have no positive finite mass")
    return weights


def rars_weighted_draws(
    weights: np.ndarray,
    draw_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    positive_rows = np.flatnonzero(weights > 0).astype(np.int64)
    probabilities = weights[positive_rows].astype(np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        positive_rows,
        size=draw_count,
        replace=True,
        p=probabilities,
    ).astype(np.int64)
    unique, counts = np.unique(draws, return_counts=True)
    return unique.astype(np.int64), counts.astype(np.int64)


def build_sidecar_codes(
    index: Any,
    embeddings: np.memmap,
    bases: dict[str, np.ndarray],
    output_dir: Path,
    *,
    n_docs: int,
    rank: int,
    batch_size: int,
    checkpoint_rows: int,
) -> dict[str, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: {
            "scales": output_dir / name / "scales.float32.npy",
            "codes": output_dir / name / "codes.rank16.int8.memmap",
        }
        for name in bases
    }
    if all(value["scales"].is_file() and value["codes"].is_file() for value in paths.values()):
        return paths

    progress_path = output_dir / "sidecar_encoding_progress.json"
    if progress_path.exists():
        progress = read_json(progress_path)
    else:
        progress = {
            "protocol_id": PROTOCOL_ID,
            "stage": "maxabs",
            "rows_completed": 0,
            "max_abs": {name: [0.0] * rank for name in bases},
            "test_qrels_accessed": False,
        }

    if progress["stage"] == "maxabs":
        max_abs = {
            name: np.asarray(progress["max_abs"][name], dtype=np.float32)
            for name in bases
        }
        start_row = int(progress["rows_completed"])
        last_checkpoint = start_row
        for start in range(start_row, n_docs, batch_size):
            end = min(start + batch_size, n_docs)
            rows = np.arange(start, end, dtype=np.int64)
            residual = reconstruct_residuals(index, embeddings, rows)
            for name, basis in bases.items():
                coefficients = residual @ basis
                max_abs[name] = np.maximum(
                    max_abs[name],
                    np.max(np.abs(coefficients), axis=0),
                )
            if end - last_checkpoint >= checkpoint_rows or end == n_docs:
                progress.update({
                    "rows_completed": end,
                    "max_abs": {name: value.tolist() for name, value in max_abs.items()},
                })
                atomic_write_json(progress_path, progress)
                last_checkpoint = end
                print(f"sidecar maxabs {end:,}/{n_docs:,}")
        for name, value in max_abs.items():
            paths[name]["scales"].parent.mkdir(parents=True, exist_ok=True)
            atomic_save_npy(paths[name]["scales"], (value + 1e-12) / 127.0)
        progress = {
            "protocol_id": PROTOCOL_ID,
            "stage": "codes",
            "rows_completed": 0,
            "test_qrels_accessed": False,
        }
        atomic_write_json(progress_path, progress)

    scales = {
        name: np.load(value["scales"]).astype(np.float32)
        for name, value in paths.items()
    }
    start_row = int(progress["rows_completed"])
    code_memmaps: dict[str, np.memmap] = {}
    for name, value in paths.items():
        temporary = value["codes"].with_name(value["codes"].name + ".part")
        code_memmaps[name] = np.memmap(
            temporary,
            dtype=np.int8,
            mode="r+" if start_row else "w+",
            shape=(n_docs, rank),
        )
    last_checkpoint = start_row
    for start in range(start_row, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        rows = np.arange(start, end, dtype=np.int64)
        residual = reconstruct_residuals(index, embeddings, rows)
        for name, basis in bases.items():
            coefficients = residual @ basis
            code_memmaps[name][start:end] = np.clip(
                np.rint(coefficients / scales[name][None, :]),
                -127,
                127,
            ).astype(np.int8)
        if end - last_checkpoint >= checkpoint_rows or end == n_docs:
            for value in code_memmaps.values():
                value.flush()
            progress["rows_completed"] = end
            atomic_write_json(progress_path, progress)
            last_checkpoint = end
            print(f"sidecar codes {end:,}/{n_docs:,}")
    for value in code_memmaps.values():
        value.flush()
    del code_memmaps
    for value in paths.values():
        temporary = value["codes"].with_name(value["codes"].name + ".part")
        temporary.replace(value["codes"])
    progress_path.unlink(missing_ok=True)
    return paths


def stable_descending_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), axis=1, kind="stable")


def proxy_metrics(
    corrected_scores: np.ndarray,
    exact_scores: np.ndarray,
    final_k: int,
) -> tuple[float, float]:
    finite = np.isfinite(exact_scores) & np.isfinite(corrected_scores)
    mse = float(np.mean((corrected_scores[finite] - exact_scores[finite]) ** 2))
    exact_order = stable_descending_order(exact_scores)
    corrected_order = stable_descending_order(corrected_scores)
    overlap = np.mean([
        len(
            set(exact_order[row, :final_k].tolist())
            & set(corrected_order[row, :final_k].tolist())
        ) / final_k
        for row in range(len(exact_scores))
    ])
    return mse, float(overlap)


def correction_matrix(
    queries: np.ndarray,
    ann_rows: np.ndarray,
    basis: np.ndarray,
    codes: np.memmap,
    scales: np.ndarray,
    top_b: int,
) -> np.ndarray:
    result = np.zeros(ann_rows.shape, dtype=np.float32)
    ids = np.asarray(ann_rows[:, :top_b], dtype=np.int64)
    valid = ids >= 0
    safe_ids = np.where(valid, ids, 0)
    coefficients = codes[safe_ids].astype(np.float32) * scales[None, None, :]
    query_projection = np.asarray(queries, dtype=np.float32) @ basis
    values = np.einsum(
        "qbr,qr->qb",
        coefficients,
        query_projection,
        optimize=True,
    ).astype(np.float32)
    values[~valid] = 0.0
    result[:, :top_b] = values
    return result


def validation_grid(
    method: str,
    queries: np.ndarray,
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    exact_scores: np.ndarray,
    basis: np.ndarray,
    codes: np.memmap,
    scales: np.ndarray,
    alphas: Iterable[float],
    top_b_values: Iterable[int],
    final_k: int,
) -> list[dict[str, Any]]:
    base_mse, base_overlap = proxy_metrics(ann_scores, exact_scores, final_k)
    rows: list[dict[str, Any]] = []
    for top_b in top_b_values:
        correction = correction_matrix(
            queries,
            ann_rows,
            basis,
            codes,
            scales,
            int(top_b),
        )
        for alpha in alphas:
            corrected = ann_scores + float(alpha) * correction
            mse, overlap = proxy_metrics(corrected, exact_scores, final_k)
            rows.append({
                "method": method,
                "alpha": float(alpha),
                "top_b": int(top_b),
                "base_mse": base_mse,
                "corrected_mse": mse,
                "mse_reduction_pct": (base_mse - mse) / base_mse * 100.0,
                "base_top10_overlap": base_overlap,
                "corrected_top10_overlap": overlap,
                "overlap_gain": overlap - base_overlap,
            })
    return rows


def select_registered_configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Validation grid is empty")
    maximum_gain = max(float(row["overlap_gain"]) for row in rows)
    threshold = 0.90 * maximum_gain
    eligible = [
        row for row in rows
        if float(row["overlap_gain"]) > threshold
        or np.isclose(float(row["overlap_gain"]), threshold, rtol=0.0, atol=1e-12)
    ]
    if not eligible:
        raise ValueError("No configuration passes the registered gain threshold")
    smallest_top_b = min(int(row["top_b"]) for row in eligible)
    eligible = [row for row in eligible if int(row["top_b"]) == smallest_top_b]
    best = sorted(
        eligible,
        key=lambda row: (
            -float(row["corrected_top10_overlap"]),
            abs(float(row["alpha"])),
            float(row["alpha"]),
        ),
    )[0].copy()
    best["maximum_validation_overlap_gain"] = maximum_gain
    best["selection_threshold"] = threshold
    return best


def write_grid_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_selected_config(
    method_id: str,
    best: dict[str, Any],
    protocol: dict[str, Any],
    *,
    basis_training: dict[str, Any],
) -> dict[str, Any]:
    base = protocol["base_index"]
    return {
        "protocol_id": PROTOCOL_ID,
        "method_id": method_id,
        "selection_split": "validation",
        "basis_training": basis_training,
        "rank": int(protocol["shared_sidecar"]["rank"]),
        "coefficient_dtype": "int8",
        "quantizer": protocol["shared_sidecar"]["quantizer"],
        "alpha": float(best["alpha"]),
        "top_b": int(best["top_b"]),
        "candidate_k": int(base["candidate_k"]),
        "final_k": int(base["final_k"]),
        "base_index": {
            "nlist": int(base["nlist"]),
            "nprobe": int(base["nprobe"]),
            "m": int(base["m"]),
            "nbits": int(base["nbits"]),
        },
        "validation_proxy": {
            "base_top10_overlap": float(best["base_top10_overlap"]),
            "corrected_top10_overlap": float(best["corrected_top10_overlap"]),
            "overlap_gain": float(best["overlap_gain"]),
            "mse_reduction_pct": float(best["mse_reduction_pct"]),
            "maximum_overlap_gain": float(best["maximum_validation_overlap_gain"]),
            "selection_threshold": float(best["selection_threshold"]),
        },
        "train_qrels_relevance_values_used": False,
        "validation_qrels_used": False,
        "test_qrels_accessed": False,
        "test_retrieval_performed": False,
        "test_outcomes_observed": False,
    }


def run_training(
    artifact_root: Path,
    repo: Path,
    protocol_path: Path,
    *,
    search_batch_size: int,
    exact_batch_size: int,
    residual_batch_size: int,
    checkpoint_rows: int,
    compute_device: str,
) -> dict[str, Any]:
    verify_stage0_gate(artifact_root, repo, protocol_path)
    assert_t4()
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    output_dir = artifact_root / "stage2" / "sidecars"
    complete_manifest_path = output_dir / "sidecar_training_manifest.json"
    if complete_manifest_path.is_file():
        return read_json(complete_manifest_path)

    corpus_manifest_path = artifact_root / "stage1" / "corpus" / "corpus_artifacts_manifest.json"
    index_manifest_path = artifact_root / "stage1" / "index" / "index_manifest.json"
    vector_manifest_path = artifact_root / "stage1" / "queries" / "train_validation_query_vector_manifest.json"
    corpus_manifest = read_json(corpus_manifest_path)
    index_manifest = read_json(index_manifest_path)
    vector_manifest = read_json(vector_manifest_path)
    for payload, label in [
        (corpus_manifest, "corpus"),
        (index_manifest, "index"),
        (vector_manifest, "query vector"),
    ]:
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"Unexpected {label} manifest protocol")
        if payload.get("test_qrels_accessed") is not False:
            raise ValueError(f"Unsafe {label} manifest test flag")

    fit_manifest_path = artifact_root / "stage1" / "query_splits" / "train_query_manifest.json"
    val_manifest_path = artifact_root / "stage1" / "query_splits" / "validation_query_manifest.json"
    fit_manifest = validate_partition(fit_manifest_path, "fit")
    val_manifest = validate_partition(val_manifest_path, "validation")
    fit_qids = [str(value) for value in fit_manifest["query_ids"]]
    val_qids = [str(value) for value in val_manifest["query_ids"]]
    if not set(fit_qids).isdisjoint(val_qids):
        raise ValueError("Fit/validation query IDs overlap")

    n_docs = int(corpus_manifest["document_count"])
    dim = int(corpus_manifest["dimension"])
    rank = int(protocol["shared_sidecar"]["rank"])
    embeddings_path = Path(corpus_manifest["document_embeddings"]["path"])
    index_path = Path(index_manifest["index"]["path"])
    vector_path = Path(vector_manifest["vectors"]["path"])
    embeddings = np.memmap(
        embeddings_path,
        dtype=np.float16,
        mode="r",
        shape=(n_docs, dim),
    )
    all_queries = np.load(vector_path, mmap_mode="r")
    if all_queries.shape != (len(fit_qids) + len(val_qids), dim):
        raise ValueError(f"Unexpected query vector shape: {all_queries.shape}")
    fit_queries = np.asarray(all_queries[:len(fit_qids)], dtype=np.float32)
    val_queries = np.asarray(all_queries[len(fit_qids):], dtype=np.float32)

    faiss = import_faiss()
    index = faiss.read_index(str(index_path))
    validate_index(index, faiss, n_docs=n_docs, dim=dim)
    index.nprobe = int(protocol["base_index"]["nprobe"])
    if hasattr(index, "make_direct_map"):
        index.make_direct_map()

    cache_root = output_dir / "candidate_cache"
    fit_rows_path, fit_scores_path = build_ann_cache(
        faiss,
        index,
        fit_queries,
        cache_root / "fit",
        top_k=int(protocol["base_index"]["candidate_k"]),
        nprobe=int(protocol["base_index"]["nprobe"]),
        batch_size=search_batch_size,
    )
    val_rows_path, val_scores_path = build_ann_cache(
        faiss,
        index,
        val_queries,
        cache_root / "validation",
        top_k=int(protocol["base_index"]["candidate_k"]),
        nprobe=int(protocol["base_index"]["nprobe"]),
        batch_size=search_batch_size,
    )
    fit_exact_path = build_exact_score_cache(
        fit_queries,
        embeddings,
        fit_rows_path,
        cache_root / "fit" / "exact_scores.float32.npy",
        batch_size=exact_batch_size,
    )
    val_exact_path = build_exact_score_cache(
        val_queries,
        embeddings,
        val_rows_path,
        cache_root / "validation" / "exact_scores.float32.npy",
        batch_size=exact_batch_size,
    )

    basis_dir = output_dir / "bases"
    basis_dir.mkdir(parents=True, exist_ok=True)
    pca_basis_path = basis_dir / "pca_rank16.float32.npy"
    pca_rows_path = basis_dir / "pca_sample_rows.int64.npy"
    if pca_basis_path.exists() and pca_rows_path.exists():
        pca_basis = np.load(pca_basis_path).astype(np.float32)
        pca_rows = np.load(pca_rows_path).astype(np.int64)
    else:
        pca_rows = pca_sample_rows(
            n_docs,
            int(protocol["pca"]["residual_sample_max"]),
            int(protocol["pca"]["sample_seed"]),
        )
        pca_basis = residual_covariance_basis(
            index,
            embeddings,
            pca_rows,
            rank=rank,
            batch_size=residual_batch_size,
            compute_device=compute_device,
        )
        atomic_save_npy(pca_rows_path, pca_rows)
        atomic_save_npy(pca_basis_path, pca_basis)

    rars_basis_path = basis_dir / "rars_score_error_rank16.float32.npy"
    rars_rows_path = basis_dir / "rars_sample_rows.int64.npy"
    rars_counts_path = basis_dir / "rars_sample_counts.int64.npy"
    if rars_basis_path.exists() and rars_rows_path.exists() and rars_counts_path.exists():
        rars_basis = np.load(rars_basis_path).astype(np.float32)
        rars_rows = np.load(rars_rows_path).astype(np.int64)
        rars_counts = np.load(rars_counts_path).astype(np.int64)
    else:
        weights = aggregate_score_error_weights(
            np.load(fit_rows_path, mmap_mode="r"),
            np.load(fit_scores_path, mmap_mode="r"),
            np.load(fit_exact_path, mmap_mode="r"),
            n_docs,
        )
        rars_rows, rars_counts = rars_weighted_draws(
            weights,
            int(protocol["rars"]["residual_sample_draws"]),
            int(protocol["base_index"]["training_seed"]),
        )
        rars_basis = residual_covariance_basis(
            index,
            embeddings,
            rars_rows,
            counts=rars_counts,
            rank=rank,
            batch_size=residual_batch_size,
            compute_device=compute_device,
        )
        atomic_save_npy(rars_rows_path, rars_rows)
        atomic_save_npy(rars_counts_path, rars_counts)
        atomic_save_npy(rars_basis_path, rars_basis)

    if pca_basis.shape != (dim, rank) or rars_basis.shape != (dim, rank):
        raise ValueError("Unexpected PCA/RARS basis shape")
    sidecar_paths = build_sidecar_codes(
        index,
        embeddings,
        {"pca": pca_basis, "rars": rars_basis},
        output_dir / "encoded",
        n_docs=n_docs,
        rank=rank,
        batch_size=residual_batch_size,
        checkpoint_rows=checkpoint_rows,
    )

    val_rows = np.load(val_rows_path, mmap_mode="r")
    val_ann = np.load(val_scores_path, mmap_mode="r")
    val_exact = np.load(val_exact_path, mmap_mode="r")
    alphas = [float(value) for value in protocol["validation"]["alphas"]]
    top_b_values = [int(value) for value in protocol["validation"]["top_b"]]
    configs: dict[str, dict[str, Any]] = {}
    grid_paths: dict[str, Path] = {}
    for name, method_id, basis in [
        ("pca", "pca_r16_int8", pca_basis),
        ("rars", "rars_r16_int8", rars_basis),
    ]:
        codes = np.memmap(
            sidecar_paths[name]["codes"],
            dtype=np.int8,
            mode="r",
            shape=(n_docs, rank),
        )
        scales = np.load(sidecar_paths[name]["scales"]).astype(np.float32)
        grid = validation_grid(
            name,
            val_queries,
            val_rows,
            val_ann,
            val_exact,
            basis,
            codes,
            scales,
            alphas,
            top_b_values,
            int(protocol["base_index"]["final_k"]),
        )
        grid_path = output_dir / name / "validation_selection.csv"
        write_grid_csv(grid_path, grid)
        grid_paths[name] = grid_path
        best = select_registered_configuration(grid)
        if name == "pca":
            basis_training = {
                "basis": "unweighted_residual_pca",
                "sampling": protocol["pca"]["sampling"],
                "sample_count": len(pca_rows),
                "sample_seed": int(protocol["pca"]["sample_seed"]),
                "query_labels_used": False,
                "qrels_used": False,
            }
        else:
            basis_training = {
                "basis": "score_error_weighted",
                "weight": protocol["rars"]["weight"],
                "sampling": protocol["rars"]["sampling"],
                "draw_count": int(protocol["rars"]["residual_sample_draws"]),
                "unique_sample_rows": len(rars_rows),
                "sample_seed": int(protocol["base_index"]["training_seed"]),
                "fit_query_count": len(fit_qids),
                "query_labels_used": False,
                "qrels_used": False,
            }
        config = build_selected_config(
            method_id,
            best,
            protocol,
            basis_training=basis_training,
        )
        config_path = output_dir / name / "selected_config.json"
        atomic_write_json(config_path, config)
        configs[name] = config

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "package": "beir_nq_pca_rars_train_validation_v1",
        "corpus_document_count": n_docs,
        "fit_query_count": len(fit_qids),
        "validation_query_count": len(val_qids),
        "selected_configs": {
            name: {
                "method_id": value["method_id"],
                "alpha": value["alpha"],
                "top_b": value["top_b"],
            }
            for name, value in configs.items()
        },
        "files": {
            "pca_config": file_record(output_dir / "pca" / "selected_config.json"),
            "pca_validation_grid": file_record(grid_paths["pca"]),
            "pca_basis": file_record(pca_basis_path),
            "pca_scales": file_record(sidecar_paths["pca"]["scales"]),
            "pca_codes": file_record(sidecar_paths["pca"]["codes"]),
            "rars_config": file_record(output_dir / "rars" / "selected_config.json"),
            "rars_validation_grid": file_record(grid_paths["rars"]),
            "rars_basis": file_record(rars_basis_path),
            "rars_scales": file_record(sidecar_paths["rars"]["scales"]),
            "rars_codes": file_record(sidecar_paths["rars"]["codes"]),
            "fit_ann_rows": file_record(fit_rows_path),
            "fit_ann_scores": file_record(fit_scores_path),
            "fit_exact_scores": file_record(fit_exact_path),
            "validation_ann_rows": file_record(val_rows_path),
            "validation_ann_scores": file_record(val_scores_path),
            "validation_exact_scores": file_record(val_exact_path),
        },
        "train_qrels_relevance_values_used": False,
        "validation_qrels_used": False,
        "test_qrels_accessed": False,
        "test_retrieval_performed": False,
        "test_outcomes_observed": False,
        "completed_utc": utc_now(),
    }
    atomic_write_json(complete_manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--search-batch-size", type=int, default=256)
    parser.add_argument("--exact-batch-size", type=int, default=64)
    parser.add_argument("--residual-batch-size", type=int, default=20_000)
    parser.add_argument("--checkpoint-rows", type=int, default=100_000)
    parser.add_argument("--compute-device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        args.artifact_root,
        args.repo,
        args.protocol,
        search_batch_size=args.search_batch_size,
        exact_batch_size=args.exact_batch_size,
        residual_batch_size=args.residual_batch_size,
        checkpoint_rows=args.checkpoint_rows,
        compute_device=args.compute_device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
