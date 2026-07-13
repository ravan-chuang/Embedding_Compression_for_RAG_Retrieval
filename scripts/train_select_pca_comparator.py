#!/usr/bin/env python3
"""Fit and select the preregistered PCA residual-sidecar comparator.

This script intentionally never loads the held-out test split or external qrels.
It:

1. loads the frozen IVF-PQ index and train/validation split metadata;
2. reconstructs document residuals from the frozen index;
3. fits an ordinary unweighted rank-k PCA basis from a deterministic document sample;
4. builds a storage-matched int8 sidecar;
5. selects alpha and Top-B on validation proxy metrics only;
6. writes a frozen PCA configuration and audit manifest.

The current 1,000-query held-out result and 863-query sensitivity subset are
prohibited inputs for this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Faiss is required. Install faiss-gpu or faiss-cpu in the experiment environment."
    ) from exc


ALPHAS = np.asarray(
    [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1, 0.0,
     0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    dtype=np.float32,
)
TOP_B_VALUES = (10, 20, 40, 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--doc-embeddings", required=True, type=Path)
    p.add_argument("--query-vectors", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--train-split", required=True, type=Path)
    p.add_argument("--validation-split", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-docs", default=1_000_000, type=int)
    p.add_argument("--dim", default=384, type=int)
    p.add_argument("--rank", default=16, type=int)
    p.add_argument("--nprobe", default=16, type=int)
    p.add_argument("--top-l", default=100, type=int)
    p.add_argument("--final-k", default=10, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--doc-batch-size", default=50_000, type=int)
    p.add_argument("--max-svd-samples", default=300_000, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_protocol(protocol: dict[str, Any], args: argparse.Namespace) -> None:
    if protocol.get("protocol_id") != "rars_pca_comparator_v1":
        raise ValueError("Unexpected protocol_id")
    if protocol.get("primary_metric") != "recall@10":
        raise ValueError("Primary metric must remain recall@10")

    shared = protocol["shared"]
    expected = {
        "dimension": args.dim,
        "nlist": 512,
        "nprobe": args.nprobe,
        "nbits": 8,
        "candidate_k": args.top_l,
        "final_k": args.final_k,
        "rank": args.rank,
        "coefficient_dtype": "int8",
        "max_svd_samples": args.max_svd_samples,
        "basis_seed": args.seed,
    }
    for key, value in expected.items():
        if shared.get(key) != value:
            raise ValueError(
                f"Protocol mismatch for {key}: {shared.get(key)!r} != {value!r}"
            )

    search = protocol["validation_search"]
    if [float(v) for v in search["alphas"]] != [float(v) for v in ALPHAS]:
        raise ValueError("Alpha search space does not match preregistration")
    if [int(v) for v in search["top_b"]] != list(TOP_B_VALUES):
        raise ValueError("Top-B search space does not match preregistration")


def validate_split(
    payload: dict[str, Any],
    expected_count: int | None = None,
) -> np.ndarray:
    qids = [str(x) for x in payload["query_ids"]]
    rows = np.asarray(payload["query_rows"], dtype=np.int64)
    if len(qids) != len(rows):
        raise ValueError("query_ids/query_rows length mismatch")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate query IDs in split")
    return rows


def build_ann_cache(
    Q: np.ndarray,
    index: Any,
    query_rows: np.ndarray,
    top_l: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    for start in range(0, len(query_rows), batch_size):
        end = min(start + batch_size, len(query_rows))
        scores, rows = index.search(
            Q[query_rows[start:end]].astype(np.float32),
            top_l,
        )
        score_parts.append(scores.astype(np.float32))
        row_parts.append(rows.astype(np.int64))
        print(f"ANN searched {end}/{len(query_rows)}")
    return np.vstack(score_parts), np.vstack(row_parts)


def build_exact_scores(
    Q: np.ndarray,
    X: np.memmap,
    query_rows: np.ndarray,
    ann_rows: np.ndarray,
) -> np.ndarray:
    exact = np.empty(ann_rows.shape, dtype=np.float32)
    for i, qrow in enumerate(query_rows):
        ids = ann_rows[i]
        valid = ids >= 0
        exact[i] = -np.inf
        if np.any(valid):
            xb = np.asarray(X[ids[valid]], dtype=np.float32)
            exact[i, valid] = xb @ Q[int(qrow)].astype(np.float32)
        if (i + 1) % 100 == 0 or i + 1 == len(query_rows):
            print(f"Exact scored {i + 1}/{len(query_rows)}")
    return exact


def load_or_build_split_cache(
    name: str,
    out_dir: Path,
    Q: np.ndarray,
    X: np.memmap,
    index: Any,
    query_rows: np.ndarray,
    top_l: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = out_dir / "candidate_cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    rows_path = cache / "ann_rows.npy"
    ann_path = cache / "ann_scores.npy"
    exact_path = cache / "exact_scores.npy"

    if rows_path.exists() and ann_path.exists():
        rows = np.load(rows_path).astype(np.int64)
        ann = np.load(ann_path).astype(np.float32)
    else:
        ann, rows = build_ann_cache(Q, index, query_rows, top_l, batch_size)
        np.save(rows_path, rows)
        np.save(ann_path, ann)

    if exact_path.exists():
        exact = np.load(exact_path).astype(np.float32)
    else:
        exact = build_exact_scores(Q, X, query_rows, rows)
        np.save(exact_path, exact)

    expected = (len(query_rows), top_l)
    for label, arr in [("rows", rows), ("ann", ann), ("exact", exact)]:
        if arr.shape != expected:
            raise ValueError(f"{name} {label} shape {arr.shape} != {expected}")
    return rows, ann, exact


def load_or_build_residuals(
    path: Path,
    index: Any,
    X: np.memmap,
    n_docs: int,
    dim: int,
    batch_size: int,
) -> np.memmap:
    if path.exists():
        return np.memmap(path, dtype=np.float32, mode="r", shape=(n_docs, dim))

    R = np.memmap(path, dtype=np.float32, mode="w+", shape=(n_docs, dim))
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        ids = np.arange(start, end, dtype=np.int64)
        xhat = index.reconstruct_batch(ids).astype(np.float32)
        R[start:end] = np.asarray(X[start:end], dtype=np.float32) - xhat
        R.flush()
        print(f"Residuals {end}/{n_docs}")
    del R
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n_docs, dim))


def orient_basis_deterministically(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=np.float32).copy()
    for j in range(B.shape[1]):
        column = B[:, j]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            B[:, j] *= -1.0
    return B


def train_unweighted_pca_basis(
    R: np.memmap,
    *,
    n_docs: int,
    rank: int,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    take = min(max_samples, n_docs)
    sampled_rows = np.sort(
        rng.choice(n_docs, size=take, replace=False).astype(np.int64)
    )
    sample = np.asarray(R[sampled_rows], dtype=np.float32)
    print("Unweighted PCA SVD sample:", sample.shape)
    _, _, vh = np.linalg.svd(sample, full_matrices=False)
    basis = orient_basis_deterministically(vh[:rank].T)
    return basis, sampled_rows


def build_int8_codes(
    R: np.memmap,
    B: np.ndarray,
    code_path: Path,
    scale_path: Path,
    n_docs: int,
    rank: int,
    batch_size: int,
) -> tuple[np.memmap, np.ndarray]:
    if code_path.exists() and scale_path.exists():
        return (
            np.memmap(code_path, dtype=np.int8, mode="r", shape=(n_docs, rank)),
            np.load(scale_path).astype(np.float32),
        )

    max_abs = np.zeros(rank, dtype=np.float32)
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        coeff = np.asarray(R[start:end], dtype=np.float32) @ B
        max_abs = np.maximum(max_abs, np.max(np.abs(coeff), axis=0))
        print(f"Maxabs {end}/{n_docs}")

    scales = (max_abs + 1e-12) / 127.0
    np.save(scale_path, scales)

    codes = np.memmap(
        code_path,
        dtype=np.int8,
        mode="w+",
        shape=(n_docs, rank),
    )
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        coeff = np.asarray(R[start:end], dtype=np.float32) @ B
        codes[start:end] = np.clip(
            np.round(coeff / scales[None, :]),
            -127,
            127,
        ).astype(np.int8)
        codes.flush()
        print(f"Codes {end}/{n_docs}")

    del codes
    return (
        np.memmap(code_path, dtype=np.int8, mode="r", shape=(n_docs, rank)),
        scales.astype(np.float32),
    )


def correction_matrix(
    Q: np.ndarray,
    query_rows: np.ndarray,
    ann_rows: np.ndarray,
    B: np.ndarray,
    codes: np.memmap,
    scales: np.ndarray,
    top_b: int,
) -> np.ndarray:
    corr = np.zeros(ann_rows.shape, dtype=np.float32)
    for i, qrow in enumerate(query_rows):
        ids = ann_rows[i, :top_b]
        valid = ids >= 0
        if np.any(valid):
            q_proj = Q[int(qrow)].astype(np.float32) @ B
            coeff = codes[ids[valid]].astype(np.float32) * scales[None, :]
            corr[i, :top_b][valid] = coeff @ q_proj
    return corr


def proxy_metrics(
    corrected: np.ndarray,
    exact: np.ndarray,
    final_k: int,
) -> tuple[float, float]:
    finite = np.isfinite(exact)
    mse = float(np.mean((corrected[finite] - exact[finite]) ** 2))
    exact_order = np.argsort(-exact, axis=1)
    corr_order = np.argsort(-corrected, axis=1)
    overlaps = [
        len(set(exact_order[i, :final_k]) & set(corr_order[i, :final_k]))
        / final_k
        for i in range(len(exact))
    ]
    return mse, float(np.mean(overlaps))


def select_registered_configuration(results: pd.DataFrame) -> dict[str, Any]:
    required = {
        "alpha",
        "top_b",
        "corrected_top10_overlap",
        "overlap_gain",
        "mse_reduction_pct",
    }
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing selection columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("No validation rows")

    max_gain = float(results["overlap_gain"].max())
    threshold = 0.90 * max_gain
    eligible = results[results["overlap_gain"] >= threshold].copy()
    min_top_b = int(eligible["top_b"].min())
    eligible = eligible[eligible["top_b"] == min_top_b].copy()

    eligible["abs_alpha"] = eligible["alpha"].abs()
    best = (
        eligible.sort_values(
            [
                "corrected_top10_overlap",
                "abs_alpha",
                "alpha",
                "mse_reduction_pct",
            ],
            ascending=[False, True, True, False],
        )
        .iloc[0]
        .to_dict()
    )
    best.pop("abs_alpha", None)
    best["maximum_validation_overlap_gain"] = max_gain
    best["selection_threshold"] = threshold
    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_json(args.protocol)
    validate_protocol(protocol, args)

    train_payload = load_json(args.train_split)
    val_payload = load_json(args.validation_split)
    train_rows = validate_split(train_payload, 4980)
    val_rows = validate_split(val_payload, 1000)

    train_qids = set(map(str, train_payload["query_ids"]))
    val_qids = set(map(str, val_payload["query_ids"]))
    if not train_qids.isdisjoint(val_qids):
        raise ValueError("Train and validation splits overlap")

    X = np.memmap(
        args.doc_embeddings,
        dtype=np.float16,
        mode="r",
        shape=(args.n_docs, args.dim),
    )
    Q = np.load(args.query_vectors, mmap_mode="r")
    if Q.shape != (6980, args.dim):
        raise ValueError(f"Unexpected query matrix shape: {Q.shape}")

    index = faiss.read_index(str(args.index))
    index.nprobe = args.nprobe
    if hasattr(index, "make_direct_map"):
        index.make_direct_map()
        print("Initialized Faiss direct map for reconstruction.")

    # Train cache is retained only for protocol symmetry and auditability.
    load_or_build_split_cache(
        "train",
        args.output_dir,
        Q,
        X,
        index,
        train_rows,
        args.top_l,
        args.batch_size,
    )
    val_ann_rows, val_ann, val_exact = load_or_build_split_cache(
        "validation",
        args.output_dir,
        Q,
        X,
        index,
        val_rows,
        args.top_l,
        args.batch_size,
    )

    R = load_or_build_residuals(
        args.output_dir / "residual_ivfpq_m32.float32.memmap",
        index,
        X,
        args.n_docs,
        args.dim,
        args.doc_batch_size,
    )

    basis_dir = args.output_dir / "bases"
    sidecar_dir = args.output_dir / "sidecars"
    basis_dir.mkdir(exist_ok=True)
    sidecar_dir.mkdir(exist_ok=True)

    basis_path = basis_dir / f"pca_unweighted_rank{args.rank}.float32.npy"
    sample_path = basis_dir / "pca_sample_rows.int64.npy"

    if basis_path.exists() and sample_path.exists():
        B = np.load(basis_path).astype(np.float32)
        sampled_rows = np.load(sample_path).astype(np.int64)
    else:
        B, sampled_rows = train_unweighted_pca_basis(
            R,
            n_docs=args.n_docs,
            rank=args.rank,
            max_samples=args.max_svd_samples,
            seed=args.seed,
        )
        np.save(basis_path, B)
        np.save(sample_path, sampled_rows)

    if B.shape != (args.dim, args.rank):
        raise ValueError(f"Unexpected PCA basis shape: {B.shape}")

    code_path = sidecar_dir / f"codes_pca_rank{args.rank}.int8.memmap"
    scale_path = sidecar_dir / f"scales_pca_rank{args.rank}.float32.npy"
    codes, scales = build_int8_codes(
        R,
        B,
        code_path,
        scale_path,
        args.n_docs,
        args.rank,
        args.doc_batch_size,
    )

    rows: list[dict[str, Any]] = []
    base_mse, base_overlap = proxy_metrics(
        val_ann,
        val_exact,
        args.final_k,
    )

    for top_b in TOP_B_VALUES:
        corr = correction_matrix(
            Q,
            val_rows,
            val_ann_rows,
            B,
            codes,
            scales,
            top_b,
        )
        for alpha in ALPHAS:
            corrected = val_ann + float(alpha) * corr
            mse, overlap = proxy_metrics(
                corrected,
                val_exact,
                args.final_k,
            )
            rows.append({
                "basis": "pca_unweighted",
                "alpha": float(alpha),
                "top_b": int(top_b),
                "base_mse": base_mse,
                "corrected_mse": mse,
                "mse_reduction_pct": (
                    (base_mse - mse) / base_mse * 100.0
                ),
                "base_top10_overlap": base_overlap,
                "corrected_top10_overlap": overlap,
                "overlap_gain": overlap - base_overlap,
            })

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["corrected_top10_overlap", "mse_reduction_pct"],
        ascending=False,
    )
    validation_path = args.output_dir / "validation_selection.csv"
    results.to_csv(validation_path, index=False)

    best = select_registered_configuration(results)
    selected = {
        "protocol": "rars_pca_comparator_v1",
        "method": "pca_r16_int8",
        "selection_split": "validation",
        "basis_variant": "pca_unweighted",
        "basis_training": {
            "query_labels_used": False,
            "qrels_used": False,
            "retrieval_scores_used": False,
            "sample_source": "document_residual_rows",
            "sample_count": int(len(sampled_rows)),
            "sample_seed": args.seed,
        },
        "rank": args.rank,
        "alpha": float(best["alpha"]),
        "top_b": int(best["top_b"]),
        "candidate_k": args.top_l,
        "final_k": args.final_k,
        "base_index": {
            "nlist": 512,
            "nprobe": args.nprobe,
            "m": 32,
            "nbits": 8,
        },
        "validation_proxy": {
            "corrected_top10_overlap": float(
                best["corrected_top10_overlap"]
            ),
            "overlap_gain": float(best["overlap_gain"]),
            "mse_reduction_pct": float(best["mse_reduction_pct"]),
            "maximum_overlap_gain": float(
                best["maximum_validation_overlap_gain"]
            ),
            "selection_threshold": float(best["selection_threshold"]),
        },
        "external_evaluated": False,
        "current_heldout_used_for_selection": False,
        "sensitivity_subset_used_for_selection": False,
    }

    selected_path = args.output_dir / "selected_pca_config.json"
    write_json(selected_path, selected)

    manifest = {
        "package": "rars_pca_comparator_train_validation_v1",
        "protocol_id": "rars_pca_comparator_v1",
        "heldout_split_loaded": False,
        "external_qrels_loaded": False,
        "files": [],
    }
    for path in sorted(args.output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        item = {
            "path": str(path.relative_to(args.output_dir)),
            "bytes": path.stat().st_size,
            "sha256": None,
        }
        if path.stat().st_size < 200 * 1024 * 1024:
            item["sha256"] = sha256_file(path)
        manifest["files"].append(item)

    manifest["inputs"] = {
        "protocol": {
            "path": str(args.protocol),
            "sha256": sha256_file(args.protocol),
        },
        "train_split": {
            "path": str(args.train_split),
            "sha256": sha256_file(args.train_split),
        },
        "validation_split": {
            "path": str(args.validation_split),
            "sha256": sha256_file(args.validation_split),
        },
        "index": {
            "path": str(args.index),
            "sha256": sha256_file(args.index),
        },
        "query_vectors": {
            "path": str(args.query_vectors),
            "sha256": sha256_file(args.query_vectors),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
