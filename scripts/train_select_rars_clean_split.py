#!/usr/bin/env python3
"""Train and select RARS on clean MS MARCO train/validation splits.

This script intentionally never loads the test split. It:

1. loads the frozen IVF-PQ index, document/query embeddings, and train/validation splits;
2. builds split-specific Top-L ANN and exact candidate-score caches;
3. fits retrieval-aware residual bases using training queries only;
4. builds int8 sidecar codes for each basis;
5. selects basis, alpha, and Top-B using validation proxy metrics only;
6. writes a frozen selected configuration and SHA-256 manifest.

The untouched test split must be evaluated by a separate script after this
configuration and artifact are committed.

Example (Colab paths):

python scripts/train_select_rars_clean_split.py \
  --doc-embeddings /content/gdrive/MyDrive/rag-pq-checkpoints/msmarco_basis_gate0_cache/embeddings.fp16.memmap \
  --query-vectors /content/gdrive/MyDrive/rag-pq-checkpoints/msmarco_basis_gate0_cache/query_vectors.fp32.npy \
  --index /content/gdrive/MyDrive/rag-pq-checkpoints/msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index \
  --pca-basis /content/gdrive/MyDrive/rag-pq-checkpoints/msmarco_1m_pq_residual_gate3/pq_residual_sidecar_rank16/basis.float32.npy \
  --train-split splits/msmarco_rars_train_split.json \
  --validation-split splits/msmarco_rars_validation_split.json \
  --output-dir /content/gdrive/MyDrive/rag-pq-checkpoints/rars_clean_split_v1
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
    p.add_argument("--pca-basis", required=True, type=Path)
    p.add_argument("--train-split", required=True, type=Path)
    p.add_argument("--validation-split", required=True, type=Path)
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


def validate_split(payload: dict[str, Any], expected_count: int | None = None) -> np.ndarray:
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
        scores, rows = index.search(Q[query_rows[start:end]].astype(np.float32), top_l)
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


def train_weighted_basis(
    R: np.memmap,
    ann_rows: np.ndarray,
    weights_2d: np.ndarray,
    rank: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    flat_docs = ann_rows.reshape(-1).astype(np.int64)
    flat_weights = weights_2d.reshape(-1).astype(np.float64)
    valid = (flat_docs >= 0) & np.isfinite(flat_weights) & (flat_weights > 0)
    df = pd.DataFrame({"doc": flat_docs[valid], "w": flat_weights[valid]})
    agg = df.groupby("doc", sort=False)["w"].sum()
    docs = agg.index.to_numpy(dtype=np.int64)
    weights = agg.to_numpy(dtype=np.float64)
    weights /= weights.sum() + 1e-12

    rng = np.random.default_rng(seed)
    take = min(max_samples, len(docs))
    sampled = (
        rng.choice(docs, size=take, replace=False, p=weights)
        if take < len(docs)
        else docs
    )
    lookup = dict(zip(docs.tolist(), weights.tolist()))
    sampled_w = np.asarray([lookup[int(d)] for d in sampled], dtype=np.float32)
    sampled_w /= sampled_w.mean() + 1e-12
    Rw = np.asarray(R[sampled], dtype=np.float32) * np.sqrt(sampled_w[:, None])
    print("Weighted SVD sample:", Rw.shape)
    _, _, vh = np.linalg.svd(Rw, full_matrices=False)
    return vh[:rank].T.astype(np.float32)


def training_weights(
    exact: np.ndarray,
    ann: np.ndarray,
    final_k: int,
) -> dict[str, np.ndarray]:
    score_error = exact - ann
    w_error = np.abs(score_error)
    w_error /= np.mean(w_error[np.isfinite(w_error)]) + 1e-12
    w_error = 1.0 + 5.0 * w_error

    order = np.argsort(-exact, axis=1)
    topk = np.zeros_like(exact, dtype=bool)
    for i in range(len(exact)):
        topk[i, order[i, :final_k]] = True
    cutoff = np.take_along_axis(exact, order[:, [final_k - 1]], axis=1)
    margin = np.abs(exact - cutoff)
    margin_weight = 1.0 / (margin + 1e-3)
    margin_weight /= np.mean(margin_weight[np.isfinite(margin_weight)]) + 1e-12
    w_boundary = 1.0 + 8.0 * topk.astype(np.float32) + 4.0 * margin_weight
    return {
        "score_error_weighted": w_error.astype(np.float32),
        "top10_boundary_weighted": w_boundary.astype(np.float32),
    }


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

    codes = np.memmap(code_path, dtype=np.int8, mode="w+", shape=(n_docs, rank))
    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        coeff = np.asarray(R[start:end], dtype=np.float32) @ B
        codes[start:end] = np.clip(
            np.round(coeff / scales[None, :]), -127, 127
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
        len(set(exact_order[i, :final_k]) & set(corr_order[i, :final_k])) / final_k
        for i in range(len(exact))
    ]
    return mse, float(np.mean(overlaps))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = load_json(args.train_split)
    val_payload = load_json(args.validation_split)
    train_rows = validate_split(train_payload, 4980)
    val_rows = validate_split(val_payload, 1000)
    if not set(train_payload["query_ids"]).isdisjoint(val_payload["query_ids"]):
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

    train_ann_rows, train_ann, train_exact = load_or_build_split_cache(
        "train", args.output_dir, Q, X, index, train_rows,
        args.top_l, args.batch_size,
    )
    val_ann_rows, val_ann, val_exact = load_or_build_split_cache(
        "validation", args.output_dir, Q, X, index, val_rows,
        args.top_l, args.batch_size,
    )

    R = load_or_build_residuals(
        args.output_dir / "residual_ivfpq_m32.float32.memmap",
        index, X, args.n_docs, args.dim, args.doc_batch_size,
    )

    basis_dir = args.output_dir / "bases"
    sidecar_dir = args.output_dir / "sidecars"
    basis_dir.mkdir(exist_ok=True)
    sidecar_dir.mkdir(exist_ok=True)

    bases: dict[str, np.ndarray] = {
        "pca_existing": np.load(args.pca_basis).astype(np.float32)
    }
    weights = training_weights(train_exact, train_ann, args.final_k)
    for name, w in weights.items():
        path = basis_dir / f"{name}_rank{args.rank}.npy"
        if path.exists():
            bases[name] = np.load(path).astype(np.float32)
        else:
            bases[name] = train_weighted_basis(
                R, train_ann_rows, w, args.rank,
                args.max_svd_samples, args.seed,
            )
            np.save(path, bases[name])

    sidecars = {}
    for name, B in bases.items():
        sidecars[name] = build_int8_codes(
            R, B,
            sidecar_dir / f"codes_{name}_rank{args.rank}.int8.memmap",
            sidecar_dir / f"scales_{name}_rank{args.rank}.float32.npy",
            args.n_docs, args.rank, args.doc_batch_size,
        )

    rows: list[dict[str, Any]] = []
    base_mse, base_overlap = proxy_metrics(val_ann, val_exact, args.final_k)
    for name, B in bases.items():
        codes, scales = sidecars[name]
        for top_b in TOP_B_VALUES:
            corr = correction_matrix(
                Q, val_rows, val_ann_rows, B, codes, scales, top_b
            )
            for alpha in ALPHAS:
                corrected = val_ann + float(alpha) * corr
                mse, overlap = proxy_metrics(corrected, val_exact, args.final_k)
                rows.append({
                    "basis": name,
                    "alpha": float(alpha),
                    "top_b": int(top_b),
                    "base_mse": base_mse,
                    "corrected_mse": mse,
                    "mse_reduction_pct": (base_mse - mse) / base_mse * 100.0,
                    "base_top10_overlap": base_overlap,
                    "corrected_top10_overlap": overlap,
                    "overlap_gain": overlap - base_overlap,
                })

    results = pd.DataFrame(rows).sort_values(
        ["corrected_top10_overlap", "mse_reduction_pct"],
        ascending=False,
    )
    results.to_csv(args.output_dir / "validation_selection.csv", index=False)
    best = results.iloc[0].to_dict()

    selected = {
        "protocol": "rars_clean_query_split_v1",
        "selection_split": "validation",
        "basis_variant": str(best["basis"]),
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
            "corrected_top10_overlap": float(best["corrected_top10_overlap"]),
            "overlap_gain": float(best["overlap_gain"]),
            "mse_reduction_pct": float(best["mse_reduction_pct"]),
        },
        "test_evaluated": False,
    }
    write_json(args.output_dir / "selected_config.json", selected)

    manifest = {
        "package": "rars_clean_split_train_validation_v1",
        "test_split_loaded": False,
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
    write_json(args.output_dir / "manifest.json", manifest)

    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
