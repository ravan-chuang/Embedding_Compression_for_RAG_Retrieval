#!/usr/bin/env python3
"""Artifact-backed MS MARCO RARS benchmark.

Measures:
- Recall@10, Success@10, MRR@10, nDCG@10
- exact-candidate Top-10 overlap
- candidate score MSE
- sidecar-only latency
- artifact storage accounting

The cached ANN rows are corpus-internal row ids. doc_ids.npy maps them to
external MS MARCO passage ids used by qrels_subset.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("artifacts/_rars_benchmark_inputs"),
    )
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=Path("artifacts/msmarco_rars_sidecar_m32_rank16"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/retrieval_aware_residual_basis/sidecar_artifact_benchmark"
        ),
    )
    p.add_argument("--top-b", type=int, nargs="+", default=[0, 20, 40])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--warmup-runs", type=int, default=3)
    p.add_argument("--timed-runs", type=int, default=20)
    p.add_argument("--query-limit", type=int, default=None)
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_qrels(raw: Any) -> dict[str, dict[str, float]]:
    """Normalize common qrels JSON layouts to qid -> docid -> relevance."""
    out: dict[str, dict[str, float]] = {}

    if isinstance(raw, dict):
        for qid, value in raw.items():
            qid = str(qid)
            rels: dict[str, float] = {}

            if isinstance(value, dict):
                for docid, rel in value.items():
                    try:
                        score = float(rel)
                    except (TypeError, ValueError):
                        continue
                    if score > 0:
                        rels[str(docid)] = score

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        docid = (
                            item.get("doc_id")
                            or item.get("docid")
                            or item.get("pid")
                            or item.get("passage_id")
                        )
                        rel = item.get(
                            "relevance",
                            item.get("score", item.get("rel", 1)),
                        )
                        if docid is not None and float(rel) > 0:
                            rels[str(docid)] = float(rel)
                    else:
                        rels[str(item)] = 1.0

            if rels:
                out[qid] = rels

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            qid = item.get("query_id", item.get("qid", item.get("query")))
            docid = item.get(
                "doc_id",
                item.get("docid", item.get("pid", item.get("passage_id"))),
            )
            rel = item.get("relevance", item.get("score", item.get("rel", 1)))
            if qid is not None and docid is not None and float(rel) > 0:
                out.setdefault(str(qid), {})[str(docid)] = float(rel)

    if not out:
        raise ValueError("Could not parse any positive qrels from qrels_subset.json")
    return out


def dcg(gains: list[float]) -> float:
    if not gains:
        return 0.0
    arr = np.asarray(gains, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, arr.size + 2))
    return float(np.sum(arr * discounts))


def correct_one_query(
    query: np.ndarray,
    candidate_rows: np.ndarray,
    ann_scores: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.ndarray,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    corrected = np.asarray(ann_scores, dtype=np.float32).copy()
    if top_b <= 0:
        return corrected

    depth = min(top_b, candidate_rows.shape[0])
    rows = np.asarray(candidate_rows[:depth], dtype=np.int64)
    valid_positions = np.flatnonzero(rows >= 0)
    if valid_positions.size == 0:
        return corrected

    valid_rows = rows[valid_positions]
    q_proj = np.asarray(query, dtype=np.float32) @ basis
    coeff = codes[valid_rows].astype(np.float32)
    coeff *= scales[None, :]
    correction = coeff @ q_proj
    corrected[valid_positions] += np.float32(alpha) * correction.astype(np.float32)
    return corrected


def evaluate(
    *,
    name: str,
    corrected_scores: np.ndarray,
    exact_scores: np.ndarray,
    ann_rows: np.ndarray,
    doc_ids: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, float]],
    top_k: int,
) -> dict[str, Any]:
    recalls: list[float] = []
    successes: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    overlaps: list[float] = []

    corrected_order = np.argsort(-corrected_scores, axis=1)
    exact_order = np.argsort(-exact_scores, axis=1)

    missing_qrels = 0

    for i, qid in enumerate(query_ids):
        rel_map = qrels.get(str(qid))
        if not rel_map:
            missing_qrels += 1
            continue

        order = corrected_order[i, :top_k]
        internal_rows = ann_rows[i, order]

        ranked_doc_ids: list[str | None] = []
        for row in internal_rows:
            row = int(row)
            ranked_doc_ids.append(None if row < 0 else str(int(doc_ids[row])))

        hits = [1 if docid is not None and docid in rel_map else 0
                for docid in ranked_doc_ids]

        recalls.append(sum(hits) / max(1, len(rel_map)))
        successes.append(float(any(hits)))

        rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        gains = [
            float(rel_map.get(docid, 0.0)) if docid is not None else 0.0
            for docid in ranked_doc_ids
        ]
        ideal = sorted(rel_map.values(), reverse=True)[:top_k]
        ideal_dcg = dcg(ideal)
        ndcgs.append(dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0)

        exact_set = set(exact_order[i, :top_k].tolist())
        corrected_set = set(corrected_order[i, :top_k].tolist())
        overlaps.append(len(exact_set & corrected_set) / float(top_k))

    if not recalls:
        raise ValueError("No held-out queries matched qrels")

    return {
        "method": name,
        "evaluated_queries": len(recalls),
        "missing_qrels_queries": missing_qrels,
        "recall@10": float(np.mean(recalls)),
        "success@10": float(np.mean(successes)),
        "mrr@10": float(np.mean(mrrs)),
        "ndcg@10": float(np.mean(ndcgs)),
        "candidate_exact_top10_overlap": float(np.mean(overlaps)),
        "candidate_score_mse": float(
            np.mean((corrected_scores - exact_scores) ** 2)
        ),
    }


def latency_stats(
    *,
    queries: np.ndarray,
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.ndarray,
    alpha: float,
    top_b: int,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, Any]:
    def run_once() -> np.ndarray:
        samples = np.empty(queries.shape[0], dtype=np.int64)
        for i in range(queries.shape[0]):
            start = time.perf_counter_ns()
            correct_one_query(
                queries[i],
                ann_rows[i],
                ann_scores[i],
                basis,
                scales,
                codes,
                alpha,
                top_b,
            )
            samples[i] = time.perf_counter_ns() - start
        return samples

    for _ in range(warmup_runs):
        run_once()

    values = np.concatenate([run_once() for _ in range(timed_runs)])
    seconds = float(values.sum() / 1e9)

    return {
        "latency_mean_ms_per_query": float(values.mean() / 1e6),
        "latency_p50_ms_per_query": float(np.percentile(values, 50) / 1e6),
        "latency_p95_ms_per_query": float(np.percentile(values, 95) / 1e6),
        "latency_p99_ms_per_query": float(np.percentile(values, 99) / 1e6),
        "sidecar_qps": float(values.size / seconds) if seconds > 0 else math.inf,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "timed_query_calls": int(values.size),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = load_json(args.benchmark_dir / "heldout_eval_split.json")
    qrels = parse_qrels(load_json(args.benchmark_dir / "qrels_subset.json"))
    config = load_json(args.sidecar_dir / "sidecar_config.json")

    query_vectors = np.load(
        args.benchmark_dir / "query_vectors.fp32.npy", mmap_mode="r"
    )
    query_rows = np.asarray(split["query_rows"], dtype=np.int64)
    query_ids = [str(x) for x in split["query_ids"]]
    ann_rows = np.load(args.benchmark_dir / "ann_rows.npy", mmap_mode="r")
    ann_scores = np.load(args.benchmark_dir / "ann_scores.npy", mmap_mode="r")
    exact_scores = np.load(
        args.benchmark_dir / "candidate_exact_scores.npy", mmap_mode="r"
    )

    basis = np.load(args.sidecar_dir / config["basis_file"]).astype(np.float32)
    scales = np.load(args.sidecar_dir / config["scales_file"]).astype(np.float32)
    codes = np.load(
        args.sidecar_dir / config["codes_file"], mmap_mode="r"
    )
    doc_ids = np.load(
        args.sidecar_dir / config["doc_ids_file"], mmap_mode="r"
    )

    if ann_rows.shape != ann_scores.shape or ann_rows.shape != exact_scores.shape:
        raise ValueError("ANN rows, ANN scores, and exact scores are misaligned")
    if ann_rows.shape[0] != len(query_rows) or len(query_rows) != len(query_ids):
        raise ValueError("Held-out query metadata is misaligned")
    if query_vectors.shape[1] != basis.shape[0]:
        raise ValueError("Query dimension and sidecar basis dimension differ")
    if codes.shape != (doc_ids.shape[0], basis.shape[1]):
        raise ValueError("Sidecar codes/doc_ids/basis are misaligned")

    if args.query_limit is not None:
        if args.query_limit <= 0:
            raise ValueError("--query-limit must be positive")
        n = min(args.query_limit, len(query_rows))
        query_rows = query_rows[:n]
        query_ids = query_ids[:n]
        ann_rows = ann_rows[:n]
        ann_scores = ann_scores[:n]
        exact_scores = exact_scores[:n]

    queries = np.asarray(query_vectors[query_rows], dtype=np.float32)
    max_top_b = int(config["max_top_b"])
    depths = sorted(set(args.top_b))

    results: list[dict[str, Any]] = []

    for top_b in depths:
        if top_b < 0 or top_b > max_top_b:
            raise ValueError(
                f"top_b={top_b} must be between 0 and {max_top_b}"
            )

        corrected = np.empty_like(ann_scores, dtype=np.float32)
        for i in range(queries.shape[0]):
            corrected[i] = correct_one_query(
                queries[i],
                ann_rows[i],
                ann_scores[i],
                basis,
                scales,
                codes,
                float(config["alpha"]),
                top_b,
            )

        name = "ivfpq_m32_base" if top_b == 0 else f"rars_top{top_b}"
        row = evaluate(
            name=name,
            corrected_scores=corrected,
            exact_scores=np.asarray(exact_scores),
            ann_rows=np.asarray(ann_rows),
            doc_ids=doc_ids,
            query_ids=query_ids,
            qrels=qrels,
            top_k=args.top_k,
        )
        row["top_b"] = top_b
        row["alpha"] = 0.0 if top_b == 0 else float(config["alpha"])
        row.update(
            latency_stats(
                queries=queries,
                ann_rows=np.asarray(ann_rows),
                ann_scores=np.asarray(ann_scores),
                basis=basis,
                scales=scales,
                codes=codes,
                alpha=float(config["alpha"]),
                top_b=top_b,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
            )
        )
        results.append(row)
        print(json.dumps(row, indent=2))

    csv_path = args.output_dir / "cached_candidate_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    artifact_files = [
        args.sidecar_dir / "basis.npy",
        args.sidecar_dir / "scales.npy",
        args.sidecar_dir / "codes.int8.npy",
        args.sidecar_dir / "doc_ids.npy",
        args.sidecar_dir / "sidecar_config.json",
        args.sidecar_dir / "manifest.json",
    ]
    sizes = {p.name: p.stat().st_size for p in artifact_files if p.exists()}
    n_docs = int(doc_ids.shape[0])

    representation_bytes = (
        sizes.get("basis.npy", 0)
        + sizes.get("scales.npy", 0)
        + sizes.get("codes.int8.npy", 0)
    )
    deployable_bytes = sum(sizes.values())

    summary = {
        "benchmark": "MS MARCO 1M cached-candidate RARS artifact",
        "query_count": int(queries.shape[0]),
        "candidate_pool": int(ann_rows.shape[1]),
        "results": results,
        "storage": {
            "documents": n_docs,
            "representation_bytes": representation_bytes,
            "representation_bytes_per_doc": representation_bytes / n_docs,
            "codes_bytes_per_doc": sizes.get("codes.int8.npy", 0) / n_docs,
            "doc_ids_bytes_per_doc": sizes.get("doc_ids.npy", 0) / n_docs,
            "deployable_artifact_bytes": deployable_bytes,
            "deployable_artifact_bytes_per_doc": deployable_bytes / n_docs,
            "file_bytes": sizes,
        },
        "hashes": {
            p.name: sha256_file(p) for p in artifact_files if p.exists()
        },
        "scope_note": (
            "Latency is sidecar correction only. It excludes embedding, Faiss "
            "search, HTTP, serialization, and document lookup."
        ),
    }

    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (args.output_dir / "README.md").write_text(
        """# RARS Sidecar Artifact Benchmark

Artifact-backed evaluation on the held-out MS MARCO 1M split.

Reported quality metrics use `qrels_subset.json`. Candidate exact-Top10 overlap
and score MSE use cached exact scores inside the same Top-100 ANN candidate pool.

Latency is sidecar-only and excludes query encoding, Faiss retrieval, HTTP,
serialization, and document lookup.
""",
        encoding="utf-8",
    )

    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
