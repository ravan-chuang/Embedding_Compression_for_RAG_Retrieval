#!/usr/bin/env python3
"""Live Faiss + RARS benchmark with true end-to-end timing.

Measures:
- live Faiss search latency
- loop-based RARS correction latency
- vectorized RARS correction latency
- estimated combined latency from separately timed components
- true end-to-end latency from one search+correction invocation
- qrels retrieval metrics
- live/cached candidate alignment

The benchmark excludes query encoding, HTTP, JSON serialization, and document lookup.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("artifacts/_rars_benchmark_inputs"),
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=Path("artifacts/msmarco_rars_sidecar_m32_rank16"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/retrieval_aware_residual_basis/live_faiss_benchmark"
        ),
    )
    parser.add_argument("--top-b", type=int, nargs="+", default=[0, 20, 40])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 14])
    parser.add_argument("--implementations", nargs="+", default=["loop", "vectorized"])
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument("--query-limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_qrels(raw: Any) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    if not isinstance(raw, dict):
        raise ValueError("qrels_subset.json must be a JSON object")

    for query_id, value in raw.items():
        relevance: dict[str, float] = {}

        if isinstance(value, dict):
            for document_id, score in value.items():
                score = float(score)
                if score > 0:
                    relevance[str(document_id)] = score

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    document_id = (
                        item.get("doc_id")
                        or item.get("docid")
                        or item.get("pid")
                        or item.get("passage_id")
                    )
                    score = item.get(
                        "relevance",
                        item.get("score", item.get("rel", 1)),
                    )
                    if document_id is not None and float(score) > 0:
                        relevance[str(document_id)] = float(score)
                else:
                    relevance[str(item)] = 1.0

        if relevance:
            output[str(query_id)] = relevance

    if not output:
        raise ValueError("No positive qrels parsed")

    return output


def dcg(gains: list[float]) -> float:
    if not gains:
        return 0.0
    values = np.asarray(gains, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, values.size + 2))
    return float(np.sum(values * discounts))


def apply_rars_loop(
    queries: np.ndarray,
    rows: np.ndarray,
    scores: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.ndarray,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    corrected = np.asarray(scores, dtype=np.float32).copy()

    if top_b <= 0:
        return corrected

    depth = min(top_b, rows.shape[1])

    for query_index in range(queries.shape[0]):
        candidate_rows = np.asarray(
            rows[query_index, :depth],
            dtype=np.int64,
        )
        valid_positions = np.flatnonzero(candidate_rows >= 0)

        if valid_positions.size == 0:
            continue

        valid_rows = candidate_rows[valid_positions]
        query_projection = queries[query_index] @ basis
        coefficients = codes[valid_rows].astype(np.float32)
        coefficients *= scales[None, :]
        correction = coefficients @ query_projection

        corrected[query_index, valid_positions] += (
            np.float32(alpha) * correction.astype(np.float32)
        )

    return corrected


def apply_rars_vectorized(
    queries: np.ndarray,
    rows: np.ndarray,
    scores: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.ndarray,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    corrected = np.asarray(scores, dtype=np.float32).copy()

    if top_b <= 0:
        return corrected

    depth = min(top_b, rows.shape[1])
    selected_rows = np.asarray(rows[:, :depth], dtype=np.int64)
    valid = selected_rows >= 0

    safe_rows = selected_rows.copy()
    safe_rows[~valid] = 0

    query_projection = np.asarray(queries, dtype=np.float32) @ basis
    coefficients = codes[safe_rows].astype(np.float32)
    coefficients *= scales[None, None, :]

    correction = np.einsum(
        "qbr,qr->qb",
        coefficients,
        query_projection,
        optimize=True,
    ).astype(np.float32)

    correction[~valid] = 0.0
    corrected[:, :depth] += np.float32(alpha) * correction

    return corrected


def evaluate(
    query_ids: list[str],
    rows: np.ndarray,
    scores: np.ndarray,
    doc_ids: np.ndarray,
    qrels: dict[str, dict[str, float]],
    final_k: int,
) -> dict[str, float]:
    recalls: list[float] = []
    successes: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    order = np.argsort(-scores, axis=1)

    for query_index, query_id in enumerate(query_ids):
        relevance = qrels.get(str(query_id))
        if not relevance:
            continue

        ranked_rows = rows[query_index, order[query_index, :final_k]]
        ranked_document_ids = [
            None if int(row) < 0 else str(int(doc_ids[int(row)]))
            for row in ranked_rows
        ]

        hits = [
            1
            if document_id is not None and document_id in relevance
            else 0
            for document_id in ranked_document_ids
        ]

        recalls.append(sum(hits) / max(1, len(relevance)))
        successes.append(float(any(hits)))

        reciprocal_rank = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal_rank = 1.0 / rank
                break
        mrrs.append(reciprocal_rank)

        gains = [
            float(relevance.get(document_id, 0.0))
            if document_id is not None
            else 0.0
            for document_id in ranked_document_ids
        ]
        ideal = sorted(relevance.values(), reverse=True)[:final_k]
        ideal_dcg = dcg(ideal)
        ndcgs.append(dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0)

    return {
        "recall@10": float(np.mean(recalls)),
        "success@10": float(np.mean(successes)),
        "mrr@10": float(np.mean(mrrs)),
        "ndcg@10": float(np.mean(ndcgs)),
    }


def latency_summary(
    values_ns: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": float(values_ns.mean() / 1e6),
        f"{prefix}_p50_ms": float(np.percentile(values_ns, 50) / 1e6),
        f"{prefix}_p95_ms": float(np.percentile(values_ns, 95) / 1e6),
        f"{prefix}_p99_ms": float(np.percentile(values_ns, 99) / 1e6),
    }


def benchmark_function(
    function: Callable[[], Any],
    warmup_runs: int,
    timed_runs: int,
) -> np.ndarray:
    for _ in range(warmup_runs):
        function()

    values = np.empty(timed_runs, dtype=np.int64)

    for run_index in range(timed_runs):
        start = time.perf_counter_ns()
        function()
        values[run_index] = time.perf_counter_ns() - start

    return values


def benchmark_paired_end_to_end(
    *,
    baseline_function: Callable[[], Any],
    method_function: Callable[[], Any],
    warmup_runs: int,
    timed_runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure baseline and method in alternating order on every repetition.

    Alternating order reduces systematic drift from thermal state, cache state,
    and background scheduling. The returned arrays are aligned by repetition.
    """
    for _ in range(warmup_runs):
        baseline_function()
        method_function()

    baseline_values = np.empty(timed_runs, dtype=np.int64)
    method_values = np.empty(timed_runs, dtype=np.int64)

    for run_index in range(timed_runs):
        if run_index % 2 == 0:
            start = time.perf_counter_ns()
            baseline_function()
            baseline_values[run_index] = time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            method_function()
            method_values[run_index] = time.perf_counter_ns() - start
        else:
            start = time.perf_counter_ns()
            method_function()
            method_values[run_index] = time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            baseline_function()
            baseline_values[run_index] = time.perf_counter_ns() - start

    return baseline_values, method_values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    implementation_map = {
        "loop": apply_rars_loop,
        "vectorized": apply_rars_vectorized,
    }

    unknown = sorted(set(args.implementations) - set(implementation_map))
    if unknown:
        raise ValueError(f"Unknown implementations: {unknown}")

    split = load_json(args.benchmark_dir / "heldout_eval_split.json")
    qrels = parse_qrels(
        load_json(args.benchmark_dir / "qrels_subset.json")
    )
    config = load_json(args.sidecar_dir / "sidecar_config.json")

    query_vectors = np.load(
        args.benchmark_dir / "query_vectors.fp32.npy",
        mmap_mode="r",
    )
    query_rows = np.asarray(split["query_rows"], dtype=np.int64)
    query_ids = [str(value) for value in split["query_ids"]]
    cached_rows = np.load(
        args.benchmark_dir / "ann_rows.npy",
        mmap_mode="r",
    )
    cached_scores = np.load(
        args.benchmark_dir / "ann_scores.npy",
        mmap_mode="r",
    )

    basis = np.load(
        args.sidecar_dir / config["basis_file"],
    ).astype(np.float32)
    scales = np.load(
        args.sidecar_dir / config["scales_file"],
    ).astype(np.float32)
    codes = np.load(
        args.sidecar_dir / config["codes_file"],
        mmap_mode="r",
    )
    doc_ids = np.load(
        args.sidecar_dir / config["doc_ids_file"],
        mmap_mode="r",
    )

    index = faiss.read_index(
        str(args.benchmark_dir / "index.faiss")
    )

    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe

    if args.query_limit is not None:
        if args.query_limit <= 0:
            raise ValueError("--query-limit must be positive")

        query_count = min(args.query_limit, len(query_rows))
        query_rows = query_rows[:query_count]
        query_ids = query_ids[:query_count]
        cached_rows = cached_rows[:query_count]
        cached_scores = cached_scores[:query_count]

    queries = np.asarray(
        query_vectors[query_rows],
        dtype=np.float32,
    )

    results: list[dict[str, Any]] = []

    for thread_count in args.threads:
        faiss.omp_set_num_threads(int(thread_count))

        live_scores, live_rows = index.search(
            queries,
            args.candidate_k,
        )

        exact_row_position_match = float(
            np.mean(live_rows == np.asarray(cached_rows))
        )
        score_close_ratio = float(
            np.mean(
                np.isclose(
                    live_scores,
                    np.asarray(cached_scores),
                    rtol=1e-5,
                    atol=1e-6,
                )
            )
        )

        if {"loop", "vectorized"}.issubset(set(args.implementations)):
            for verification_top_b in sorted(set(args.top_b)):
                loop_scores = apply_rars_loop(
                    queries,
                    live_rows,
                    live_scores,
                    basis,
                    scales,
                    codes,
                    float(config["alpha"]),
                    verification_top_b,
                )
                vectorized_scores = apply_rars_vectorized(
                    queries,
                    live_rows,
                    live_scores,
                    basis,
                    scales,
                    codes,
                    float(config["alpha"]),
                    verification_top_b,
                )
                if not np.allclose(
                    loop_scores,
                    vectorized_scores,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    max_diff = float(
                        np.max(np.abs(loop_scores - vectorized_scores))
                    )
                    raise ValueError(
                        "Loop/vectorized correction mismatch at "
                        f"top_b={verification_top_b}; max_abs_diff={max_diff}"
                    )

        search_samples = benchmark_function(
            lambda: index.search(queries, args.candidate_k),
            args.warmup_runs,
            args.timed_runs,
        )

        for implementation_name in args.implementations:
            correction_function = implementation_map[implementation_name]

            for top_b in sorted(set(args.top_b)):
                if top_b < 0 or top_b > int(config["max_top_b"]):
                    raise ValueError(
                        f"top_b={top_b} must be between 0 and "
                        f"{int(config['max_top_b'])}"
                    )

                corrected = correction_function(
                    queries,
                    live_rows,
                    live_scores,
                    basis,
                    scales,
                    codes,
                    float(config["alpha"]),
                    top_b,
                )

                correction_samples = benchmark_function(
                    lambda: correction_function(
                        queries,
                        live_rows,
                        live_scores,
                        basis,
                        scales,
                        codes,
                        float(config["alpha"]),
                        top_b,
                    ),
                    args.warmup_runs,
                    args.timed_runs,
                )

                def run_baseline_end_to_end() -> None:
                    index.search(
                        queries,
                        args.candidate_k,
                    )

                def run_method_end_to_end() -> None:
                    current_scores, current_rows = index.search(
                        queries,
                        args.candidate_k,
                    )
                    correction_function(
                        queries,
                        current_rows,
                        current_scores,
                        basis,
                        scales,
                        codes,
                        float(config["alpha"]),
                        top_b,
                    )

                paired_baseline_samples, end_to_end_samples = (
                    benchmark_paired_end_to_end(
                        baseline_function=run_baseline_end_to_end,
                        method_function=run_method_end_to_end,
                        warmup_runs=args.warmup_runs,
                        timed_runs=args.timed_runs,
                    )
                )

                estimated_combined_samples = (
                    search_samples + correction_samples
                )
                paired_delta_samples = (
                    end_to_end_samples.astype(np.int64)
                    - paired_baseline_samples.astype(np.int64)
                )

                metrics = evaluate(
                    query_ids,
                    live_rows,
                    corrected,
                    doc_ids,
                    qrels,
                    args.final_k,
                )

                method = (
                    "ivfpq_m32_base"
                    if top_b == 0
                    else f"rars_top{top_b}"
                )

                row: dict[str, Any] = {
                    "threads": int(thread_count),
                    "implementation": implementation_name,
                    "method": method,
                    "top_b": int(top_b),
                    "alpha": (
                        0.0
                        if top_b == 0
                        else float(config["alpha"])
                    ),
                    "queries_per_run": int(queries.shape[0]),
                    "candidate_k": int(args.candidate_k),
                    "nprobe": int(args.nprobe),
                    "exact_row_position_match": (
                        exact_row_position_match
                    ),
                    "score_close_ratio": score_close_ratio,
                    **metrics,
                    **latency_summary(
                        search_samples,
                        "search_batch",
                    ),
                    **latency_summary(
                        correction_samples,
                        "correction_batch",
                    ),
                    **latency_summary(
                        estimated_combined_samples,
                        "estimated_combined_batch",
                    ),
                    **latency_summary(
                        paired_baseline_samples,
                        "paired_baseline_end_to_end_batch",
                    ),
                    **latency_summary(
                        end_to_end_samples,
                        "end_to_end_batch",
                    ),
                    **latency_summary(
                        paired_delta_samples,
                        "paired_end_to_end_delta_batch",
                    ),
                }

                query_count = queries.shape[0]

                row["search_mean_ms_per_query"] = (
                    row["search_batch_mean_ms"] / query_count
                )
                row["correction_mean_ms_per_query"] = (
                    row["correction_batch_mean_ms"] / query_count
                )
                row["estimated_combined_mean_ms_per_query"] = (
                    row["estimated_combined_batch_mean_ms"]
                    / query_count
                )
                row["end_to_end_mean_ms_per_query"] = (
                    row["end_to_end_batch_mean_ms"] / query_count
                )
                row["incremental_overhead_pct"] = (
                    100.0
                    * row["correction_batch_mean_ms"]
                    / row["search_batch_mean_ms"]
                    if row["search_batch_mean_ms"] > 0
                    else 0.0
                )
                row["paired_baseline_end_to_end_mean_ms_per_query"] = (
                    row["paired_baseline_end_to_end_batch_mean_ms"]
                    / query_count
                )
                row["paired_end_to_end_delta_mean_ms_per_query"] = (
                    row["paired_end_to_end_delta_batch_mean_ms"]
                    / query_count
                )
                row["paired_end_to_end_overhead_pct"] = (
                    100.0
                    * row["paired_end_to_end_delta_batch_mean_ms"]
                    / row["paired_baseline_end_to_end_batch_mean_ms"]
                    if row["paired_baseline_end_to_end_batch_mean_ms"] > 0
                    else 0.0
                )
                row["vs_separate_search_mean_pct"] = (
                    100.0
                    * (
                        row["end_to_end_batch_mean_ms"]
                        - row["search_batch_mean_ms"]
                    )
                    / row["search_batch_mean_ms"]
                    if row["search_batch_mean_ms"] > 0
                    else 0.0
                )

                results.append(row)
                print(json.dumps(row, indent=2))

    csv_path = args.output_dir / "live_faiss_benchmark.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "benchmark": "live_faiss_rars_msmarco_1m",
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "faiss": faiss.__version__,
            "numpy": np.__version__,
        },
        "scope_note": (
            "Includes live Faiss index.search and RARS correction. "
            "Excludes query encoding, HTTP, JSON serialization, "
            "and document lookup."
        ),
        "timing_note": (
            "estimated_combined_* adds separately measured search and "
            "correction samples. end_to_end_* measures search and correction "
            "inside the same invocation. paired_baseline_end_to_end_* and "
            "paired_end_to_end_overhead_pct use alternating baseline/method "
            "measurements and are the preferred overhead statistics."
        ),
        "results": results,
    }

    (
        args.output_dir / "benchmark_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (
        args.output_dir / "README.md"
    ).write_text(
        """# Live Faiss + RARS Benchmark

Measures live Faiss IVF-PQ search and RARS correction on the aligned
MS MARCO 1M artifacts.

Two correction implementations are compared:

- `loop`: per-query Python loop
- `vectorized`: batched NumPy projection and `einsum`

`estimated_combined_*` adds separately measured search and correction samples.
`end_to_end_*` measures search and correction inside the same invocation.

For each method, baseline and method end-to-end runs are measured in alternating
order. `paired_end_to_end_overhead_pct` is the preferred overhead statistic.
`vs_separate_search_mean_pct` is retained only as a diagnostic and should not be
used as the main reported overhead.

The benchmark also verifies that loop and vectorized correction produce
numerically equivalent score matrices.

The benchmark excludes query encoding, HTTP, response serialization,
and document lookup.
""",
        encoding="utf-8",
    )

    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
