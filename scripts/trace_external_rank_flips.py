#!/usr/bin/env python3
"""Frozen rank-flip trace for selected external queries.

This script reruns only the already-frozen Base/PCA/RARS scoring pipeline.
It performs no fitting, no parameter search, and no retuning.

Outputs:
- rank_flip_trace.csv
- rank_flip_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Faiss is required.") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--query-vectors", required=True, type=Path)
    p.add_argument("--query-manifest", required=True, type=Path)
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--changed-query-csv", required=True, type=Path)
    p.add_argument("--doc-ids", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)

    p.add_argument("--pca-basis", required=True, type=Path)
    p.add_argument("--pca-scales", required=True, type=Path)
    p.add_argument("--pca-codes", required=True, type=Path)

    p.add_argument("--rars-basis", required=True, type=Path)
    p.add_argument("--rars-scales", required=True, type=Path)
    p.add_argument("--rars-codes", required=True, type=Path)

    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--candidate-k", default=100, type=int)
    p.add_argument("--trace-k", default=15, type=int)
    p.add_argument("--top-b", default=40, type=int)
    p.add_argument("--alpha", default=0.75, type=float)
    p.add_argument("--nprobe", default=16, type=int)
    p.add_argument("--n-docs", default=1_000_000, type=int)
    p.add_argument("--dim", default=384, type=int)
    p.add_argument("--rank", default=16, type=int)
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_qrels(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
            else:
                raise ValueError(f"Unsupported qrels line {line_no}")
            out.setdefault(str(qid), {})[int(docid)] = float(rel)
    return out


def load_sidecar(
    basis_path: Path,
    scales_path: Path,
    codes_path: Path,
    *,
    n_docs: int,
    dim: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.memmap]:
    basis = np.load(basis_path).astype(np.float32)
    scales = np.load(scales_path).astype(np.float32)
    if basis.shape != (dim, rank):
        raise ValueError(f"Unexpected basis shape: {basis.shape}")
    if scales.shape != (rank,):
        raise ValueError(f"Unexpected scales shape: {scales.shape}")
    if codes_path.stat().st_size != n_docs * rank:
        raise ValueError("Unexpected sidecar code size")
    codes = np.memmap(
        codes_path,
        dtype=np.int8,
        mode="r",
        shape=(n_docs, rank),
    )
    return basis, scales, codes


def corrections(
    query: np.ndarray,
    candidate_rows: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.memmap,
    *,
    top_b: int,
    alpha: float,
) -> np.ndarray:
    values = np.zeros(len(candidate_rows), dtype=np.float32)
    valid_count = min(top_b, len(candidate_rows))
    ids = candidate_rows[:valid_count]
    valid = ids >= 0
    if np.any(valid):
        q_proj = np.asarray(query, dtype=np.float32) @ basis
        coeff = codes[ids[valid]].astype(np.float32) * scales[None, :]
        values[:valid_count][valid] = float(alpha) * (coeff @ q_proj)
    return values


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.query_manifest)
    qids = [str(x) for x in manifest["query_ids"]]
    texts = [str(x) for x in manifest["query_texts"]]
    qid_to_row = {qid: i for i, qid in enumerate(qids)}
    qid_to_text = dict(zip(qids, texts))

    changed = pd.read_csv(args.changed_query_csv, dtype={"qid": str})
    changed_qids = changed.loc[
        changed["rars_vs_pca_recall_group"] != "tie",
        "qid",
    ].astype(str).tolist()

    Q = np.load(args.query_vectors, mmap_mode="r")
    if Q.shape[1] != args.dim:
        raise ValueError(f"Unexpected query-vector shape: {Q.shape}")

    doc_ids = np.memmap(
        args.doc_ids,
        dtype=np.int64,
        mode="r",
        shape=(args.n_docs,),
    )
    qrels = load_qrels(args.qrels)

    index = faiss.read_index(str(args.index))
    index.nprobe = args.nprobe

    pca_basis, pca_scales, pca_codes = load_sidecar(
        args.pca_basis,
        args.pca_scales,
        args.pca_codes,
        n_docs=args.n_docs,
        dim=args.dim,
        rank=args.rank,
    )
    rars_basis, rars_scales, rars_codes = load_sidecar(
        args.rars_basis,
        args.rars_scales,
        args.rars_codes,
        n_docs=args.n_docs,
        dim=args.dim,
        rank=args.rank,
    )

    rows: list[dict[str, Any]] = []
    summary_queries: list[dict[str, Any]] = []

    for qid in changed_qids:
        qrow = qid_to_row[qid]
        query = np.asarray(Q[qrow], dtype=np.float32)

        base_scores, candidate_rows = index.search(
            query.reshape(1, -1),
            args.candidate_k,
        )
        base_scores = base_scores[0].astype(np.float32)
        candidate_rows = candidate_rows[0].astype(np.int64)

        pca_corr = corrections(
            query,
            candidate_rows,
            pca_basis,
            pca_scales,
            pca_codes,
            top_b=args.top_b,
            alpha=args.alpha,
        )
        rars_corr = corrections(
            query,
            candidate_rows,
            rars_basis,
            rars_scales,
            rars_codes,
            top_b=args.top_b,
            alpha=args.alpha,
        )
        pca_scores = base_scores + pca_corr
        rars_scores = base_scores + rars_corr

        base_ranks = ranks_from_scores(base_scores)
        pca_ranks = ranks_from_scores(pca_scores)
        rars_ranks = ranks_from_scores(rars_scores)

        rel_map = qrels[qid]
        relevant_docs = {docid for docid, rel in rel_map.items() if rel > 0}

        union_mask = (
            (base_ranks <= args.trace_k)
            | (pca_ranks <= args.trace_k)
            | (rars_ranks <= args.trace_k)
        )

        for i in np.where(union_mask)[0]:
            internal_row = int(candidate_rows[i])
            if internal_row < 0:
                continue
            docid = int(doc_ids[internal_row])
            rows.append({
                "qid": qid,
                "query_text": qid_to_text[qid],
                "candidate_position": int(i + 1),
                "internal_row": internal_row,
                "doc_id": docid,
                "relevance": float(rel_map.get(docid, 0.0)),
                "is_positive": bool(docid in relevant_docs),
                "base_score": float(base_scores[i]),
                "pca_correction": float(pca_corr[i]),
                "pca_final_score": float(pca_scores[i]),
                "rars_correction": float(rars_corr[i]),
                "rars_final_score": float(rars_scores[i]),
                "base_rank": int(base_ranks[i]),
                "pca_rank": int(pca_ranks[i]),
                "rars_rank": int(rars_ranks[i]),
                "base_top10": bool(base_ranks[i] <= 10),
                "pca_top10": bool(pca_ranks[i] <= 10),
                "rars_top10": bool(rars_ranks[i] <= 10),
                "pca_to_rars_rank_change": int(
                    rars_ranks[i] - pca_ranks[i]
                ),
                "entered_rars_top10_vs_pca": bool(
                    pca_ranks[i] > 10 and rars_ranks[i] <= 10
                ),
                "left_rars_top10_vs_pca": bool(
                    pca_ranks[i] <= 10 and rars_ranks[i] > 10
                ),
            })

        relevant_trace = [
            row for row in rows
            if row["qid"] == qid and row["is_positive"]
        ]
        summary_queries.append({
            "qid": qid,
            "query_text": qid_to_text[qid],
            "positive_qrels": len(relevant_docs),
            "positive_docs_in_trace": len(relevant_trace),
            "positive_entered_rars_top10_vs_pca": [
                row["doc_id"]
                for row in relevant_trace
                if row["entered_rars_top10_vs_pca"]
            ],
            "positive_left_rars_top10_vs_pca": [
                row["doc_id"]
                for row in relevant_trace
                if row["left_rars_top10_vs_pca"]
            ],
            "negative_entered_rars_top10_vs_pca": [
                row["doc_id"]
                for row in rows
                if row["qid"] == qid
                and not row["is_positive"]
                and row["entered_rars_top10_vs_pca"]
            ],
        })

    trace = pd.DataFrame(rows)
    trace = trace.sort_values(
        ["qid", "rars_rank", "pca_rank", "base_rank"]
    )
    trace.to_csv(args.output_dir / "rank_flip_trace.csv", index=False)

    summary = {
        "analysis_type": "frozen_post_hoc_rank_flip_trace",
        "retrieval_rerun": True,
        "fitting_performed": False,
        "selection_performed": False,
        "retuning_performed": False,
        "candidate_k": args.candidate_k,
        "trace_k": args.trace_k,
        "top_b": args.top_b,
        "alpha": args.alpha,
        "nprobe": args.nprobe,
        "query_count": len(changed_qids),
        "queries": summary_queries,
    }
    (args.output_dir / "rank_flip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
