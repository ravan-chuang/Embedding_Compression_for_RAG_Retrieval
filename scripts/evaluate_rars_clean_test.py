#!/usr/bin/env python3
"""One-shot untouched-test evaluation for frozen clean-split RARS.

This evaluator is intentionally selection-free. It never loads train or
validation splits and contains no fitting, SVD, alpha sweep, basis sweep, or
Top-B sweep. All deployable parameters are loaded from the frozen
selected_config.json.

It evaluates:
  * frozen IVF-PQ baseline;
  * frozen RARS sidecar correction;
  * optional frozen PCA sidecar comparison.

Metrics:
  Recall@10, Success@10, MRR@10, nDCG@10, plus paired bootstrap confidence
  intervals for RARS minus IVF-PQ and (optionally) RARS minus PCA.

Supported qrels inputs:
  * JSON mapping: {"qid": {"docid": relevance, ...}, ...}
  * JSON list records with qid/query_id, docid/doc_id/pid, relevance/score
  * TSV/TREC-style text with 3 or 4 columns:
      qid docid relevance
      qid 0 docid relevance

Example:

python scripts/evaluate_rars_clean_test.py \
  --query-vectors /content/gdrive/MyDrive/rag-pq-checkpoints/msmarco_basis_gate0_cache/query_vectors.fp32.npy \
  --doc-ids /content/gdrive/MyDrive/rag-pq-checkpoints/msmarco_basis_gate0_cache/doc_ids.int64.memmap \
  --index /content/gdrive/MyDrive/rag-pq-checkpoints/msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index \
  --test-split splits/msmarco_rars_test_split.json \
  --qrels /path/to/msmarco_dev_qrels.tsv \
  --selected-config results/rars_clean_split/selected_config.json \
  --freeze-manifest results/rars_clean_split/freeze_manifest.json \
  --artifact-root /content/gdrive/MyDrive/rag-pq-checkpoints/rars_clean_split_v1 \
  --output-dir /content/gdrive/MyDrive/rag-pq-checkpoints/rars_clean_split_test_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Faiss is required in the evaluation environment.") from exc


METRIC_NAMES = ("recall@10", "success@10", "mrr@10", "ndcg@10")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--query-vectors", required=True, type=Path)
    p.add_argument("--doc-ids", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--test-split", required=True, type=Path)
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--selected-config", required=True, type=Path)
    p.add_argument("--freeze-manifest", required=True, type=Path)
    p.add_argument("--artifact-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--n-docs", default=1_000_000, type=int)
    p.add_argument("--dim", default=384, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--bootstrap-replicates", default=10_000, type=int)
    p.add_argument("--bootstrap-seed", default=20260712, type=int)

    # Optional frozen PCA comparator.
    p.add_argument("--pca-basis", type=Path)
    p.add_argument("--pca-scales", type=Path)
    p.add_argument("--pca-codes", type=Path)
    p.add_argument("--pca-alpha", default=1.0, type=float)
    p.add_argument("--pca-top-b", default=40, type=int)
    return p.parse_args()


def read_json(path: Path) -> Any:
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


def verify_frozen_files(
    manifest_path: Path,
    artifact_root: Path,
    selected_config_path: Path,
) -> dict[str, str]:
    manifest = read_json(manifest_path)
    if manifest.get("test_evaluated") is not False:
        raise ValueError("Freeze manifest must record test_evaluated=false.")

    verified: dict[str, str] = {}
    for item in manifest.get("files", []):
        rel = str(item["path"])
        expected = item.get("sha256")
        if not expected:
            continue

        # selected_config.json may be supplied from the repository snapshot.
        path = selected_config_path if rel == "selected_config.json" else artifact_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Frozen artifact missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {rel}: expected {expected}, got {actual}"
            )
        verified[rel] = actual
    return verified


def normalize_qid(value: Any) -> str:
    return str(value).strip()


def normalize_docid(value: Any) -> int:
    return int(str(value).strip())


def add_qrel(
    qrels: dict[str, dict[int, float]],
    qid: Any,
    docid: Any,
    relevance: Any,
) -> None:
    rel = float(relevance)
    q = normalize_qid(qid)
    d = normalize_docid(docid)
    qrels.setdefault(q, {})[d] = rel


def load_qrels(path: Path) -> dict[str, dict[int, float]]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        qrels: dict[str, dict[int, float]] = {}

        if isinstance(payload, dict):
            for qid, docs in payload.items():
                if isinstance(docs, dict):
                    for docid, rel in docs.items():
                        add_qrel(qrels, qid, docid, rel)
                elif isinstance(docs, list):
                    for item in docs:
                        if isinstance(item, dict):
                            docid = (
                                item.get("docid")
                                or item.get("doc_id")
                                or item.get("pid")
                                or item.get("passage_id")
                            )
                            rel = item.get("relevance", item.get("score", 1))
                            add_qrel(qrels, qid, docid, rel)
                        else:
                            add_qrel(qrels, qid, item, 1)
                else:
                    raise ValueError(f"Unsupported qrels value for qid={qid}")
            return qrels

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("JSON-list qrels must contain objects.")
                qid = item.get("qid", item.get("query_id"))
                docid = (
                    item.get("docid")
                    or item.get("doc_id")
                    or item.get("pid")
                    or item.get("passage_id")
                )
                rel = item.get("relevance", item.get("score", 1))
                add_qrel(qrels, qid, docid, rel)
            return qrels

        raise ValueError("Unsupported JSON qrels structure.")

    qrels: dict[str, dict[int, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if line_no == 1 and any(x.lower() in {"qid", "query_id"} for x in parts):
                continue
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
            else:
                raise ValueError(f"Unsupported qrels line {line_no}: {raw!r}")
            add_qrel(qrels, qid, docid, rel)
    return qrels


def load_test_split(path: Path) -> tuple[list[str], np.ndarray]:
    payload = read_json(path)
    qids = [normalize_qid(x) for x in payload["query_ids"]]
    rows = np.asarray(payload["query_rows"], dtype=np.int64)
    if len(qids) != 1000 or len(rows) != 1000:
        raise ValueError(f"Expected untouched 1000-query test split, got {len(qids)}.")
    if len(set(qids)) != len(qids):
        raise ValueError("Duplicate query IDs in test split.")
    return qids, rows


def search_index(
    index: Any,
    Q: np.ndarray,
    query_rows: np.ndarray,
    candidate_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    for start in range(0, len(query_rows), batch_size):
        end = min(start + batch_size, len(query_rows))
        scores, rows = index.search(
            np.asarray(Q[query_rows[start:end]], dtype=np.float32),
            candidate_k,
        )
        score_parts.append(scores.astype(np.float32))
        row_parts.append(rows.astype(np.int64))
        print(f"Test ANN searched {end}/{len(query_rows)}")
    return np.vstack(score_parts), np.vstack(row_parts)


def load_sidecar(
    basis_path: Path,
    scale_path: Path,
    codes_path: Path,
    n_docs: int,
    dim: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.memmap]:
    basis = np.load(basis_path).astype(np.float32)
    scales = np.load(scale_path).astype(np.float32)
    if basis.shape != (dim, rank):
        raise ValueError(f"Unexpected basis shape: {basis.shape}")
    if scales.shape != (rank,):
        raise ValueError(f"Unexpected scales shape: {scales.shape}")
    expected_bytes = n_docs * rank
    if codes_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected codes size: {codes_path.stat().st_size} != {expected_bytes}"
        )
    codes = np.memmap(codes_path, dtype=np.int8, mode="r", shape=(n_docs, rank))
    return basis, scales, codes


def apply_frozen_sidecar(
    Q: np.ndarray,
    query_rows: np.ndarray,
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.memmap,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    if not (0 < top_b <= ann_rows.shape[1]):
        raise ValueError(f"Invalid frozen top_b={top_b}")
    corrected = ann_scores.copy()

    for i, qrow in enumerate(query_rows):
        ids = ann_rows[i, :top_b]
        valid = ids >= 0
        if not np.any(valid):
            continue
        q_proj = np.asarray(Q[int(qrow)], dtype=np.float32) @ basis
        coeff = codes[ids[valid]].astype(np.float32) * scales[None, :]
        correction = coeff @ q_proj
        corrected[i, :top_b][valid] += float(alpha) * correction
    return corrected


def top_docids(
    internal_rows: np.ndarray,
    scores: np.ndarray,
    doc_ids: np.memmap,
    k: int,
) -> np.ndarray:
    order = np.argsort(-scores, axis=1)[:, :k]
    selected_rows = np.take_along_axis(internal_rows, order, axis=1)
    if np.any(selected_rows < 0):
        raise ValueError("Negative Faiss row ID entered final top-k.")
    return np.asarray(doc_ids[selected_rows], dtype=np.int64)


def per_query_metrics(
    qids: list[str],
    ranked_docids: np.ndarray,
    qrels: dict[str, dict[int, float]],
    k: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    recall = np.zeros(len(qids), dtype=np.float64)
    success = np.zeros(len(qids), dtype=np.float64)
    mrr = np.zeros(len(qids), dtype=np.float64)
    ndcg = np.zeros(len(qids), dtype=np.float64)

    missing_qrels: list[str] = []
    zero_positive: list[str] = []

    for i, qid in enumerate(qids):
        rel_map = qrels.get(qid)
        if rel_map is None:
            missing_qrels.append(qid)
            continue

        positive = {docid for docid, rel in rel_map.items() if rel > 0}
        if not positive:
            zero_positive.append(qid)
            continue

        docs = [int(x) for x in ranked_docids[i, :k]]
        binary_hits = np.asarray([1.0 if d in positive else 0.0 for d in docs])
        recall[i] = binary_hits.sum() / len(positive)
        success[i] = float(binary_hits.any())

        hit_positions = np.flatnonzero(binary_hits)
        if len(hit_positions):
            mrr[i] = 1.0 / float(hit_positions[0] + 1)

        gains = np.asarray(
            [max(float(rel_map.get(d, 0.0)), 0.0) for d in docs],
            dtype=np.float64,
        )
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum((np.power(2.0, gains) - 1.0) * discounts))

        ideal_gains = sorted(
            (max(float(rel), 0.0) for rel in rel_map.values()),
            reverse=True,
        )[:k]
        ideal = np.zeros(k, dtype=np.float64)
        ideal[: len(ideal_gains)] = ideal_gains
        idcg = float(np.sum((np.power(2.0, ideal) - 1.0) * discounts))
        ndcg[i] = dcg / idcg if idcg > 0 else 0.0

    if missing_qrels:
        raise ValueError(
            f"{len(missing_qrels)} test qids have no qrels; examples: "
            f"{missing_qrels[:10]}"
        )
    if zero_positive:
        raise ValueError(
            f"{len(zero_positive)} test qids have no positive qrels; examples: "
            f"{zero_positive[:10]}"
        )

    return (
        {
            "recall@10": recall,
            "success@10": success,
            "mrr@10": mrr,
            "ndcg@10": ndcg,
        },
        {
            "num_queries": len(qids),
            "missing_qrels": len(missing_qrels),
            "zero_positive_qrels": len(zero_positive),
        },
    )


def summarize(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def paired_bootstrap(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(next(iter(candidate.values())))
    output: dict[str, dict[str, float]] = {}

    for metric in METRIC_NAMES:
        delta = candidate[metric] - baseline[metric]
        point = float(np.mean(delta))
        boot = np.empty(replicates, dtype=np.float64)

        # Chunk bootstrap indexes to avoid a giant replicates x queries matrix.
        done = 0
        chunk = 500
        while done < replicates:
            count = min(chunk, replicates - done)
            indexes = rng.integers(0, n, size=(count, n))
            boot[done : done + count] = np.mean(delta[indexes], axis=1)
            done += count

        lo, hi = np.quantile(boot, [0.025, 0.975])
        output[metric] = {
            "difference": point,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "probability_difference_gt_0": float(np.mean(boot > 0)),
        }
    return output


def save_per_query_csv(
    path: Path,
    qids: list[str],
    systems: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = ["qid"]
    for system in systems:
        fields.extend(f"{system}_{metric}" for metric in METRIC_NAMES)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, qid in enumerate(qids):
            row: dict[str, Any] = {"qid": qid}
            for system, metrics in systems.items():
                for metric in METRIC_NAMES:
                    row[f"{system}_{metric}"] = float(metrics[metric][i])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(args.selected_config)
    if config.get("test_evaluated") is not False:
        raise ValueError("Selected config must be frozen before test evaluation.")
    if config.get("selection_split") != "validation":
        raise ValueError("Selected config was not selected on validation.")
    if config.get("protocol") != "rars_clean_query_split_v1":
        raise ValueError("Unexpected protocol.")

    verified = verify_frozen_files(
        args.freeze_manifest,
        args.artifact_root,
        args.selected_config,
    )

    qids, query_rows = load_test_split(args.test_split)
    Q = np.load(args.query_vectors, mmap_mode="r")
    if Q.shape != (6980, args.dim):
        raise ValueError(f"Unexpected query vectors shape: {Q.shape}")

    qrels = load_qrels(args.qrels)
    doc_ids = np.memmap(
        args.doc_ids,
        dtype=np.int64,
        mode="r",
        shape=(args.n_docs,),
    )

    candidate_k = int(config["candidate_k"])
    final_k = int(config["final_k"])
    rank = int(config["rank"])
    nprobe = int(config["base_index"]["nprobe"])

    index = faiss.read_index(str(args.index))
    index.nprobe = nprobe
    ann_scores, ann_rows = search_index(
        index, Q, query_rows, candidate_k, args.batch_size
    )

    rars_name = str(config["basis_variant"])
    rars_basis_path = (
        args.artifact_root / "bases" / f"{rars_name}_rank{rank}.npy"
    )
    rars_scale_path = (
        args.artifact_root / "sidecars" / f"scales_{rars_name}_rank{rank}.float32.npy"
    )
    rars_codes_path = (
        args.artifact_root / "sidecars" / f"codes_{rars_name}_rank{rank}.int8.memmap"
    )
    rars_basis, rars_scales, rars_codes = load_sidecar(
        rars_basis_path,
        rars_scale_path,
        rars_codes_path,
        args.n_docs,
        args.dim,
        rank,
    )
    rars_scores = apply_frozen_sidecar(
        Q,
        query_rows,
        ann_rows,
        ann_scores,
        rars_basis,
        rars_scales,
        rars_codes,
        float(config["alpha"]),
        int(config["top_b"]),
    )

    ranked = {
        "ivfpq_m32": top_docids(ann_rows, ann_scores, doc_ids, final_k),
        "rars_frozen": top_docids(ann_rows, rars_scores, doc_ids, final_k),
    }

    pca_requested = any(
        x is not None for x in (args.pca_basis, args.pca_scales, args.pca_codes)
    )
    if pca_requested:
        if not all(
            x is not None for x in (args.pca_basis, args.pca_scales, args.pca_codes)
        ):
            raise ValueError(
                "PCA comparison requires --pca-basis, --pca-scales and --pca-codes."
            )
        pca_basis, pca_scales, pca_codes = load_sidecar(
            args.pca_basis,
            args.pca_scales,
            args.pca_codes,
            args.n_docs,
            args.dim,
            rank,
        )
        pca_scores = apply_frozen_sidecar(
            Q,
            query_rows,
            ann_rows,
            ann_scores,
            pca_basis,
            pca_scales,
            pca_codes,
            args.pca_alpha,
            args.pca_top_b,
        )
        ranked["pca_frozen"] = top_docids(
            ann_rows, pca_scores, doc_ids, final_k
        )

    per_system: dict[str, dict[str, np.ndarray]] = {}
    qrels_info: dict[str, Any] | None = None
    for system, ranked_docs in ranked.items():
        metrics, info = per_query_metrics(qids, ranked_docs, qrels, final_k)
        per_system[system] = metrics
        qrels_info = info

    summaries = {
        system: summarize(metrics) for system, metrics in per_system.items()
    }
    comparisons = {
        "rars_frozen_minus_ivfpq_m32": paired_bootstrap(
            per_system["rars_frozen"],
            per_system["ivfpq_m32"],
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
    }
    if "pca_frozen" in per_system:
        comparisons["rars_frozen_minus_pca_frozen"] = paired_bootstrap(
            per_system["rars_frozen"],
            per_system["pca_frozen"],
            args.bootstrap_replicates,
            args.bootstrap_seed + 1,
        )

    result = {
        "protocol": "rars_clean_query_split_v1",
        "evaluation_split": "untouched_test",
        "test_query_count": len(qids),
        "selected_config": config,
        "verified_freeze_sha256": verified,
        "qrels": qrels_info,
        "metrics": summaries,
        "paired_bootstrap": comparisons,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
    }

    write_json(args.output_dir / "test_metrics.json", result)
    save_per_query_csv(
        args.output_dir / "test_per_query_metrics.csv",
        qids,
        per_system,
    )

    # Record output and immutable input hashes after evaluation.
    audit = {
        "test_evaluator": sha256_file(Path(__file__)),
        "test_split": sha256_file(args.test_split),
        "qrels": sha256_file(args.qrels),
        "selected_config": sha256_file(args.selected_config),
        "freeze_manifest": sha256_file(args.freeze_manifest),
        "index": sha256_file(args.index),
        "query_vectors": sha256_file(args.query_vectors),
        "doc_ids": sha256_file(args.doc_ids),
        "rars_basis": sha256_file(rars_basis_path),
        "rars_scales": sha256_file(rars_scale_path),
        # Codes are large, but hashing once after the one-shot evaluation gives
        # the final audit record.
        "rars_codes": sha256_file(rars_codes_path),
        "test_metrics": sha256_file(args.output_dir / "test_metrics.json"),
        "test_per_query_metrics": sha256_file(
            args.output_dir / "test_per_query_metrics.csv"
        ),
    }
    write_json(args.output_dir / "test_audit_manifest.json", audit)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
