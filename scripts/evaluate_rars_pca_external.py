#!/usr/bin/env python3
"""One-shot external evaluator for frozen Base, PCA, and RARS systems.

This evaluator is selection-free. It never fits a basis, derives scales,
searches alpha/Top-B, or reads train/validation splits.

Required systems:
- frozen IVF-PQ M32 baseline;
- frozen rank-16 int8 PCA sidecar;
- frozen rank-16 int8 RARS sidecar.

The evaluator verifies the preregistered protocol and external dataset manifest
before loading qrels and producing metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Faiss is required in the evaluation environment.") from exc


METRIC_NAMES = ("recall@10", "success@10", "mrr@10", "ndcg@10")
SYSTEM_NAMES = ("base_m32", "pca_r16_int8", "rars_r16_int8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--query-vectors", required=True, type=Path)
    p.add_argument("--query-manifest", required=True, type=Path)
    p.add_argument("--qrels", required=True, type=Path)
    p.add_argument("--doc-ids", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)

    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--external-manifest", required=True, type=Path)

    p.add_argument("--pca-config", required=True, type=Path)
    p.add_argument("--pca-basis", required=True, type=Path)
    p.add_argument("--pca-scales", required=True, type=Path)
    p.add_argument("--pca-codes", required=True, type=Path)

    p.add_argument("--rars-config", required=True, type=Path)
    p.add_argument("--rars-basis", required=True, type=Path)
    p.add_argument("--rars-scales", required=True, type=Path)
    p.add_argument("--rars-codes", required=True, type=Path)

    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-docs", default=1_000_000, type=int)
    p.add_argument("--dim", default=384, type=int)
    p.add_argument("--batch-size", default=64, type=int)
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


def verify_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {label}: expected {expected}, got {actual}"
        )
    return actual


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != "rars_pca_comparator_v1":
        raise ValueError("Unexpected protocol_id")
    if protocol.get("primary_metric") != "recall@10":
        raise ValueError("Primary metric must remain recall@10")
    if protocol.get("primary_contrast") != "rars_r16_int8_minus_pca_r16_int8":
        raise ValueError("Primary contrast must remain RARS minus PCA")
    bootstrap = protocol.get("bootstrap", {})
    if int(bootstrap.get("replicates", -1)) != 20_000:
        raise ValueError("Bootstrap replicates must remain 20000")
    if int(bootstrap.get("seed", -1)) != 20260712:
        raise ValueError("Bootstrap seed must remain 20260712")


def validate_frozen_config(
    config: dict[str, Any],
    *,
    expected_method: str,
) -> dict[str, Any]:
    if expected_method == "pca_r16_int8":
        if config.get("method") != expected_method:
            raise ValueError("Unexpected PCA method")
        if config.get("external_evaluated") is not False:
            raise ValueError("PCA config must record external_evaluated=false")
    else:
        if config.get("basis_variant") != "score_error_weighted":
            raise ValueError("Unexpected RARS basis variant")
        if config.get("test_evaluated") is not False:
            raise ValueError("RARS config must record test_evaluated=false")

    expected = {
        "rank": 16,
        "alpha": 0.75,
        "top_b": 40,
        "candidate_k": 100,
        "final_k": 10,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"{expected_method} config mismatch for {key}: "
                f"{config.get(key)!r} != {value!r}"
            )

    base = config.get("base_index", {})
    for key, value in {"nlist": 512, "nprobe": 16, "m": 32, "nbits": 8}.items():
        if base.get(key) != value:
            raise ValueError(
                f"{expected_method} base-index mismatch for {key}"
            )
    return config


def validate_external_manifest(
    manifest: dict[str, Any],
    *,
    protocol: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    if manifest.get("status") != "frozen_before_qrels_evaluation":
        raise ValueError(
            "External manifest must be frozen before qrels evaluation"
        )
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("External manifest protocol mismatch")
    if manifest.get("primary_metric") != "recall@10":
        raise ValueError("External primary metric mismatch")
    if manifest.get("primary_contrast") != (
        "rars_r16_int8_minus_pca_r16_int8"
    ):
        raise ValueError("External primary contrast mismatch")
    if manifest.get("prior_overlap_audit_complete") is not True:
        raise ValueError("Prior-query overlap audit is not complete")
    if manifest.get("outcome_inspection_before_freeze") is not False:
        raise ValueError("Manifest must record no outcome inspection before freeze")

    frozen_files = manifest.get("files", {})
    for label, path in paths.items():
        entry = frozen_files.get(label)
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"Missing frozen hash for {label}")
        verify_hash(path, str(entry["sha256"]), label)


def load_query_manifest(
    path: Path,
    query_vectors: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    payload = read_json(path)
    qids = [str(x).strip() for x in payload["query_ids"]]
    rows = np.asarray(payload.get("query_rows", list(range(len(qids)))), dtype=np.int64)

    if len(qids) == 0:
        raise ValueError("External query set is empty")
    if len(qids) != len(rows):
        raise ValueError("query_ids/query_rows length mismatch")
    if len(set(qids)) != len(qids):
        raise ValueError("Duplicate external query IDs")
    if np.any(rows < 0) or np.any(rows >= len(query_vectors)):
        raise ValueError("External query row outside query-vector matrix")
    if len(set(rows.tolist())) != len(rows):
        raise ValueError("Duplicate external query rows")
    return qids, rows


def normalize_docid(value: Any) -> int:
    return int(str(value).strip())


def add_qrel(
    qrels: dict[str, dict[int, float]],
    qid: Any,
    docid: Any,
    relevance: Any,
) -> None:
    qrels.setdefault(str(qid).strip(), {})[
        normalize_docid(docid)
    ] = float(relevance)


def load_qrels(path: Path) -> dict[str, dict[int, float]]:
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        qrels: dict[str, dict[int, float]] = {}
        if isinstance(payload, dict):
            for qid, docs in payload.items():
                if not isinstance(docs, dict):
                    raise ValueError("JSON qrels values must be mappings")
                for docid, rel in docs.items():
                    add_qrel(qrels, qid, docid, rel)
            return qrels
        if isinstance(payload, list):
            for item in payload:
                add_qrel(
                    qrels,
                    item.get("qid", item.get("query_id")),
                    item.get("docid", item.get("doc_id", item.get("pid"))),
                    item.get("relevance", item.get("score", 1)),
                )
            return qrels
        raise ValueError("Unsupported JSON qrels structure")

    qrels: dict[str, dict[int, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if line_no == 1 and any(
                x.lower() in {"qid", "query_id"} for x in parts
            ):
                continue
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
            else:
                raise ValueError(f"Unsupported qrels line {line_no}")
            add_qrel(qrels, qid, docid, rel)
    return qrels


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
    expected_bytes = n_docs * rank
    if codes_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected codes size: {codes_path.stat().st_size} != {expected_bytes}"
        )
    codes = np.memmap(
        codes_path,
        dtype=np.int8,
        mode="r",
        shape=(n_docs, rank),
    )
    return basis, scales, codes


def search_index(
    index: Any,
    queries: np.ndarray,
    *,
    candidate_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        scores, rows = index.search(
            np.asarray(queries[start:end], dtype=np.float32),
            candidate_k,
        )
        score_parts.append(scores.astype(np.float32))
        row_parts.append(rows.astype(np.int64))
        print(f"External ANN searched {end}/{len(queries)}")
    return np.vstack(score_parts), np.vstack(row_parts)


def apply_sidecar(
    queries: np.ndarray,
    ann_rows: np.ndarray,
    ann_scores: np.ndarray,
    basis: np.ndarray,
    scales: np.ndarray,
    codes: np.memmap,
    *,
    alpha: float,
    top_b: int,
) -> np.ndarray:
    corrected = ann_scores.copy()
    for i, query in enumerate(queries):
        ids = ann_rows[i, :top_b]
        valid = ids >= 0
        if not np.any(valid):
            continue
        q_proj = np.asarray(query, dtype=np.float32) @ basis
        coeff = codes[ids[valid]].astype(np.float32) * scales[None, :]
        corrected[i, :top_b][valid] += float(alpha) * (coeff @ q_proj)
    return corrected


def top_docids(
    internal_rows: np.ndarray,
    scores: np.ndarray,
    doc_ids: np.memmap,
    *,
    k: int,
) -> np.ndarray:
    order = np.argsort(-scores, axis=1)[:, :k]
    selected_rows = np.take_along_axis(internal_rows, order, axis=1)
    if np.any(selected_rows < 0):
        raise ValueError("Negative Faiss row ID entered final top-k")
    return np.asarray(doc_ids[selected_rows], dtype=np.int64)


def per_query_metrics(
    qids: list[str],
    ranked_docids: np.ndarray,
    qrels: dict[str, dict[int, float]],
    *,
    k: int,
) -> dict[str, np.ndarray]:
    output = {
        "recall@10": np.zeros(len(qids), dtype=np.float64),
        "success@10": np.zeros(len(qids), dtype=np.float64),
        "mrr@10": np.zeros(len(qids), dtype=np.float64),
        "ndcg@10": np.zeros(len(qids), dtype=np.float64),
    }

    missing = [qid for qid in qids if qid not in qrels]
    if missing:
        raise ValueError(
            f"{len(missing)} external queries have no qrels; examples={missing[:5]}"
        )

    for i, qid in enumerate(qids):
        rel_map = qrels[qid]
        positives = {docid for docid, rel in rel_map.items() if rel > 0}
        if not positives:
            raise ValueError(f"External query {qid} has no positive qrels")

        docs = [int(x) for x in ranked_docids[i, :k]]
        hits = [doc in positives for doc in docs]
        output["recall@10"][i] = sum(hits) / len(positives)
        output["success@10"][i] = float(any(hits))

        reciprocal = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal = 1.0 / rank
                break
        output["mrr@10"][i] = reciprocal

        gains = np.asarray(
            [max(0.0, float(rel_map.get(doc, 0.0))) for doc in docs],
            dtype=np.float64,
        )
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum((2.0 ** gains - 1.0) * discounts))
        ideal = sorted(
            [max(0.0, float(v)) for v in rel_map.values()],
            reverse=True,
        )[:k]
        ideal += [0.0] * (k - len(ideal))
        idcg = float(
            np.sum((2.0 ** np.asarray(ideal) - 1.0) * discounts)
        )
        output["ndcg@10"][i] = dcg / idcg if idcg > 0 else 0.0

    return output


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("Paired arrays must have the same shape")
    diff = np.asarray(left - right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(replicates, dtype=np.float64)
    n = len(diff)
    for start in range(0, replicates, 1000):
        end = min(start + 1000, replicates)
        indices = rng.integers(0, n, size=(end - start, n))
        sampled_means[start:end] = diff[indices].mean(axis=1)
    low, high = np.percentile(sampled_means, [2.5, 97.5])
    return {
        "difference": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_difference_gt_zero": float(
            np.mean(sampled_means > 0.0)
        ),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def write_per_query_csv(
    path: Path,
    qids: list[str],
    systems: dict[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["qid"]
    for system in SYSTEM_NAMES:
        fields.extend(f"{system}_{metric}" for metric in METRIC_NAMES)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, qid in enumerate(qids):
            row: dict[str, Any] = {"qid": qid}
            for system in SYSTEM_NAMES:
                for metric in METRIC_NAMES:
                    row[f"{system}_{metric}"] = float(
                        systems[system][metric][i]
                    )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = read_json(args.protocol)
    validate_protocol(protocol)

    pca_config = validate_frozen_config(
        read_json(args.pca_config),
        expected_method="pca_r16_int8",
    )
    rars_config = validate_frozen_config(
        read_json(args.rars_config),
        expected_method="rars_r16_int8",
    )

    external_manifest = read_json(args.external_manifest)
    frozen_paths = {
        "query_vectors": args.query_vectors,
        "query_manifest": args.query_manifest,
        "qrels": args.qrels,
        "doc_ids": args.doc_ids,
        "index": args.index,
        "pca_config": args.pca_config,
        "pca_basis": args.pca_basis,
        "pca_scales": args.pca_scales,
        "pca_codes": args.pca_codes,
        "rars_config": args.rars_config,
        "rars_basis": args.rars_basis,
        "rars_scales": args.rars_scales,
        "rars_codes": args.rars_codes,
        "evaluator": Path(__file__).resolve(),
        "protocol": args.protocol,
    }
    validate_external_manifest(
        external_manifest,
        protocol=protocol,
        paths=frozen_paths,
    )

    Q_all = np.load(args.query_vectors, mmap_mode="r")
    if Q_all.ndim != 2 or Q_all.shape[1] != args.dim:
        raise ValueError(f"Unexpected query-vector shape: {Q_all.shape}")
    qids, query_rows = load_query_manifest(args.query_manifest, Q_all)
    queries = np.asarray(Q_all[query_rows], dtype=np.float32)

    qrels = load_qrels(args.qrels)
    doc_ids = np.memmap(
        args.doc_ids,
        dtype=np.int64,
        mode="r",
        shape=(args.n_docs,),
    )

    index = faiss.read_index(str(args.index))
    index.nprobe = 16
    ann_scores, ann_rows = search_index(
        index,
        queries,
        candidate_k=100,
        batch_size=args.batch_size,
    )

    pca_basis, pca_scales, pca_codes = load_sidecar(
        args.pca_basis,
        args.pca_scales,
        args.pca_codes,
        n_docs=args.n_docs,
        dim=args.dim,
        rank=16,
    )
    rars_basis, rars_scales, rars_codes = load_sidecar(
        args.rars_basis,
        args.rars_scales,
        args.rars_codes,
        n_docs=args.n_docs,
        dim=args.dim,
        rank=16,
    )

    pca_scores = apply_sidecar(
        queries,
        ann_rows,
        ann_scores,
        pca_basis,
        pca_scales,
        pca_codes,
        alpha=float(pca_config["alpha"]),
        top_b=int(pca_config["top_b"]),
    )
    rars_scores = apply_sidecar(
        queries,
        ann_rows,
        ann_scores,
        rars_basis,
        rars_scales,
        rars_codes,
        alpha=float(rars_config["alpha"]),
        top_b=int(rars_config["top_b"]),
    )

    rankings = {
        "base_m32": top_docids(ann_rows, ann_scores, doc_ids, k=10),
        "pca_r16_int8": top_docids(ann_rows, pca_scores, doc_ids, k=10),
        "rars_r16_int8": top_docids(ann_rows, rars_scores, doc_ids, k=10),
    }
    metrics = {
        system: per_query_metrics(qids, ranked, qrels, k=10)
        for system, ranked in rankings.items()
    }

    aggregate = {
        system: {
            metric: float(values.mean())
            for metric, values in system_metrics.items()
        }
        for system, system_metrics in metrics.items()
    }

    bootstrap_replicates = 20_000
    bootstrap_seed = 20260712
    contrasts: dict[str, Any] = {}
    for contrast_name, left, right in [
        ("rars_minus_pca", "rars_r16_int8", "pca_r16_int8"),
        ("pca_minus_base", "pca_r16_int8", "base_m32"),
        ("rars_minus_base", "rars_r16_int8", "base_m32"),
    ]:
        contrasts[contrast_name] = {}
        for offset, metric in enumerate(METRIC_NAMES):
            contrasts[contrast_name][metric] = paired_bootstrap(
                metrics[left][metric],
                metrics[right][metric],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + offset,
            )

    write_per_query_csv(
        args.output_dir / "per_query_metrics.csv",
        qids,
        metrics,
    )
    write_json(args.output_dir / "metrics.json", aggregate)
    write_json(args.output_dir / "paired_bootstrap.json", contrasts)

    audit = {
        "protocol_id": protocol["protocol_id"],
        "external_manifest_sha256": sha256_file(args.external_manifest),
        "query_count": len(qids),
        "systems": list(SYSTEM_NAMES),
        "primary_metric": "recall@10",
        "primary_contrast": "rars_minus_pca",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed_base": bootstrap_seed,
        "selection_performed": False,
        "fitting_performed": False,
        "files": {},
    }
    for path in [
        args.output_dir / "metrics.json",
        args.output_dir / "paired_bootstrap.json",
        args.output_dir / "per_query_metrics.csv",
    ]:
        audit["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(args.output_dir / "audit_manifest.json", audit)

    print(json.dumps({
        "aggregate": aggregate,
        "primary": contrasts["rars_minus_pca"]["recall@10"],
    }, indent=2))


if __name__ == "__main__":
    main()
