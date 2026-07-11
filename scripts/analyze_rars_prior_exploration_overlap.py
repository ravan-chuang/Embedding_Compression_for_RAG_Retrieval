#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("recall@10", "success@10", "mrr@10", "ndcg@10")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exclude queries used by an earlier exploratory experiment from a "
            "saved held-out per-query result file and recompute paired-bootstrap statistics."
        )
    )
    p.add_argument("--old-split", required=True, type=Path)
    p.add_argument("--per-query", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bootstrap-replicates", type=int, default=20_000)
    p.add_argument("--bootstrap-seed", type=int, default=20260712)
    return p.parse_args()


def load_old_qids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    else:
        for key in ("query_ids", "qids"):
            if key in payload:
                values = payload[key]
                break
        else:
            raise KeyError("Old split JSON must contain query_ids or qids.")
    return {str(v) for v in values}


def bootstrap_mean_ci(
    delta: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 500,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    boot = np.empty(replicates, dtype=np.float64)

    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        idx = rng.integers(0, n, size=(end - start, n))
        boot[start:end] = delta[idx].mean(axis=1)

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "probability_gt_0": float((boot > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    old_qids = load_old_qids(args.old_split)

    df = pd.read_csv(args.per_query, dtype={"qid": str})
    required = {"qid"}
    for metric in METRICS:
        required.update({
            f"ivfpq_m32_{metric}",
            f"rars_frozen_{metric}",
        })
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    overlap_mask = df["qid"].isin(old_qids)
    clean = df.loc[~overlap_mask].reset_index(drop=True)
    overlap = df.loc[overlap_mask].reset_index(drop=True)

    metrics: dict[str, dict[str, float]] = {}
    overlap_metrics: dict[str, dict[str, float]] = {}

    for i, metric in enumerate(METRICS):
        base_col = f"ivfpq_m32_{metric}"
        rars_col = f"rars_frozen_{metric}"

        base = clean[base_col].to_numpy(dtype=np.float64)
        rars = clean[rars_col].to_numpy(dtype=np.float64)
        delta = rars - base

        metrics[metric] = {
            "base": float(base.mean()),
            "rars": float(rars.mean()),
            "difference": float(delta.mean()),
            **bootstrap_mean_ci(
                delta,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + i,
            ),
        }

        if len(overlap):
            old_base = overlap[base_col].to_numpy(dtype=np.float64)
            old_rars = overlap[rars_col].to_numpy(dtype=np.float64)
            overlap_metrics[metric] = {
                "base": float(old_base.mean()),
                "rars": float(old_rars.mean()),
                "difference": float((old_rars - old_base).mean()),
            }

    output = {
        "analysis": "prior_exploration_excluded_sensitivity",
        "interpretation": (
            "Post-hoc contamination audit. This subset analysis does not create a "
            "new untouched test set and must not be used for model selection."
        ),
        "total_heldout_queries": int(len(df)),
        "prior_explored_overlap": int(overlap_mask.sum()),
        "remaining_prior_unseen_queries": int(len(clean)),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed_base": args.bootstrap_seed,
        "metrics_prior_unseen_subset": metrics,
        "metrics_prior_explored_overlap": overlap_metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
