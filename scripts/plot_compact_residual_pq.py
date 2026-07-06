#!/usr/bin/env python3
"""Generate compact Residual-PQ figures from committed result artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LEGACY_RESULTS = (
    ROOT
    / "results"
    / "fixed_budget_residual_pq"
    / "fixed_budget_residual_pq_comparison.csv"
)

COMPACT_RESULTS = (
    ROOT
    / "results"
    / "compact_residual_pq_sidecar"
    / "compact_residual_pq_heldout_results.csv"
)

FIGURE_DIR = ROOT / "figures"
QUALITY_FIGURE = FIGURE_DIR / "compact_residual_pq_quality.png"
COVERAGE_FIGURE = ROOT / "figures" / "compact_residual_pq_coverage.png"


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)

    legacy = pd.read_csv(LEGACY_RESULTS, encoding="utf-8-sig")
    compact = pd.read_csv(COMPACT_RESULTS, encoding="utf-8-sig")

    require_columns(
        legacy,
        [
            "method",
            "policy",
            "candidate_depth",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
        ],
        "legacy comparison CSV",
    )

    require_columns(
        compact,
        [
            "layout_name",
            "selected_fraction",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
        ],
        "compact result CSV",
    )

    base = legacy[
        legacy["method"] == "base_ivfpq_m32"
    ].iloc[0]

    uniform = legacy[
        legacy["method"] == "uniform_ivfpq_m48"
    ].iloc[0]

    legacy_16b = legacy[
        (legacy["method"] == "fixed_budget_residual_pq_m32")
        & (legacy["policy"] == "reconstruction_error_sidecar")
        & (legacy["candidate_depth"] == 50)
        & (legacy["residual_pq_m"] == 16)
    ].iloc[0]

    compact_4bit = compact[
        compact["layout_name"] == "compact_4bit_m32_fp16_codebook"
    ].iloc[0]

    compact_8bit = compact[
        compact["layout_name"] == "compact_8bit_m16_fp16_codebook"
    ].iloc[0]

    methods = [
        {
            "label": "Base\nM=32",
            "coverage": 0.0,
            "recall": float(base["recall_at_10"]),
            "mrr": float(base["mrr_at_10"]),
            "ndcg": float(base["ndcg_at_10"]),
        },
        {
            "label": "Legacy\n16B sidecar",
            "coverage": float(legacy_16b["selected_fraction"]),
            "recall": float(legacy_16b["recall_at_10"]),
            "mrr": float(legacy_16b["mrr_at_10"]),
            "ndcg": float(legacy_16b["ndcg_at_10"]),
        },
        {
            "label": "Compact-4bit\nM_r=32",
            "coverage": float(compact_4bit["selected_fraction"]),
            "recall": float(compact_4bit["recall_at_10"]),
            "mrr": float(compact_4bit["mrr_at_10"]),
            "ndcg": float(compact_4bit["ndcg_at_10"]),
        },
        {
            "label": "Compact-8bit\nM_r=16",
            "coverage": float(compact_8bit["selected_fraction"]),
            "recall": float(compact_8bit["recall_at_10"]),
            "mrr": float(compact_8bit["mrr_at_10"]),
            "ndcg": float(compact_8bit["ndcg_at_10"]),
        },
        {
            "label": "Uniform\nM=48",
            "coverage": 1.0,
            "recall": float(uniform["recall_at_10"]),
            "mrr": float(uniform["mrr_at_10"]),
            "ndcg": float(uniform["ndcg_at_10"]),
        },
    ]

    labels = [item["label"] for item in methods]
    positions = np.arange(len(methods))
    width = 0.24

    plt.figure(figsize=(12, 6))
    plt.bar(
        positions - width,
        [item["recall"] for item in methods],
        width=width,
        label="Recall@10",
    )
    plt.bar(
        positions,
        [item["mrr"] for item in methods],
        width=width,
        label="MRR@10",
    )
    plt.bar(
        positions + width,
        [item["ndcg"] for item in methods],
        width=width,
        label="nDCG@10",
    )

    plt.xticks(positions, labels)
    plt.ylim(0.0, 0.39)
    plt.ylabel("Held-out FiQA metric")
    plt.title("Compact Residual-PQ narrows the fixed-budget quality gap")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(QUALITY_FIGURE, dpi=220)
    plt.close()

    coverage_methods = methods[1:]

    plt.figure(figsize=(10, 6))
    plt.scatter(
        [item["coverage"] for item in coverage_methods],
        [item["recall"] for item in coverage_methods],
        s=100,
    )

    for item in coverage_methods:
        plt.annotate(
            (
                f"{item['label'].replace(chr(10), ' ')}\n"
                f"coverage={item['coverage']:.1%}\n"
                f"R@10={item['recall']:.4f}"
            ),
            (item["coverage"], item["recall"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Fraction of documents receiving the extra code budget")
    plt.ylabel("Recall@10")
    plt.title("Coverage-versus-quality under the 48 bytes/vector budget")
    plt.xlim(0.35, 1.05)
    plt.ylim(0.30, 0.36)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COVERAGE_FIGURE, dpi=220)
    plt.close()

    print(f"Generated: {QUALITY_FIGURE.relative_to(ROOT)}")
    print(f"Generated: {COVERAGE_FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
