#!/usr/bin/env python3
"""Regenerate fixed-budget Residual-PQ figures from committed CSV artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "fixed_budget_residual_pq"
FIGURE_DIR = ROOT / "figures"

QUALITY_FIGURE = FIGURE_DIR / "fixed_budget_residual_pq_quality.png"
COVERAGE_FIGURE = FIGURE_DIR / "residual_pq_coverage_tradeoff.png"


def method_label(row: pd.Series) -> str:
    method = str(row["method"])

    if method == "base_ivfpq_m32":
        return "Base M=32"

    if method == "uniform_ivfpq_m48":
        return "Uniform M=48"

    residual_m = int(row["residual_pq_m"])
    return f"{residual_m}B Residual-PQ\nerror / Top-50"


def plot_quality_tradeoff(comparison: pd.DataFrame) -> None:
    selected = comparison[
        comparison["method"].isin(
            ["base_ivfpq_m32", "uniform_ivfpq_m48"]
        )
        | (
            (comparison["method"] == "fixed_budget_residual_pq_m32")
            & (comparison["policy"] == "reconstruction_error_sidecar")
            & (comparison["candidate_depth"] == 50)
        )
    ].dropna(
        subset=["total_bytes_per_vector", "recall_at_10"]
    ).sort_values("total_bytes_per_vector")

    plt.figure(figsize=(10, 6))
    plt.scatter(
        selected["total_bytes_per_vector"],
        selected["recall_at_10"],
        s=75,
    )

    for _, row in selected.iterrows():
        plt.annotate(
            method_label(row),
            (row["total_bytes_per_vector"], row["recall_at_10"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Total storage (bytes/vector)")
    plt.ylabel("Recall@10")
    plt.title("Fixed-budget Residual-PQ quality trade-off on held-out FiQA")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(QUALITY_FIGURE, dpi=200)
    plt.close()


def plot_coverage_tradeoff(
    heldout: pd.DataFrame,
    storage: pd.DataFrame,
) -> None:
    error_top50 = heldout[
        (heldout["policy"] == "reconstruction_error_sidecar")
        & (heldout["candidate_depth"] == 50)
    ][["residual_pq_m", "recall_at_10", "ndcg_at_10"]].copy()

    error_storage = storage[
        storage["policy"] == "reconstruction_error_sidecar"
    ].copy()

    coverage = error_storage.merge(
        error_top50,
        on="residual_pq_m",
        how="inner",
    ).sort_values("residual_pq_m")

    plt.figure(figsize=(10, 6))
    plt.scatter(
        coverage["residual_pq_m"],
        coverage["selected_fraction"],
        s=90,
    )

    for _, row in coverage.iterrows():
        plt.annotate(
            (
                f"{int(row['residual_pq_m'])}B\n"
                f"coverage={row['selected_fraction']:.1%}\n"
                f"R@10={row['recall_at_10']:.4f}"
            ),
            (row["residual_pq_m"], row["selected_fraction"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Residual-PQ code bytes per selected document")
    plt.ylabel("Selected-document coverage")
    plt.title("Residual precision versus coverage at 48 bytes/vector")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COVERAGE_FIGURE, dpi=200)
    plt.close()


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)

    comparison = pd.read_csv(
        RESULT_DIR / "fixed_budget_residual_pq_comparison.csv"
    )
    heldout = pd.read_csv(
        RESULT_DIR / "fixed_budget_residual_pq_heldout_results.csv"
    )
    storage = pd.read_csv(
        RESULT_DIR / "fixed_budget_residual_pq_storage_config.csv"
    )

    plot_quality_tradeoff(comparison)
    plot_coverage_tradeoff(heldout, storage)

    print(f"Generated: {QUALITY_FIGURE.relative_to(ROOT)}")
    print(f"Generated: {COVERAGE_FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
