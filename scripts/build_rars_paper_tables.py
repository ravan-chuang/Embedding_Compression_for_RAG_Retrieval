#!/usr/bin/env python3
"""Build paper-ready RARS tables from committed result artifacts.

Outputs CSV, LaTeX, and a Markdown summary under results/paper_tables/.

The script intentionally keeps unavailable metrics blank rather than fabricating
or mixing incomparable evaluation protocols.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper_tables",
    )
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def load_json(path: Path) -> Any:
    return json.loads(require(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path))


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def bool_mark(value: bool) -> str:
    return "Yes" if value else "No"


def save_table(
    frame: pd.DataFrame,
    stem: str,
    output_dir: Path,
    *,
    caption: str,
    label: str,
    float_format: str = "%.6f",
) -> None:
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"

    frame.to_csv(csv_path, index=False)

    latex = frame.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        float_format=float_format,
        caption=caption,
        label=label,
        position="t",
    )
    tex_path.write_text(latex, encoding="utf-8")


def build_main_table() -> pd.DataFrame:
    rars = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "rars_final_comparison_qrels.csv"
    )
    live = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "live_faiss_benchmark"
        / "live_faiss_benchmark.csv"
    )
    cross = read_csv(
        ROOT
        / "results"
        / "pq_residual_sidecar_cross_setting"
        / "cross_setting_summary.csv"
    )
    storage = load_json(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "sidecar_artifact_benchmark"
        / "benchmark_summary.json"
    )["storage"]

    msmarco = cross.loc[
        cross["setting_id"] == "msmarco_1m_bge_small"
    ].iloc[0]

    live14 = live[
        (live["threads"] == 14)
        & (live["implementation"] == "vectorized")
    ].set_index("method")

    def row_from_rars(
        source_method: str,
        display_method: str,
        *,
        frozen: bool,
        rewrite: bool,
        extra_bytes: float,
        deployable_bytes: float | None,
        correction_method: str | None,
        note: str,
    ) -> dict[str, Any]:
        source = rars.loc[rars["Method"] == source_method].iloc[0]
        correction_us = np.nan
        correction_pct = np.nan
        paired_pct = np.nan

        if correction_method is not None:
            live_row = live14.loc[correction_method]
            correction_us = (
                float(live_row["correction_mean_ms_per_query"]) * 1000.0
            )
            correction_pct = float(
                live_row["incremental_overhead_pct"]
            )
            paired_pct = float(
                live_row["paired_end_to_end_overhead_pct"]
            )

        return {
            "Method": display_method,
            "Frozen index": bool_mark(frozen),
            "Rewrite PQ codes": bool_mark(rewrite),
            "Extra representation B/doc": extra_bytes,
            "Deployable artifact B/doc": deployable_bytes,
            "Recall@10": float(source["Recall@10"]),
            "Success@10": float(source["Success@10"]),
            "MRR@10": float(source["MRR@10"]),
            "nDCG@10": float(source["nDCG@10"]),
            "Correction us/query": correction_us,
            "Correction/Faiss %": correction_pct,
            "Paired E2E overhead %": paired_pct,
            "Note": note,
        }

    rows = [
        row_from_rars(
            "frozen_ivfpq_m32",
            "IVF-PQ M32",
            frozen=True,
            rewrite=False,
            extra_bytes=0.0,
            deployable_bytes=0.0,
            correction_method=None,
            note="Frozen base index",
        ),
        row_from_rars(
            "pca_existing_alpha1_top40",
            "PCA sidecar Top40",
            frozen=True,
            rewrite=False,
            extra_bytes=16.0,
            deployable_bytes=np.nan,
            correction_method=None,
            note="Post-hoc PCA residual sidecar",
        ),
        row_from_rars(
            "score_error_weighted_alpha075_top40",
            "RARS Top40",
            frozen=True,
            rewrite=False,
            extra_bytes=float(storage["representation_bytes_per_doc"]),
            deployable_bytes=float(
                storage["deployable_artifact_bytes_per_doc"]
            ),
            correction_method="rars_top40",
            note="Quality-max RARS operating point",
        ),
    ]

    top20_live = live14.loc["rars_top20"]
    rows.insert(
        2,
        {
            "Method": "RARS Top20",
            "Frozen index": "Yes",
            "Rewrite PQ codes": "No",
            "Extra representation B/doc": float(
                storage["representation_bytes_per_doc"]
            ),
            "Deployable artifact B/doc": float(
                storage["deployable_artifact_bytes_per_doc"]
            ),
            "Recall@10": float(top20_live["recall@10"]),
            "Success@10": float(top20_live["success@10"]),
            "MRR@10": float(top20_live["mrr@10"]),
            "nDCG@10": float(top20_live["ndcg@10"]),
            "Correction us/query": float(
                top20_live["correction_mean_ms_per_query"]
            )
            * 1000.0,
            "Correction/Faiss %": float(
                top20_live["incremental_overhead_pct"]
            ),
            "Paired E2E overhead %": float(
                top20_live["paired_end_to_end_overhead_pct"]
            ),
            "Note": "Preferred deployable operating point",
        },
    )

    rows.extend(
        [
            {
                "Method": "IVF-PQ M48",
                "Frozen index": "No",
                "Rewrite PQ codes": "Yes",
                "Extra representation B/doc": 16.0,
                "Deployable artifact B/doc": np.nan,
                "Recall@10": float(msmarco["m48_recall_at_10"]),
                "Success@10": float(msmarco["m48_success_at_10"]),
                "MRR@10": np.nan,
                "nDCG@10": np.nan,
                "Correction us/query": np.nan,
                "Correction/Faiss %": np.nan,
                "Paired E2E overhead %": np.nan,
                "Note": "Higher-rate index; requires re-encoding",
            },
            {
                "Method": "Exact candidate oracle",
                "Frozen index": "Yes",
                "Rewrite PQ codes": "No",
                "Extra representation B/doc": np.nan,
                "Deployable artifact B/doc": np.nan,
                "Recall@10": float(msmarco["oracle_recall_at_10"]),
                "Success@10": float(msmarco["oracle_success_at_10"]),
                "MRR@10": np.nan,
                "nDCG@10": np.nan,
                "Correction us/query": np.nan,
                "Correction/Faiss %": np.nan,
                "Paired E2E overhead %": np.nan,
                "Note": "Top-100 candidate rescoring ceiling",
            },
        ]
    )

    return pd.DataFrame(rows)


def build_pca_transfer_table() -> pd.DataFrame:
    """Legacy cross-setting validation for the PCA residual sidecar."""
    source = read_csv(
        ROOT
        / "results"
        / "pq_residual_sidecar_cross_setting"
        / "cross_setting_summary.csv"
    )

    return pd.DataFrame(
        {
            "Setting": source["setting_id"],
            "Dataset": source["dataset"],
            "Embedding model": source["embedding_model"],
            "Base Recall@10": source["base_recall_at_10"],
            "PCA Top40 Recall@10": source["sidecar_recall_at_10"],
            "Gain": source["sidecar_gain"],
            "95% CI low": source["bootstrap_ci_95_low"],
            "95% CI high": source["bootstrap_ci_95_high"],
            "M48 Recall@10": source["m48_recall_at_10"],
            "Oracle Recall@10": source["oracle_recall_at_10"],
            "Oracle-gap recovery": source["oracle_gap_recovery"],
            "Significance": source["significance_note"],
        }
    )


def build_rars_cross_setting_table() -> pd.DataFrame:
    """Build a method-consistent RARS/PCA table across committed settings."""
    rows = []

    msmarco = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "rars_final_comparison_qrels.csv"
    ).set_index("Method")

    for method_name, display, alpha, top_b in [
        ("frozen_ivfpq_m32", "Base M32", 0.0, 0),
        ("pca_existing_alpha1_top40", "PCA Top40", 1.0, 40),
        (
            "score_error_weighted_alpha075_top40",
            "RARS Top40",
            0.75,
            40,
        ),
    ]:
        row = msmarco.loc[method_name]
        rows.append(
            {
                "Setting": "msmarco_1m_bge_small",
                "Dataset": "MS MARCO",
                "Embedding model": "BAAI/bge-small-en-v1.5",
                "Selection": "final held-out",
                "Method": display,
                "Alpha": alpha,
                "Top-B": top_b,
                "Queries": 1000,
                "Recall@10": float(row["Recall@10"]),
                "Success@10": float(row["Success@10"]),
                "MRR@10": float(row["MRR@10"]),
                "nDCG@10": float(row["nDCG@10"]),
            }
        )

    bge = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "fiqa_bge_small_transfer"
        / "qrels_final_metrics.csv"
    ).set_index("name")

    for method_name, display, alpha, top_b in [
        ("ivfpq_m32_base_top100_candidates", "Base M32", 0.0, 0),
        ("pca_current_alpha1_top40", "PCA Top40", 1.0, 40),
        (
            "score_error_weighted_alpha1_top20",
            "RARS Top20",
            1.0,
            20,
        ),
        (
            "score_error_weighted_alpha1_top40",
            "RARS Top40",
            1.0,
            40,
        ),
    ]:
        row = bge.loc[method_name]
        rows.append(
            {
                "Setting": "fiqa_bge_small",
                "Dataset": "FiQA",
                "Embedding model": "BAAI/bge-small-en-v1.5",
                "Selection": "fixed transfer",
                "Method": display,
                "Alpha": alpha,
                "Top-B": top_b,
                "Queries": int(row["queries"]),
                "Recall@10": float(row["recall@10"]),
                "Success@10": float(row["success@10"]),
                "MRR@10": float(row["mrr@10"]),
                "nDCG@10": float(row["ndcg@10"]),
            }
        )

    minilm = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "fiqa_minilm_transfer"
        / "qrels_final_metrics_extended.csv"
    ).set_index("name")

    for method_name, display, alpha, top_b, selection in [
        (
            "ivfpq_m32_base_top100_candidates",
            "Base M32",
            0.0,
            0,
            "fixed transfer",
        ),
        (
            "pca_current_alpha1_top40",
            "PCA Top40",
            1.0,
            40,
            "fixed transfer",
        ),
        (
            "score_error_weighted_alpha1_top20",
            "RARS Top20",
            1.0,
            20,
            "fixed transfer",
        ),
        (
            "score_error_weighted_alpha1_top40",
            "RARS Top40",
            1.0,
            40,
            "fixed transfer",
        ),
        (
            "score_error_weighted_bestproxy_alpha0.75_top20",
            "RARS Top20",
            0.75,
            20,
            "proxy-selected",
        ),
        (
            "score_error_weighted_bestproxy_alpha0.75_top40",
            "RARS Top40",
            0.75,
            40,
            "proxy-selected",
        ),
    ]:
        row = minilm.loc[method_name]
        rows.append(
            {
                "Setting": "fiqa_minilm",
                "Dataset": "FiQA",
                "Embedding model": "all-MiniLM-L6-v2",
                "Selection": selection,
                "Method": display,
                "Alpha": alpha,
                "Top-B": top_b,
                "Queries": int(row["queries"]),
                "Recall@10": float(row["recall@10"]),
                "Success@10": float(row["success@10"]),
                "MRR@10": float(row["mrr@10"]),
                "nDCG@10": float(row["ndcg@10"]),
            }
        )

    output = pd.DataFrame(rows)

    base_by_setting = (
        output.loc[output["Method"] == "Base M32"]
        .set_index("Setting")["Recall@10"]
        .to_dict()
    )

    output["Recall gain vs base"] = output.apply(
        lambda row: (
            row["Recall@10"] - base_by_setting[row["Setting"]]
        ),
        axis=1,
    )

    return output

def build_system_table() -> pd.DataFrame:
    live = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "live_faiss_benchmark"
        / "live_faiss_benchmark.csv"
    )

    output = live[
        (live["implementation"] == "vectorized")
        & (live["threads"].isin([1, 14]))
    ].copy()

    output["Search us/query"] = (
        output["search_mean_ms_per_query"] * 1000.0
    )
    output["Correction us/query"] = (
        output["correction_mean_ms_per_query"] * 1000.0
    )
    output["Estimated combined us/query"] = (
        output["estimated_combined_mean_ms_per_query"] * 1000.0
    )

    base_mask = output["method"] == "ivfpq_m32_base"
    output.loc[base_mask, "Correction us/query"] = 0.0
    output.loc[base_mask, "incremental_overhead_pct"] = 0.0
    output.loc[
        base_mask,
        "paired_end_to_end_overhead_pct",
    ] = np.nan

    return output[
        [
            "threads",
            "method",
            "recall@10",
            "mrr@10",
            "Search us/query",
            "Correction us/query",
            "Estimated combined us/query",
            "incremental_overhead_pct",
            "paired_end_to_end_overhead_pct",
        ]
    ].rename(
        columns={
            "threads": "Threads",
            "method": "Method",
            "recall@10": "Recall@10",
            "mrr@10": "MRR@10",
            "incremental_overhead_pct": "Correction/Faiss %",
            "paired_end_to_end_overhead_pct": (
                "Paired E2E overhead %"
            ),
        }
    )

def build_ablation_table() -> pd.DataFrame:
    alpha = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "score_error_weighted_alpha_qrels_sweep.csv"
    )
    topb = read_csv(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "score_error_weighted_topb_qrels_ablation.csv"
    )

    alpha_out = pd.DataFrame(
        {
            "Ablation": "alpha",
            "Value": alpha["alpha"],
            "Basis": alpha["basis_name"],
            "Top-B": alpha["top_b"],
            "Recall@10": alpha["recall_at_10"],
            "Success@10": alpha["success_at_10"],
            "MRR@10": alpha["mrr_at_10"],
            "nDCG@10": alpha["ndcg_at_10"],
            "Recall gain": alpha["recall_gain_over_base"],
        }
    )

    topb_out = pd.DataFrame(
        {
            "Ablation": "Top-B (RARS score-error basis)",
            "Value": topb["top_b"],
            "Basis": topb["basis_name"],
            "Top-B": topb["top_b"],
            "Recall@10": topb["recall_at_10"],
            "Success@10": topb["success_at_10"],
            "MRR@10": topb["mrr_at_10"],
            "nDCG@10": topb["ndcg_at_10"],
            "Recall gain": topb["recall_gain_over_top0"],
        }
    )

    return pd.concat([alpha_out, topb_out], ignore_index=True)


def build_significance_table() -> pd.DataFrame:
    source = load_json(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "paired_bootstrap_best_rars_score_vs_pca.json"
    )

    rows = []
    for metric_name, result in source.items():
        rows.append(
            {
                "Comparison": "RARS score-error a=0.75 Top40 vs PCA a=1 Top40",
                "Metric": metric_name.replace("_at_", "@").replace("_", " "),
                "Mean difference": result["observed_mean_diff"],
                "95% CI low": result["ci95_low"],
                "95% CI high": result["ci95_high"],
                "Bootstrap P(diff<=0)": result["fraction_le_zero"],
                "Positive queries": result["positive_queries"],
                "Zero queries": result["zero_queries"],
                "Negative queries": result["negative_queries"],
                "Bootstrap samples": result["n_boot"],
            }
        )

    return pd.DataFrame(rows)


def build_external_system_table() -> pd.DataFrame:
    """Build the frozen external aggregate table without mixing protocols."""
    root = (
        ROOT
        / "results"
        / "external_confirmation"
        / "trec_dl_2019_msmarco_1m_restricted"
    )
    metrics = load_json(root / "evaluation_v1" / "metrics.json")
    manifest = load_json(root / "external_confirmation_manifest.json")

    rows = []
    for system_id, display_name in [
        ("base_m32", "Base IVF-PQ M32"),
        ("pca_r16_int8", "PCA rank-16 int8"),
        ("rars_r16_int8", "RARS rank-16 int8"),
    ]:
        result = metrics[system_id]
        rows.append(
            {
                "System": display_name,
                "Queries": int(manifest["query_count"]),
                "Recall@10": result["recall@10"],
                "Success@10": result["success@10"],
                "MRR@10": result["mrr@10"],
                "nDCG@10": result["ndcg@10"],
                "Evaluation": "TREC DL 2019 / frozen 1M corpus restriction",
            }
        )

    return pd.DataFrame(rows)


def build_external_contrast_table() -> pd.DataFrame:
    """Build the preregistered external RARS-minus-PCA contrast table."""
    source = load_json(
        ROOT
        / "results"
        / "external_confirmation"
        / "trec_dl_2019_msmarco_1m_restricted"
        / "evaluation_v1"
        / "paired_bootstrap.json"
    )["rars_minus_pca"]

    rows = []
    for metric in ["recall@10", "success@10", "mrr@10", "ndcg@10"]:
        result = source[metric]
        rows.append(
            {
                "Contrast": "RARS minus PCA",
                "Metric": metric.replace("ndcg", "nDCG").replace(
                    "recall", "Recall"
                ).replace("success", "Success").replace("mrr", "MRR"),
                "Mean difference": result["difference"],
                "95% CI low": result["ci_low"],
                "95% CI high": result["ci_high"],
                "Bootstrap P(diff>0)": result[
                    "probability_difference_gt_zero"
                ],
                "Bootstrap samples": result["replicates"],
                "Seed": result["seed"],
            }
        )

    return pd.DataFrame(rows)


def build_storage_table() -> pd.DataFrame:
    storage = load_json(
        ROOT
        / "results"
        / "retrieval_aware_residual_basis"
        / "sidecar_artifact_benchmark"
        / "benchmark_summary.json"
    )["storage"]

    return pd.DataFrame(
        [
            {
                "Component": "RARS representation",
                "Bytes total": storage["representation_bytes"],
                "Bytes/document": storage[
                    "representation_bytes_per_doc"
                ],
                "Accounting": "codes + shared basis/scales",
            },
            {
                "Component": "Int8 codes",
                "Bytes total": storage["file_bytes"]["codes.int8.npy"],
                "Bytes/document": storage["codes_bytes_per_doc"],
                "Accounting": "rank-16 int8 coefficients",
            },
            {
                "Component": "External document IDs",
                "Bytes total": storage["file_bytes"]["doc_ids.npy"],
                "Bytes/document": storage["doc_ids_bytes_per_doc"],
                "Accounting": "serving metadata",
            },
            {
                "Component": "Complete deployable artifact",
                "Bytes total": storage["deployable_artifact_bytes"],
                "Bytes/document": storage[
                    "deployable_artifact_bytes_per_doc"
                ],
                "Accounting": "representation + IDs + config/manifest",
            },
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "paper_main_table": (
            build_main_table(),
            "MS MARCO 1M frozen-index retrofit comparison.",
            "tab:rars-main",
        ),
        "paper_rars_cross_setting_table": (
            build_rars_cross_setting_table(),
            "Cross-setting RARS and PCA comparison.",
            "tab:rars-cross-setting",
        ),
        "paper_pca_transfer_table": (
            build_pca_transfer_table(),
            "Legacy PCA residual-sidecar transfer validation.",
            "tab:pca-transfer",
        ),
        "paper_system_table": (
            build_system_table(),
            "Live Faiss and vectorized RARS latency.",
            "tab:rars-system",
        ),
        "paper_ablation_table": (
            build_ablation_table(),
            "RARS alpha and correction-depth ablations.",
            "tab:rars-ablation",
        ),
        "paper_significance_table": (
            build_significance_table(),
            "Developmental paired bootstrap comparison of RARS and PCA.",
            "tab:rars-significance",
        ),
        "paper_external_system_table": (
            build_external_system_table(),
            (
                "Frozen TREC DL 2019 evaluation restricted to the "
                "MS MARCO 1M indexed corpus."
            ),
            "tab:rars-external-systems",
        ),
        "paper_external_contrast_table": (
            build_external_contrast_table(),
            (
                "Frozen external RARS-minus-PCA paired-bootstrap "
                "contrasts."
            ),
            "tab:rars-external-contrasts",
        ),
        "paper_storage_table": (
            build_storage_table(),
            "Serialized RARS storage accounting.",
            "tab:rars-storage",
        ),
    }

    for stem, (frame, caption, label) in tables.items():
        save_table(
            frame,
            stem,
            args.output_dir,
            caption=caption,
            label=label,
        )

    readme = """# RARS Paper Tables

Generated by:

```bash
python scripts/build_rars_paper_tables.py
```

## Files

- `paper_main_table.*`: MS MARCO 1M main comparison
- `paper_rars_cross_setting_table.*`: method-consistent RARS/PCA transfer summary
- `paper_pca_transfer_table.*`: legacy PCA residual-sidecar transfer summary
- `paper_system_table.*`: live Faiss latency and overhead
- `paper_ablation_table.*`: alpha and Top-B ablations
- `paper_significance_table.*`: developmental paired-bootstrap results
- `paper_external_system_table.*`: frozen external aggregate metrics
- `paper_external_contrast_table.*`: frozen external RARS-minus-PCA contrasts
- `paper_storage_table.*`: serialized storage accounting

## Interpretation constraints

- `IVF-PQ M48` is not a frozen-index retrofit; it requires re-encoding.
- Missing M48/oracle MRR and nDCG values are intentionally left blank because
  the committed cross-setting artifact only provides Recall and Success.
- `paper_pca_transfer_table` contains the legacy PCA-only cross-setting artifact.
- `paper_rars_cross_setting_table` distinguishes fixed-transfer and proxy-selected configurations.
- `Correction/Faiss %` is based on independently timed correction and search.
- `Paired E2E overhead %` is reported separately because multi-threaded timing
  is sensitive to scheduling noise.
- Small non-zero or negative Top0 deltas are timing noise.
- The developmental RARS-vs-PCA table uses the earlier MS MARCO query pool and
  must not be presented as external confirmation.
- The preregistered external Recall@10 contrast is negative and its confidence
  interval crosses zero; the external primary hypothesis was not supported.
- The external set contains 42 eligible queries and only the judgments covered
  by the frozen 1M corpus. It is not an official full-corpus TREC result.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Generated {len(tables)} paper tables in {args.output_dir}")
    for stem in tables:
        print(f"- {stem}.csv")
        print(f"- {stem}.tex")


if __name__ == "__main__":
    main()
