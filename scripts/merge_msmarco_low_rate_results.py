from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {
        "name": "pilot_m24_m48",
        "summary": ROOT / "results/msmarco_low_rate_pareto/1m_pilot_m24_m48/msmarco_1m_low_rate_pq_opq_pareto_summary.csv",
        "artifacts": ROOT / "results/msmarco_low_rate_pareto/1m_pilot_m24_m48/msmarco_1m_low_rate_pq_opq_artifacts.csv",
        "metadata": ROOT / "results/msmarco_low_rate_pareto/1m_pilot_m24_m48/msmarco_1m_low_rate_pq_opq_pareto_metadata.json",
    },
    {
        "name": "full_m32_m64_m96",
        "summary": ROOT / "results/msmarco_low_rate_pareto_results_full_m32_m64_m96/msmarco_1m_low_rate_pq_opq_pareto_summary.csv",
        "artifacts": ROOT / "results/msmarco_low_rate_pareto_results_full_m32_m64_m96/msmarco_1m_low_rate_pq_opq_artifacts.csv",
        "metadata": ROOT / "results/msmarco_low_rate_pareto_results_full_m32_m64_m96/msmarco_1m_low_rate_pq_opq_pareto_metadata.json",
    },
]

OUT_DIR = ROOT / "results/msmarco_low_rate_pareto/1m_full_m24_m96"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("M", "").strip()]


def normalize(row: dict[str, str], source: str) -> dict[str, str]:
    row = dict(row)
    row["M"] = str(int(float(row["M"])))
    if row.get("nprobe", "").strip():
        row["nprobe"] = str(int(float(row["nprobe"])))
    row["source_run"] = source
    return row


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pareto_flags(
    rows: list[dict[str, str]],
    cost_key: str,
) -> set[tuple[str, str, str]]:
    """Maximize Recall@10; minimize cost_key."""
    frontier: set[tuple[str, str, str]] = set()

    for candidate in rows:
        candidate_quality = as_float(candidate, "recall_at_10")
        candidate_cost = as_float(candidate, cost_key)

        dominated = False
        for other in rows:
            if other is candidate:
                continue

            other_quality = as_float(other, "recall_at_10")
            other_cost = as_float(other, cost_key)

            better_or_equal = (
                other_quality >= candidate_quality
                and other_cost <= candidate_cost
            )
            strictly_better = (
                other_quality > candidate_quality
                or other_cost < candidate_cost
            )

            if better_or_equal and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.add((
                candidate["method_key"],
                candidate["M"],
                candidate["nprobe"],
            ))

    return frontier


def key_of(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["method_key"], row["M"], row["nprobe"])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_pareto(
    rows: list[dict[str, str]],
    x_key: str,
    x_label: str,
    title: str,
    output_name: str,
    log_x: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))

    for method_key, marker, label in [
        ("ivfpq", "o", "GPU IVF-PQ"),
        ("native_opq_ivfpq", "s", "OPQMatrix + GPU IVF-PQ"),
    ]:
        subset = [r for r in rows if r["method_key"] == method_key]
        plt.scatter(
            [as_float(r, x_key) for r in subset],
            [as_float(r, "recall_at_10") for r in subset],
            marker=marker,
            label=label,
        )

        for r in subset:
            plt.annotate(
                f'M={r["M"]}, np={r["nprobe"]}',
                (as_float(r, x_key), as_float(r, "recall_at_10")),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

    if log_x:
        plt.xscale("log")

    plt.xlabel(x_label)
    plt.ylabel("Recall@10")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / output_name, dpi=180)
    plt.close()


def main() -> None:
    metadata_by_source = {
        source["name"]: json.loads(source["metadata"].read_text(encoding="utf-8"))
        for source in SOURCES
    }

    ignored_metadata_keys = {
        "M_values",
        "generated_at",
        "result_dir",
        "run_label",
    }

    first_name = SOURCES[0]["name"]
    reference = metadata_by_source[first_name]

    for source_name, metadata in metadata_by_source.items():
        for key in sorted(set(reference) | set(metadata)):
            if key in ignored_metadata_keys:
                continue
            if reference.get(key) != metadata.get(key):
                raise ValueError(
                    f"Metadata mismatch for '{key}': "
                    f"{first_name}={reference.get(key)!r}, "
                    f"{source_name}={metadata.get(key)!r}"
                )

    summary_rows: list[dict[str, str]] = []
    artifact_rows: list[dict[str, str]] = []

    for source in SOURCES:
        summary_rows.extend(
            normalize(row, source["name"])
            for row in read_csv(source["summary"])
        )
        artifact_rows.extend(
            normalize(row, source["name"])
            for row in read_csv(source["artifacts"])
        )

    summary_rows.sort(
        key=lambda r: (
            int(r["M"]),
            r["method_key"],
            int(r["nprobe"]),
        )
    )
    artifact_rows.sort(key=lambda r: (int(r["M"]), r["method_key"]))

    expected_m = [24, 32, 48, 64, 96]
    actual_m = sorted({int(row["M"]) for row in summary_rows})

    if actual_m != expected_m:
        raise ValueError(f"Unexpected M values: {actual_m}")

    if len(summary_rows) != 40:
        raise ValueError(f"Expected 40 summary rows, got {len(summary_rows)}")

    if len(artifact_rows) != 10:
        raise ValueError(f"Expected 10 artifact rows, got {len(artifact_rows)}")

    storage_frontier = pareto_flags(summary_rows, "serialized_total_bytes")

    qps_frontier: set[tuple[str, str, str]] = set()
    for candidate in summary_rows:
        candidate_quality = as_float(candidate, "recall_at_10")
        candidate_qps = as_float(candidate, "qps")
        dominated = False

        for other in summary_rows:
            if other is candidate:
                continue

            other_quality = as_float(other, "recall_at_10")
            other_qps = as_float(other, "qps")

            if (
                other_quality >= candidate_quality
                and other_qps >= candidate_qps
                and (other_quality > candidate_quality or other_qps > candidate_qps)
            ):
                dominated = True
                break

        if not dominated:
            qps_frontier.add(key_of(candidate))

    for row in summary_rows:
        row["pareto_quality_storage"] = str(key_of(row) in storage_frontier).lower()
        row["pareto_quality_qps"] = str(key_of(row) in qps_frontier).lower()

    write_csv(
        OUT_DIR / "msmarco_1m_low_rate_pq_opq_full_summary.csv",
        summary_rows,
    )
    write_csv(
        OUT_DIR / "msmarco_1m_low_rate_pq_opq_full_artifacts.csv",
        artifact_rows,
    )

    combined_metadata = {
        **reference,
        "M_values": expected_m,
        "result_scope": "Merged M=24/32/48/64/96 IVF-PQ and OPQ IVF-PQ study",
        "source_runs": {
            name: {
                "M_values": metadata["M_values"],
                "metadata_path": str(
                    next(s for s in SOURCES if s["name"] == name)["metadata"]
                    .relative_to(ROOT)
                ),
            }
            for name, metadata in metadata_by_source.items()
        },
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_rows": len(summary_rows),
        "artifact_rows": len(artifact_rows),
        "metadata_consistency_check": "passed",
    }

    (OUT_DIR / "msmarco_1m_low_rate_pq_opq_full_metadata.json").write_text(
        json.dumps(combined_metadata, indent=2),
        encoding="utf-8",
    )

    plot_pareto(
        summary_rows,
        "serialized_total_bytes",
        "Serialized total index bytes",
        "MS MARCO 1M: Recall@10 vs Serialized Storage",
        "quality_storage_pareto.png",
        log_x=True,
    )
    plot_pareto(
        summary_rows,
        "qps",
        "Queries per second",
        "MS MARCO 1M: Recall@10 vs Throughput",
        "quality_qps_pareto.png",
        log_x=True,
    )

    nprobe64 = [
        row for row in summary_rows
        if row["nprobe"] == "64"
    ]

    by_method_m = {
        (row["method_key"], row["M"]): row
        for row in nprobe64
    }

    lines = [
        "# MS MARCO 1M Low-Rate PQ / OPQ Full Sweep",
        "",
        "## Scope",
        "",
        "- Corpus: 1,000,000 MS MARCO passages",
        f"- Embedding model: `{reference['embedding_model']}`",
        f"- Embedding dimension: {reference['embedding_dimension']}",
        f"- IVF nlist: {reference['nlist']}",
        f"- PQ nbits: {reference['nbits']}",
        "- M values: 24, 32, 48, 64, 96",
        "- nprobe values: 4, 16, 32, 64",
        "- Methods: GPU IVF-PQ and Native Faiss OPQMatrix + GPU IVF-PQ",
        "",
        "## Merge Validation",
        "",
        "- Metadata consistency check: passed",
        "- Combined benchmark points: 40",
        "- Combined artifacts: 10",
        "",
        "## nprobe=64: OPQ Incremental Value",
        "",
        "| M | Plain Recall@10 | OPQ Recall@10 | Δ Recall@10 | Plain build (s) | OPQ build (s) | Build multiplier |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for m in expected_m:
        plain = by_method_m[("ivfpq", str(m))]
        opq = by_method_m[("native_opq_ivfpq", str(m))]
        delta = as_float(opq, "recall_at_10") - as_float(plain, "recall_at_10")
        multiplier = as_float(opq, "build_seconds") / as_float(plain, "build_seconds")

        lines.append(
            f"| {m} | {as_float(plain, 'recall_at_10'):.4f} | "
            f"{as_float(opq, 'recall_at_10'):.4f} | {delta:+.4f} | "
            f"{as_float(plain, 'build_seconds'):.1f} | "
            f"{as_float(opq, 'build_seconds'):.1f} | {multiplier:.1f}× |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Within this fixed configuration, OPQ provides the clearest Recall@10 recovery at lower code rates and its incremental benefit contracts as M increases. The storage overhead of the OPQ transform is small, while its offline training/build cost rises substantially. These observations are specific to this corpus, embedding model, index configuration, and evaluation workload.",
        "",
        "## Files",
        "",
        "- `msmarco_1m_low_rate_pq_opq_full_summary.csv`",
        "- `msmarco_1m_low_rate_pq_opq_full_artifacts.csv`",
        "- `msmarco_1m_low_rate_pq_opq_full_metadata.json`",
        "- `quality_storage_pareto.png`",
        "- `quality_qps_pareto.png`",
    ])

    (OUT_DIR / "msmarco_1m_low_rate_pq_opq_full_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Merge complete.")
    print(f"Output directory: {OUT_DIR.relative_to(ROOT)}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Artifact rows: {len(artifact_rows)}")
    print("M values:", actual_m)
    print("Storage Pareto points:", len(storage_frontier))
    print("QPS Pareto points:", len(qps_frontier))


if __name__ == "__main__":
    main()
