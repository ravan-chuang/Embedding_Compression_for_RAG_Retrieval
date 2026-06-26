"""Rebuild API document metadata from FiQA / BEIR using exported doc_ids.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FiQA and rebuild documents.jsonl in the exact index row order."
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/fiqa_ivfpq_m96",
        help="Directory containing doc_ids.json and index.faiss.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory used to cache the FiQA BEIR dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing documents.jsonl.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    doc_ids_path = artifact_dir / "doc_ids.json"
    output_path = artifact_dir / "documents.jsonl"

    if not doc_ids_path.exists():
        raise FileNotFoundError(
            f"Missing {doc_ids_path}. Export or extract doc_ids.json from the original artifact bundle first."
        )
    if output_path.exists() and not args.force:
        print(f"{output_path} already exists; nothing to do. Use --force to overwrite.")
        return

    doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))

    data_path = util.download_and_unzip(FIQA_URL, args.data_dir)
    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")

    missing = [doc_id for doc_id in doc_ids if doc_id not in corpus]
    if missing:
        raise ValueError(
            f"{len(missing)} exported doc IDs were not found in FiQA. "
            f"First missing ID: {missing[0]}"
        )

    with output_path.open("w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            row = corpus[doc_id]
            f.write(
                json.dumps(
                    {
                        "doc_id": str(doc_id),
                        "title": str(row.get("title") or ""),
                        "text": str(row.get("text") or ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(doc_ids):,} rows to {output_path}")


if __name__ == "__main__":
    main()
