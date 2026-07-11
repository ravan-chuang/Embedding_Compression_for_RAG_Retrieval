#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def load_qids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        payload: Any = json.loads(text)
        values = payload if isinstance(payload, list) else payload["qids"]
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    qids = [str(v) for v in values]
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate query IDs found.")
    return qids


def write_json(path: Path, value: Any) -> str:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", default=Path("splits"), type=Path)
    p.add_argument("--seed", default=20260712, type=int)
    args = p.parse_args()

    qids = load_qids(args.input)
    if len(qids) != 6980:
        raise ValueError(f"Expected 6980 query IDs, found {len(qids)}")

    shuffled = sorted(qids)
    random.Random(args.seed).shuffle(shuffled)
    splits = {
        "train": shuffled[:4980],
        "validation": shuffled[4980:5980],
        "test": shuffled[5980:],
    }

    assert all(len(v) == n for v, n in zip(splits.values(), [4980, 1000, 1000]))
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["validation"]).isdisjoint(splits["test"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in splits.items():
        path = args.output_dir / f"msmarco_rars_{name}_qids.json"
        files[name] = {
            "path": str(path),
            "count": len(values),
            "sha256": write_json(path, values),
        }

    manifest = {
        "protocol": "msmarco_rars_clean_query_split_v1",
        "seed": args.seed,
        "source": str(args.input),
        "source_count": len(qids),
        "splits": files,
    }
    write_json(args.output_dir / "msmarco_rars_split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
