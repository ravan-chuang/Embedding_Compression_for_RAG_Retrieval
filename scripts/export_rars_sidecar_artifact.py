#!/usr/bin/env python
"""Export a deployable RARS / residual sidecar artifact.

This script packages notebook-produced sidecar files into a serving-friendly
artifact directory:

    sidecar_config.json
    basis.npy
    scales.npy
    codes.int8.npy
    doc_ids.npy

The exported artifact is consumed by app.sidecar.RARSSidecar.

Example:

    python scripts/export_rars_sidecar_artifact.py \
      --basis /path/to/basis_score_error_weighted_rank16.npy \
      --scales /path/to/scales_score_error_weighted_rank16.float32.npy \
      --codes /path/to/codes_score_error_weighted_rank16.int8.npy \
      --doc-ids /path/to/doc_ids.int64.npy \
      --output-dir artifacts/msmarco_rars_sidecar_m32_rank16 \
      --alpha 0.75 \
      --default-top-b 20 \
      --max-top-b 40 \
      --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_array_shape_dtype(path: Path) -> tuple[tuple[int, ...], str]:
    arr = np.load(path, mmap_mode="r")
    return tuple(int(x) for x in arr.shape), str(arr.dtype)


def copy_or_convert_npy(src: Path, dst: Path, *, dtype: str | None = None) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    arr = np.load(src, mmap_mode="r")

    if dtype is not None and str(arr.dtype) != dtype:
        out = np.asarray(arr, dtype=np.dtype(dtype))
        np.save(dst, out)
        return

    # Keep .npy format exactly if possible.
    shutil.copyfile(src, dst)


def build_config(
    *,
    dim: int,
    rank: int,
    alpha: float,
    default_top_b: int,
    max_top_b: int,
    code_dtype: str,
    doc_id_dtype: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "rars_residual_sidecar",
        "version": 1,
        "dim": int(dim),
        "rank": int(rank),
        "alpha": float(alpha),
        "default_top_b": int(default_top_b),
        "max_top_b": int(max_top_b),
        "code_dtype": str(code_dtype),
        "doc_id_dtype": str(doc_id_dtype),
        "basis_file": "basis.npy",
        "scales_file": "scales.npy",
        "codes_file": "codes.int8.npy",
        "doc_ids_file": "doc_ids.npy",
        "score_formula": "corrected_score = ann_score + alpha * q^T B a_x",
        "notes": [
            "candidate rows must use corpus-internal row ids",
            "only the first top_b ANN candidates are corrected before reranking",
            "doc_ids.npy maps corpus-internal row ids to external document ids",
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deployable RARS residual sidecar artifact."
    )

    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--scales", required=True, type=Path)
    parser.add_argument("--codes", required=True, type=Path)
    parser.add_argument("--doc-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--default-top-b", type=int, default=20)
    parser.add_argument("--max-top-b", type=int, default=40)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in output-dir.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for name in ["basis", "scales", "codes", "doc_ids"]:
        path = getattr(args, name if name != "doc_ids" else "doc_ids", None)
        if path is None:
            continue

    input_paths = {
        "basis": args.basis,
        "scales": args.scales,
        "codes": args.codes,
        "doc_ids": args.doc_ids,
    }

    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    target_files = [
        out / "basis.npy",
        out / "scales.npy",
        out / "codes.int8.npy",
        out / "doc_ids.npy",
        out / "sidecar_config.json",
        out / "manifest.json",
    ]

    existing = [p for p in target_files if p.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "output artifact already contains files. Use --force to overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )

    basis_shape, basis_dtype = load_array_shape_dtype(args.basis)
    scales_shape, scales_dtype = load_array_shape_dtype(args.scales)
    codes_shape, codes_dtype = load_array_shape_dtype(args.codes)
    doc_ids_shape, doc_ids_dtype = load_array_shape_dtype(args.doc_ids)

    if len(basis_shape) != 2:
        raise ValueError(f"basis must be 2D, got {basis_shape}")

    dim, rank = basis_shape

    if scales_shape != (rank,):
        raise ValueError(
            f"scales shape mismatch: expected {(rank,)}, got {scales_shape}"
        )

    if len(codes_shape) != 2 or codes_shape[1] != rank:
        raise ValueError(
            f"codes shape mismatch: expected [num_docs, {rank}], got {codes_shape}"
        )

    if len(doc_ids_shape) != 1 or doc_ids_shape[0] != codes_shape[0]:
        raise ValueError(
            f"doc_ids shape mismatch: expected {(codes_shape[0],)}, got {doc_ids_shape}"
        )

    if args.default_top_b < 0 or args.max_top_b < 0:
        raise ValueError("top_b values must be non-negative")

    if args.default_top_b > args.max_top_b:
        raise ValueError(
            f"default_top_b={args.default_top_b} exceeds max_top_b={args.max_top_b}"
        )

    copy_or_convert_npy(args.basis, out / "basis.npy", dtype="float32")
    copy_or_convert_npy(args.scales, out / "scales.npy", dtype="float32")
    copy_or_convert_npy(args.codes, out / "codes.int8.npy", dtype="int8")
    copy_or_convert_npy(args.doc_ids, out / "doc_ids.npy", dtype=doc_ids_dtype)

    config = build_config(
        dim=dim,
        rank=rank,
        alpha=args.alpha,
        default_top_b=args.default_top_b,
        max_top_b=args.max_top_b,
        code_dtype="int8",
        doc_id_dtype=doc_ids_dtype,
    )
    write_json(out / "sidecar_config.json", config)

    exported_files = [
        "basis.npy",
        "scales.npy",
        "codes.int8.npy",
        "doc_ids.npy",
        "sidecar_config.json",
    ]

    manifest = {
        "artifact_type": "rars_residual_sidecar",
        "version": 1,
        "source_files": {k: str(v) for k, v in input_paths.items()},
        "exported_files": {},
        "shapes": {
            "basis": list(load_array_shape_dtype(out / "basis.npy")[0]),
            "scales": list(load_array_shape_dtype(out / "scales.npy")[0]),
            "codes": list(load_array_shape_dtype(out / "codes.int8.npy")[0]),
            "doc_ids": list(load_array_shape_dtype(out / "doc_ids.npy")[0]),
        },
        "dtypes": {
            "basis": load_array_shape_dtype(out / "basis.npy")[1],
            "scales": load_array_shape_dtype(out / "scales.npy")[1],
            "codes": load_array_shape_dtype(out / "codes.int8.npy")[1],
            "doc_ids": load_array_shape_dtype(out / "doc_ids.npy")[1],
        },
        "config": config,
    }

    for filename in exported_files:
        p = out / filename
        manifest["exported_files"][filename] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    write_json(out / "manifest.json", manifest)

    print("exported RARS sidecar artifact:", out)
    print("basis:", manifest["shapes"]["basis"], manifest["dtypes"]["basis"])
    print("scales:", manifest["shapes"]["scales"], manifest["dtypes"]["scales"])
    print("codes:", manifest["shapes"]["codes"], manifest["dtypes"]["codes"])
    print("doc_ids:", manifest["shapes"]["doc_ids"], manifest["dtypes"]["doc_ids"])
    print("config:", out / "sidecar_config.json")
    print("manifest:", out / "manifest.json")


if __name__ == "__main__":
    main()