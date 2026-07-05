#!/usr/bin/env python3
"""Validate a bitmap + rank-prefix compact residual-sidecar layout."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class CompactLayout:
    n_docs: int
    block_size: int
    bitmap_bytes: int
    rank_prefix_bytes: int
    selection_index_bytes: int


def build_layout(n_docs: int, block_size: int = 256) -> CompactLayout:
    if n_docs <= 0:
        raise ValueError("n_docs must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    bitmap_bytes = ceil(n_docs / 8)
    block_count = ceil(n_docs / block_size)

    # One uint32 prefix per block start, plus one terminal total.
    rank_prefix_bytes = (block_count + 1) * 4

    return CompactLayout(
        n_docs=n_docs,
        block_size=block_size,
        bitmap_bytes=bitmap_bytes,
        rank_prefix_bytes=rank_prefix_bytes,
        selection_index_bytes=bitmap_bytes + rank_prefix_bytes,
    )


def build_bitmap_and_prefix(
    selected_ids: np.ndarray,
    n_docs: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_ids = np.asarray(selected_ids, dtype=np.int64)

    if selected_ids.ndim != 1:
        raise ValueError("selected_ids must be one-dimensional.")
    if len(selected_ids) and (
        selected_ids[0] < 0 or selected_ids[-1] >= n_docs
    ):
        raise ValueError("selected_ids contains an out-of-range ID.")
    if len(selected_ids) > 1 and np.any(selected_ids[1:] <= selected_ids[:-1]):
        raise ValueError("selected_ids must be sorted and unique.")

    bitmap = np.zeros(ceil(n_docs / 8), dtype=np.uint8)

    for doc_id in selected_ids:
        byte_index = int(doc_id) // 8
        bit_index = int(doc_id) % 8
        bitmap[byte_index] |= np.uint8(1 << bit_index)

    block_count = ceil(n_docs / block_size)
    prefix = np.zeros(block_count + 1, dtype=np.uint32)

    selected_set = set(int(x) for x in selected_ids)

    running = 0
    for block in range(block_count):
        prefix[block] = running
        start = block * block_size
        end = min(start + block_size, n_docs)

        for doc_id in range(start, end):
            if doc_id in selected_set:
                running += 1

    prefix[block_count] = running

    if int(prefix[-1]) != len(selected_ids):
        raise AssertionError("Prefix total does not match selected count.")

    return bitmap, prefix


def is_selected(bitmap: np.ndarray, doc_id: int) -> bool:
    byte_index = doc_id // 8
    bit_index = doc_id % 8
    return bool((int(bitmap[byte_index]) >> bit_index) & 1)


def rank_before(
    bitmap: np.ndarray,
    prefix: np.ndarray,
    doc_id: int,
    block_size: int,
) -> int:
    """Number of selected IDs strictly smaller than doc_id."""
    block = doc_id // block_size
    block_start = block * block_size
    rank = int(prefix[block])

    for current_id in range(block_start, doc_id):
        if is_selected(bitmap, current_id):
            rank += 1

    return rank


def lookup_slot(
    bitmap: np.ndarray,
    prefix: np.ndarray,
    doc_id: int,
    block_size: int,
) -> int | None:
    if not is_selected(bitmap, doc_id):
        return None
    return rank_before(bitmap, prefix, doc_id, block_size)


def maximum_selected_count(
    *,
    n_docs: int,
    total_budget_bytes_per_vector: float,
    base_bytes_per_vector: float,
    codebook_bytes: int,
    selection_index_bytes: int,
    residual_payload_bytes_per_selected_doc: int,
) -> int:
    total_budget_bytes = int(round(total_budget_bytes_per_vector * n_docs))
    base_bytes = int(round(base_bytes_per_vector * n_docs))

    available_payload_bytes = (
        total_budget_bytes
        - base_bytes
        - codebook_bytes
        - selection_index_bytes
    )

    return max(
        0,
        min(
            n_docs,
            available_payload_bytes // residual_payload_bytes_per_selected_doc,
        ),
    )


def main() -> None:
    n_docs = 57_638
    base_bytes = 32.0
    total_budget = 48.0

    layout = build_layout(n_docs=n_docs, block_size=256)

    print("Compact selection index")
    print(f"bitmap bytes:       {layout.bitmap_bytes:,}")
    print(f"rank-prefix bytes:  {layout.rank_prefix_bytes:,}")
    print(f"selection index:    {layout.selection_index_bytes:,}")
    print(
        "selection index avg:"
        f" {layout.selection_index_bytes / n_docs:.6f} B/vector"
    )

    configs = [
        {
            "name": "legacy_8bit_m16_fp32_codebook_uint32_ids",
            "codebook_bytes": 384 * 256 * 4,
            "selection_index_bytes": 0,
            "payload_bytes": 16 + 4,
        },
        {
            "name": "compact_8bit_m16_fp16_codebook",
            "codebook_bytes": 384 * 256 * 2,
            "selection_index_bytes": layout.selection_index_bytes,
            "payload_bytes": 16,
        },
        {
            "name": "compact_4bit_m32_fp16_codebook",
            "codebook_bytes": 384 * 16 * 2,
            "selection_index_bytes": layout.selection_index_bytes,
            "payload_bytes": 16,
        },
    ]

    print("\nBudget table")
    print(
        "name, codebook_bytes, index_bytes, payload_bytes, "
        "selected_count, selected_fraction, total_B_per_vector"
    )

    for config in configs:
        selected_count = maximum_selected_count(
            n_docs=n_docs,
            total_budget_bytes_per_vector=total_budget,
            base_bytes_per_vector=base_bytes,
            codebook_bytes=config["codebook_bytes"],
            selection_index_bytes=config["selection_index_bytes"],
            residual_payload_bytes_per_selected_doc=config["payload_bytes"],
        )

        total_bytes = (
            base_bytes * n_docs
            + config["codebook_bytes"]
            + config["selection_index_bytes"]
            + selected_count * config["payload_bytes"]
        ) / n_docs

        print(
            f"{config['name']}, "
            f"{config['codebook_bytes']}, "
            f"{config['selection_index_bytes']}, "
            f"{config['payload_bytes']}, "
            f"{selected_count}, "
            f"{selected_count / n_docs:.6f}, "
            f"{total_bytes:.6f}"
        )

    rng = np.random.default_rng(20260706)
    selected_ids = np.sort(
        rng.choice(n_docs, size=10_000, replace=False).astype(np.int64)
    )

    bitmap, prefix = build_bitmap_and_prefix(
        selected_ids=selected_ids,
        n_docs=n_docs,
        block_size=layout.block_size,
    )

    legacy_lookup = {
        int(doc_id): position
        for position, doc_id in enumerate(selected_ids)
    }

    probe_ids = np.concatenate(
        [
            selected_ids[:100],
            rng.integers(0, n_docs, size=1_000, dtype=np.int64),
        ]
    )

    for doc_id in probe_ids:
        compact_slot = lookup_slot(
            bitmap=bitmap,
            prefix=prefix,
            doc_id=int(doc_id),
            block_size=layout.block_size,
        )
        legacy_slot = legacy_lookup.get(int(doc_id))

        if compact_slot != legacy_slot:
            raise AssertionError(
                f"Lookup mismatch for doc_id={doc_id}: "
                f"compact={compact_slot}, legacy={legacy_slot}"
            )

    print("\nLookup equivalence: PASS")
    print(
        "Validated bitmap + rank-prefix slots against "
        "a legacy document-ID dictionary."
    )


if __name__ == "__main__":
    main()
