#!/usr/bin/env python3
"""Train a development-only rank-16 boundary-loss sidecar.

The input bundle contains query vectors, frozen ANN rows/scores, binary candidate
relevance labels, and document residuals.  Its manifest must identify the split
as train or validation and must not reference the closed NQ test artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ALLOWED_SPLIT_ROLES = {"train", "validation"}
FORBIDDEN_MARKERS = ("nq test", "official test", "stage3/evaluation", "posthoc_diagnosis")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_development_manifest(manifest: dict[str, Any]) -> None:
    role = str(manifest.get("split_role", "")).casefold()
    if role not in ALLOWED_SPLIT_ROLES:
        raise ValueError(f"Boundary-loss development requires train/validation, got {role!r}")
    if manifest.get("test_qrels_accessed") is not False:
        raise ValueError("Manifest must explicitly state test_qrels_accessed=false")
    serialized = json.dumps(manifest, sort_keys=True).casefold()
    marker = next((value for value in FORBIDDEN_MARKERS if value in serialized), None)
    if marker is not None:
        raise ValueError(f"Closed-test marker is forbidden in v2 development: {marker}")


def build_boundary_pairs(
    labels: np.ndarray,
    ann_scores: np.ndarray,
    *,
    final_k: int = 10,
    negative_window: int = 10,
) -> np.ndarray:
    """Return (query, positive-position, negative-position) hard-boundary pairs."""
    labels = np.asarray(labels)
    scores = np.asarray(ann_scores)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("labels and ann_scores must be matching [queries, candidates] arrays")
    pairs: list[tuple[int, int, int]] = []
    for query_index in range(len(labels)):
        order = np.argsort(-scores[query_index], kind="stable")
        positives = np.flatnonzero(labels[query_index] > 0)
        boundary = order[: min(len(order), final_k + negative_window)]
        negatives = boundary[labels[query_index, boundary] <= 0]
        if not len(positives) or not len(negatives):
            continue
        for positive in positives:
            for negative in negatives:
                pairs.append((query_index, int(positive), int(negative)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 3)


def load_bundle(bundle_dir: Path) -> dict[str, np.ndarray]:
    manifest = read_json(bundle_dir / "manifest.json")
    validate_development_manifest(manifest)
    arrays = {
        name: np.load(bundle_dir / filename, mmap_mode="r")
        for name, filename in {
            "queries": "query_vectors.float32.npy",
            "ann_rows": "ann_rows.int64.npy",
            "ann_scores": "ann_scores.float32.npy",
            "labels": "candidate_relevance.uint8.npy",
            "residuals": "document_residuals.float32.npy",
        }.items()
    }
    q, c = arrays["ann_rows"].shape
    if arrays["queries"].shape[0] != q or arrays["ann_scores"].shape != (q, c):
        raise ValueError("Query and candidate array shapes do not agree")
    if arrays["labels"].shape != (q, c):
        raise ValueError("Candidate relevance labels do not match ANN candidates")
    return arrays


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    arrays = load_bundle(args.bundle_dir)
    pairs = build_boundary_pairs(
        arrays["labels"], arrays["ann_scores"], final_k=args.final_k,
        negative_window=args.negative_window,
    )
    if not len(pairs):
        raise ValueError("No relevance-boundary training pairs were found")
    rng = np.random.default_rng(args.seed)
    dimension = int(arrays["queries"].shape[1])
    device = torch.device(args.device)
    query_projection = torch.nn.Linear(dimension, args.rank, bias=False, device=device)
    document_projection = torch.nn.Linear(dimension, args.rank, bias=False, device=device)
    query_gate = torch.nn.Linear(dimension, 1, device=device)
    torch.nn.init.orthogonal_(query_projection.weight)
    torch.nn.init.orthogonal_(document_projection.weight)
    optimizer = torch.optim.AdamW(
        [*query_projection.parameters(), *document_projection.parameters(), *query_gate.parameters()],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    def fake_int8(values: torch.Tensor) -> torch.Tensor:
        maximum = values.detach().abs().amax(dim=0, keepdim=True).clamp_min(1e-8)
        scale = maximum / 127.0
        quantized = torch.clamp(torch.round(values / scale), -127, 127) * scale
        return values + (quantized - values).detach()

    losses: list[float] = []
    for _epoch in range(args.epochs):
        rng.shuffle(pairs)
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            qi, pi, ni = batch.T
            positive_rows = np.asarray(arrays["ann_rows"][qi, pi], dtype=np.int64)
            negative_rows = np.asarray(arrays["ann_rows"][qi, ni], dtype=np.int64)
            q = torch.as_tensor(np.asarray(arrays["queries"][qi]), device=device)
            rp = torch.as_tensor(np.asarray(arrays["residuals"][positive_rows]), device=device)
            rn = torch.as_tensor(np.asarray(arrays["residuals"][negative_rows]), device=device)
            base_margin = torch.as_tensor(
                np.asarray(arrays["ann_scores"][qi, pi] - arrays["ann_scores"][qi, ni]),
                device=device,
            )
            zq = query_projection(q)
            gate = torch.sigmoid(query_gate(q)).squeeze(1)
            cp = (zq * fake_int8(document_projection(rp))).sum(dim=1)
            cn = (zq * fake_int8(document_projection(rn))).sum(dim=1)
            corrected_margin = base_margin + gate * (cp - cn)
            loss = functional.softplus(args.margin - corrected_margin).mean()
            loss = loss + args.correction_l2 * ((cp.square() + cn.square()).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "query_projection.float32.npy", query_projection.weight.detach().cpu().numpy().T)
    np.save(args.output_dir / "document_projection.float32.npy", document_projection.weight.detach().cpu().numpy().T)
    torch.save(query_gate.state_dict(), args.output_dir / "query_gate.pt")
    summary = {
        "status": "development_feasibility_training_complete",
        "protocol_id": "rars_v2_boundary_loss_feasibility_v1",
        "rank": args.rank,
        "pair_count": int(len(pairs)),
        "epochs": args.epochs,
        "initial_loss": float(np.mean(losses[: max(1, min(len(losses), 10))])),
        "final_loss": float(np.mean(losses[-max(1, min(len(losses), 10)) :])),
        "test_qrels_accessed": False,
        "nq_test_retuning_authorized": False,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--negative-window", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correction-l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
