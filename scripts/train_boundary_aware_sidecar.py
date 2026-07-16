#!/usr/bin/env python3
"""Train a fixed-budget residual sidecar with a Top-k boundary objective.

This is a development-only method trainer.  It consumes ANN candidate caches and
full-precision candidate scores from a *development* collection, builds hard
pairs around the exact Top-k boundary, and learns a rank-k residual basis.  It
does not read qrels and deliberately refuses BEIR NQ, which is reserved for the
already-recorded post-hoc diagnosis.

The learned correction for document residual ``r`` and query ``q`` is

    correction(q, r) = (q @ B) dot fake_int8(r @ B)

where ``B`` is orthonormalized in every forward pass.  Fake int8 quantization
uses a straight-through estimator so the training objective matches the
deployed rank-16/int8 sidecar more closely than a float-only fit.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairBatch:
    """Flattened hard pairs referring to rows within candidate matrices."""

    query: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.query), len(self.positive), len(self.negative), len(self.weight)
        }
        if len(lengths) != 1:
            raise ValueError("Hard-pair arrays must have equal length")

    def __len__(self) -> int:
        return len(self.query)


def stable_descending_order(scores: np.ndarray) -> np.ndarray:
    """Sort descending while preserving candidate order for score ties."""

    return np.argsort(-np.asarray(scores), axis=1, kind="stable")


def validate_development_dataset(name: str) -> str:
    """Prevent post-hoc NQ results from becoming method-selection data."""

    normalized = "".join(character for character in name.lower() if character.isalnum())
    if normalized in {"nq", "beirnq", "naturalquestions"} or "beirnq" in normalized:
        raise ValueError(
            "BEIR NQ is locked for post-hoc diagnosis and cannot train this method"
        )
    if not normalized:
        raise ValueError("A development dataset name is required")
    return name


def build_boundary_pairs(
    exact_scores: np.ndarray,
    ann_scores: np.ndarray,
    *,
    final_k: int = 10,
    top_b: int = 40,
    positives_per_query: int = 4,
    negatives_per_positive: int = 4,
    margin_temperature: float = 0.05,
) -> PairBatch:
    """Create deterministic hard pairs straddling the exact Top-k boundary.

    Positives are the exact ranks closest to the lower edge of Top-k.  Negative
    examples are chosen from ranks ``k+1..top_b`` with base-order inversions
    first, then by smallest exact-score margin.  Weights emphasize small teacher
    margins and pairs that the ANN score currently orders incorrectly.
    """

    exact = np.asarray(exact_scores, dtype=np.float32)
    ann = np.asarray(ann_scores, dtype=np.float32)
    if exact.shape != ann.shape or exact.ndim != 2:
        raise ValueError("exact_scores and ann_scores must be matching 2-D arrays")
    if not 0 < final_k < top_b <= exact.shape[1]:
        raise ValueError("Require 0 < final_k < top_b <= candidate count")
    if positives_per_query <= 0 or negatives_per_positive <= 0:
        raise ValueError("Pair counts must be positive")
    if margin_temperature <= 0:
        raise ValueError("margin_temperature must be positive")

    query_rows: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    weights: list[float] = []
    exact_order = stable_descending_order(exact[:, :top_b])

    for query_index in range(len(exact)):
        order = exact_order[query_index]
        positive_pool = order[max(0, final_k - positives_per_query):final_k]
        negative_pool = order[final_k:top_b]
        for positive in positive_pool:
            teacher_margin = exact[query_index, positive] - exact[query_index, negative_pool]
            finite = np.isfinite(teacher_margin) & (teacher_margin >= 0)
            if not np.any(finite):
                continue
            candidates = negative_pool[finite]
            margins = teacher_margin[finite]
            inversions = ann[query_index, positive] <= ann[query_index, candidates]
            # Inversions first, then the smallest exact margin, then candidate row.
            selection_order = np.lexsort((candidates, margins, ~inversions))
            for selected in selection_order[:negatives_per_positive]:
                negative = int(candidates[selected])
                margin = float(margins[selected])
                inverted = bool(inversions[selected])
                boundary_weight = float(np.exp(-margin / margin_temperature))
                query_rows.append(query_index)
                positives.append(int(positive))
                negatives.append(negative)
                weights.append((1.0 + float(inverted)) * (0.25 + boundary_weight))

    if not query_rows:
        raise ValueError("No finite boundary pairs could be constructed")
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array /= np.mean(weight_array)
    return PairBatch(
        query=np.asarray(query_rows, dtype=np.int64),
        positive=np.asarray(positives, dtype=np.int64),
        negative=np.asarray(negatives, dtype=np.int64),
        weight=weight_array,
    )


def pairwise_softplus_numpy(
    corrected_scores: np.ndarray,
    pairs: PairBatch,
) -> float:
    """Reference implementation of the weighted boundary loss."""

    scores = np.asarray(corrected_scores, dtype=np.float32)
    margin = (
        scores[pairs.query, pairs.positive]
        - scores[pairs.query, pairs.negative]
    )
    return float(np.mean(pairs.weight * np.logaddexp(0.0, -margin)))


def topk_overlap(corrected: np.ndarray, exact: np.ndarray, final_k: int) -> float:
    corrected_order = stable_descending_order(corrected)
    exact_order = stable_descending_order(exact)
    overlaps = [
        len(set(corrected_order[i, :final_k]) & set(exact_order[i, :final_k]))
        / final_k
        for i in range(len(exact))
    ]
    return float(np.mean(overlaps))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_queries(path: Path, rows_path: Path | None) -> np.ndarray:
    queries = np.load(path, mmap_mode="r")
    if rows_path is not None:
        rows = np.load(rows_path)
        queries = np.asarray(queries[rows], dtype=np.float32)
    return np.asarray(queries, dtype=np.float32)


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run QAT boundary training.  Torch is imported only in this code path."""

    validate_development_dataset(args.development_dataset)
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - Colab dependency
        raise RuntimeError("Training requires Colab's CUDA-enabled PyTorch") from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no GPU is available")

    queries = _load_queries(args.query_vectors, args.query_rows)
    candidate_rows = np.load(args.ann_rows, mmap_mode="r")
    ann_scores = np.load(args.ann_scores, mmap_mode="r")
    exact_scores = np.load(args.exact_scores, mmap_mode="r")
    expected = tuple(candidate_rows.shape)
    if tuple(ann_scores.shape) != expected or tuple(exact_scores.shape) != expected:
        raise ValueError("Candidate row and score cache shapes do not match")
    if len(queries) != expected[0]:
        raise ValueError("Query count does not match candidate caches")

    residuals = np.memmap(
        args.residuals,
        dtype=np.dtype(args.residual_dtype),
        mode="r",
        shape=(args.n_docs, args.dim),
    )
    initial = np.load(args.initial_basis).astype(np.float32)
    if initial.shape != (args.dim, args.rank):
        raise ValueError(f"Initial basis shape {initial.shape} is not {(args.dim, args.rank)}")

    pairs = build_boundary_pairs(
        exact_scores,
        ann_scores,
        final_k=args.final_k,
        top_b=args.top_b,
        positives_per_query=args.positives_per_query,
        negatives_per_positive=args.negatives_per_positive,
        margin_temperature=args.margin_temperature,
    )
    basis_parameter = torch.nn.Parameter(torch.from_numpy(initial).to(device))
    optimizer = torch.optim.AdamW([basis_parameter], lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []

    def fake_int8(values: Any) -> Any:
        scale = values.detach().abs().amax(dim=0).clamp_min(1e-8) / 127.0
        quantized = torch.clamp(torch.round(values / scale), -127, 127) * scale
        return values + (quantized - values).detach()

    indices = np.arange(len(pairs), dtype=np.int64)
    for epoch in range(args.epochs):
        rng.shuffle(indices)
        losses: list[float] = []
        for start in range(0, len(indices), args.pair_batch_size):
            chosen = indices[start:start + args.pair_batch_size]
            q_index = pairs.query[chosen]
            positive_index = pairs.positive[chosen]
            negative_index = pairs.negative[chosen]
            positive_docs = np.asarray(candidate_rows[q_index, positive_index], dtype=np.int64)
            negative_docs = np.asarray(candidate_rows[q_index, negative_index], dtype=np.int64)
            if np.any(positive_docs < 0) or np.any(negative_docs < 0):
                raise ValueError("Hard pairs contain invalid candidate document rows")

            q = torch.from_numpy(queries[q_index]).to(device)
            positive_r = torch.from_numpy(
                np.asarray(residuals[positive_docs], dtype=np.float32)
            ).to(device)
            negative_r = torch.from_numpy(
                np.asarray(residuals[negative_docs], dtype=np.float32)
            ).to(device)
            base_margin = torch.from_numpy(
                np.asarray(
                    ann_scores[q_index, positive_index]
                    - ann_scores[q_index, negative_index],
                    dtype=np.float32,
                )
            ).to(device)
            weight = torch.from_numpy(pairs.weight[chosen]).to(device)

            basis, _ = torch.linalg.qr(basis_parameter, mode="reduced")
            q_projection = q @ basis
            positive_code = fake_int8(positive_r @ basis)
            negative_code = fake_int8(negative_r @ basis)
            corrected_margin = base_margin + args.alpha * torch.sum(
                q_projection * (positive_code - negative_code), dim=1
            )
            loss = torch.mean(weight * functional.softplus(-corrected_margin))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([basis_parameter], args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        history.append({"epoch": epoch + 1, "pairwise_loss": float(np.mean(losses))})
        print(f"epoch {epoch + 1}/{args.epochs}: loss={history[-1]['pairwise_loss']:.6f}")

    with torch.no_grad():
        basis, _ = torch.linalg.qr(basis_parameter, mode="reduced")
        learned = basis.cpu().numpy().astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = args.output_dir / f"boundary_aware_rank{args.rank}.float32.npy"
    atomic_save_npy(basis_path, learned)
    atomic_write_json(args.output_dir / "training_history.json", history)
    config = {
        "method": "boundary_aware_residual_sidecar_v1",
        "development_dataset": args.development_dataset,
        "qrels_used": False,
        "beir_nq_used_for_training": False,
        "rank": args.rank,
        "coefficient_dtype": "int8_qat_ste",
        "top_b": args.top_b,
        "final_k": args.final_k,
        "alpha": args.alpha,
        "hard_pair_count": len(pairs),
        "loss": "weighted_pairwise_softplus",
        "basis": str(basis_path),
        "selection_status": "development_only_not_frozen",
    }
    atomic_write_json(args.output_dir / "training_config.json", config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-dataset", required=True)
    parser.add_argument("--query-vectors", required=True, type=Path)
    parser.add_argument("--query-rows", type=Path)
    parser.add_argument("--ann-rows", required=True, type=Path)
    parser.add_argument("--ann-scores", required=True, type=Path)
    parser.add_argument("--exact-scores", required=True, type=Path)
    parser.add_argument("--residuals", required=True, type=Path)
    parser.add_argument("--initial-basis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-docs", required=True, type=int)
    parser.add_argument("--dim", default=384, type=int)
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--top-b", default=40, type=int)
    parser.add_argument("--final-k", default=10, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--positives-per-query", default=4, type=int)
    parser.add_argument("--negatives-per-positive", default=4, type=int)
    parser.add_argument("--margin-temperature", default=0.05, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--pair-batch-size", default=2048, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--max-grad-norm", default=5.0, type=float)
    parser.add_argument("--residual-dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2))


if __name__ == "__main__":
    main()
