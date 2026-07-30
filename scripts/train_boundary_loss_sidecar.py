#!/usr/bin/env python3
"""Train and validate the relevance-supervised RARS-v2 sidecar.

The trainer is development-only.  It accepts train/validation bundles with
frozen ANN candidates, relevance labels, and document residuals.  Closed test
artifacts are rejected by manifest checks.  The document payload remains a
rank-16 signed-int8 code; its per-dimension scales are learned during QAT and
then reused unchanged for full-corpus encoding and validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v2_boundary_loss_feasibility_v1"
ALLOWED_SPLIT_ROLES = {"train", "validation"}
FORBIDDEN_MARKERS = (
    "nq test",
    "official test",
    "stage3/evaluation",
    "stage3/posthoc_diagnosis",
    "posthoc_diagnosis",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def validate_development_manifest(
    manifest: dict[str, Any], expected_role: str | None = None
) -> None:
    role = str(manifest.get("split_role", "")).casefold()
    if role not in ALLOWED_SPLIT_ROLES:
        raise ValueError(
            f"Boundary-loss development requires train/validation, got {role!r}"
        )
    if expected_role is not None and role != expected_role:
        raise ValueError(f"Expected a {expected_role} bundle, found {role}")
    if manifest.get("test_qrels_accessed") is not False:
        raise ValueError("Manifest must explicitly state test_qrels_accessed=false")
    if manifest.get("nq_test_retuning_authorized", False) is not False:
        raise ValueError("NQ test retuning must remain unauthorized")
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
    max_negatives_per_positive: int = 8,
    correction_depth: int = 40,
) -> np.ndarray:
    """Return deterministic (query, positive, negative) relevance pairs.

    Non-relevant documents are drawn from ANN ranks 1 through
    ``final_k + negative_window`` and ordered by proximity to the Top-k cutoff.
    Limiting negatives prevents a query with many judged positives from
    dominating the objective.
    """

    labels = np.asarray(labels)
    scores = np.asarray(ann_scores)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError(
            "labels and ann_scores must be matching [queries, candidates] arrays"
        )
    if (
        final_k <= 0
        or negative_window <= 0
        or max_negatives_per_positive <= 0
        or not final_k < correction_depth <= scores.shape[1]
    ):
        raise ValueError("Boundary-pair parameters must be positive")
    pairs: list[tuple[int, int, int]] = []
    for query_index in range(len(labels)):
        order = np.argsort(-scores[query_index], kind="stable")
        correctable = np.arange(scores.shape[1]) < correction_depth
        positives = np.flatnonzero((labels[query_index] > 0) & correctable)
        # Hardest negatives are closest to the final-k score cutoff.
        cutoff_position = min(final_k - 1, len(order) - 1)
        cutoff = scores[query_index, order[cutoff_position]]
        if not len(positives):
            continue
        for positive in positives:
            positive_rank = int(np.flatnonzero(order == positive)[0])
            if positive_rank >= final_k:
                pool = order[:final_k]
            else:
                pool = order[final_k:correction_depth]
            negatives = pool[labels[query_index, pool] <= 0]
            negatives = negatives[
                np.argsort(
                    np.abs(scores[query_index, negatives] - cutoff), kind="stable"
                )
            ][:max_negatives_per_positive]
            for negative in negatives:
                pairs.append((query_index, int(positive), int(negative)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 3)


def load_bundle(
    bundle_dir: Path,
    *,
    expected_role: str | None = None,
    require_relevant_counts: bool = False,
) -> dict[str, np.ndarray]:
    manifest = read_json(bundle_dir / "manifest.json")
    validate_development_manifest(manifest, expected_role)
    filenames = {
        "queries": "query_vectors.float32.npy",
        "ann_rows": "ann_rows.int64.npy",
        "ann_scores": "ann_scores.float32.npy",
        "labels": "candidate_relevance.uint8.npy",
    }
    arrays = {
        name: np.load(bundle_dir / filename, mmap_mode="r")
        for name, filename in filenames.items()
    }
    full_residuals = bundle_dir / "document_residuals.float32.npy"
    candidate_residuals = bundle_dir / "candidate_residuals.float32.npy"
    if full_residuals.exists():
        arrays["residuals"] = np.load(full_residuals, mmap_mode="r")
        arrays["residual_lookup"] = arrays["ann_rows"]
        arrays["residual_scope"] = np.asarray("full_corpus")
    elif candidate_residuals.exists():
        arrays["residuals"] = np.load(candidate_residuals, mmap_mode="r")
        arrays["residual_lookup"] = np.load(
            bundle_dir / "ann_residual_rows.int64.npy", mmap_mode="r"
        )
        arrays["residual_scope"] = np.asarray("candidate_union")
    else:
        raise ValueError("Bundle has no document or candidate residual array")

    counts_path = bundle_dir / "relevant_counts.int32.npy"
    if counts_path.exists():
        arrays["relevant_counts"] = np.load(counts_path, mmap_mode="r")
    elif require_relevant_counts:
        raise ValueError(
            "Validation Recall@k requires relevant_counts.int32.npy with full-qrels totals"
        )
    q, c = arrays["ann_rows"].shape
    if arrays["queries"].shape[0] != q or arrays["ann_scores"].shape != (q, c):
        raise ValueError("Query and candidate array shapes do not agree")
    if arrays["labels"].shape != (q, c):
        raise ValueError("Candidate relevance labels do not match ANN candidates")
    if arrays["residual_lookup"].shape != (q, c):
        raise ValueError("Residual lookup does not match ANN candidates")
    if arrays["residuals"].ndim != 2:
        raise ValueError("Document residuals must be a [documents, dimension] array")
    if "relevant_counts" in arrays:
        counts = np.asarray(arrays["relevant_counts"])
        if counts.shape != (q,) or np.any(counts <= 0):
            raise ValueError("Relevant counts must be positive and match query count")
    for name in ("pca", "rars"):
        path = bundle_dir / f"{name}_scores.float32.npy"
        if path.exists():
            values = np.load(path, mmap_mode="r")
            if values.shape != (q, c):
                raise ValueError(f"{name} baseline scores do not match candidates")
            arrays[f"{name}_scores"] = values
    return arrays


def calibrate_scales(
    residuals: np.ndarray,
    document_projection: np.ndarray,
    *,
    sample_rows: int = 100_000,
    percentile: float = 99.9,
    seed: int = 42,
) -> np.ndarray:
    """Deterministically initialize one deployable int8 scale per coefficient."""

    residuals = np.asarray(residuals)
    projection = np.asarray(document_projection, dtype=np.float32)
    if residuals.ndim != 2 or projection.ndim != 2:
        raise ValueError("Residuals and projection must be matrices")
    if residuals.shape[1] != projection.shape[0]:
        raise ValueError("Residual and projection dimensions do not match")
    rng = np.random.default_rng(seed)
    take = min(int(sample_rows), len(residuals))
    rows = np.sort(rng.choice(len(residuals), size=take, replace=False))
    values = np.asarray(residuals[rows], dtype=np.float32) @ projection
    maximum = np.percentile(np.abs(values), percentile, axis=0)
    return np.maximum(maximum / 127.0, 1e-8).astype(np.float32)


def quantize_coefficients(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if values.shape[-1] != len(scales) or np.any(scales <= 0):
        raise ValueError("Invalid coefficient scales")
    return np.clip(np.rint(values / scales), -127, 127).astype(np.int8)


def encode_document_codes(
    residuals: np.ndarray,
    projection: np.ndarray,
    scales: np.ndarray,
    output_path: Path,
    *,
    batch_size: int,
) -> Path:
    shape = (len(residuals), projection.shape[1])
    temporary = output_path.with_name(output_path.name + ".part")
    codes = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int8, shape=shape)
    for start in range(0, len(residuals), batch_size):
        end = min(start + batch_size, len(residuals))
        coefficients = np.asarray(residuals[start:end], dtype=np.float32) @ projection
        codes[start:end] = quantize_coefficients(coefficients, scales)
        codes.flush()
        if end % 250_000 == 0 or end == len(residuals):
            print(f"encoded document codes {end:,}/{len(residuals):,}")
    del codes
    temporary.replace(output_path)
    return output_path


def corrected_candidate_scores(
    arrays: dict[str, np.ndarray],
    query_projection: np.ndarray,
    document_projection: np.ndarray,
    gate_weight: np.ndarray,
    gate_bias: float,
    *,
    top_b: int,
    scales: np.ndarray | None,
    max_correction: float,
) -> np.ndarray:
    """Score validation candidates in FP32 or with fixed int8 scales."""

    queries = np.asarray(arrays["queries"], dtype=np.float32)
    rows = np.asarray(arrays["ann_rows"], dtype=np.int64)
    scores = np.asarray(arrays["ann_scores"], dtype=np.float32).copy()
    depth = min(top_b, rows.shape[1])
    candidate_rows = np.asarray(arrays["residual_lookup"][:, :depth], dtype=np.int64)
    if np.any(candidate_rows < 0):
        raise ValueError("Top-b candidates contain invalid document rows")
    residual = np.asarray(arrays["residuals"][candidate_rows], dtype=np.float32)
    coefficients = np.einsum("qcd,dr->qcr", residual, document_projection)
    if scales is not None:
        codes = quantize_coefficients(coefficients, scales)
        coefficients = codes.astype(np.float32) * scales
    q_projection = queries @ query_projection
    raw_correction = np.einsum("qr,qcr->qc", q_projection, coefficients)
    correction = max_correction * np.tanh(raw_correction / max_correction)
    gate = 1.0 / (1.0 + np.exp(-(queries @ gate_weight + gate_bias)))
    scores[:, :depth] += gate[:, None] * correction
    return scores


def recall_at_k_per_query(
    scores: np.ndarray,
    labels: np.ndarray,
    relevant_counts: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(np.asarray(labels), order, axis=1).sum(axis=1)
    return hits.astype(np.float64) / np.asarray(relevant_counts, dtype=np.float64)


def validation_summary(
    arrays: dict[str, np.ndarray],
    fp32_scores: np.ndarray,
    int8_scores: np.ndarray,
    *,
    final_k: int,
) -> dict[str, Any]:
    counts = np.asarray(arrays["relevant_counts"])
    base = recall_at_k_per_query(
        arrays["ann_scores"], arrays["labels"], counts, k=final_k
    )
    fp32 = recall_at_k_per_query(fp32_scores, arrays["labels"], counts, k=final_k)
    int8 = recall_at_k_per_query(int8_scores, arrays["labels"], counts, k=final_k)
    fp32_gain = float(np.mean(fp32 - base))
    int8_gain = float(np.mean(int8 - base))
    retained = None if fp32_gain <= 0 else int8_gain / fp32_gain
    delta = int8 - base
    summary = {
        "query_count": len(base),
        "base_recall_at_10": float(np.mean(base)),
        "boundary_fp32_recall_at_10": float(np.mean(fp32)),
        "boundary_int8_recall_at_10": float(np.mean(int8)),
        "fp32_gain_over_base": fp32_gain,
        "int8_gain_over_base": int8_gain,
        "int8_fraction_of_fp32_gain": retained,
        "improved_queries": int(np.sum(delta > 0)),
        "harmed_queries": int(np.sum(delta < 0)),
        "unchanged_queries": int(np.sum(delta == 0)),
    }
    for name in ("pca", "rars"):
        key = f"{name}_scores"
        if key not in arrays:
            continue
        baseline = recall_at_k_per_query(
            arrays[key], arrays["labels"], counts, k=final_k
        )
        recall = float(np.mean(baseline))
        summary[f"{name}_recall_at_10"] = recall
        summary[f"int8_gain_over_{name}"] = float(np.mean(int8 - baseline))
    if "pca_recall_at_10" in summary:
        summary["beats_storage_matched_pca"] = bool(
            summary["boundary_int8_recall_at_10"] > summary["pca_recall_at_10"]
        )
    return summary


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    arrays = load_bundle(args.bundle_dir, expected_role="train")
    selection = (
        load_bundle(
            args.selection_bundle_dir,
            expected_role="validation",
            require_relevant_counts=True,
        )
        if args.selection_bundle_dir is not None
        else None
    )
    validation = (
        load_bundle(
            args.validation_bundle_dir,
            expected_role="validation",
            require_relevant_counts=True,
        )
        if args.validation_bundle_dir is not None
        else None
    )
    pairs = build_boundary_pairs(
        arrays["labels"],
        arrays["ann_scores"],
        final_k=args.final_k,
        negative_window=args.negative_window,
        max_negatives_per_positive=args.max_negatives_per_positive,
        correction_depth=args.top_b,
    )
    if not len(pairs):
        raise ValueError("No relevance-boundary training pairs were found")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dimension = int(arrays["queries"].shape[1])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no GPU is available")

    query_projection = torch.nn.Linear(dimension, args.rank, bias=False, device=device)
    document_projection = torch.nn.Linear(dimension, args.rank, bias=False, device=device)
    query_gate = torch.nn.Linear(dimension, 1, device=device)
    torch.nn.init.orthogonal_(query_projection.weight)
    torch.nn.init.orthogonal_(document_projection.weight)
    torch.nn.init.zeros_(query_gate.weight)
    torch.nn.init.constant_(query_gate.bias, args.initial_gate_bias)
    initial_document_projection = document_projection.weight.detach().cpu().numpy().T
    initial_scales = calibrate_scales(
        arrays["residuals"],
        initial_document_projection,
        sample_rows=args.scale_sample_rows,
        percentile=args.scale_percentile,
        seed=args.seed,
    )
    log_scales = torch.nn.Parameter(torch.log(torch.from_numpy(initial_scales).to(device)))
    optimizer = torch.optim.AdamW(
        [
            *query_projection.parameters(),
            *document_projection.parameters(),
            *query_gate.parameters(),
            log_scales,
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def fake_int8(values: Any) -> Any:
        scales = torch.exp(log_scales).clamp(1e-8, 1.0)
        normalized = values / scales
        rounded = normalized + (torch.round(normalized) - normalized).detach()
        return torch.clamp(rounded, -127, 127) * scales

    def model_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        return (
            query_projection.weight.detach().cpu().numpy().T.astype(np.float32),
            document_projection.weight.detach().cpu().numpy().T.astype(np.float32),
            query_gate.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
            float(query_gate.bias.detach().cpu().numpy()[0]),
            np.exp(log_scales.detach().cpu().numpy()).clip(1e-8, 1.0).astype(np.float32),
        )

    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_selection_recall = -np.inf
    selected_epoch = args.epochs
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        losses: list[float] = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            qi, pi, ni = batch.T
            positive_rows = np.asarray(
                arrays["residual_lookup"][qi, pi], dtype=np.int64
            )
            negative_rows = np.asarray(
                arrays["residual_lookup"][qi, ni], dtype=np.int64
            )
            if np.any(positive_rows < 0) or np.any(negative_rows < 0):
                raise ValueError("Boundary pairs contain invalid document rows")
            q = torch.as_tensor(
                np.asarray(arrays["queries"][qi], dtype=np.float32), device=device
            )
            rp = torch.as_tensor(
                np.asarray(arrays["residuals"][positive_rows], dtype=np.float32),
                device=device,
            )
            rn = torch.as_tensor(
                np.asarray(arrays["residuals"][negative_rows], dtype=np.float32),
                device=device,
            )
            base_margin = torch.as_tensor(
                np.asarray(
                    arrays["ann_scores"][qi, pi] - arrays["ann_scores"][qi, ni],
                    dtype=np.float32,
                ),
                device=device,
            )
            zq = query_projection(q)
            gate = torch.sigmoid(query_gate(q)).squeeze(1)
            raw_cp = (zq * fake_int8(document_projection(rp))).sum(dim=1)
            raw_cn = (zq * fake_int8(document_projection(rn))).sum(dim=1)
            cp = args.max_correction * torch.tanh(raw_cp / args.max_correction)
            cn = args.max_correction * torch.tanh(raw_cn / args.max_correction)
            corrected_margin = base_margin + gate * (cp - cn)
            loss = functional.softplus(args.margin - corrected_margin).mean()
            loss = loss + args.correction_l2 * (cp.square() + cn.square()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    *query_projection.parameters(),
                    *document_projection.parameters(),
                    *query_gate.parameters(),
                    log_scales,
                ],
                args.max_grad_norm,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_record: dict[str, float] = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
        }
        if selection is not None:
            qm, dm, gw, gb, current_scales = model_arrays()
            selection_scores = corrected_candidate_scores(
                selection, qm, dm, gw, gb, top_b=args.top_b,
                scales=current_scales, max_correction=args.max_correction,
            )
            selection_recall = float(np.mean(recall_at_k_per_query(
                selection_scores, selection["labels"],
                selection["relevant_counts"], k=args.final_k,
            )))
            epoch_record["selection_int8_recall_at_10"] = selection_recall
            if selection_recall > best_selection_recall + 1e-12:
                best_selection_recall = selection_recall
                selected_epoch = epoch + 1
                best_state = {
                    "query_projection": {
                        key: value.detach().cpu().clone()
                        for key, value in query_projection.state_dict().items()
                    },
                    "document_projection": {
                        key: value.detach().cpu().clone()
                        for key, value in document_projection.state_dict().items()
                    },
                    "query_gate": {
                        key: value.detach().cpu().clone()
                        for key, value in query_gate.state_dict().items()
                    },
                    "log_scales": log_scales.detach().cpu().clone(),
                }
        history.append(epoch_record)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={history[-1]['loss']:.6f}")

    if best_state is not None:
        query_projection.load_state_dict(best_state["query_projection"])
        document_projection.load_state_dict(best_state["document_projection"])
        query_gate.load_state_dict(best_state["query_gate"])
        with torch.no_grad():
            log_scales.copy_(best_state["log_scales"].to(device))
    query_matrix, document_matrix, gate_weight, gate_bias, scales = model_arrays()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(args.output_dir / "query_projection.float32.npy", query_matrix)
    atomic_save_npy(args.output_dir / "document_projection.float32.npy", document_matrix)
    atomic_save_npy(args.output_dir / "document_scales.float32.npy", scales)
    atomic_save_npy(args.output_dir / "query_gate_weight.float32.npy", gate_weight)
    atomic_save_npy(
        args.output_dir / "query_gate_bias.float32.npy",
        np.asarray([gate_bias], dtype=np.float32),
    )
    atomic_write_json(args.output_dir / "training_history.json", history)

    codes_path = args.output_dir / "document_codes.int8.npy"
    residual_scope = str(np.asarray(arrays["residual_scope"]).item())
    if not args.skip_full_encoding and residual_scope != "full_corpus":
        raise ValueError(
            "Candidate-union bundles require --skip-full-encoding; export full "
            "codes later by streaming the frozen index and corpus embeddings"
        )
    if not args.skip_full_encoding:
        encode_document_codes(
            arrays["residuals"],
            document_matrix,
            scales,
            codes_path,
            batch_size=args.encode_batch_size,
        )

    validation_metrics = None
    if validation is not None:
        fp32_scores = corrected_candidate_scores(
            validation,
            query_matrix,
            document_matrix,
            gate_weight,
            gate_bias,
            top_b=args.top_b,
            scales=None,
            max_correction=args.max_correction,
        )
        int8_scores = corrected_candidate_scores(
            validation,
            query_matrix,
            document_matrix,
            gate_weight,
            gate_bias,
            top_b=args.top_b,
            scales=scales,
            max_correction=args.max_correction,
        )
        validation_metrics = validation_summary(
            validation, fp32_scores, int8_scores, final_k=args.final_k
        )
        atomic_write_json(args.output_dir / "validation_metrics.json", validation_metrics)

    summary = {
        "status": "development_feasibility_training_complete",
        "protocol_id": PROTOCOL_ID,
        "primary_supervision": "development_relevance_boundary",
        "rank": args.rank,
        "document_payload_bytes": args.rank,
        "top_b": args.top_b,
        "pair_count": int(len(pairs)),
        "epochs": args.epochs,
        "selected_epoch": selected_epoch,
        "selection_best_int8_recall_at_10": (
            None if selection is None else float(best_selection_recall)
        ),
        "max_correction": args.max_correction,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "quantization": "learned_fixed_per_coefficient_int8_scales",
        "document_codes_written": not args.skip_full_encoding,
        "training_residual_scope": residual_scope,
        "validation": validation_metrics,
        "test_qrels_accessed": False,
        "nq_test_retuning_authorized": False,
    }
    atomic_write_json(args.output_dir / "training_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--validation-bundle-dir", type=Path)
    parser.add_argument("--selection-bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--top-b", type=int, default=40)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--negative-window", type=int, default=10)
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--encode-batch-size", type=int, default=50_000)
    parser.add_argument("--scale-sample-rows", type=int, default=100_000)
    parser.add_argument("--scale-percentile", type=float, default=99.9)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correction-l2", type=float, default=1e-3)
    parser.add_argument("--max-correction", type=float, default=0.05)
    parser.add_argument("--initial-gate-bias", type=float, default=-2.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-full-encoding", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2))


if __name__ == "__main__":
    main()
