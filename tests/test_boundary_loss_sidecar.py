from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_boundary_loss_sidecar.py"
SPEC = importlib.util.spec_from_file_location("boundary_loss_sidecar", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_accepts_development_train_split() -> None:
    MODULE.validate_development_manifest({
        "split_role": "train",
        "source": "BEIR NQ train archive",
        "test_qrels_accessed": False,
    })


@pytest.mark.parametrize("role", ["test", "evaluation", "posthoc"])
def test_manifest_rejects_non_development_roles(role: str) -> None:
    with pytest.raises(ValueError):
        MODULE.validate_development_manifest({
            "split_role": role,
            "source": "development",
            "test_qrels_accessed": False,
        })


def test_manifest_rejects_closed_test_marker() -> None:
    with pytest.raises(ValueError, match="Closed-test marker"):
        MODULE.validate_development_manifest({
            "split_role": "validation",
            "source": "stage3/evaluation/test_query_vectors.float32.npy",
            "test_qrels_accessed": False,
        })


def test_boundary_pairs_use_relevant_positive_and_boundary_negative() -> None:
    scores = np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    labels = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)
    pairs = MODULE.build_boundary_pairs(labels, scores, final_k=2, negative_window=1)
    assert pairs.tolist() == [[0, 2, 1], [0, 2, 0]]


def test_boundary_pairs_skip_queries_without_candidate_positive() -> None:
    scores = np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    labels = np.asarray([[0, 0], [1, 0]], dtype=np.uint8)
    pairs = MODULE.build_boundary_pairs(labels, scores, final_k=1, negative_window=1)
    assert pairs.tolist() == [[1, 0, 1]]


def test_fixed_int8_scales_are_deterministic_and_reused() -> None:
    residuals = np.asarray(
        [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0], [0.5, -0.5]],
        dtype=np.float32,
    )
    projection = np.eye(2, dtype=np.float32)
    first = MODULE.calibrate_scales(
        residuals, projection, sample_rows=4, percentile=100.0, seed=7
    )
    second = MODULE.calibrate_scales(
        residuals, projection, sample_rows=4, percentile=100.0, seed=7
    )
    assert np.array_equal(first, second)
    codes = MODULE.quantize_coefficients(residuals, first)
    assert codes.dtype == np.int8
    assert np.max(np.abs(codes[:, 0])) == 127
    assert np.max(np.abs(codes[:, 1])) == 127


def test_recall_and_validation_summary_use_full_relevant_counts() -> None:
    arrays = {
        "ann_scores": np.asarray([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32),
        "labels": np.asarray([[0, 1, 0], [1, 0, 0]], dtype=np.uint8),
        "relevant_counts": np.asarray([2, 1], dtype=np.int32),
        "pca_scores": np.asarray([[0.8, 0.9, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32),
    }
    fp32 = np.asarray([[0.8, 0.9, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32)
    int8 = fp32.copy()
    summary = MODULE.validation_summary(arrays, fp32, int8, final_k=1)
    assert summary["base_recall_at_10"] == 0.5
    assert summary["boundary_int8_recall_at_10"] == 0.75
    assert summary["int8_gain_over_base"] == 0.25
    assert summary["improved_queries"] == 1
    assert summary["harmed_queries"] == 0
    assert summary["pca_recall_at_10"] == 0.75
    assert summary["beats_storage_matched_pca"] is False


def test_candidate_union_bundle_maps_ann_rows_to_local_residuals(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"split_role":"train","test_qrels_accessed":false,'
        '"nq_test_retuning_authorized":false}',
        encoding="utf-8",
    )
    np.save(tmp_path / "query_vectors.float32.npy", np.zeros((1, 2), np.float32))
    np.save(tmp_path / "ann_rows.int64.npy", np.asarray([[10, 20]], np.int64))
    np.save(tmp_path / "ann_scores.float32.npy", np.asarray([[0.2, 0.1]], np.float32))
    np.save(tmp_path / "candidate_relevance.uint8.npy", np.asarray([[1, 0]], np.uint8))
    np.save(tmp_path / "candidate_residuals.float32.npy", np.eye(2, dtype=np.float32))
    np.save(tmp_path / "ann_residual_rows.int64.npy", np.asarray([[1, 0]], np.int64))
    bundle = MODULE.load_bundle(tmp_path, expected_role="train")
    assert bundle["residual_scope"].item() == "candidate_union"
    assert bundle["residual_lookup"].tolist() == [[1, 0]]
