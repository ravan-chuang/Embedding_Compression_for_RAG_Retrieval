from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rars_v2_2_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v2_2_core", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_pca(tmp_path: Path, basis: np.ndarray, *, alpha: float = 0.75):
    basis_path = tmp_path / "basis.npy"
    config_path = tmp_path / "config.json"
    np.save(basis_path, basis.astype(np.float32))
    config_path.write_text(json.dumps({"rank": 2, "top_b": 3, "alpha": alpha}))
    return basis_path, config_path


def _toy_arrays() -> dict[str, np.ndarray]:
    return {
        "queries": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "ann_scores": np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32),
        "residual_lookup": np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        "residuals": np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [8.0, 0.0]],
            dtype=np.float32,
        ),
    }


def test_pca_warm_start_matches_unbounded_pca_correction(tmp_path: Path) -> None:
    basis = np.eye(2, dtype=np.float32)
    basis_path, config_path = _write_pca(tmp_path, basis, alpha=0.5)
    wq, wd, alpha = MODULE.load_pca_warm_start(
        basis_path, config_path, dimension=2, rank=2, top_b=3
    )
    query = np.asarray([2.0, -1.0], dtype=np.float32)
    residual = np.asarray([0.5, 3.0], dtype=np.float32)
    learned = float((query @ wq) @ (residual @ wd))
    expected = float(alpha * query @ basis @ basis.T @ residual)
    assert np.isclose(learned, expected)


def test_pca_warm_start_rejects_nonorthogonal_basis(tmp_path: Path) -> None:
    basis_path, config_path = _write_pca(
        tmp_path, np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="orthonormal"):
        MODULE.load_pca_warm_start(
            basis_path, config_path, dimension=2, rank=2, top_b=3
        )


def test_no_gate_fp32_scorer_matches_tanh_and_respects_top_b() -> None:
    arrays = _toy_arrays()
    result = MODULE.score_candidates_fp32(
        arrays,
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
        top_b=3,
        max_correction=0.05,
        batch_size=1,
    )
    raw = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
    expected = arrays["ann_scores"].copy()
    expected[0, :3] += 0.05 * np.tanh(raw / 0.05)
    assert np.allclose(result, expected)
    assert result[0, 3] == arrays["ann_scores"][0, 3]


def test_pca_fp32_comparator_uses_residual_direct_formula() -> None:
    arrays = _toy_arrays()
    basis = np.eye(2, dtype=np.float32)
    result = MODULE.pca_fp32_scores(
        arrays, basis, alpha=0.5, top_b=3, batch_size=1
    )
    assert np.allclose(result[0, :3], [0.9, 0.3, -0.3])
    np.testing.assert_array_equal(
        result[:, 3:], arrays["ann_scores"][:, 3:]
    )


def test_dynamic_miner_builds_promotion_and_protection_pairs() -> None:
    scores = np.asarray(
        [[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]],
        dtype=np.float32,
    )
    labels = np.asarray([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.uint8)
    pairs = MODULE.mine_dynamic_boundary_pairs(
        labels,
        scores,
        final_k=2,
        top_b=4,
        max_negatives_per_positive=2,
        promotion_mix=0.8,
    )
    observed = set(zip(
        pairs.query.tolist(),
        pairs.positive.tolist(),
        pairs.negative.tolist(),
        pairs.kind.tolist(),
    ))
    assert (0, 2, 0, int(MODULE.PROMOTION)) in observed
    assert (0, 2, 1, int(MODULE.PROMOTION)) in observed
    assert (1, 0, 2, int(MODULE.PROTECTION)) in observed
    promotion_fraction = float(
        np.sum(pairs.weight[pairs.kind == MODULE.PROMOTION])
        / np.sum(pairs.weight)
    )
    assert np.isclose(promotion_fraction, 0.8)


def test_pair_weights_equalize_queries_with_different_pair_counts() -> None:
    query = np.asarray([0, 1, 1, 1, 1], dtype=np.int64)
    kind = np.asarray([0, 0, 0, 0, 0], dtype=np.uint8)
    weights = MODULE._macro_query_weights(query, kind, promotion_mix=0.8)
    assert np.isclose(np.sum(weights[query == 0]), np.sum(weights[query == 1]))
    assert np.isclose(np.mean(weights), 1.0)


def test_dynamic_mining_changes_when_current_ranking_changes() -> None:
    labels = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)
    before = MODULE.mine_dynamic_boundary_pairs(
        labels,
        np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
        final_k=2,
        top_b=4,
    )
    after = MODULE.mine_dynamic_boundary_pairs(
        labels,
        np.asarray([[0.9, 0.7, 1.0, 0.6]], dtype=np.float32),
        final_k=2,
        top_b=4,
    )
    assert before.promotion_count > 0
    assert after.promotion_count == 0
    assert after.protection_count > 0


def test_dynamic_mining_is_stable_under_ties() -> None:
    labels = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)
    scores = np.ones((1, 4), dtype=np.float32)
    first = MODULE.mine_dynamic_boundary_pairs(
        labels, scores, final_k=2, top_b=4
    )
    second = MODULE.mine_dynamic_boundary_pairs(
        labels, scores, final_k=2, top_b=4
    )
    assert np.array_equal(first.positive, second.positive)
    assert np.array_equal(first.negative, second.negative)
    assert np.array_equal(first.weight, second.weight)


def _valid_manifest(role_id: str) -> dict[str, object]:
    return {
        "protocol_id": MODULE.PROTOCOL_ID,
        "role_id": role_id,
        "split_role": "train" if role_id == "inner_train" else "validation",
        "evidence_status": "DEVELOPMENT_ONLY",
        "source_commit": "0" * 40,
        "query_ids_sha256": "a" * 64,
        "query_rows_sha256": "b" * 64,
        "split_audit_sha256": "c" * 64,
        "source_bundle_manifest_sha256": "d" * 64,
        "source_builder_sha256": "e" * 64,
        "bundle_freezer_sha256": "f" * 64,
        "protocol_sha256": "1" * 64,
        "data_access": {
            "outer_outcomes_used": False,
            "closed_test_relevance_values_used": False,
        },
    }


def test_manifest_rejects_outer_bundle_even_when_called_validation() -> None:
    manifest = _valid_manifest("inner_validation")
    manifest["role_id"] = "outer_validation"
    with pytest.raises(ValueError, match="Expected role_id"):
        MODULE.validate_bundle_manifest(
            manifest, expected_role_id=MODULE.SELECTION_ROLE_ID
        )


def test_run_fingerprint_changes_for_any_configuration_change() -> None:
    payload = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": "1" * 40,
        "trainer_sha256": "2" * 64,
        "train_bundle_manifest_sha256": "3" * 64,
        "selection_bundle_manifest_sha256": "4" * 64,
        "pca_basis_sha256": "5" * 64,
        "pca_config_sha256": "6" * 64,
        "configuration": {"seed": 42, "top_b": 40},
    }
    first = MODULE.build_run_fingerprint(payload)
    changed = json.loads(json.dumps(payload))
    changed["configuration"]["seed"] = 43
    assert MODULE.build_run_fingerprint(changed) != first
