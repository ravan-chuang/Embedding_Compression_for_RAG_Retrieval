from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/rars_v16_causal_generalization_core.py"
SPEC = importlib.util.spec_from_file_location(
    "rars_v16_causal_generalization_core", PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SOURCE = PATH.read_text(encoding="utf-8")


def test_fold_assignment_is_order_independent_and_rejects_duplicates() -> None:
    ids = ["q-4", "q-1", "q-9", "q-2"]
    first = dict(zip(ids, MODULE.deterministic_fold_ids(ids, fold_count=7)))
    reversed_ids = list(reversed(ids))
    second = dict(
        zip(
            reversed_ids,
            MODULE.deterministic_fold_ids(reversed_ids, fold_count=7),
        )
    )
    assert first == second
    assert MODULE.deterministic_query_priority("q-1") == (
        MODULE.deterministic_query_priority("q-1")
    )
    with pytest.raises(ValueError, match="unique"):
        MODULE.deterministic_fold_ids(["q-1", "q-1"])


def test_fp32_scorer_is_stable_on_ties_and_only_changes_frozen_top_b() -> None:
    queries = np.asarray([[1.0, 1.0]], dtype=np.float16)
    rows = np.asarray([[12, 10, 11, -1]], dtype=np.int64)
    lookup = np.asarray([[2, 0, 1, -1]], dtype=np.int64)
    base = np.asarray([[0.8, 0.8, 0.8, 99.0]], dtype=np.float32)
    residuals = np.asarray([[1, 0], [0, 1], [8, 8]], dtype=np.float32)
    basis = np.eye(2, dtype=np.float64)
    scored = MODULE.score_fp32_sidecar_candidates(
        queries,
        rows,
        lookup,
        base,
        residuals,
        basis,
        alpha=0.5,
        top_b=2,
    )
    # The invalid row is never selected; row ids 10 then 11 win the valid tie.
    assert np.allclose(scored, [[0.8, 1.3, 1.3, 99.0]])
    assert scored.dtype == np.float32


def test_score_error_basis_aggregates_duplicate_rows_without_pandas() -> None:
    residuals = np.asarray(
        [[4.0, 0.0], [0.0, 2.0], [0.0, 1.0]], dtype=np.float32
    )
    rows = np.asarray([[0, 1], [1, -1]], dtype=np.int64)
    errors = np.asarray([[0.1, 5.0], [5.0, 100.0]], dtype=np.float32)
    weights = MODULE.aggregate_score_error_weights(
        rows, errors, residual_count=3
    )
    assert np.allclose(weights, [0.1, 10.0, 0.0])
    basis = MODULE.fit_score_error_weighted_basis(
        residuals, errors, residual_rows=rows, rank=1
    )
    assert basis.dtype == np.float32
    assert np.allclose(basis.T @ basis, np.eye(1), atol=1e-6)
    assert abs(float(basis[1, 0])) > 0.99
    assert "import pandas" not in SOURCE


def test_subspace_alignment_is_invariant_to_basis_rotation() -> None:
    first = np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.float64)
    rotation = np.asarray(
        [[0.0, -1.0], [1.0, 0.0]], dtype=np.float64
    )
    aligned = MODULE.subspace_alignment_metrics(first, first @ rotation)
    assert np.allclose(aligned["principal_cosines"], [1.0, 1.0])
    assert aligned["projection_frobenius_distance"] < 1e-7

    orthogonal = MODULE.subspace_alignment_metrics(
        np.asarray([[1.0], [0.0], [0.0]]),
        np.asarray([[0.0], [0.0], [1.0]]),
    )
    assert np.isclose(orthogonal["minimum_cosine"], 0.0)
    assert np.isclose(
        orthogonal["maximum_principal_angle_radians"], np.pi / 2
    )


def test_paired_inference_is_deterministic_and_reports_contrast_support() -> None:
    treatment = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0])
    baseline = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0])
    kwargs = {
        "bootstrap_replicates": 500,
        "bootstrap_seed": 11,
        "randomization_replicates": 1000,
        "randomization_seed": 12,
    }
    first = MODULE.paired_query_inference(treatment, baseline, **kwargs)
    second = MODULE.paired_query_inference(treatment, baseline, **kwargs)
    assert first == second
    assert first["improved_queries"] == 2
    assert first["harmed_queries"] == 1
    assert np.isclose(first["net_share"], 0.2)


def test_candidate_gap_decomposition_exposes_recovery_harm_and_overshoot() -> None:
    result = MODULE.candidate_gap_decomposition(
        method=[0.8, 0.4, 0.9],
        baseline=[0.5, 0.5, 0.5],
        ceiling=[1.0, 1.0, 0.8],
    )
    assert np.isclose(result["candidate_headroom"], 1.3 / 3)
    assert np.isclose(result["method_gain"], 0.6 / 3)
    assert np.isclose(result["gap_recovery_fraction"], 0.6 / 1.3)
    assert result["recovered_queries"] == 2
    assert result["harmed_queries"] == 1
    assert result["mean_overshoot"] > 0


def _observed(**changes: float | int) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "n_queries": 2000,
        "headroom": 0.03,
        "capacity_gain": 0.0,
        "coding_gap": 0.0,
        "objective_gain": 0.0,
        "domain_interaction": 0.0,
        "pooled_recovery": 0.0,
        "pooled_gain": 0.0,
        "improved_queries": 40,
        "harmed_queries": 10,
        "gap_recovery": 0.2,
        "worst_domain_gain": 0.0,
    }
    values.update(changes)
    return values


def test_causal_decision_covers_mechanism_and_stop_outcomes() -> None:
    thresholds = MODULE.DEFAULT_CAUSAL_THRESHOLDS
    capacity = MODULE.causal_decision(
        _observed(capacity_gain=0.006), thresholds
    )
    assert capacity["decision"] == "CAPACITY_BOTTLENECK_SUPPORTED"
    assert capacity["capacity_bottleneck_supported"]

    coding = MODULE.causal_decision(_observed(coding_gap=0.006), thresholds)
    assert coding["decision"] == "CODING_BOTTLENECK_SUPPORTED"

    domain = MODULE.causal_decision(
        _observed(
            objective_gain=0.006,
            domain_interaction=0.006,
            worst_domain_gain=-0.003,
        ),
        thresholds,
    )
    assert domain["decision"] == "DOMAIN_SHIFT_SUPPORTED"

    repaired = MODULE.causal_decision(
        _observed(objective_gain=0.006, pooled_recovery=0.6),
        thresholds,
    )
    assert repaired["decision"] == "OBJECTIVE_REPAIR_SUPPORTED"

    frozen = MODULE.causal_decision(_observed(headroom=0.009), thresholds)
    assert frozen["decision"] == "STOP_FROZEN_CANDIDATE_METHOD"

    uniform = MODULE.causal_decision(
        _observed(pooled_recovery=0.6), thresholds
    )
    assert uniform["decision"] == "STOP_LEARNING_CLAIM_KEEP_UNIFORM_RPQ"

    stopped = MODULE.causal_decision(_observed(), thresholds)
    assert stopped["decision"] == "STOP_RARS_METHOD_EXPANSION"
    assert stopped["required_improved_queries"] == 20

    scaled = MODULE.causal_decision(
        _observed(n_queries=5000, improved_queries=49), thresholds
    )
    assert scaled["required_improved_queries"] == 50
    assert not scaled["improved_query_support_passes"]
