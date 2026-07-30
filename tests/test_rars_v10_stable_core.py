from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/rars_v10_stable_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v10_stable_core", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _batch() -> tuple[object, np.ndarray]:
    queries = np.asarray(
        [[1.0, 0.2, -0.1], [0.1, 0.9, 0.3], [0.5, -0.2, 0.8]],
        dtype=np.float64,
    )
    residual_differences = np.asarray(
        [[0.4, -0.2, 0.1], [-0.1, 0.5, 0.2], [0.2, 0.1, 0.6]],
        dtype=np.float64,
    )
    anchor = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    exact = np.einsum("pd,pd->p", queries, residual_differences)
    pca_prediction = np.einsum(
        "pr,pr->p", queries @ anchor, residual_differences @ anchor
    )
    batch = MODULE.ObjectiveBatch(
        queries=queries,
        residual_differences=residual_differences,
        exact_residual_margin=exact,
        base_margin=np.asarray([-0.1, 0.02, -0.03]),
        balanced_weight=np.asarray([1 / 3, 1 / 3, 1 / 3]),
        harm_weight=np.ones(3),
        query_index=np.asarray([0, 1, 2]),
        pca_prediction=pca_prediction,
    )
    return batch, anchor


def _objective() -> dict[str, float]:
    return {
        "alpha": 0.75,
        "distillation_weight": 0.25,
        "cutoff_weight": 1.0,
        "harm_weight": 2.0,
        "anchor_weight": 0.5,
        "huber_delta": 0.02,
        "cutoff_temperature": 0.02,
        "margin_floor": 0.0,
        "harm_scale": 0.02,
        "cvar_fraction": 0.5,
    }


def test_query_equal_weights_gives_each_query_unit_mass() -> None:
    query = np.asarray([0, 0, 1, 2, 2, 2])
    weights = MODULE.query_equal_weights(query, [1, 3, 2, 1, 1, 2])
    for value in np.unique(query):
        assert np.isclose(weights[query == value].sum(), 1.0)


def test_fp32_coefficient_ceiling_uses_only_base_top_b() -> None:
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    rows = np.asarray([[10, 11, 12]], dtype=np.int64)
    lookup = np.asarray([[0, 1, 2]], dtype=np.int64)
    base = np.asarray([[0.8, 0.7, 0.6]], dtype=np.float32)
    residuals = np.asarray([[0.2, 0.0], [0.4, 0.0], [9.0, 0.0]], dtype=np.float32)
    basis = np.eye(2, dtype=np.float32)
    scores = MODULE.score_float_sidecar_candidates(
        queries,
        rows,
        lookup,
        base,
        residuals,
        basis,
        alpha=0.5,
        top_b=2,
    )
    assert np.allclose(scores, [[0.9, 0.9, 0.6]])


def test_gradient_audit_matches_retracted_finite_difference() -> None:
    batch, anchor = _batch()
    audit = MODULE.gradient_direction_audit(
        anchor,
        batch,
        anchor,
        _objective(),
        epsilon=1e-5,
        maximum_relative_error=5e-4,
    )
    assert audit["status"] == "PASS", audit
    assert audit["relative_error"] <= 5e-4


def test_optimizer_is_monotone_orthonormal_and_pca_bounded() -> None:
    batch, anchor = _batch()
    basis, history = MODULE.fit_stable_basis(
        batch,
        anchor,
        _objective(),
        maximum_steps=30,
        initial_step_size=0.05,
        backtracking_factor=0.5,
        armijo_constant=1e-4,
        maximum_backtracks=20,
        maximum_principal_angle=20.0,
        gradient_tolerance=1e-10,
    )
    assert np.allclose(basis.T @ basis, np.eye(2), atol=2e-6)
    assert MODULE.maximum_principal_angle_degrees(anchor, basis) <= 20.0
    accepted = [row for row in history if row["accepted"]]
    losses = [row["post_retraction_loss"] for row in accepted]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(losses, losses[1:]))
    assert losses[-1] <= losses[0]


def test_paired_inference_uses_explicit_separate_seeds() -> None:
    treatment = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0])
    baseline = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0])
    first = MODULE.paired_inference(
        treatment,
        baseline,
        bootstrap_replicates=500,
        bootstrap_seed=11,
        randomization_replicates=1000,
        randomization_seed=12,
    )
    second = MODULE.paired_inference(
        treatment,
        baseline,
        bootstrap_replicates=500,
        bootstrap_seed=11,
        randomization_replicates=1000,
        randomization_seed=12,
    )
    assert first == second
    assert first["bootstrap_seed"] == 11
    assert first["randomization_seed"] == 12
    assert first["improved_queries"] == 2
    assert first["harmed_queries"] == 1


def test_decision_requires_stability_and_pca_superiority() -> None:
    comparison = {
        "mean_difference": 0.01,
        "lower": 0.002,
        "randomization_p_value_one_sided": 0.01,
        "improved_queries": 40,
        "harmed_queries": 10,
    }
    thresholds = {
        "minimum_recall_at_10_gain_over_base": 0.01,
        "minimum_recall_at_10_gain_over_pca": 0.005,
        "bootstrap_lower_must_exceed": 0.0,
        "maximum_randomization_p_value": 0.025,
        "minimum_improved_queries_over_pca": 30,
        "minimum_net_improved_queries_over_pca": 15,
        "minimum_worst_fold_gain_over_pca": 0.0,
        "minimum_candidate_gap_recovery_fraction": 0.15,
        "minimum_mrr_change_vs_pca": -0.002,
        "minimum_ndcg_change_vs_pca": -0.002,
        "go_decision": "GO_TO_FRESH_EXTERNAL_V10_PROTOCOL",
        "stop_decision": "STOP_V10_NO_STABLE_PCA_ADVANTAGE",
    }
    passed = MODULE.stable_development_decision(
        v10_vs_base=comparison,
        v10_vs_pca=comparison,
        fold_gains_over_pca=[0.001, 0.002, 0.003, 0.001, 0.004],
        gap_recovery=0.2,
        pca_mrr=0.5,
        v10_mrr=0.51,
        pca_ndcg=0.55,
        v10_ndcg=0.56,
        optimizer_audits_pass=True,
        accepted_losses_monotone=True,
        thresholds=thresholds,
    )
    assert passed["decision"] == "GO_TO_FRESH_EXTERNAL_V10_PROTOCOL"
    failed = MODULE.stable_development_decision(
        v10_vs_base=comparison,
        v10_vs_pca=comparison,
        fold_gains_over_pca=[0.001, -0.001, 0.003, 0.001, 0.004],
        gap_recovery=0.2,
        pca_mrr=0.5,
        v10_mrr=0.51,
        pca_ndcg=0.55,
        v10_ndcg=0.56,
        optimizer_audits_pass=True,
        accepted_losses_monotone=True,
        thresholds=thresholds,
    )
    assert failed["decision"] == "STOP_V10_NO_STABLE_PCA_ADVANTAGE"
    assert "worst_fold_nonnegative" in failed["failed_gates"]


def test_scalar_headroom_gate_does_not_train_a_codebook() -> None:
    comparison = {
        "mean_difference": 0.004,
        "lower": 0.001,
        "randomization_p_value_one_sided": 0.02,
        "improved_queries": 25,
        "harmed_queries": 10,
    }
    thresholds = {
        "minimum_recall_at_10_gain": 0.003,
        "bootstrap_lower_must_exceed": 0.0,
        "maximum_randomization_p_value": 0.05,
        "minimum_improved_queries": 20,
        "minimum_net_improved_queries": 10,
        "go_decision": "GO_TO_SEPARATE_AVQ_CODEBOOK_PROTOCOL",
        "stop_decision": "STOP_AVQ_CODEBOOK_NO_SCALAR_HEADROOM",
    }
    result = MODULE.scalar_quantization_headroom_decision(
        comparison, thresholds
    )
    assert result["decision"] == "GO_TO_SEPARATE_AVQ_CODEBOOK_PROTOCOL"
    assert result["diagnostic_only"] is True
    assert result["codebook_training_performed"] is False
