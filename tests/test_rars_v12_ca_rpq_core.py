from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from rars_v8_cutoff_sidecar_core import CutoffPairBatch  # noqa: E402

PATH = SCRIPTS / "rars_v12_ca_rpq_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v12_ca_rpq_core", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pairs() -> CutoffPairBatch:
    return CutoffPairBatch(
        query=np.asarray([0, 1], dtype=np.int64),
        positive_position=np.asarray([1, 2], dtype=np.int64),
        challenger_position=np.asarray([0, 0], dtype=np.int64),
        positive_residual_row=np.asarray([1, 2], dtype=np.int64),
        challenger_residual_row=np.asarray([0, 3], dtype=np.int64),
        base_margin=np.asarray([-0.1, 0.1], dtype=np.float32),
        teacher_margin=np.asarray([0.2, 0.3], dtype=np.float32),
        target_residual_margin=np.asarray([0.3, 0.2], dtype=np.float32),
        raw_weight=np.asarray([1.0, 1.0], dtype=np.float32),
        balanced_weight=np.asarray([0.5, 0.5], dtype=np.float32),
        kind=np.asarray([0, 1], dtype=np.uint8),
    )


def test_fresh_query_priority_and_folds_are_order_independent() -> None:
    qids = ["10", "20", "30", "40"]
    first = dict(zip(qids, MODULE.deterministic_fold_ids(qids)))
    reversed_qids = list(reversed(qids))
    second = dict(zip(reversed_qids, MODULE.deterministic_fold_ids(reversed_qids)))
    assert first == second
    assert MODULE.deterministic_query_priority("10") == MODULE.deterministic_query_priority("10")


def test_assign_product_codes_uses_each_four_dimensional_block() -> None:
    values = np.asarray([[0.0, 0.0, 9.0, 9.0], [5.0, 5.0, 0.0, 0.0]], dtype=np.float32)
    books = np.zeros((2, 256, 2), dtype=np.float32)
    books[0, 1] = 5.0
    books[1, 2] = 9.0
    codes = MODULE.assign_product_codes(values, books, batch_size=1)
    assert codes.dtype == np.uint8
    assert np.array_equal(codes, [[0, 2], [1, 0]])


def test_cutoff_block_weights_are_positive_bounded_and_pair_local() -> None:
    queries = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    basis = np.eye(4, dtype=np.float32)
    weights, summary = MODULE.build_cutoff_block_weights(
        queries,
        basis,
        _pairs(),
        residual_count=5,
        subquantizers=2,
        cutoff_boost=4.0,
        protection_multiplier=2.0,
        maximum_weight=10.0,
    )
    assert weights.shape == (5, 2)
    assert np.all(weights >= 1.0)
    assert np.all(weights <= 10.0)
    assert np.array_equal(weights[4], [1.0, 1.0])
    assert weights[0, 0] > 1.0
    assert weights[3, 1] > 1.0
    assert summary["active_residual_rows"] == 4


def test_closed_form_update_reduces_objective_and_obeys_drift_limit() -> None:
    values = np.asarray(
        [[0.0, 0.0], [0.2, 0.1], [4.0, 4.0], [4.2, 3.9]], dtype=np.float32
    )
    books = np.zeros((1, 256, 2), dtype=np.float32)
    books[0, 1] = [4.0, 4.0]
    codes = np.asarray([[0], [0], [1], [1]], dtype=np.uint8)
    weights = np.asarray([[1.0], [10.0], [1.0], [10.0]], dtype=np.float32)
    updated, summary = MODULE.fit_anchored_cutoff_codebooks(
        values,
        codes,
        books,
        weights,
        anchor_pseudocount=2.0,
        maximum_drift_fraction=0.25,
    )
    assert updated.shape == books.shape
    assert summary["fixed_assignment_objective_after"] <= summary[
        "fixed_assignment_objective_before"
    ]
    assert summary["maximum_centroid_drift_fraction"] <= 0.25 + 1e-7
    assert summary["payload_bytes_per_document"] == 1
    assert not np.array_equal(updated, books)


def _comparison(gain: float) -> dict[str, float | int]:
    return {
        "mean_difference": gain,
        "lower": 0.001,
        "randomization_p_value_one_sided": 0.01,
        "improved_queries": 40,
        "harmed_queries": 10,
    }


def _thresholds() -> dict[str, float | int | str]:
    return {
        "minimum_recall_gain_over_unsupervised": 0.003,
        "minimum_recall_gain_over_base": 0.01,
        "bootstrap_lower_must_exceed": 0.0,
        "maximum_randomization_p_value": 0.05,
        "minimum_improved_queries": 20,
        "minimum_net_improved_queries": 10,
        "minimum_each_seed_gain": 0.0,
        "minimum_median_seed_gain": 0.002,
        "minimum_worst_fold_gain": 0.0,
        "minimum_candidate_gap_recovery_fraction": 0.2,
        "minimum_mrr_change": -0.002,
        "minimum_ndcg_change": -0.002,
        "maximum_centroid_drift_fraction": 0.25,
        "go_decision": "GO_TO_FRESH_CA_RPQ_CONFIRMATION_PROTOCOL",
        "stop_decision": "STOP_CA_RPQ_NO_STABLE_ADVANTAGE",
    }


def test_gate_requires_seed_fold_payload_and_harm_stability() -> None:
    common = {
        "primary_vs_unsupervised": _comparison(0.004),
        "primary_vs_base": _comparison(0.02),
        "seed_gains": [0.003, 0.004, 0.002],
        "fold_gains": [0.001] * 5,
        "candidate_gap_recovery": 0.3,
        "unsupervised_mrr": 0.4,
        "ca_mrr": 0.401,
        "unsupervised_ndcg": 0.5,
        "ca_ndcg": 0.501,
        "payload_bytes_per_document": 16,
        "full_corpus_codes_materialized": True,
        "all_objectives_nonincreasing": True,
        "maximum_centroid_drift_fraction": 0.2,
        "thresholds": _thresholds(),
    }
    passed = MODULE.ca_rpq_decision(**common)
    assert passed["decision"] == "GO_TO_FRESH_CA_RPQ_CONFIRMATION_PROTOCOL"
    assert not passed["failed_gates"]

    unstable = MODULE.ca_rpq_decision(**{**common, "seed_gains": [0.004, -0.001, 0.003]})
    assert unstable["decision"] == "STOP_CA_RPQ_NO_STABLE_ADVANTAGE"
    assert "all_seed_gains_nonnegative" in unstable["failed_gates"]

    no_payload = MODULE.ca_rpq_decision(**{**common, "payload_bytes_per_document": 15})
    assert "payload_exactly_sixteen_bytes" in no_payload["failed_gates"]
