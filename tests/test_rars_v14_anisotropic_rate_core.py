from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rars_v14_anisotropic_rate_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v14_core", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dynamic_programming_allocates_exact_nonuniform_128_bits() -> None:
    sensitivity = np.ones(16, dtype=np.float64)
    sensitivity[0] = 64.0
    sensitivity[-1] = 1.0 / 64.0
    allocation, summary = MODULE.allocate_bits_dynamic_programming(
        sensitivity,
        total_bits=128,
        minimum_bits=6,
        maximum_bits=10,
        block_dimension=4,
    )
    assert allocation.shape == (16,)
    assert allocation.dtype == np.int64
    assert allocation.sum() == 128
    assert allocation.min() >= 6 and allocation.max() <= 10
    assert allocation[0] > allocation[-1]
    assert summary["nonuniform"] is True
    assert summary["proxy_reduction_fraction"] > 0


def test_variable_codes_round_trip_exactly_at_128_bits() -> None:
    bits = np.asarray([6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 9, 9])
    assert bits.sum() == 128
    rng = np.random.default_rng(7)
    codes = np.column_stack(
        [rng.integers(0, 1 << int(width), size=37) for width in bits]
    ).astype(np.uint16)
    packed = MODULE.pack_variable_codes(codes, bits)
    assert packed.shape == (37, 16)
    assert packed.dtype == np.uint8
    assert np.array_equal(MODULE.unpack_variable_codes(packed, bits), codes)


def test_variable_code_packing_rejects_out_of_range_code() -> None:
    bits = np.full(16, 8, dtype=np.int64)
    codes = np.zeros((1, 16), dtype=np.uint16)
    codes[0, 3] = 256
    try:
        MODULE.pack_variable_codes(codes, bits)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("Out-of-range product code was accepted")


def test_query_metric_is_spd_trace_normalized_and_label_free() -> None:
    queries = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]], dtype=np.float32
    )
    basis = np.eye(4, dtype=np.float32)
    rows = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    scores = np.asarray([[0.9, 0.8, 0.1], [0.7, 0.6, 0.2]], dtype=np.float32)
    transforms, summary = MODULE.fit_query_metric_transforms(
        queries,
        basis,
        rows,
        scores,
        top_b=3,
        final_k=2,
        cutoff_boost=4.0,
        margin_temperature=0.02,
        ridge_fraction=0.001,
        block_dimension=4,
    )
    assert transforms.shape == (1, 4, 4)
    metric = transforms[0] @ transforms[0].T
    assert np.all(np.linalg.eigvalsh(metric) > 0)
    assert np.isclose(np.trace(metric), 4.0, rtol=0.0, atol=1e-5)
    assert summary["labels_used"] is False


def test_variable_sidecar_scoring_preserves_unselected_candidates() -> None:
    queries = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    rows = np.asarray([[10, 20, 30]], dtype=np.int64)
    lookup = np.asarray([[0, 1, 2]], dtype=np.int64)
    base = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)
    basis = np.eye(4, dtype=np.float32)
    bits = np.asarray([8], dtype=np.int64)
    codes = np.asarray([[1], [2], [3]], dtype=np.uint16)
    packed = MODULE.pack_variable_codes(codes, bits)
    book = np.zeros((256, 4), dtype=np.float32)
    book[1, 0] = 0.1
    book[2, 0] = 0.2
    book[3, 0] = 9.0
    output = MODULE.score_variable_sidecar_candidates(
        queries,
        rows,
        lookup,
        base,
        basis,
        packed,
        bits,
        [book],
        alpha=1.0,
        top_b=2,
    )
    assert np.allclose(output[0, :2], [1.0, 1.0])
    assert output[0, 2] == base[0, 2]


def test_consensus_requires_repeatable_query_directions() -> None:
    baseline = np.zeros((3, 5), dtype=np.float64)
    challenger = np.asarray(
        [[1, 1, 0, -1, 0], [1, 0, 0, -1, 0], [0, 0, 1, 0, 0]],
        dtype=np.float64,
    )
    assert MODULE.multi_seed_consensus(challenger, baseline) == {
        "improved_in_at_least_two_seeds": 1,
        "harmed_in_at_least_two_seeds": 1,
        "improved_in_all_three_seeds": 0,
        "harmed_in_all_three_seeds": 0,
    }


def test_decision_fails_uniform_allocations_even_with_good_metrics() -> None:
    comparison = {
        "mean_difference": 0.01,
        "lower": 0.005,
        "randomization_p_value_one_sided": 0.001,
        "improved_queries": 50,
        "harmed_queries": 5,
    }
    thresholds = {
        "minimum_recall_gain_over_v13_uniform_rpq": 0.003,
        "minimum_recall_gain_over_uniform_whitened": 0.001,
        "minimum_recall_gain_over_pca16": 0.003,
        "minimum_recall_gain_over_base": 0.01,
        "bootstrap_lower_must_exceed": 0.0,
        "maximum_randomization_p_value": 0.05,
        "minimum_improved_queries": 30,
        "minimum_net_improved_queries": 15,
        "minimum_each_seed_gain": 0.0,
        "minimum_median_seed_gain": 0.002,
        "minimum_worst_fold_gain": 0.0,
        "minimum_candidate_gap_recovery_fraction": 0.2,
        "minimum_mrr_change": -0.002,
        "minimum_ndcg_change": -0.002,
        "minimum_queries_improved_in_at_least_two_seeds": 10,
        "maximum_queries_harmed_in_at_least_two_seeds": 10,
        "require_exact_total_bits": 128,
        "require_nonuniform_allocation": True,
        "go_decision": "GO",
        "stop_decision": "STOP",
    }
    result = MODULE.anisotropic_rate_decision(
        primary_vs_uniform_rpq=comparison,
        primary_vs_uniform_whitened=comparison,
        primary_vs_pca16=comparison,
        primary_vs_base=comparison,
        seed_gains=[0.01, 0.01, 0.01],
        fold_gains=[0.01] * 5,
        candidate_gap_recovery=0.5,
        uniform_rpq_mrr=0.1,
        challenger_mrr=0.11,
        uniform_rpq_ndcg=0.1,
        challenger_ndcg=0.11,
        consensus={
            "improved_in_at_least_two_seeds": 20,
            "harmed_in_at_least_two_seeds": 0,
        },
        allocations=[[8] * 16] * 6,
        payload_bytes_per_document=16,
        full_corpus_codes_materialized=True,
        thresholds=thresholds,
    )
    assert result["decision"] == "STOP"
    assert result["failed_gates"] == ["nonuniform_allocations"]
