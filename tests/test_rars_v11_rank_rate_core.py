from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/rars_v11_rank_rate_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v11_rank_rate_core", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SOURCE = PATH.read_text(encoding="utf-8")


def test_signed_int4_round_trip_uses_exactly_half_a_byte_per_value() -> None:
    values = np.asarray(
        [[-7, -1, 0, 1, 7, 3], [6, -6, 2, -2, 4, -4]], dtype=np.int8
    )
    packed = MODULE.pack_signed_int4(values)
    assert packed.dtype == np.uint8
    assert packed.shape == (2, 3)
    assert np.array_equal(MODULE.unpack_signed_int4(packed), values)


def test_rank32_int4_encoder_has_16_byte_payload() -> None:
    residuals = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    basis = np.eye(32, dtype=np.float32)
    scales = MODULE.fit_int4_scales(residuals, basis)
    packed, diagnostic = MODULE.encode_residuals_int4(
        residuals, basis, scales
    )
    assert packed.shape == (2, 16)
    assert diagnostic["payload_bytes_per_document"] == 16
    assert diagnostic["saturation_fraction"] == 0.0


def test_product_scorer_uses_sixteen_byte_codes_and_only_base_top_b() -> None:
    queries = np.asarray([[1.0, 2.0]], dtype=np.float32)
    rows = np.asarray([[10, 11, 12]], dtype=np.int64)
    lookup = np.asarray([[0, 1, 2]], dtype=np.int64)
    base = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)
    basis = np.eye(2, dtype=np.float32)
    codes = np.zeros((3, 2), dtype=np.uint8)
    codes[0] = [1, 2]
    codes[1] = [3, 4]
    codes[2] = [5, 6]
    codebooks = np.zeros((2, 256, 1), dtype=np.float32)
    codebooks[0, :, 0] = np.arange(256, dtype=np.float32) / 10
    codebooks[1, :, 0] = np.arange(256, dtype=np.float32) / 20
    scores = MODULE.score_product_sidecar_candidates(
        queries,
        rows,
        lookup,
        base,
        basis,
        codes,
        codebooks,
        alpha=0.5,
        top_b=2,
    )
    assert np.allclose(scores, [[1.05, 1.15, 0.7]])


def _comparison(gain: float = 0.01) -> dict[str, float | int]:
    return {
        "mean_difference": gain,
        "lower": 0.001,
        "randomization_p_value_one_sided": 0.01,
        "improved_queries": 40,
        "harmed_queries": 10,
    }


def _thresholds() -> dict[str, float | int | str]:
    return {
        "minimum_rank64_fp32_gain_over_pca": 0.005,
        "minimum_rank64_rpq_gain_over_pca": 0.003,
        "minimum_rank64_rpq_gain_over_base": 0.01,
        "minimum_rank64_headroom_retention_fraction": 0.5,
        "bootstrap_lower_must_exceed": 0.0,
        "maximum_randomization_p_value": 0.05,
        "minimum_improved_queries": 20,
        "minimum_net_improved_queries": 10,
        "minimum_worst_fold_gain_over_pca": 0.0,
        "minimum_candidate_gap_recovery_fraction": 0.2,
        "minimum_mrr_change_vs_pca": -0.002,
        "minimum_ndcg_change_vs_pca": -0.002,
        "go_decision": "GO_TO_SEPARATE_CA_RPQ_CUTOFF_PROTOCOL",
        "stop_no_rank_headroom_decision": "STOP_LINEAR_RANK_EXPANSION_NO_HEADROOM",
        "stop_rpq_encoding_decision": "STOP_RPQ_16B_CANNOT_RETAIN_HEADROOM",
    }


def test_rank_rate_gate_requires_capacity_before_encoding() -> None:
    common = {
        "rank64_rpq_vs_base": _comparison(0.02),
        "fold_gains_over_pca": [0.002] * 5,
        "gap_recovery": 0.25,
        "pca_mrr": 0.4,
        "rpq_mrr": 0.401,
        "pca_ndcg": 0.5,
        "rpq_ndcg": 0.501,
        "thresholds": _thresholds(),
    }
    passed = MODULE.rank_rate_decision(
        rank64_fp32_vs_pca=_comparison(0.01),
        rank64_rpq_vs_pca=_comparison(0.006),
        **common,
    )
    assert passed["decision"] == "GO_TO_SEPARATE_CA_RPQ_CUTOFF_PROTOCOL"
    assert np.isclose(passed["rank64_headroom_retention_fraction"], 0.6)

    no_capacity = MODULE.rank_rate_decision(
        rank64_fp32_vs_pca=_comparison(0.004),
        rank64_rpq_vs_pca=_comparison(0.006),
        **common,
    )
    assert no_capacity["decision"] == "STOP_LINEAR_RANK_EXPANSION_NO_HEADROOM"

    poor_encoding = MODULE.rank_rate_decision(
        rank64_fp32_vs_pca=_comparison(0.01),
        rank64_rpq_vs_pca=_comparison(0.002),
        **common,
    )
    assert poor_encoding["decision"] == "STOP_RPQ_16B_CANNOT_RETAIN_HEADROOM"


def test_paired_inference_is_deterministic_and_counts_support() -> None:
    treatment = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0])
    baseline = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0])
    kwargs = {
        "bootstrap_replicates": 500,
        "bootstrap_seed": 11,
        "randomization_replicates": 1000,
        "randomization_seed": 12,
    }
    first = MODULE.paired_inference(treatment, baseline, **kwargs)
    second = MODULE.paired_inference(treatment, baseline, **kwargs)
    assert first == second
    assert first["improved_queries"] == 2
    assert first["harmed_queries"] == 1


def test_faiss_training_validation_does_not_depend_on_is_trained_binding() -> None:
    assert "quantizer.is_trained" not in SOURCE
    assert "expected_centroids" in SOURCE
    assert "product-quantizer round trip is invalid" in SOURCE
