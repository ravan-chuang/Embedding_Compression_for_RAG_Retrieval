import numpy as np

from scripts.hc_rars_phase1_core import (
    RPQCodec,
    build_query_lut,
    decode_codes,
    encode_projected,
    fit_rpq_codec,
    score_codes_with_lut,
    score_rpq_candidates,
)


def test_lut_scoring_matches_decoded_dot_product():
    rng = np.random.default_rng(7)
    codebooks = rng.normal(size=(4, 8, 2)).astype(np.float32)
    coefficients = rng.normal(size=(25, 8)).astype(np.float32)
    query = rng.normal(size=8).astype(np.float32)
    codes = encode_projected(coefficients, codebooks)
    decoded = decode_codes(codes, codebooks)
    lut_scores = score_codes_with_lut(codes, build_query_lut(query, codebooks))
    np.testing.assert_allclose(lut_scores, decoded @ query, atol=1e-5, rtol=1e-5)


def test_fit_codec_is_deterministic():
    rng = np.random.default_rng(11)
    residuals = rng.normal(size=(96, 12)).astype(np.float32)
    first = fit_rpq_codec(
        residuals, rank=8, block_count=4, centroid_count=8, seed=17, max_iterations=8
    )
    second = fit_rpq_codec(
        residuals, rank=8, block_count=4, centroid_count=8, seed=17, max_iterations=8
    )
    np.testing.assert_array_equal(first.basis, second.basis)
    np.testing.assert_array_equal(first.codebooks, second.codebooks)


def test_candidate_correction_changes_only_top_b():
    basis = np.eye(4, dtype=np.float32)
    codebooks = np.array([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ], dtype=np.float32)
    codec = RPQCodec(basis=basis, codebooks=codebooks)
    codes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    queries = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rows = np.array([[0, 1, 2]], dtype=np.int64)
    base = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
    corrected = score_rpq_candidates(
        queries, rows, base, codes, codec, alpha=1.0, top_b=2
    )
    assert corrected[0, 2] == base[0, 2]
    assert corrected[0, 0] > base[0, 0]
    assert corrected[0, 1] > base[0, 1]
