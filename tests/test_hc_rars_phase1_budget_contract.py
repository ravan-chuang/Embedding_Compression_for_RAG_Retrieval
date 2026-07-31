import numpy as np
import pytest

from scripts.hc_rars_phase1_core import RPQCodec, validate_codec


def test_production_shape_is_exactly_16_bytes_per_document():
    codec = RPQCodec(
        basis=np.eye(384, 64, dtype=np.float32),
        codebooks=np.zeros((16, 256, 4), dtype=np.float32),
    )
    validate_codec(codec, expected_payload_bytes=16)
    assert codec.payload_bytes_per_document == 16
    codes = np.zeros((1000, 16), dtype=np.uint8)
    assert codes.nbytes == 1000 * 16


def test_wrong_block_count_fails_budget_contract():
    codec = RPQCodec(
        basis=np.eye(384, 64, dtype=np.float32),
        codebooks=np.zeros((8, 256, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="payload budget"):
        validate_codec(codec, expected_payload_bytes=16)
