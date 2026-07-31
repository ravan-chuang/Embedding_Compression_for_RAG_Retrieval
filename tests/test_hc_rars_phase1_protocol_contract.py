import json
from pathlib import Path


def test_phase1_protocol_is_frozen_and_budget_matched():
    path = Path("protocols/hc_rars_phase1_rpq64_16b_v1.json")
    protocol = json.loads(path.read_text())
    assert protocol["status"] == "FROZEN_BEFORE_FIRST_PHASE1_RUN"
    method = protocol["method"]
    assert method["rank"] == 64
    assert method["block_count"] == 16
    assert method["block_dimension"] == 4
    assert method["centroids_per_block"] == 256
    assert method["code_dtype"] == "uint8"
    assert method["payload_bytes_per_document"] == 16
    assert protocol["test_policy"]["forbid_fit_on_test"] is True
    assert protocol["test_policy"]["forbid_alpha_sweep_on_test"] is True
