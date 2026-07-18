from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_rars_v2_2_replication_packet.py"
SPEC = importlib.util.spec_from_file_location("verify_rars_v2_2_packet", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_committed_rars_v2_2_replication_packet_verifies() -> None:
    report = MODULE.verify_packet(
        ROOT / "results" / "rars_v2_2_fp32_replication",
        ROOT / "notebooks" / "MSMARCO_RARS_v2_2_FP32_Replication.ipynb",
    )

    assert report["status"] == "REPLICATION_COMPLETE"
    assert report["decision"] == "UNSTABLE_NO_QAT"
    assert report["registered_aggregate_outputs"] == 19
    assert report["runner_artifacts"] == 12
    assert report["verified_seeds"] == [42, 43, 44]
    assert report["query_count"] == 1019
