from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_rars_v5_pq_aware_100k_packet.py"
PACKET = ROOT / "results/rars_v5_pq_aware_100k_pilot"
SPEC = importlib.util.spec_from_file_location("verify_rars_v5_packet", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v5_closure_packet_verifies() -> None:
    summary = MODULE.verify_packet(PACKET)
    assert summary["status"] == "RARS_V5_PQ_AWARE_100K_CLOSURE_VERIFIED"
    assert summary["formal_decision"] == "STOP_PQ_AWARE_100K_PILOT"
    assert summary["selected_epoch"] == 3
    assert summary["query_count"] == 728
    assert summary["improved_queries"] == 2
    assert summary["harmed_queries"] == 0
    assert len(summary["failed_gates"]) == 5


def test_v5_per_query_packet_is_exactly_two_sparse_improvements() -> None:
    per_query = json.loads(
        (PACKET / "per_query_recall_at_100.json").read_text(encoding="utf-8")
    )
    delta = per_query["delta"]
    assert len(delta) == 728
    assert delta.count(1.0) == 2
    assert delta.count(0.0) == 726
    assert all(value >= 0.0 for value in delta)


def test_v5_packet_keeps_external_binary_hashes_without_claiming_payloads() -> None:
    complete = json.loads(
        (PACKET / "training_complete.json").read_text(encoding="utf-8")
    )
    audit = json.loads((PACKET / "artifact_audit.json").read_text(encoding="utf-8"))
    assert set(audit["artifacts"]) == set(complete["outputs"])
    assert audit["registered_file_count"] == 8
    assert audit["all_registered_bytes_and_sha256_verified"] is True
    assert not any((PACKET / filename).exists() for filename in complete["outputs"])
