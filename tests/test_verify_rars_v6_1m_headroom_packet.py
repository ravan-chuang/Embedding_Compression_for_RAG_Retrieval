from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rars_v6_headroom_core import decompose_recall_gaps, diagnostic_gate_decision
from scripts.verify_rars_v6_1m_headroom_packet import file_record, verify_packet


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _packet(root: Path) -> Path:
    root.mkdir()
    query_count = 2307
    top_shape = (query_count, 200)
    arrays: dict[str, np.ndarray] = {}
    for method, offset in (("base_pq", 0), ("ivf_exact", 1), ("full_exact", 2)):
        rows = np.tile(np.arange(200, dtype=np.int64), (query_count, 1)) + offset * 1000
        scores = np.tile(
            np.linspace(1.0, 0.0, 200, dtype=np.float32), (query_count, 1)
        )
        arrays[f"{method}_top_rows.int64.npy"] = rows
        arrays[f"{method}_top_scores.float32.npy"] = scores
    arrays["probed_ivf_lists.int64.npy"] = np.tile(
        np.arange(16, dtype=np.int64), (query_count, 1)
    )
    recall_values = {
        "base_pq": (0.60, 0.80),
        "ivf_exact": (0.70, 0.82),
        "full_exact": (0.80, 0.90),
    }
    for method, (r10, r100) in recall_values.items():
        arrays[f"{method}_recall_at_10.float64.npy"] = np.full(query_count, r10)
        arrays[f"{method}_recall_at_100.float64.npy"] = np.full(query_count, r100)
    for filename, value in arrays.items():
        np.save(root / filename, value)
    output_records = {filename: file_record(root / filename) for filename in arrays}
    decomposition = decompose_recall_gaps(
        arrays["full_exact_recall_at_100.float64.npy"],
        arrays["ivf_exact_recall_at_100.float64.npy"],
        arrays["base_pq_recall_at_100.float64.npy"],
    )
    decomposition["qrels_corpus_coverage"] = 1.0
    uncapped = {
        "triplets": 600,
        "distinct_flip_queries": 120,
        "distinct_positive_documents": 120,
        "distinct_unjudged_documents": 500,
        "distinct_flip_documents": 620,
        "effective_sample_size": 500.0,
        "max_query_weight_share": 0.01,
    }
    gate = diagnostic_gate_decision(
        pq_specific_r100_gap=decomposition["pq_specific_r100_gap"],
        uncapped_triplets=uncapped["triplets"],
        distinct_flip_queries=uncapped["distinct_flip_queries"],
        effective_sample_size=uncapped["effective_sample_size"],
        max_query_weight_share=uncapped["max_query_weight_share"],
        qrels_corpus_coverage=1.0,
    )
    mean_recall = {
        "base_pq": {"recall_at_10": 0.6, "recall_at_100": 0.8},
        "same_ivf_exact": {"recall_at_10": 0.7, "recall_at_100": 0.82},
        "full_exact": {"recall_at_10": 0.8, "recall_at_100": 0.9},
    }
    result = {
        "schema_version": 1,
        "protocol_id": "rars_v6_1m_headroom_v1",
        "status": "RARS_V6_1M_HEADROOM_COMPLETE",
        "source_commit": "26a7717b964eed979b3bf7a3149d0d24e9bce3f1",
        "evidence_role": "oracle_design",
        "training_performed": False,
        "adapter_used": False,
        "rars_used": False,
        "future_or_audit_role_opened": False,
        "qrels_mapping": {"qrels_corpus_coverage": 1.0},
        "mean_recall": mean_recall,
        "recall_gap_decomposition": decomposition,
        "flip_support": {"uncapped": uncapped, "capped": uncapped},
        "formal_decision": "GO_TO_V6_LOSS_IMPLEMENTATION",
        "signal_gate": gate,
        "outputs": output_records,
    }
    result_path = root / "headroom_result.json"
    _write_json(result_path, result)
    complete = {
        "schema_version": 1,
        "protocol_id": "rars_v6_1m_headroom_v1",
        "status": "RARS_V6_1M_HEADROOM_COMPLETE",
        "source_commit": "26a7717b964eed979b3bf7a3149d0d24e9bce3f1",
        "formal_decision": "GO_TO_V6_LOSS_IMPLEMENTATION",
        "signal_gate": gate,
        "result": file_record(result_path),
        "outputs": output_records,
        "corpus_tensor_persisted": False,
        "training_performed": False,
        "adapter_used": False,
        "rars_used": False,
        "future_or_audit_role_opened": False,
    }
    _write_json(root / "headroom_complete.json", complete)
    return root


def test_v6_packet_verifier_recomputes_gate_and_means(tmp_path: Path) -> None:
    summary = verify_packet(_packet(tmp_path / "packet"))
    assert summary["status"] == "RARS_V6_1M_HEADROOM_PACKET_VERIFIED"
    assert summary["formal_decision"] == "GO_TO_V6_LOSS_IMPLEMENTATION"
    assert summary["verified_output_count"] == 13


def test_v6_packet_verifier_rejects_array_tampering(tmp_path: Path) -> None:
    packet = _packet(tmp_path / "packet")
    path = packet / "base_pq_recall_at_100.float64.npy"
    value = np.load(path)
    value[0] = 0.0
    np.save(path, value)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_packet(packet)

