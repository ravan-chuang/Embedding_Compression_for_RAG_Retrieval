from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_rars_v6_1m_headroom as MODULE


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_positive_qrels_parser_and_padding_preserve_all_positives(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qrels.json"
    _write_json(
        path,
        {
            "q1": {"10": 1, "11": 0},
            "q2": [{"doc_id": 20, "relevance": 2}, 21],
        },
    )
    qrels = MODULE.load_positive_qrels(path)
    values, valid = MODULE.pad_qrels_for_queries(["q2", "q1"], qrels)
    assert values.tolist() == [[20, 21], [10, -1]]
    assert valid.tolist() == [[True, True], [True, False]]


def test_flip_union_excludes_routing_misses_from_pq_supervision() -> None:
    base = np.asarray(
        [np.arange(100, 220), np.arange(300, 420)], dtype=np.int64
    )
    ivf_exact = np.asarray(
        [np.arange(150, 270), np.arange(350, 470)], dtype=np.int64
    )
    positives = np.asarray([[105, 999], [360, -1]], dtype=np.int64)
    valid = np.asarray([[1, 1], [1, 0]], dtype=bool)
    probed = [np.arange(0, 500), np.arange(300, 800)]

    rows, eligible, summary = MODULE.build_flip_candidate_union(
        base, ivf_exact, positives, valid, probed
    )

    assert eligible.tolist() == [[True, False], [True, False]]
    assert 999 not in rows[0]
    assert 105 in rows[0]
    assert 360 in rows[1]
    assert summary["total_positive_qrels"] == 3
    assert summary["routed_positive_qrels"] == 2
    assert summary["routing_missed_positive_qrels"] == 1


def test_gate_adapter_uses_uncapped_support_and_qrels_coverage() -> None:
    decomposition = {
        "pq_specific_r100_gap": 0.01,
        "qrels_corpus_coverage": 1.0,
    }
    support = {
        "uncapped": {
            "triplets": 500,
            "distinct_flip_queries": 100,
            "effective_sample_size": 250.0,
            "max_query_weight_share": 0.02,
        }
    }
    decision = MODULE._gate_call(decomposition, support)
    assert decision["decision"] == "GO_TO_V6_LOSS_IMPLEMENTATION"
    assert decision["training_authorized"] is False


def test_signal_gate_protocol_cannot_drift_from_executable() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1]
        / "protocols/rars_v6_1m_headroom_v1.json").read_text(encoding="utf-8")
    )
    MODULE.validate_signal_gate_contract(protocol)
    protocol["signal_gate"]["minimum_uncapped_flip_triplets"] = 499
    with pytest.raises(ValueError, match="disagree"):
        MODULE.validate_signal_gate_contract(protocol)


def test_output_contract_uses_stable_non_top200_filenames() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1]
        / "protocols/rars_v6_1m_headroom_v1.json").read_text(encoding="utf-8")
    )
    required = set(protocol["required_outputs"])
    assert "headroom_result.json" in required
    assert "ivf_exact_top_rows.int64.npy" in required
    assert not any("top200" in name for name in required)


def test_faiss_validation_downcasts_generic_ivf_wrapper_before_reading_pq() -> None:
    generic = SimpleNamespace()
    concrete = SimpleNamespace(
        d=384,
        nlist=512,
        nprobe=1,
        pq=SimpleNamespace(M=32, nbits=8),
        ntotal=1_000_000,
        metric_type=7,
        is_trained=True,
    )

    class FakeFaiss:
        METRIC_INNER_PRODUCT = 7

        @staticmethod
        def extract_index_ivf(index: object) -> object:
            assert index == "frozen-index"
            return generic

        @staticmethod
        def downcast_index(index: object) -> object:
            assert index is generic
            return concrete

    ivf, observed = MODULE.validate_faiss_index("frozen-index", FakeFaiss)
    assert ivf is concrete
    assert observed["subquantizers"] == 32
    assert observed["nbits"] == 8
    assert observed["runtime_nprobe"] == 16


def test_torch_topk_merge_gathers_document_rows_along_candidate_axis() -> None:
    class FakeTorch:
        @staticmethod
        def topk(
            value: np.ndarray,
            *,
            k: int,
            dim: int,
            largest: bool,
            sorted: bool,
        ) -> tuple[np.ndarray, np.ndarray]:
            assert dim == 1 and largest and sorted
            positions = np.argsort(-value, axis=1)[:, :k]
            return np.take_along_axis(value, positions, axis=1), positions

        @staticmethod
        def gather(
            value: np.ndarray, dim: int, positions: np.ndarray
        ) -> np.ndarray:
            assert dim == 1
            return np.take_along_axis(value, positions, axis=1)

        @staticmethod
        def cat(values: tuple[np.ndarray, ...], dim: int) -> np.ndarray:
            return np.concatenate(values, axis=dim)

    scores, rows = MODULE._merge_torch_topk(
        None,
        None,
        np.asarray([[0.2, 0.9, 0.5]], dtype=np.float32),
        np.asarray([[10, 11, 12]], dtype=np.int64),
        k=2,
        torch=FakeTorch,
    )
    assert np.allclose(scores, [[0.9, 0.5]])
    assert rows.tolist() == [[11, 12]]

    merged_scores, merged_rows = MODULE._merge_torch_topk(
        scores,
        rows,
        np.asarray([[0.8, 0.1]], dtype=np.float32),
        np.asarray([[20, 21]], dtype=np.int64),
        k=2,
        torch=FakeTorch,
    )
    assert np.allclose(merged_scores, [[0.9, 0.8]])
    assert merged_rows.tolist() == [[11, 20]]
