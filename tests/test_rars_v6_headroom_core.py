from __future__ import annotations

import numpy as np
import pytest

from scripts import rars_v6_headroom_core as MODULE


def test_protocol_id_is_frozen() -> None:
    assert MODULE.PROTOCOL_ID == "rars_v6_1m_headroom_v1"


def test_qrels_mapping_preserves_missing_positive_in_coverage() -> None:
    mapping = MODULE.map_qrels_doc_ids_to_corpus_rows(
        np.asarray([30, 10, 20], dtype=np.int64),
        np.asarray([[10, 99], [30, -1]], dtype=np.int64),
        np.asarray([[1, 1], [1, 0]], dtype=bool),
    )
    assert mapping.rows.tolist() == [[1, -1], [0, -1]]
    assert mapping.in_corpus.tolist() == [[True, False], [True, False]]
    assert mapping.coverage["total_positive_qrels"] == 3
    assert mapping.coverage["total_positive_qrels_missing_from_corpus"] == 1
    assert mapping.coverage["qrels_corpus_coverage"] == pytest.approx(2 / 3)
    assert mapping.coverage["queries_with_incomplete_corpus_coverage"] == [0]


def test_qrels_mapping_rejects_ambiguous_ids() -> None:
    with pytest.raises(ValueError, match="corpus_doc_ids must be unique"):
        MODULE.map_qrels_doc_ids_to_corpus_rows(
            np.asarray([1, 1]), np.asarray([[1]]), np.asarray([[1]], dtype=bool)
        )


def test_qrels_mapping_accepts_ordered_query_dictionary_form() -> None:
    mapping = MODULE.map_qrels_doc_ids_to_corpus_rows(
        ["q2", "q1"],
        {"q1": {20}, "q2": {30, 10}},
        np.asarray([10, 20, 30], dtype=np.int64),
    )
    assert mapping.positive_rows.tolist() == [[0, 2], [1, -1]]
    assert mapping.positive_valid.tolist() == [[True, True], [True, False]]
    assert mapping.query_count == 2
    assert mapping.positive_count == 3
    assert mapping.mapped_positive_count == 3
    assert mapping.corpus_coverage == 1.0
    with pytest.raises(ValueError, match="qrels must be unique"):
        MODULE.map_qrels_doc_ids_to_corpus_rows(
            np.asarray([1, 2]),
            np.asarray([[1, 1]]),
            np.asarray([[1, 1]], dtype=bool),
        )


def test_known_positive_recall_counts_missing_corpus_qrel_as_miss() -> None:
    recall = MODULE.known_positive_recall_at_k(
        np.asarray([[4, 2, -1], [7, 3, 9]], dtype=np.int64),
        np.asarray([[2, -1], [5, -1]], dtype=np.int64),
        np.asarray([[1, 1], [1, 0]], dtype=bool),
        k=2,
    )
    assert recall.tolist() == [0.5, 0.0]


def test_recall_gap_decomposition_separates_routing_from_pq() -> None:
    result = MODULE.decompose_recall_gaps(
        np.asarray([1.0, 0.8]),
        np.asarray([0.9, 0.7]),
        np.asarray([0.8, 0.6]),
    )
    assert result["full_exact_recall_at_100"] == pytest.approx(0.9)
    assert result["ivf_routing_r100_gap"] == pytest.approx(0.1)
    assert result["pq_specific_r100_gap"] == pytest.approx(0.1)
    assert result["total_r100_gap"] == pytest.approx(0.2)
    assert result["pq_fraction_of_total_r100_gap"] == pytest.approx(0.5)


def _flip_fixture() -> tuple[np.ndarray, ...]:
    # PQ ranks rows 11, 12, 13 around k=2.  Row 10 is a known positive whose
    # exact score beats all three but whose PQ score loses to rows 11 and 12.
    rows = np.asarray([[10, 11, 12, 13, 14]], dtype=np.int64)
    exact = np.asarray([[1.0, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    pq = np.asarray([[0.75, 0.95, 0.85, 0.65, 0.4]], dtype=np.float32)
    positives = np.asarray([[10]], dtype=np.int64)
    valid = np.asarray([[1]], dtype=bool)
    return rows, exact, pq, positives, valid


def test_flip_miner_retains_uncapped_then_caps_unjudged_pairs() -> None:
    result = MODULE.mine_pq_induced_flip_triplets(
        *_flip_fixture(),
        pool_k=2,
        negative_window=2,
        max_unjudged_per_positive=1,
    )
    assert result.uncapped.unjudged_row.tolist() == [11, 12]
    assert np.all(result.uncapped.exact_margin > 0)
    assert np.all(result.uncapped.pq_margin <= 0)
    assert len(result.capped) == 1
    assert result.capped.unjudged_row.tolist() == [11]
    assert result.support["uncapped"]["triplets"] == 2
    assert result.support["uncapped"]["distinct_flip_queries"] == 1
    assert "unjudged_candidate" in MODULE.FlipTripletBatch.__dataclass_fields__


def test_flip_miner_is_deterministic_under_candidate_column_permutation() -> None:
    fixture = _flip_fixture()
    left = MODULE.mine_pq_induced_flip_triplets(
        *fixture, pool_k=2, negative_window=2, max_unjudged_per_positive=2
    )
    permutation = np.asarray([4, 2, 0, 3, 1])
    right = MODULE.mine_pq_induced_flip_triplets(
        fixture[0][:, permutation],
        fixture[1][:, permutation],
        fixture[2][:, permutation],
        fixture[3],
        fixture[4],
        pool_k=2,
        negative_window=2,
        max_unjudged_per_positive=2,
    )
    assert left.uncapped.positive_row.tolist() == right.uncapped.positive_row.tolist()
    assert left.uncapped.unjudged_row.tolist() == right.uncapped.unjudged_row.tolist()
    assert np.allclose(left.uncapped.weight, right.uncapped.weight)


def test_flip_support_reports_weight_ess_and_query_concentration() -> None:
    batch = MODULE.FlipTripletBatch(
        query=np.asarray([0, 0, 1]),
        positive_candidate=np.asarray([0, 0, 0]),
        unjudged_candidate=np.asarray([1, 2, 1]),
        positive_row=np.asarray([10, 10, 20]),
        unjudged_row=np.asarray([11, 12, 21]),
        exact_margin=np.asarray([0.1, 0.2, 0.1]),
        pq_margin=np.asarray([-0.1, -0.2, -0.1]),
        weight=np.asarray([1.0, 1.0, 2.0]),
    )
    summary = MODULE.summarize_flip_triplets(batch)
    assert summary["effective_sample_size"] == pytest.approx(16 / 6)
    assert summary["max_query_weight_share"] == pytest.approx(0.5)
    assert summary["distinct_flip_queries"] == 2
    assert summary["distinct_flip_documents"] == 5


def test_diagnostic_gate_passes_only_when_every_threshold_passes() -> None:
    passed = MODULE.diagnostic_gate_decision(
        pq_specific_r100_gap=0.005,
        uncapped_triplets=500,
        distinct_flip_queries=100,
        effective_sample_size=250,
        max_query_weight_share=0.02,
        qrels_corpus_coverage=1.0,
    )
    assert passed["decision"] == "GO_TO_V6_LOSS_IMPLEMENTATION"
    assert passed["go_to_loss_implementation"] is True
    assert passed["training_authorized"] is False
    assert passed["failed_gates"] == []

    stopped = MODULE.diagnostic_gate_decision(
        pq_specific_r100_gap=0.0049,
        uncapped_triplets=500,
        distinct_flip_queries=100,
        effective_sample_size=250,
        max_query_weight_share=0.02,
        qrels_corpus_coverage=1.0,
    )
    assert stopped["decision"] == "STOP_NO_DISTRIBUTED_PQ_HEADROOM"
    assert stopped["failed_gates"] == ["pq_specific_r100_gap"]
