#!/usr/bin/env python3
"""Deterministic NumPy core for the RARS-v6 1M headroom diagnostic.

The diagnostic is deliberately read-only: it maps positive qrels into corpus
rows, decomposes retrieval loss into IVF-routing and PQ-specific components,
and measures the amount and concentration of ranking supervision available at
the deployed Top-100 boundary.  A document without a positive judgment is
called *unjudged* throughout this module; absence from qrels is not treated as
an explicit non-relevance judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v6_1m_headroom_v1"

GATE_THRESHOLDS = {
    "minimum_pq_specific_r100_gap": 0.005,
    "minimum_uncapped_triplets": 500,
    "minimum_distinct_flip_queries": 100,
    "minimum_effective_sample_size": 250.0,
    "maximum_query_weight_share": 0.02,
    "required_qrels_corpus_coverage": 1.0,
}


def _integer_array(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array")
    return array.astype(np.int64, copy=False)


@dataclass(frozen=True)
class QrelsRowMapping:
    """Padded qrels mapped to corpus rows, including missing-positive audit."""

    rows: np.ndarray
    qrels_valid: np.ndarray
    in_corpus: np.ndarray
    coverage: dict[str, Any]

    def __post_init__(self) -> None:
        if not (
            self.rows.shape == self.qrels_valid.shape == self.in_corpus.shape
        ):
            raise ValueError("Mapped-qrels arrays must have identical shapes")
        if self.rows.ndim != 2:
            raise ValueError("Mapped qrels must be matrices")

    @property
    def positive_rows(self) -> np.ndarray:
        return self.rows

    @property
    def positive_valid(self) -> np.ndarray:
        return self.qrels_valid

    @property
    def query_count(self) -> int:
        return int(self.coverage["query_count"])

    @property
    def positive_count(self) -> int:
        return int(self.coverage["total_positive_qrels"])

    @property
    def mapped_positive_count(self) -> int:
        return int(self.coverage["total_positive_qrels_in_corpus"])

    @property
    def corpus_coverage(self) -> float:
        return float(self.coverage["qrels_corpus_coverage"])


def map_qrels_doc_ids_to_corpus_rows(
    corpus_doc_ids: Any,
    qrels_doc_ids: Any,
    qrels_valid: Any,
) -> QrelsRowMapping:
    """Map padded positive-qrel document IDs to zero-based corpus rows.

    Missing qrels receive row ``-1`` while remaining valid denominator entries.
    This makes corpus coverage auditable and lets recall count such entries as
    misses rather than silently shrinking its denominator.
    """

    # Convenience form used by the evaluator: (ordered_qids, qrels_by_qid,
    # corpus_doc_ids).  The numerical form below remains useful to tests and
    # to callers that have already materialized padded arrays.
    if isinstance(qrels_doc_ids, dict):
        query_ids = [str(value) for value in corpus_doc_ids]
        if not query_ids or len(query_ids) != len(set(query_ids)):
            raise ValueError("Query IDs must be non-empty and unique")
        by_query: list[np.ndarray] = []
        for query_id in query_ids:
            raw = qrels_doc_ids.get(query_id, set())
            if not isinstance(raw, (set, list, tuple, np.ndarray)):
                raise ValueError("Each qrels mapping value must be a document-ID set")
            values = np.asarray(sorted(int(value) for value in raw), dtype=np.int64)
            if not len(values):
                raise ValueError("Every requested query must have a positive qrel")
            by_query.append(values)
        width = max(len(value) for value in by_query)
        padded = np.full((len(by_query), width), -1, dtype=np.int64)
        padded_valid = np.zeros_like(padded, dtype=bool)
        for query_index, values in enumerate(by_query):
            padded[query_index, : len(values)] = values
            padded_valid[query_index, : len(values)] = True
        corpus_doc_ids, qrels_doc_ids, qrels_valid = (
            qrels_valid,
            padded,
            padded_valid,
        )

    corpus = _integer_array(corpus_doc_ids, name="corpus_doc_ids", ndim=1)
    qrels = _integer_array(qrels_doc_ids, name="qrels_doc_ids", ndim=2)
    valid = np.asarray(qrels_valid, dtype=bool)
    if not len(corpus) or valid.shape != qrels.shape:
        raise ValueError("Corpus IDs must be non-empty and qrels masks must match")
    if np.unique(corpus).size != corpus.size:
        raise ValueError("corpus_doc_ids must be unique")
    for query_index in range(len(qrels)):
        values = qrels[query_index, valid[query_index]]
        if np.unique(values).size != values.size:
            raise ValueError("Positive qrels must be unique within every query")

    order = np.argsort(corpus, kind="stable")
    sorted_ids = corpus[order]
    rows = np.full(qrels.shape, -1, dtype=np.int64)
    in_corpus = np.zeros(qrels.shape, dtype=bool)
    flattened_valid = np.flatnonzero(valid.ravel())
    if len(flattened_valid):
        wanted = qrels.ravel()[flattened_valid]
        positions = np.searchsorted(sorted_ids, wanted)
        found = positions < len(sorted_ids)
        safe_positions = np.minimum(positions, len(sorted_ids) - 1)
        found &= sorted_ids[safe_positions] == wanted
        mapped = np.full(len(wanted), -1, dtype=np.int64)
        mapped[found] = order[positions[found]]
        rows.ravel()[flattened_valid] = mapped
        in_corpus.ravel()[flattened_valid] = found

    per_query_total = valid.sum(axis=1).astype(np.int64)
    per_query_found = in_corpus.sum(axis=1).astype(np.int64)
    total = int(per_query_total.sum())
    found_total = int(per_query_found.sum())
    coverage = {
        "query_count": int(len(qrels)),
        "queries_without_positive_qrels": np.flatnonzero(
            per_query_total == 0
        ).astype(np.int64).tolist(),
        "queries_with_incomplete_corpus_coverage": np.flatnonzero(
            per_query_found != per_query_total
        ).astype(np.int64).tolist(),
        "total_positive_qrels": total,
        "total_positive_qrels_in_corpus": found_total,
        "total_positive_qrels_missing_from_corpus": total - found_total,
        "qrels_corpus_coverage": float(found_total / total) if total else 0.0,
        "positive_qrels_per_query": per_query_total.tolist(),
        "positive_qrels_in_corpus_per_query": per_query_found.tolist(),
    }
    return QrelsRowMapping(rows, valid, in_corpus, coverage)


def known_positive_recall_at_k(
    retrieved_rows: np.ndarray,
    positive_rows: np.ndarray,
    positive_valid: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Return per-query recall against known positive corpus rows.

    A valid positive with row ``-1`` is a qrel missing from the corpus and is
    retained in the denominator.  Retrieved ``-1`` values are padding and can
    never match it.
    """

    retrieved = _integer_array(retrieved_rows, name="retrieved_rows", ndim=2)
    positives = _integer_array(positive_rows, name="positive_rows", ndim=2)
    valid = np.asarray(positive_valid, dtype=bool)
    if positives.shape != valid.shape or len(retrieved) != len(positives):
        raise ValueError("Positive rows/masks must match retrieval queries")
    if not 0 < k <= retrieved.shape[1]:
        raise ValueError("k is outside the retrieved matrix")
    if np.any(retrieved < -1) or np.any(positives[valid] < -1):
        raise ValueError("Corpus rows must be non-negative or use -1 padding")
    counts = valid.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("Every query must have at least one positive qrel")
    for row in retrieved:
        present = row[row >= 0]
        if np.unique(present).size != present.size:
            raise ValueError("Retrieved corpus rows must be unique per query")
    for query_index in range(len(positives)):
        present = positives[query_index, valid[query_index]]
        present = present[present >= 0]
        if np.unique(present).size != present.size:
            raise ValueError("Mapped positive rows must be unique per query")

    top = retrieved[:, :k]
    matched = (
        (top[:, :, None] == positives[:, None, :])
        & valid[:, None, :]
        & (positives[:, None, :] >= 0)
        & (top[:, :, None] >= 0)
    )
    hits = np.any(matched, axis=2).sum(axis=1)
    return hits.astype(np.float64) / counts.astype(np.float64)


def decompose_recall_gaps(
    full_exact: np.ndarray,
    ivf_exact: np.ndarray,
    base_pq: np.ndarray,
) -> dict[str, float]:
    """Decompose mean recall loss into IVF-routing and PQ-specific gaps."""

    arrays = [np.asarray(value, dtype=np.float64) for value in (
        full_exact, ivf_exact, base_pq
    )]
    if any(value.ndim != 1 or not len(value) for value in arrays):
        raise ValueError("Recall inputs must be non-empty vectors")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("Recall vectors must have identical shapes")
    if any(
        not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or np.any(value > 1.0)
        for value in arrays
    ):
        raise ValueError("Recall values must be finite and within [0, 1]")
    full_mean, ivf_mean, pq_mean = (float(value.mean()) for value in arrays)
    routing = full_mean - ivf_mean
    pq_gap = ivf_mean - pq_mean
    total = full_mean - pq_mean
    return {
        "query_count": int(len(arrays[0])),
        "full_exact_recall_at_100": full_mean,
        "ivf_exact_recall_at_100": ivf_mean,
        "base_pq_recall_at_100": pq_mean,
        "ivf_routing_r100_gap": routing,
        "pq_specific_r100_gap": pq_gap,
        "total_r100_gap": total,
        "pq_fraction_of_total_r100_gap": (
            float(pq_gap / total) if total > 0.0 else 0.0
        ),
    }


@dataclass(frozen=True)
class FlipTripletBatch:
    """Query, known-positive, and unjudged rows whose order PQ reverses."""

    query: np.ndarray
    positive_candidate: np.ndarray
    unjudged_candidate: np.ndarray
    positive_row: np.ndarray
    unjudged_row: np.ndarray
    exact_margin: np.ndarray
    pq_margin: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.query,
            self.positive_candidate,
            self.unjudged_candidate,
            self.positive_row,
            self.unjudged_row,
            self.exact_margin,
            self.pq_margin,
            self.weight,
        )
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("Flip-triplet arrays must have identical lengths")
        if any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("Flip-triplet fields must be vectors")
        if len(self.weight) and (
            np.any(~np.isfinite(self.weight)) or np.any(self.weight <= 0)
        ):
            raise ValueError("Flip-triplet weights must be finite and positive")

    def __len__(self) -> int:
        return len(self.query)


@dataclass(frozen=True)
class FlipMiningResult:
    uncapped: FlipTripletBatch
    capped: FlipTripletBatch
    support: dict[str, Any]


def _empty_triplets() -> FlipTripletBatch:
    integer = np.empty(0, dtype=np.int64)
    floating = np.empty(0, dtype=np.float32)
    return FlipTripletBatch(
        integer, integer.copy(), integer.copy(), integer.copy(), integer.copy(),
        floating, floating.copy(), floating.copy()
    )


def _take_triplets(batch: FlipTripletBatch, indices: np.ndarray) -> FlipTripletBatch:
    chosen = np.asarray(indices, dtype=np.int64)
    return FlipTripletBatch(*(
        np.asarray(getattr(batch, field))[chosen]
        for field in batch.__dataclass_fields__
    ))


def summarize_flip_triplets(batch: FlipTripletBatch) -> dict[str, Any]:
    """Summarize support, pair-weight ESS, and query concentration."""

    count = len(batch)
    if not count:
        return {
            "triplets": 0,
            "distinct_flip_queries": 0,
            "distinct_positive_documents": 0,
            "distinct_unjudged_documents": 0,
            "distinct_flip_documents": 0,
            "effective_sample_size": 0.0,
            "max_query_weight_share": 0.0,
        }
    weights = np.asarray(batch.weight, dtype=np.float64)
    weight_sum = float(weights.sum())
    ess = float(weight_sum * weight_sum / np.square(weights).sum())
    queries, inverse = np.unique(batch.query, return_inverse=True)
    query_weights = np.bincount(inverse, weights=weights, minlength=len(queries))
    all_docs = np.concatenate([batch.positive_row, batch.unjudged_row])
    return {
        "triplets": int(count),
        "distinct_flip_queries": int(len(queries)),
        "distinct_positive_documents": int(np.unique(batch.positive_row).size),
        "distinct_unjudged_documents": int(np.unique(batch.unjudged_row).size),
        "distinct_flip_documents": int(np.unique(all_docs).size),
        "effective_sample_size": ess,
        "max_query_weight_share": float(query_weights.max() / weight_sum),
    }


def mine_pq_induced_flip_triplets(
    candidate_rows: np.ndarray,
    exact_scores: np.ndarray,
    pq_scores: np.ndarray,
    positive_rows: np.ndarray,
    positive_valid: np.ndarray,
    *,
    pool_k: int = 100,
    negative_window: int = 16,
    max_unjudged_per_positive: int = 4,
    margin_temperature: float = 0.05,
    damage_scale: float = 8.0,
    flip_bonus: float = 2.0,
) -> FlipMiningResult:
    """Mine uncapped and capped exact-to-PQ pair reversals near Top-k.

    The boundary pool is selected by the deployed PQ order in
    ``[pool_k-negative_window, pool_k+negative_window)``.  A triplet is kept
    only when exact scoring ranks its known positive above an unjudged document
    while PQ scoring reverses that order.  Capping is per query-positive pair.
    """

    rows = _integer_array(candidate_rows, name="candidate_rows", ndim=2)
    exact = np.asarray(exact_scores, dtype=np.float32)
    pq = np.asarray(pq_scores, dtype=np.float32)
    positives = _integer_array(positive_rows, name="positive_rows", ndim=2)
    positive_mask = np.asarray(positive_valid, dtype=bool)
    if exact.shape != rows.shape or pq.shape != rows.shape:
        raise ValueError("Candidate rows and score matrices must match")
    if positives.shape != positive_mask.shape or len(positives) != len(rows):
        raise ValueError("Positive rows/masks must match candidate queries")
    if pool_k <= 0 or negative_window <= 0 or max_unjudged_per_positive <= 0:
        raise ValueError("Pool and mining counts must be positive")
    if margin_temperature <= 0 or damage_scale < 0 or flip_bonus < 0:
        raise ValueError("Triplet weighting parameters are invalid")
    if np.any(rows < -1) or np.any(positives[positive_mask] < 0):
        raise ValueError("Valid corpus rows must be non-negative")
    if np.any(np.isnan(exact)) or np.any(np.isposinf(exact)):
        raise ValueError("Exact scores contain unsupported non-finite values")
    if np.any(np.isnan(pq)) or np.any(np.isposinf(pq)):
        raise ValueError("PQ scores contain unsupported non-finite values")

    records: list[tuple[int, int, int, int, int, float, float, float]] = []
    for query_index in range(len(rows)):
        valid_candidates = rows[query_index] >= 0
        if np.any(~np.isfinite(exact[query_index, valid_candidates])) or np.any(
            ~np.isfinite(pq[query_index, valid_candidates])
        ):
            raise ValueError("Valid candidate scores must be finite")
        valid_indices = np.flatnonzero(valid_candidates)
        valid_rows = rows[query_index, valid_candidates]
        if np.unique(valid_rows).size != valid_rows.size:
            raise ValueError("Candidate corpus rows must be unique per query")
        if len(valid_indices) <= pool_k:
            raise ValueError("Every query needs more valid candidates than pool_k")

        known = positives[query_index, positive_mask[query_index]]
        if np.unique(known).size != known.size:
            raise ValueError("Positive corpus rows must be unique per query")
        row_to_candidate = {
            int(row): int(index) for row, index in zip(valid_rows, valid_indices)
        }
        missing = [int(row) for row in known if int(row) not in row_to_candidate]
        if missing:
            raise ValueError("Candidate union omits a known positive corpus row")

        pq_order_local = np.lexsort((
            rows[query_index, valid_indices],
            -pq[query_index, valid_indices],
        ))
        pq_order = valid_indices[pq_order_local]
        lo = max(0, pool_k - negative_window)
        hi = min(len(pq_order), pool_k + negative_window)
        boundary = pq_order[lo:hi]
        known_set = set(int(value) for value in known)
        unjudged = np.asarray(
            [index for index in boundary if int(rows[query_index, index]) not in known_set],
            dtype=np.int64,
        )
        if not len(unjudged):
            continue

        positive_candidates = sorted(
            (row_to_candidate[int(row)] for row in known),
            key=lambda index: int(rows[query_index, index]),
        )
        for positive in positive_candidates:
            exact_margin = exact[query_index, positive] - exact[query_index, unjudged]
            pq_margin = pq[query_index, positive] - pq[query_index, unjudged]
            flipped = (exact_margin > 0.0) & (pq_margin <= 0.0)
            for offset in np.flatnonzero(flipped):
                candidate = int(unjudged[offset])
                exact_value = float(exact_margin[offset])
                pq_value = float(pq_margin[offset])
                damage = max(0.0, exact_value - pq_value)
                boundary_weight = float(np.exp(-exact_value / margin_temperature))
                weight = 1.0 + boundary_weight + damage_scale * damage + flip_bonus
                records.append((
                    query_index,
                    int(positive),
                    candidate,
                    int(rows[query_index, positive]),
                    int(rows[query_index, candidate]),
                    exact_value,
                    pq_value,
                    weight,
                ))

    if records:
        columns = tuple(zip(*records))
        uncapped = FlipTripletBatch(
            *(np.asarray(columns[index], dtype=np.int64) for index in range(5)),
            *(np.asarray(columns[index], dtype=np.float32) for index in range(5, 8)),
        )
    else:
        uncapped = _empty_triplets()

    selected: list[int] = []
    if len(uncapped):
        groups = np.stack([uncapped.query, uncapped.positive_row], axis=1)
        for group in np.unique(groups, axis=0):
            members = np.flatnonzero(
                (uncapped.query == group[0]) & (uncapped.positive_row == group[1])
            )
            local = np.lexsort((
                uncapped.unjudged_row[members],
                uncapped.exact_margin[members],
                -uncapped.weight[members],
            ))
            selected.extend(members[local[:max_unjudged_per_positive]].tolist())
    capped = _take_triplets(uncapped, np.asarray(selected, dtype=np.int64))
    support = {
        "pool_k": int(pool_k),
        "negative_window": int(negative_window),
        "max_unjudged_per_positive": int(max_unjudged_per_positive),
        "uncapped": summarize_flip_triplets(uncapped),
        "capped": summarize_flip_triplets(capped),
    }
    return FlipMiningResult(uncapped, capped, support)


def diagnostic_gate_decision(
    *,
    pq_specific_r100_gap: float,
    uncapped_triplets: int,
    distinct_flip_queries: int,
    effective_sample_size: float,
    max_query_weight_share: float,
    qrels_corpus_coverage: float,
) -> dict[str, Any]:
    """Apply the preregistered v6 headroom gate without training a model."""

    scalar_values = (
        pq_specific_r100_gap,
        effective_sample_size,
        max_query_weight_share,
        qrels_corpus_coverage,
    )
    if any(not np.isfinite(float(value)) for value in scalar_values):
        raise ValueError("Gate inputs must be finite")
    for name, value in (
        ("uncapped_triplets", uncapped_triplets),
        ("distinct_flip_queries", distinct_flip_queries),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < 0:
            raise ValueError(f"{name} cannot be negative")
    if pq_specific_r100_gap < -1.0 or pq_specific_r100_gap > 1.0:
        raise ValueError("PQ-specific recall gap is outside [-1, 1]")
    if effective_sample_size < 0.0:
        raise ValueError("Effective sample size cannot be negative")
    if not 0.0 <= max_query_weight_share <= 1.0:
        raise ValueError("Query weight share must be within [0, 1]")
    if not 0.0 <= qrels_corpus_coverage <= 1.0:
        raise ValueError("Qrels corpus coverage must be within [0, 1]")

    observed = {
        "pq_specific_r100_gap": float(pq_specific_r100_gap),
        "uncapped_triplets": int(uncapped_triplets),
        "distinct_flip_queries": int(distinct_flip_queries),
        "effective_sample_size": float(effective_sample_size),
        "max_query_weight_share": float(max_query_weight_share),
        "qrels_corpus_coverage": float(qrels_corpus_coverage),
    }
    passed = {
        "pq_specific_r100_gap": observed["pq_specific_r100_gap"]
        >= GATE_THRESHOLDS["minimum_pq_specific_r100_gap"],
        "uncapped_triplets": observed["uncapped_triplets"]
        >= GATE_THRESHOLDS["minimum_uncapped_triplets"],
        "distinct_flip_queries": observed["distinct_flip_queries"]
        >= GATE_THRESHOLDS["minimum_distinct_flip_queries"],
        "effective_sample_size": observed["effective_sample_size"]
        >= GATE_THRESHOLDS["minimum_effective_sample_size"],
        "max_query_weight_share": observed["max_query_weight_share"]
        <= GATE_THRESHOLDS["maximum_query_weight_share"],
        "qrels_corpus_coverage": observed["qrels_corpus_coverage"]
        == GATE_THRESHOLDS["required_qrels_corpus_coverage"],
    }
    go = all(passed.values())
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "HEADROOM_DIAGNOSTIC_COMPLETE",
        "decision": (
            "GO_TO_V6_LOSS_IMPLEMENTATION"
            if go
            else "STOP_NO_DISTRIBUTED_PQ_HEADROOM"
        ),
        "go_to_loss_implementation": go,
        "training_authorized": False,
        "observed": observed,
        "thresholds": dict(GATE_THRESHOLDS),
        "gates": {name: bool(value) for name, value in passed.items()},
        "failed_gates": [name for name, value in passed.items() if not value],
    }
