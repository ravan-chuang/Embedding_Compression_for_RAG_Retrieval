from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rars_v3_oracle_core.py"
SPEC = importlib.util.spec_from_file_location("rars_v3_oracle_core", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _parent_qids() -> list[str]:
    path = (
        ROOT
        / "results/rars_v2_2_fp32_replication/provenance/"
        "input-audit-00a0dee30767/inner_train/query_manifest.json"
    )
    return [str(value) for value in json.loads(path.read_text())["query_ids"]]


def _newline_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def test_frozen_v3_split_counts_and_hashes() -> None:
    qids = _parent_qids()
    roles = MODULE.split_development_qids(qids)
    expected = {
        MODULE.DESIGN_ROLE_ID: (
            2307,
            "952219a667ecfcd8cad5e18475531ceba73720054fae3344227dbdd008491fb6",
        ),
        MODULE.AUDIT_ROLE_ID: (
            851,
            "60a760cf5e8b8a4f9c238bc52a24448d3699e271d99fe15af69d80aec39ecc8a",
        ),
        MODULE.FUTURE_ROLE_ID: (
            803,
            "41a1908b386cdf8bcd5e2905b88d99e66e3e1b392916c55643f36bd215ae95a8",
        ),
    }
    for role_id, (count, expected_hash) in expected.items():
        selected = [qids[int(index)] for index in roles[role_id]]
        assert len(selected) == count
        assert _newline_hash(selected) == expected_hash
    all_indices = np.concatenate(list(roles.values()))
    assert len(np.unique(all_indices)) == len(qids)


def test_frozen_design_fold_counts() -> None:
    qids = _parent_qids()
    design = MODULE.split_development_qids(qids)[MODULE.DESIGN_ROLE_ID]
    folds = MODULE.design_fold_ids([qids[int(index)] for index in design])
    assert np.bincount(folds, minlength=5).tolist() == [463, 468, 470, 436, 470]


def test_progressive_fit_returns_oriented_basis_and_full_spectrum() -> None:
    residuals = np.asarray(
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    basis, scales, rows, singular_values = MODULE.fit_progressive_pca(
        residuals, rank=2, max_samples=4, seed=7, scale_batch_size=2
    )
    assert basis.shape == (3, 2)
    assert scales.shape == (2,)
    assert rows.shape == (4,)
    assert singular_values.shape == (3,)
    assert np.all(np.diff(singular_values) <= 0)
    for column in basis.T:
        pivot = int(np.argmax(np.abs(column)))
        assert column[pivot] >= 0


def test_stable_topk_breaks_score_ties_by_document_id() -> None:
    scores = np.asarray([[1.0, 1.0, 0.5]], dtype=np.float32)
    document_ids = np.asarray([[20, 10, 30]], dtype=np.int64)
    top = MODULE.stable_topk(scores, document_ids, 2)
    assert top.tolist() == [[1, 0]]


@pytest.mark.parametrize(
    ("scores", "document_ids", "message"),
    [
        ([[1.0, np.nan, 0.0]], [[1, 2, 3]], "finite"),
        ([[1.0, 0.5, 0.0]], [[1, 1, 3]], "unique"),
    ],
)
def test_stable_topk_rejects_nonfinite_scores_and_duplicate_docids(
    scores: list[list[float]], document_ids: list[list[int]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE.stable_topk(np.asarray(scores), np.asarray(document_ids), 1)


def test_recall_at_k_supports_multiple_relevant_documents() -> None:
    scores = np.asarray([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], np.float32)
    docids = np.asarray([[1, 2, 3], [1, 2, 3]], np.int64)
    labels = np.asarray([[1, 0, 1], [0, 1, 0]], np.uint8)
    counts = np.asarray([2, 1], np.int32)
    recall = MODULE.recall_at_k_per_query(
        scores, docids, labels, counts, k=2
    )
    np.testing.assert_allclose(recall, [0.5, 1.0])


def test_progressive_scores_are_prefixes_and_zero_is_base() -> None:
    queries = np.asarray([[1.0, 1.0]], np.float32)
    ann = np.asarray([[0.0, 0.5, 0.2]], np.float32)
    lookup = np.asarray([[0, 1, 2]], np.int64)
    residuals = np.asarray([[1.0, 2.0], [3.0, 4.0], [8.0, 8.0]], np.float32)
    result = MODULE.progressive_tier_scores(
        queries,
        ann,
        lookup,
        residuals,
        np.eye(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        tiers=(0, 1, 2),
        alpha=1.0,
        top_b=2,
        batch_size=1,
    )
    np.testing.assert_array_equal(result[:, 0], ann)
    np.testing.assert_allclose(result[0, 1], [1.0, 3.5, 0.2])
    np.testing.assert_allclose(result[0, 2], [3.0, 7.5, 0.2])


def test_exact_residual_scores_respects_depth() -> None:
    queries = np.asarray([[1.0, 0.0]], np.float32)
    ann = np.asarray([[0.4, 0.3, 0.2]], np.float32)
    lookup = np.asarray([[0, 1, 2]], np.int64)
    residuals = np.asarray([[0.5, 0.0], [-0.2, 0.0], [9.0, 0.0]], np.float32)
    scores = MODULE.exact_residual_scores(
        queries, ann, lookup, residuals, top_b=2, batch_size=1
    )
    np.testing.assert_allclose(scores, [[0.9, 0.1, 0.2]])


@pytest.mark.parametrize("scorer", ["progressive", "exact"])
def test_residual_scorers_reject_lookup_upper_bound(scorer: str) -> None:
    queries = np.asarray([[1.0, 0.0]], np.float32)
    ann = np.asarray([[0.4, 0.3, 0.2]], np.float32)
    lookup = np.asarray([[0, 3, 1]], np.int64)
    residuals = np.asarray([[0.5, 0.0], [-0.2, 0.0], [9.0, 0.0]], np.float32)
    with pytest.raises(ValueError, match="out-of-bounds"):
        if scorer == "progressive":
            MODULE.progressive_tier_scores(
                queries,
                ann,
                lookup,
                residuals,
                np.eye(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                tiers=(0, 1, 2),
                alpha=1.0,
                top_b=2,
            )
        else:
            MODULE.exact_residual_scores(
                queries, ann, lookup, residuals, top_b=2
            )


def _brute_force_query(
    scores: np.ndarray,
    costs: np.ndarray,
    labels: np.ndarray,
    docids: np.ndarray,
    *,
    k: int,
    top_b: int,
    budget: int,
) -> tuple[int, int]:
    best_hits = -1
    best_bytes = budget + 1
    for tiers in itertools.product(range(len(costs)), repeat=top_b):
        used = sum(int(costs[value]) for value in tiers)
        if used > budget:
            continue
        selected_scores = scores[0].copy()
        for candidate, tier in enumerate(tiers):
            selected_scores[candidate] = scores[tier, candidate]
        order = np.lexsort((docids, -selected_scores))[:k]
        hits = int(np.sum(labels[order] > 0))
        if hits > best_hits or (hits == best_hits and used < best_bytes):
            best_hits, best_bytes = hits, used
    return best_hits, best_bytes


@pytest.mark.parametrize("seed", range(10))
def test_exact_access_oracle_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    tier_count, candidate_count, top_b = 3, 5, 3
    base = np.sort(rng.normal(size=candidate_count))[::-1]
    scores = np.repeat(base[None, :], tier_count, axis=0)
    scores[:, :top_b] += rng.normal(scale=0.25, size=(tier_count, top_b))
    costs = np.asarray([0, 1, 2], np.int64)
    labels = (rng.random(candidate_count) < 0.4).astype(np.uint8)
    docids = rng.permutation(np.arange(100, 100 + candidate_count)).astype(np.int64)
    expected = _brute_force_query(
        scores, costs, labels, docids, k=2, top_b=top_b, budget=3
    )
    actual = MODULE._exact_query_access_oracle(
        scores,
        costs,
        labels,
        docids,
        final_k=2,
        top_b=top_b,
        budget_bytes=3,
    )
    assert actual[:2] == expected
    assert int(np.sum(actual[2])) == actual[1]
    assert int(np.sum(labels[actual[3]] > 0)) == actual[0]


def test_access_oracle_never_exceeds_budget() -> None:
    tier_scores = np.asarray(
        [
            [
                [0.9, 0.8, 0.7, 0.6],
                [0.9, 1.0, 0.7, 0.6],
                [0.9, 1.2, 0.7, 0.6],
                [0.9, 1.4, 0.7, 0.6],
            ]
        ],
        dtype=np.float32,
    )
    result = MODULE.exact_accessed_byte_oracle(
        tier_scores,
        (0, 8, 16, 32),
        np.asarray([[0, 1, 0, 0]], np.uint8),
        np.asarray([[1, 2, 3, 4]], np.int64),
        np.asarray([1], np.int32),
        final_k=1,
        top_b=2,
        budget_bytes=8,
    )
    assert result.recall_at_k.tolist() == [1.0]
    assert result.accessed_bytes.tolist() == [8]
    assert int(result.rate_assignments.sum()) == 8


@pytest.mark.parametrize(
    "costs",
    [
        (8, 16, 32),
        (0, 16, 8),
        (0, 8, 8),
        (0, 8.0, 16),
        (0, 8, 40_000),
    ],
)
def test_access_oracle_rejects_invalid_tier_cost_contract(
    costs: tuple[object, ...]
) -> None:
    tier_scores = np.zeros((1, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Tier byte costs"):
        MODULE.exact_accessed_byte_oracle(
            tier_scores,
            costs,
            np.asarray([[1, 0, 0, 0]], np.uint8),
            np.asarray([[1, 2, 3, 4]], np.int64),
            np.asarray([1], np.int32),
            final_k=1,
            top_b=2,
            budget_bytes=16,
        )


def test_access_oracle_rejects_nonfinite_scores_and_duplicate_docids() -> None:
    tier_scores = np.zeros((1, 3, 4), dtype=np.float32)
    tier_scores[0, 1, 0] = np.inf
    common = {
        "labels": np.asarray([[1, 0, 0, 0]], np.uint8),
        "relevant_counts": np.asarray([1], np.int32),
        "final_k": 1,
        "top_b": 2,
        "budget_bytes": 16,
    }
    with pytest.raises(ValueError, match="finite"):
        MODULE.exact_accessed_byte_oracle(
            tier_scores,
            (0, 8, 16),
            document_ids=np.asarray([[1, 2, 3, 4]], np.int64),
            **common,
        )
    with pytest.raises(ValueError, match="unique"):
        MODULE.exact_accessed_byte_oracle(
            np.zeros_like(tier_scores),
            (0, 8, 16),
            document_ids=np.asarray([[1, 1, 3, 4]], np.int64),
            **common,
        )


def _valid_manifest(role_id: str) -> dict[str, object]:
    return {
        "protocol_id": MODULE.PROTOCOL_ID,
        "role_id": role_id,
        "evidence_status": "DEVELOPMENT_ONLY",
        "source_commit": "0" * 40,
        "query_ids_sha256": "a" * 64,
        "query_rows_sha256": "b" * 64,
        "split_audit_sha256": "c" * 64,
        "builder_sha256": "d" * 64,
        "protocol_sha256": "e" * 64,
        "parent_v2_2_manifest_sha256": "f" * 64,
        "data_access": {
            "v2_2_inner_validation_values_used": False,
            "outer_relevance_values_used": False,
            "clean_test_relevance_values_used": False,
            "nq_relevance_values_used": False,
            "trec_relevance_values_used": False,
            "future_method_holdout_relevance_values_used": False,
        },
    }


def test_manifest_rejects_closed_role_and_forbidden_access() -> None:
    manifest = _valid_manifest(MODULE.AUDIT_ROLE_ID)
    MODULE.validate_bundle_manifest(
        manifest, expected_role_id=MODULE.AUDIT_ROLE_ID
    )
    manifest["role_id"] = "inner_validation"
    with pytest.raises(ValueError, match="Expected role_id"):
        MODULE.validate_bundle_manifest(
            manifest, expected_role_id=MODULE.AUDIT_ROLE_ID
        )
    manifest = _valid_manifest(MODULE.AUDIT_ROLE_ID)
    manifest["data_access"]["outer_relevance_values_used"] = True
    with pytest.raises(ValueError, match="Forbidden data-access"):
        MODULE.validate_bundle_manifest(
            manifest, expected_role_id=MODULE.AUDIT_ROLE_ID
        )


def test_gain_diagnostics_measures_support_harm_and_concentration() -> None:
    diagnostics = MODULE.gain_diagnostics(
        np.asarray([1.0, 0.5, -0.25, 0.0], np.float64)
    )
    assert diagnostics["improved"] == 2
    assert diagnostics["harmed"] == 1
    assert diagnostics["harm_to_positive_mass_ratio"] == pytest.approx(1 / 6)
    assert diagnostics["effective_positive_support"] == pytest.approx(1.8)


def test_paired_bootstrap_requires_identical_finite_vectors() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        MODULE.paired_bootstrap_mean_delta(
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0], [2.0]]),
            replicates=10,
            seed=1,
        )
    for left, right in (
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, np.inf]),
    ):
        with pytest.raises(ValueError, match="finite"):
            MODULE.paired_bootstrap_mean_delta(
                np.asarray(left), np.asarray(right), replicates=10, seed=1
            )


def test_compression_recovery_is_explicitly_reference_relative() -> None:
    document_ids = np.asarray([[1, 2, 3]], dtype=np.int64)
    base_scores = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)
    comparator_scores = np.asarray([[0.7, 0.9, 0.8]], dtype=np.float32)
    exact_scores = np.asarray([[0.7, 0.8, 0.9]], dtype=np.float32)
    oracle_membership = np.asarray([[False, False, True]])
    exact_recall = np.asarray([1.0])
    oracle_recall = np.asarray([0.75])
    base = MODULE.compression_recovery_diagnostics(
        base_scores,
        exact_scores,
        oracle_recall,
        np.asarray([0.0]),
        exact_recall,
        oracle_membership,
        document_ids,
        k=1,
        reference_name="base",
    )
    comparator = MODULE.compression_recovery_diagnostics(
        comparator_scores,
        exact_scores,
        oracle_recall,
        np.asarray([0.5]),
        exact_recall,
        oracle_membership,
        document_ids,
        k=1,
        reference_name="comparator",
    )
    assert base["reference_name"] == "base"
    assert base["counterfactual_recovery_fraction"] == pytest.approx(0.75)
    assert comparator["reference_name"] == "comparator"
    assert comparator["counterfactual_recovery_fraction"] == pytest.approx(0.5)


def _gate_inputs() -> dict[str, object]:
    protocol = json.loads(
        (ROOT / "protocols/rars_v3_oracle_first_feasibility_v1.json").read_text()
    )
    count = 100
    return {
        "oracle_recall": np.full(count, 0.02),
        "comparator_recall": np.zeros(count),
        "exact40_recall": np.full(count, 0.03),
        "base_recall": np.zeros(count),
        "base_relative_cfr8": 0.3,
        "base_relative_cfr16": 0.5,
        "base_relative_alignment16": 0.8,
        "comparator_relative_cfr8": 0.3,
        "comparator_relative_cfr16": 0.5,
        "comparator_relative_alignment16": 0.8,
        "design_fold_gains": [0.01] * 5,
        "bootstrap": {
            "replicates": 20_000,
            "seed": 20_260_719,
            "confidence": 0.95,
            "lower": 0.01,
            "upper": 0.03,
            "point_estimate": 0.02,
        },
        "thresholds": protocol["access_gate"],
    }


def test_gate_go_kill_and_sparse_stop() -> None:
    inputs = _gate_inputs()
    go = MODULE.decide_oracle_gate(**inputs)
    assert go["decision"] == "GO_TO_STATIC_STORAGE_ORACLE"
    assert go["compression_recovery"]["base_relative"]["oracle16_cfr"] == 0.5
    assert (
        go["compression_recovery"]["comparator_relative"]["oracle16_cfr"]
        == 0.5
    )

    killed_inputs = dict(inputs)
    killed_inputs["exact40_recall"] = np.full(100, 0.005)
    killed = MODULE.decide_oracle_gate(**killed_inputs)
    assert killed["decision"] == "KILL_NO_SCORE_HEADROOM"

    sparse = np.zeros(100)
    sparse[:2] = 1.0
    stopped_inputs = dict(inputs)
    stopped_inputs["oracle_recall"] = sparse
    stopped = MODULE.decide_oracle_gate(**stopped_inputs)
    assert stopped["decision"] == "STOP_NO_HEADROOM"
    assert stopped["checks"]["positive_support"] is False


def test_gate_uses_comparator_relative_recovery_as_hard_evidence() -> None:
    inputs = _gate_inputs()
    inputs["base_relative_cfr8"] = 0.0
    inputs["base_relative_cfr16"] = 0.0
    inputs["base_relative_alignment16"] = 0.0
    assert (
        MODULE.decide_oracle_gate(**inputs)["decision"]
        == "GO_TO_STATIC_STORAGE_ORACLE"
    )

    inputs["base_relative_cfr8"] = 1.0
    inputs["base_relative_cfr16"] = 1.0
    inputs["base_relative_alignment16"] = 1.0
    inputs["comparator_relative_cfr8"] = 0.1
    stopped = MODULE.decide_oracle_gate(**inputs)
    assert stopped["decision"] == "STOP_NO_HEADROOM"
    assert stopped["checks"]["oracle8_comparator_relative_cfr"] is False


def test_gate_rejects_nonfinite_recall_recovery_bootstrap_and_folds() -> None:
    invalid_cases: list[tuple[str, object]] = [
        ("oracle_recall", np.concatenate(([np.nan], np.full(99, 0.02)))),
        ("comparator_recall", np.concatenate(([np.inf], np.zeros(99)))),
        ("exact40_recall", np.concatenate(([np.inf], np.full(99, 0.03)))),
        ("base_recall", np.concatenate(([np.nan], np.zeros(99)))),
        ("base_relative_cfr8", np.nan),
        ("base_relative_cfr16", np.inf),
        ("base_relative_alignment16", np.nan),
        ("comparator_relative_cfr8", np.inf),
        ("comparator_relative_cfr16", np.nan),
        ("comparator_relative_alignment16", np.inf),
        ("design_fold_gains", [0.01, 0.01, np.nan, 0.01, 0.01]),
    ]
    for field, value in invalid_cases:
        inputs = _gate_inputs()
        inputs[field] = value
        with pytest.raises(ValueError, match="finite"):
            MODULE.decide_oracle_gate(**inputs)
    for bootstrap_field in (
        "point_estimate",
        "lower",
        "upper",
        "confidence",
        "replicates",
        "seed",
    ):
        inputs = _gate_inputs()
        inputs["bootstrap"] = dict(inputs["bootstrap"])
        inputs["bootstrap"][bootstrap_field] = np.nan
        with pytest.raises(ValueError, match="finite"):
            MODULE.decide_oracle_gate(**inputs)


def test_run_fingerprint_changes_when_budget_changes() -> None:
    payload = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": "0" * 40,
        "configuration": {"budget": 640},
    }
    changed = json.loads(json.dumps(payload))
    changed["configuration"]["budget"] = 320
    assert MODULE.build_run_fingerprint(payload) != MODULE.build_run_fingerprint(changed)
