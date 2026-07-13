from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_external_confirmation_set.py"
)
SPEC = importlib.util.spec_from_file_location("external_prep", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_query_text() -> None:
    assert MODULE.normalize_query_text("  What   Is AI? ") == "what is ai?"


def test_overlap_audit_uses_id_and_text() -> None:
    result = MODULE.audit_overlap(
        ["1", "2", "3"],
        ["alpha", "same text", "gamma"],
        {"1"},
        {"same text"},
    )
    assert result["excluded_qids"] == ["1", "2"]
    assert result["eligible_qids"] == ["3"]
    assert result["outcome_values_used"] is False


def test_qrels_coverage_audit() -> None:
    result = MODULE.audit_qrels_coverage(
        ["q1", "q2"],
        {"q1": {10, 20}, "q2": {30}},
        {10, 30},
    )
    assert result["total_positive_qrels"] == 3
    assert result["total_positive_in_indexed_corpus"] == 2
    assert result["total_positive_missing_from_indexed_corpus"] == 1
    assert result["positive_qrels_corpus_coverage"] == 2 / 3
