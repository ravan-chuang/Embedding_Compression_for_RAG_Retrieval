from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_rars_v4_tristate_labels.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_rars_v4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_graded_mapping_preserves_explicit_negative_and_missing_is_unjudged() -> None:
    parsed, schema = MODULE._parse_role_judgments(
        {
            "q1": {"10": 1, "11": 0},
            "q2": {"20": 2, "21": -1},
        },
        ["q1", "q2"],
    )
    assert parsed["q1"] == {10: 1.0, 11: 0.0}
    assert schema["explicit_negative_source_rows"] == 2
    assert schema["all_role_entries_graded_mapping"] is True
    assert schema["explicit_negative_semantics_preserved"] is True


def test_positive_only_list_is_not_certified_as_explicit_negative_semantics() -> None:
    _, schema = MODULE._parse_role_judgments(
        {"q1": ["10", "11"], "q2": ["20"]}, ["q1", "q2"]
    )
    assert schema["positive_only_list_query_count"] == 2
    assert schema["explicit_negative_source_rows"] == 0
    assert schema["explicit_negative_semantics_preserved"] is False


def test_mapping_without_any_grade_zero_is_not_enough_to_certify_semantics() -> None:
    _, schema = MODULE._parse_role_judgments(
        {"q1": {"10": 1}, "q2": {"20": 1}}, ["q1", "q2"]
    )
    assert schema["all_role_entries_graded_mapping"] is True
    assert schema["explicit_negative_semantics_preserved"] is False


def test_missing_role_query_fails_closed() -> None:
    try:
        MODULE._parse_role_judgments({"q1": {"10": 1}}, ["q1", "q2"])
    except ValueError as error:
        assert "lacks 1 role queries" in str(error)
    else:
        raise AssertionError("Missing role query was accepted")

