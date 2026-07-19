from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rars_v4_tristate_action_feasibility.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rars_v4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _protocol() -> dict:
    return json.loads(
        (ROOT / "protocols/rars_v4_tristate_action_feasibility_v1.json").read_text()
    )


def test_v3_parent_closure_is_recursively_verified(tmp_path: Path) -> None:
    protocol = _protocol()
    output = tmp_path
    (output / "placeholder.txt").write_text("frozen\n")
    MODULE.atomic_json(output / "design_freeze.json", {"status": "frozen"})
    observed = protocol["parent_lineage"]["v3_observed_metrics"]
    summary = {
        "status": "ORACLE_COMPLETE",
        "source_commit": protocol["parent_lineage"]["v3_implementation_commit"],
        "run_fingerprint": "fingerprint",
        "formal_decision": "STOP_NO_HEADROOM",
        "selected_primary_comparator": {
            "method": "residual_norm_top20_rank32_else0"
        },
        "mean_recall_at_10": {
            "base": observed["base_recall_at_10"],
            "primary_comparator": observed["primary_comparator_recall_at_10"],
            "Exact40": observed["exact40_recall_at_10"],
            "Oracle16": observed["oracle16_recall_at_10"],
        },
        "oracle_budget_curve": {
            "Oracle16": {
                "mean_recall_gain_over_primary_comparator": observed[
                    "oracle16_gain_over_comparator"
                ]
            }
        },
        "counterfactual_recovery": {
            "comparator_relative": {
                "Oracle16": {
                    "counterfactual_recovery_fraction": observed[
                        "oracle16_comparator_relative_cfr"
                    ],
                    "positive_gain_mass_with_exact_distance_reduction_fraction": observed[
                        "oracle16_exact40_membership_alignment"
                    ],
                }
            }
        },
    }
    MODULE.atomic_json(output / "oracle_summary.json", summary)
    complete = {
        "status": "ORACLE_COMPLETE",
        "protocol_id": MODULE.V3_PROTOCOL_ID,
        "source_commit": protocol["parent_lineage"]["v3_implementation_commit"],
        "run_fingerprint": "fingerprint",
        "design_freeze": MODULE.file_record(output / "design_freeze.json"),
        "outputs": {"placeholder.txt": MODULE.file_record(output / "placeholder.txt")},
    }
    MODULE.atomic_json(output / "oracle_complete.json", complete)
    verified = MODULE.verify_v3_complete_run(output, protocol)
    assert verified["run_fingerprint"] == "fingerprint"

    (output / "placeholder.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="hash changed|byte count changed"):
        MODULE.verify_v3_complete_run(output, protocol)


def test_registered_output_path_cannot_escape_output_root() -> None:
    for value in ("../secret", "/tmp/secret", ""):
        with pytest.raises(ValueError, match="Unsafe"):
            MODULE._safe_relative_path(value)


def test_source_contains_no_future_holdout_bundle_or_label_load() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "future_method_holdout_accessed\": False" in source
    for forbidden in (
        "future_method_holdout/candidate",
        "future_method_holdout/v4",
        "candidate_relevance.uint8.npy",
        "train_boundary_loss",
        "torch",
    ):
        assert forbidden not in source


def test_two_phases_map_only_to_observed_v3_roles() -> None:
    assert MODULE._phase_role("design") == MODULE.DESIGN_ROLE_ID
    assert MODULE._phase_role("audit") == MODULE.AUDIT_ROLE_ID
    with pytest.raises(ValueError, match="Unsupported"):
        MODULE._phase_role("future")

