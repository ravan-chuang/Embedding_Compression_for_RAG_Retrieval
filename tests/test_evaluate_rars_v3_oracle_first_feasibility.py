from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
SCRIPT = ROOT / "scripts/evaluate_rars_v3_oracle_first_feasibility.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rars_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_binary_rank32_policy_uses_exactly_half_top_b() -> None:
    base = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5]], np.float32)
    tiers = np.repeat(base[:, None, :], 4, axis=1)
    tiers[:, 3, :4] += np.asarray([[1.0, 2.0, 3.0, 4.0]], np.float32)
    docids = np.asarray([[50, 40, 30, 20, 10]], np.int64)
    priority = np.asarray([[1.0, 4.0, 3.0, 2.0]], np.float32)
    scores = MODULE.binary_rank32_policy_scores(
        tiers, docids, priority, top_b=4, selected_count=2
    )
    np.testing.assert_allclose(scores, [[0.9, 2.8, 3.7, 0.6, 0.5]])


def test_primary_baseline_selection_uses_recall_then_name() -> None:
    recalls = {
        name: np.asarray([0.5, 0.5], np.float64)
        for name in MODULE.BASELINE_NAMES
    }
    selection = MODULE.select_primary_baseline(recalls, accessed_bytes=640)
    assert selection["selected"]["method"] == sorted(MODULE.BASELINE_NAMES)[0]
    recalls["frozen_pca_rank16_int8"] = np.asarray([1.0, 1.0])
    selection = MODULE.select_primary_baseline(recalls, accessed_bytes=640)
    assert selection["selected"]["method"] == "frozen_pca_rank16_int8"


def test_membership_event_diagnostics_tracks_recovery_and_harm() -> None:
    docids = np.asarray([[1, 2, 3, 4]], np.int64)
    base = np.asarray([[0.9, 0.8, 0.7, 0.6]], np.float32)
    exact = np.asarray([[0.9, 0.7, 1.0, 0.6]], np.float32)
    labels = np.asarray([[0, 0, 1, 0]], np.uint8)
    oracle_membership = np.asarray([[False, False, True, False]])
    result = MODULE.membership_event_diagnostics(
        base,
        exact,
        oracle_membership,
        docids,
        labels,
        final_k=1,
    )
    assert result["compression_relevant_drop_events"] == 1
    assert result["recovered_relevant_drop_events"] == 1
    assert result["relevant_drop_recovery_rate"] == 1.0


def test_exact_solver_preflight_matches_brute_force() -> None:
    result = MODULE.exact_solver_preflight()
    assert result["status"] == "EXACT_SOLVER_BRUTE_FORCE_PREFLIGHT_PASSED"
    assert len(result["cases"]) == 2
    assert all(case["passed"] for case in result["cases"])


def test_registered_outputs_support_safe_provenance_paths_and_detect_changes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "provenance" / "split_audit.json"
        MODULE.atomic_text(path, "{}\n")
        records = {"provenance/split_audit.json": MODULE.file_record(path)}
        MODULE.verify_registered_outputs(root, records)
        MODULE.atomic_text(path, '{"changed":true}\n')
        try:
            MODULE.verify_registered_outputs(root, records)
        except ValueError as error:
            assert "hash changed" in str(error) or "byte count changed" in str(error)
        else:
            raise AssertionError("Mutated registered provenance was accepted")


def test_two_phase_dispatch_never_falls_back_to_legacy_runner() -> None:
    original_design = MODULE.run_design_phase
    original_audit = MODULE.run_audit_phase
    calls: list[str] = []
    try:
        MODULE.run_design_phase = lambda args: calls.append("design") or {"phase": "design"}
        MODULE.run_audit_phase = lambda args: calls.append("audit") or {"phase": "audit"}
        assert MODULE.run(SimpleNamespace(phase="design")) == {"phase": "design"}
        assert MODULE.run(SimpleNamespace(phase="audit")) == {"phase": "audit"}
    finally:
        MODULE.run_design_phase = original_design
        MODULE.run_audit_phase = original_audit
    assert calls == ["design", "audit"]


def test_oracle0_contract_and_gain_per_byte_curve() -> None:
    scores = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)
    docids = np.asarray([[1, 2, 3]], dtype=np.int64)
    oracle0 = SimpleNamespace(
        recall_at_k=np.asarray([0.5], dtype=np.float64),
        accessed_bytes=np.asarray([0], dtype=np.int32),
        rate_assignments=np.zeros((1, 2), dtype=np.int16),
        topk_membership=np.asarray([[True, False, False]]),
    )
    contract = MODULE.validate_oracle0_contract(
        oracle0,
        base_recall=np.asarray([0.5], dtype=np.float64),
        base_scores=scores,
        document_ids=docids,
        final_k=1,
    )
    assert contract["passed"] is True
    oracle16 = SimpleNamespace(
        recall_at_k=np.asarray([0.75], dtype=np.float64),
        accessed_bytes=np.asarray([16], dtype=np.int32),
    )
    curve = MODULE.summarize_oracle_budget_curve(
        {"Oracle0": oracle0, "Oracle16": oracle16},
        ("Oracle0", "Oracle16"),
        [0, 16],
        np.asarray([0.5], dtype=np.float64),
    )
    assert curve["Oracle0"]["gain_per_mean_accessed_byte"] is None
    assert curve["Oracle16"]["mean_recall_gain_over_primary_comparator"] == 0.25
    assert curve["Oracle16"]["gain_per_mean_accessed_byte"] == 0.25 / 16
    bad = SimpleNamespace(**vars(oracle0))
    bad.accessed_bytes = np.asarray([8], dtype=np.int32)
    with pytest.raises(AssertionError, match="nonzero"):
        MODULE.validate_oracle0_contract(
            bad,
            base_recall=np.asarray([0.5], dtype=np.float64),
            base_scores=scores,
            document_ids=docids,
            final_k=1,
        )


def test_bootstrap_rejects_broadcasting_and_records_linear_quantiles() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        MODULE.paired_bootstrap_statistics(
            np.asarray([1.0]),
            np.asarray([1.0, 2.0]),
            replicates=10,
            seed=7,
            confidence=0.95,
        )
    summary, values = MODULE.paired_bootstrap_statistics(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 0.0]),
        replicates=20,
        seed=7,
        confidence=0.95,
    )
    assert values.shape == (20,)
    assert summary["quantile_method"] == "linear"


def _lineage_fixture(root: Path, *, status: str, future_qid: str) -> dict:
    source_commit = "a" * 40
    role_qids = {
        MODULE.DESIGN_ROLE_ID: ["10"],
        MODULE.AUDIT_ROLE_ID: ["20"],
        "future_method_holdout": [future_qid],
    }
    role_rows = {
        MODULE.DESIGN_ROLE_ID: [0],
        MODULE.AUDIT_ROLE_ID: [1],
        "future_method_holdout": [2],
    }
    for role_id in role_qids:
        role_dir = root / role_id
        role_dir.mkdir(parents=True)
        MODULE.atomic_json(
            role_dir / "query_manifest.json",
            {
                "query_ids": role_qids[role_id],
                "query_rows": role_rows[role_id],
                "parent_inner_train_indices": role_rows[role_id],
                **(
                    {
                        "candidate_arrays_created": False,
                        "labels_materialized": False,
                        "metrics_computed": False,
                    }
                    if role_id == "future_method_holdout"
                    else {}
                ),
            },
        )
    started = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": source_commit,
        "qrels_input_accepted": False,
        "parent_label_payload_bytes_read": False,
    }
    split_roles = {
        role_id: {
            "query_count": 1,
            "query_ids_canonical_sha256": MODULE.canonical_sha256(qids),
            "query_ids_source_order_newline_sha256": MODULE._newline_sha256(qids),
            "query_ids_numeric_sorted_newline_sha256": (
                MODULE._numeric_sorted_newline_sha256(qids)
            ),
            "query_rows_sha256": MODULE.array_sha256(
                np.asarray(role_rows[role_id], dtype=np.int64)
            ),
            "parent_role_indices_sha256": MODULE.array_sha256(
                np.asarray(role_rows[role_id], dtype=np.int64)
            ),
            "candidate_retrieval_performed": False,
            "labels_materialized": False,
        }
        for role_id, qids in role_qids.items()
    }
    identity_names = (
        MODULE.DESIGN_ROLE_ID,
        MODULE.AUDIT_ROLE_ID,
        "future_method_holdout",
        "v2_2_inner_validation",
        "burned_outer",
        "clean_test",
    )
    pairwise = {
        f"{left}_vs_{right}": {"qid_overlap": 0, "row_overlap": 0}
        for left, right in MODULE.itertools.combinations(identity_names, 2)
    }
    split = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": source_commit,
        "qrels_opened_or_parsed": False,
        "parent_label_payload_bytes_read": False,
        "parent_label_values_loaded_or_sliced": False,
        "all_required_assertions_passed": True,
        "roles": split_roles,
        "pairwise_overlap": pairwise,
    }
    MODULE.atomic_json(root / "v3_oracle_bundle_build_started.json", started)
    MODULE.atomic_json(root / "v3_oracle_split_audit.json", split)
    future_query_path = root / "future_method_holdout/query_manifest.json"
    future = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": source_commit,
        "role_id": "future_method_holdout",
        "query_count": 1,
        "query_ids_sha256": MODULE.canonical_sha256([future_qid]),
        "query_rows_sha256": MODULE.array_sha256(np.asarray([2], dtype=np.int64)),
        "parent_role_indices_sha256": MODULE.array_sha256(
            np.asarray([2], dtype=np.int64)
        ),
        "query_manifest": MODULE.file_record(future_query_path),
        "candidate_arrays_created": False,
        "labels_materialized": False,
        "metrics_computed": False,
    }
    future_path = root / "future_method_holdout/v3_identity_manifest.json"
    MODULE.atomic_json(future_path, future)
    summary = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": source_commit,
        "status": status,
        "split_audit": MODULE.file_record(root / "v3_oracle_split_audit.json"),
        "roles": {
            MODULE.DESIGN_ROLE_ID: {"candidate_manifest": {"registered": True}},
            MODULE.AUDIT_ROLE_ID: {"candidate_manifest": {"registered": True}},
        },
        "future_method_holdout": {
            "identity_manifest": MODULE.file_record(future_path),
            "candidate_arrays_created": False,
            "labels_materialized": False,
            "metrics_computed": False,
        },
        "parent_candidate_payloads_hash_verified": True,
        "qrels_opened_or_parsed": False,
        "parent_label_payload_bytes_read": False,
        "parent_label_values_loaded_or_sliced": False,
        "faiss_imported_or_search_performed": False,
        "pca_fit_or_score_recomputation_performed": False,
        "closed_role_outcomes_computed": False,
    }
    MODULE.atomic_json(root / "v3_oracle_bundle_freeze_summary.json", summary)
    roles = {}
    for role_id, qids in role_qids.items():
        roles[role_id] = {
            "query_count": 1,
            "source_order_newline_qid_sha256": MODULE._newline_sha256(qids),
            "numeric_sorted_newline_qid_sha256": MODULE._numeric_sorted_newline_sha256(
                qids
            ),
        }
    return {
        "source_commit": source_commit,
        "protocol": {"data_policy": {"roles": roles}},
    }


def test_bundle_root_lineage_rejects_legacy_status_and_future_overlap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _lineage_fixture(
            root,
            status="V3_ORACLE_BUNDLES_FROZEN",
            future_qid="30",
        )
        with pytest.raises(ValueError, match="status"):
            MODULE.validate_bundle_root_lineage(
                root,
                design_manifest_path=None,
                audit_manifest_path=None,
                protocol=fixture["protocol"],
                source_commit=fixture["source_commit"],
                source_hashes={},
            )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fixture = _lineage_fixture(
            root,
            status="V3_QRELS_FREE_CANDIDATE_BUNDLES_FROZEN",
            future_qid="10",
        )
        with pytest.raises(ValueError, match="overlap"):
            MODULE.validate_bundle_root_lineage(
                root,
                design_manifest_path=None,
                audit_manifest_path=None,
                protocol=fixture["protocol"],
                source_commit=fixture["source_commit"],
                source_hashes={},
            )


def test_complete_reuse_rebinds_design_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        freeze_path = output / "design_freeze.json"
        MODULE.atomic_json(freeze_path, {"status": "real"})
        source_commit = "b" * 40
        freeze = {"run_fingerprint": "fingerprint", "external_inputs": {}}
        MODULE.atomic_json(
            output / "oracle_summary.json",
            {"status": "ORACLE_COMPLETE", "run_fingerprint": "fingerprint"},
        )
        MODULE.atomic_json(
            output / "oracle_complete.json",
            {
                "status": "ORACLE_COMPLETE",
                "protocol_id": MODULE.PROTOCOL_ID,
                "source_commit": source_commit,
                "run_fingerprint": "fingerprint",
                "design_freeze": {"bytes": 0, "sha256": "wrong"},
                "outputs": {"placeholder": {}},
                "external_inputs": {},
            },
        )
        monkeypatch.setattr(MODULE, "verify_design_freeze", lambda *args, **kwargs: freeze)
        monkeypatch.setattr(MODULE, "verify_registered_outputs", lambda *args, **kwargs: None)
        monkeypatch.setattr(MODULE, "verify_external_inputs", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match="design freeze"):
            MODULE._reuse_audit_complete(
                output,
                protocol={},
                source_commit=source_commit,
                source_hashes={},
            )


def test_design_freeze_rebinds_selection_folds_environment_and_contracts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        source_commit = "c" * 40
        source_hashes = {"protocol_sha256": "d" * 64}
        protocol = {
            "progressive_representation": {"rank": 32},
            "registered_matched_baselines": {"eligible": ["x"]},
            "matched_access_oracle": {
                "budget_curve_bytes_per_query": [0, 320, 640, 1280]
            },
            "access_gate": {"minimum": 0.01},
            "metric_contract": {"bootstrap": {"replicates": 20}},
        }
        MODULE.atomic_text(output / "numpy_config.txt", "blas=test\n")
        environment = {"python_version": "3.12.13", "numpy_version": "1.26.4"}
        MODULE.atomic_json(
            output / "execution_environment.json",
            {
                **environment,
                "numpy_config": MODULE.file_record(output / "numpy_config.txt"),
            },
        )
        selection = {
            "selected": {"method": "x", "mean_recall_at_10": 0.5},
            "design_fold_gains": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        MODULE.atomic_json(output / "design_primary_comparator.json", selection)
        registered = {
            "numpy_config.txt": MODULE.file_record(output / "numpy_config.txt"),
            "execution_environment.json": MODULE.file_record(
                output / "execution_environment.json"
            ),
            "design_primary_comparator.json": MODULE.file_record(
                output / "design_primary_comparator.json"
            ),
        }
        design_candidate = {"bytes": 1, "sha256": "1" * 64}
        design_labels = {"bytes": 2, "sha256": "2" * 64}
        audit_candidate = {"bytes": 3, "sha256": "3" * 64}
        fingerprint_payload = {
            "protocol_id": MODULE.PROTOCOL_ID,
            "design_candidate_manifest": design_candidate,
            "design_role_labels_manifest": design_labels,
            "audit_candidate_manifest_registration": audit_candidate,
        }
        freeze = {
            "status": "DESIGN_ARTIFACTS_FROZEN_BEFORE_AUDIT_LOAD",
            "protocol_id": MODULE.PROTOCOL_ID,
            "source_commit": source_commit,
            "audit_bundle_loaded_before_this_freeze": False,
            "audit_role_labels_materialized_before_this_freeze": False,
            "future_method_holdout_accessed": False,
            "source_hashes": source_hashes,
            "registered_outputs": registered,
            "fingerprint_payload": fingerprint_payload,
            "run_fingerprint": MODULE.build_run_fingerprint(fingerprint_payload),
            "contracts": {
                "progressive_representation": protocol["progressive_representation"],
                "eligible_baselines": protocol["registered_matched_baselines"],
                "matched_access_oracle": protocol["matched_access_oracle"],
                "access_gate": protocol["access_gate"],
                "bootstrap": protocol["metric_contract"]["bootstrap"],
                "budgets_bytes_per_query": [0, 320, 640, 1280],
            },
            "selected_primary_comparator": selection["selected"],
            "design_fold_gains": selection["design_fold_gains"],
            "execution_environment": environment,
            "design_bundle_manifest": design_candidate,
            "design_role_labels_manifest": design_labels,
            "audit_bundle_manifest_registered_but_arrays_unloaded": audit_candidate,
        }
        MODULE.atomic_json(output / "design_freeze.json", freeze)
        verified = MODULE.verify_design_freeze(
            output,
            protocol=protocol,
            source_commit=source_commit,
            source_hashes=source_hashes,
        )
        assert verified["selected_primary_comparator"]["method"] == "x"
        freeze["design_fold_gains"][0] = -1.0
        MODULE.atomic_json(output / "design_freeze.json", freeze)
        with pytest.raises(ValueError, match="fold gains"):
            MODULE.verify_design_freeze(
                output,
                protocol=protocol,
                source_commit=source_commit,
                source_hashes=source_hashes,
            )
