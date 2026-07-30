from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/MSMARCO_RARS_v3_Oracle_First_Feasibility.ipynb"
TRAINING_COMMIT = "bb9b106e69b9a453756fd800665f701614ce67b3"
IMPLEMENTATION_COMMIT = "05c2ae43b7d11783460822d10c590240dab1a399"
EXPECTED_IDS = [
    "v3-title",
    "v3-context",
    "v3-setup",
    "v3-checkout-tests",
    "v3-preflight",
    "v3-data",
    "v3-parent-rematerialize",
    "v3-candidate-freeze",
    "v3-design-labels",
    "v3-design-phase",
    "v3-results",
    "v3-audit-phase",
    "v3-final-report",
    "v3-takeaways",
]


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def _cells() -> dict[str, dict]:
    return {cell["id"]: cell for cell in _notebook()["cells"]}


def _code() -> str:
    return "\n".join(
        _source(cell)
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v3_notebook_is_clean_ordered_and_dual_pinned() -> None:
    notebook = _notebook()
    cells = notebook["cells"]
    ids = [cell["id"] for cell in cells]
    assert ids == EXPECTED_IDS
    assert len(ids) == len(set(ids))
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    code = _code()
    assert f"TRAINING_COMMIT = '{TRAINING_COMMIT}'" in code
    assert f"V3_IMPLEMENTATION_COMMIT = '{IMPLEMENTATION_COMMIT}'" in code
    assert len(IMPLEMENTATION_COMMIT) == 40
    int(IMPLEMENTATION_COMMIT, 16)
    metadata = notebook["metadata"]
    assert metadata["accelerator"] == "GPU"
    assert metadata["colab"]["gpuType"] == "T4"
    assert metadata["kernelspec"]["name"] == "python3"


def test_v3_notebook_uses_isolated_numpy_subprocess() -> None:
    cells = _cells()
    setup = _source(cells["v3-setup"])
    preflight = _source(cells["v3-preflight"])
    assert "numpy==1.26.4" in setup
    assert "'--target', str(NUMPY_TARGET)" in setup
    assert "EXPERIMENT_ENV['PYTHONPATH']" in setup
    assert "env=EXPERIMENT_ENV" in setup
    assert "current_environment['python_version'] == contract['python_version']" in preflight
    assert "current_environment['numpy_version'] == contract['numpy_version']" in preflight
    assert "is_relative_to(" in preflight
    assert "NUMPY_TARGET.resolve()" in preflight

    for cell in _notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(_source(cell))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name.split(".")[0] in {"numpy", "faiss", "torch"}
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in {"numpy", "faiss", "torch"}

    for cell_id in (
        "v3-checkout-tests",
        "v3-parent-rematerialize",
        "v3-candidate-freeze",
        "v3-design-labels",
        "v3-design-phase",
        "v3-audit-phase",
    ):
        source = _source(cells[cell_id])
        assert "env=EXPERIMENT_ENV" in source


def test_v3_notebook_rematerializes_exact_parent_inner_only() -> None:
    cells = _cells()
    setup = _source(cells["v3-setup"])
    parent = _source(cells["v3-parent-rematerialize"])
    assert "/content/Embedding_Compression_for_RAG_Retrieval_rars_v2_2" in setup
    assert "f'rars-v2.2-{TRAINING_COMMIT[:12]}'" in setup
    assert "'--inner-only'" in parent
    assert "'--qrels'" in parent
    assert "'--index'" in parent
    assert "--cached-full-residuals" not in parent
    for field in (
        "parent_inner_train_manifest_sha256",
        "parent_inner_train_source_manifest_sha256",
        "parent_inner_train_query_manifest_sha256",
        "closed_inner_validation_query_manifest_sha256",
        "parent_v2_2_split_audit_sha256",
    ):
        assert field in parent
    assert "for key, actual in parent_hashes.items()" in parent
    assert "assert actual == parent[key]" in parent


def test_v3_candidate_freeze_has_no_label_or_retrieval_inputs() -> None:
    candidate = _source(_cells()["v3-candidate-freeze"])
    for forbidden in ("'--qrels'", "'--index'", "'--pca-"):
        assert forbidden not in candidate
    for assertion in (
        "parent_candidate_payloads_hash_verified",
        "parent_label_payload_bytes_read",
        "qrels_opened_or_parsed",
        "faiss_imported_or_search_performed",
        "pca_fit_or_score_recomputation_performed",
    ):
        assert assertion in candidate
    assert "future_files == {'query_manifest.json', 'v3_identity_manifest.json'}" in candidate
    assert "ROLE_LABEL_FILES" in candidate


def test_v3_notebook_enforces_design_then_audit_chronology() -> None:
    cells = _notebook()["cells"]
    positions = {cell["id"]: index for index, cell in enumerate(cells)}
    assert positions["v3-candidate-freeze"] < positions["v3-design-labels"]
    assert positions["v3-design-labels"] < positions["v3-design-phase"]
    assert positions["v3-design-phase"] < positions["v3-audit-phase"]
    code = _code()
    assert code.count("'--role', 'oracle_design'") == 1
    assert "'--role', 'oracle_audit'" not in code
    design = _source(_cells()["v3-design-phase"])
    audit = _source(_cells()["v3-audit-phase"])
    assert "audit_role_labels_materialized_before_this_freeze" in design
    assert "ROLE_LABEL_FILES" in design
    assert "'--design-label-manifest'" not in audit
    assert "'--parent-inner-train-bundle'" in audit


def test_v3_notebook_preserves_durable_output_and_complete_reuse() -> None:
    code = _code()
    assert "shutil.rmtree(OUTPUT)" not in code
    assert code.count("'--reuse-complete'") == 2
    assert "OUTPUT = DRIVE / 'rars-v3-oracle-first' / V3_IMPLEMENTATION_COMMIT[:12]" in code
    assert "f'rars-v3-{V3_IMPLEMENTATION_COMMIT[:12]}'" in code
    final_report = _source(_cells()["v3-final-report"])
    assert "for relative_name, record in complete['outputs'].items()" in final_report
    assert "not relative.is_absolute() and '..' not in relative.parts" in final_report
    assert "verify_record(OUTPUT / 'design_freeze.json', complete['design_freeze'])" in final_report
    assert "complete['run_fingerprint'] == summary['run_fingerprint']" in final_report


def test_v3_notebook_reports_full_registered_curve() -> None:
    report = _source(_cells()["v3-final-report"])
    for name in ("Oracle0", "Oracle8", "Oracle16", "Oracle32"):
        assert name in report
    assert "oracle_budget_curve" in report
    assert "oracle0_contract" in report
    assert "comparator_relative_recovery" in report
    assert "bootstrap_oracle16_vs_primary_comparator" in report
    assert "gate_checks" in report
    markdown = "\n".join(
        _source(cell)
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "markdown"
    ).casefold()
    assert "development-only" in markdown
    assert "non-deployable" in markdown
    assert "not a persistent-storage result" in markdown
    assert "not independent confirmation" in markdown


def test_v3_notebook_never_trains_or_accesses_external_outcomes() -> None:
    code = _code().casefold()
    for forbidden in (
        "train_boundary_loss_sidecar_v2_2.py",
        "train_select_beir_nq_sidecars.py",
        "evaluate_beir",
        "evaluate_trec",
        "qat",
        "learned_allocator",
        "static_storage_oracle.py",
    ):
        assert forbidden not in code
    audit = _source(_cells()["v3-audit-phase"])
    assert "inner_validation" not in audit
    assert "oracle_audit" in audit
