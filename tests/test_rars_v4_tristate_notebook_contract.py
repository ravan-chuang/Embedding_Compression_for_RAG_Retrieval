from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/MSMARCO_RARS_v4_Tristate_Action_Feasibility.ipynb"
TRAINING_COMMIT = "bb9b106e69b9a453756fd800665f701614ce67b3"
V3_IMPLEMENTATION_COMMIT = "05c2ae43b7d11783460822d10c590240dab1a399"
V4_IMPLEMENTATION_COMMIT = "bbdf8656881fd32d1961610d1c0b5c6d989fcc7a"
EXPECTED_IDS = [
    "v4-title",
    "v4-context",
    "v4-setup",
    "v4-checkout-tests",
    "v4-preflight",
    "v4-data",
    "v4-parent-rematerialize",
    "v4-candidate-freeze",
    "v4-design-labels",
    "v4-design-phase",
    "v4-results",
    "v4-audit-phase",
    "v4-final-report",
    "v4-takeaways",
]


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _cells() -> dict[str, dict]:
    return {cell["id"]: cell for cell in _notebook()["cells"]}


def _code() -> str:
    return "\n".join(
        _source(cell)
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )


def test_v4_notebook_is_clean_ordered_and_pinned() -> None:
    notebook = _notebook()
    assert [cell["id"] for cell in notebook["cells"]] == EXPECTED_IDS
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    code = _code()
    for name, commit in (
        ("TRAINING_COMMIT", TRAINING_COMMIT),
        ("V3_IMPLEMENTATION_COMMIT", V3_IMPLEMENTATION_COMMIT),
        ("V4_IMPLEMENTATION_COMMIT", V4_IMPLEMENTATION_COMMIT),
    ):
        assert f"{name} = '{commit}'" in code
        assert len(commit) == 40
        int(commit, 16)
    metadata = notebook["metadata"]
    assert metadata["accelerator"] == "GPU"
    assert metadata["colab"]["gpuType"] == "T4"
    assert metadata["kernelspec"]["name"] == "python3"


def test_v4_notebook_uses_isolated_numpy_subprocess() -> None:
    cells = _cells()
    setup = _source(cells["v4-setup"])
    preflight = _source(cells["v4-preflight"])
    assert "numpy==1.26.4" in setup
    assert "'--target', str(NUMPY_TARGET)" in setup
    assert "EXPERIMENT_ENV['PYTHONPATH']" in setup
    assert "NUMPY_TARGET.resolve()" in setup
    assert "env=EXPERIMENT_ENV" in preflight
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
        "v4-checkout-tests",
        "v4-preflight",
        "v4-parent-rematerialize",
        "v4-candidate-freeze",
        "v4-design-labels",
        "v4-design-phase",
        "v4-audit-phase",
    ):
        assert "env=EXPERIMENT_ENV" in _source(cells[cell_id])


def test_v4_notebook_reuses_exact_v3_closure_and_parent_lineage() -> None:
    cells = _cells()
    preflight = _source(cells["v4-preflight"])
    for filename in (
        "oracle_complete.json",
        "oracle_summary.json",
        "design_freeze.json",
        "progressive_svd_rank32.float32.npy",
        "progressive_svd_rank32_scales.float32.npy",
    ):
        assert filename in preflight
    parent = _source(cells["v4-parent-rematerialize"])
    assert "'--inner-only'" in parent
    assert "'--qrels'" in parent
    assert "'--index'" in parent
    for field in (
        "parent_inner_train_manifest_sha256",
        "parent_inner_train_source_manifest_sha256",
        "parent_inner_train_query_manifest_sha256",
        "closed_inner_validation_query_manifest_sha256",
        "parent_v2_2_split_audit_sha256",
    ):
        assert field in parent


def test_v4_notebook_preserves_tristate_semantics() -> None:
    design_labels = _source(_cells()["v4-design-labels"])
    assert "materialize_rars_v4_tristate_labels.py" in design_labels
    assert "qrels_subset.json" in design_labels
    assert "binary_candidate_relevance_read" in design_labels
    assert "missing_rows_interpreted_as_explicit_negative" in design_labels
    assert "'v4_design_observed'" in design_labels
    assert "'--role', 'v4_diagnostic_audit'" not in design_labels
    assert "not AUDIT_LABEL_DIR.exists()" in design_labels


def test_v4_notebook_enforces_design_then_conditional_audit() -> None:
    cells = _notebook()["cells"]
    positions = {cell["id"]: index for index, cell in enumerate(cells)}
    assert positions["v4-design-labels"] < positions["v4-design-phase"]
    assert positions["v4-design-phase"] < positions["v4-audit-phase"]
    design = _source(_cells()["v4-design-phase"])
    audit = _source(_cells()["v4-audit-phase"])
    assert "RUN_DIAGNOSTIC_AUDIT" in design
    assert "DESIGN_GO_TO_DIAGNOSTIC_AUDIT" in design
    assert audit.startswith("if RUN_DIAGNOSTIC_AUDIT:")
    assert "'--design-freeze'" in audit
    assert "assert not AUDIT_LABEL_DIR.exists()" in audit
    assert "'v4_diagnostic_audit'" in audit


def test_v4_notebook_keeps_future_role_identity_only() -> None:
    candidate = _source(_cells()["v4-candidate-freeze"])
    assert "future_method_holdout" in candidate
    assert "candidate_arrays_created'] is False" in candidate
    assert "future_files == {'query_manifest.json', 'v3_identity_manifest.json'}" in candidate
    code = _code()
    assert "V3_BUNDLES / 'future_method_holdout' / 'candidate" not in code
    assert "V3_BUNDLES / 'future_method_holdout' / 'v4" not in code
    assert "future_method_holdout_accessed'] is False" in code


def test_v4_notebook_preserves_durable_output_and_verifies_records() -> None:
    code = _code()
    assert "shutil.rmtree(OUTPUT)" not in code
    assert code.count("'--reuse-complete'") == 2
    assert (
        "OUTPUT = DRIVE / 'rars-v4-tristate-action-feasibility' / V4_IMPLEMENTATION_COMMIT[:12]"
        in code
    )
    report = _source(_cells()["v4-final-report"])
    assert "for relative_name, record in marker['registered_outputs'].items()" in report
    assert "not relative.is_absolute() and '..' not in relative.parts" in report
    assert "marker['run_fingerprint'] == final_summary['run_fingerprint']" in report


def test_v4_notebook_never_trains_or_opens_external_outcomes() -> None:
    code = _code().casefold()
    for forbidden in (
        "train_boundary_loss_sidecar_v2_2.py",
        "train_select_beir_nq_sidecars.py",
        "evaluate_beir",
        "evaluate_trec",
        "learned_allocator",
        "optimizer.step",
        "torch.optim",
    ):
        assert forbidden not in code
    markdown = "\n".join(
        _source(cell)
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "markdown"
    ).casefold()
    assert "external confirmation" in markdown
    assert "no phase-0 result is training" in markdown
    assert "unjudged" in markdown
    assert "explicit non-relevant" in markdown
