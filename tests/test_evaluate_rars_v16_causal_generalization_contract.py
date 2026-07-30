from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_rars_v16_causal_generalization.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("evaluate_rars_v16", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v16_cli_is_manifest_driven_and_corpus_free() -> None:
    for argument in (
        "--domain-manifest",
        "--output-dir",
        "--source-commit",
    ):
        assert f'parser.add_argument("{argument}"' in SOURCE
    assert '"--protocol"' in SOURCE
    for forbidden in (
        "--embeddings",
        "--doc-ids",
        "--index",
        "--qrels",
        "--future-method-holdout",
        "--oracle-audit",
    ):
        assert forbidden not in SOURCE


def test_v16_runs_all_six_preregistered_factor_contrasts() -> None:
    for contrast in (
        '"candidate_headroom"',
        '"rank_capacity"',
        '"int8_coding_loss"',
        '"objective_value"',
        '"fit_domain_interaction"',
        '"pooled_repair"',
    ):
        assert contrast in SOURCE
    for method in (
        '"same_candidate_exact"',
        '"pca_local_r16_fp32"',
        '"pca_local_r16_int8"',
        '"pca_local_r64_fp32"',
        '"rars_source_r16_int8"',
        '"rars_local_r16_int8"',
        '"rars_pooled_r16_int8"',
    ):
        assert method in SOURCE


def test_v16_uses_v8_cutoff_pairs_and_storage_matched_int8() -> None:
    for symbol in (
        "mine_cutoff_pairs",
        "fit_cutoff_aware_basis",
        "query_role_balanced_weights",
        "fit_int8_scales",
        "encode_residuals_int8",
        "score_sidecar_candidates",
    ):
        assert symbol in SOURCE
    assert '"scales_fit_without_labels": True' in SOURCE
    assert '"evaluation_used_for_selection": False' in SOURCE
    assert '"confirmatory_claim_allowed": False' in SOURCE


def test_v16_refuses_closed_paths_and_nonempty_output(tmp_path: Path) -> None:
    for value in (
        tmp_path / "future_method_holdout",
        tmp_path / "oracle_audit",
        tmp_path / "beir_nq_confirmation",
        tmp_path / "trec_dl_2019",
    ):
        with pytest.raises(ValueError, match="forbidden closed-test"):
            MODULE._reject_forbidden_closed_path(value, "fixture")
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        MODULE._prepare_empty_output(output)


def _write_npy(path: Path, value: np.ndarray) -> dict[str, object]:
    np.save(path, value)
    return MODULE.file_record(path)


def test_v16_bundle_loader_checks_hash_shape_encoder_and_role(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "fiqa_bge_same_encoder" / "fit"
    bundle.mkdir(parents=True)
    qids = bundle / "query_ids.utf8.txt"
    qids.write_text("q1\nq2\n", encoding="utf-8")
    files: dict[str, dict[str, object]] = {
        "query_ids.utf8.txt": MODULE.file_record(qids)
    }
    files["fold_ids.int64.npy"] = _write_npy(
        bundle / "fold_ids.int64.npy", np.asarray([0, 1], dtype=np.int64)
    )
    files["query_vectors.float32.npy"] = _write_npy(
        bundle / "query_vectors.float32.npy", np.eye(2, dtype=np.float32)
    )
    rows = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    files["ann_rows.int64.npy"] = _write_npy(bundle / "ann_rows.int64.npy", rows)
    files["ann_scores.float32.npy"] = _write_npy(
        bundle / "ann_scores.float32.npy",
        np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
    )
    files["ann_residual_rows.int64.npy"] = _write_npy(
        bundle / "ann_residual_rows.int64.npy", rows
    )
    files["candidate_residuals.float32.npy"] = _write_npy(
        bundle / "candidate_residuals.float32.npy",
        np.asarray([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
    )
    files["candidate_relevance.uint8.npy"] = _write_npy(
        bundle / "candidate_relevance.uint8.npy",
        np.asarray([[1, 0], [1, 0]], dtype=np.uint8),
    )
    files["relevant_counts.int32.npy"] = _write_npy(
        bundle / "relevant_counts.int32.npy",
        np.asarray([1, 1], dtype=np.int32),
    )
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": MODULE.PROTOCOL_ID,
                "status": "RARS_V16_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS",
                "domain_id": "fiqa_bge_same_encoder",
                "evidence_role": "fit",
                "encoder_id": "encoder",
                "encoder_revision": "revision",
                "dimension": 2,
                "query_count": 2,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    loaded = MODULE.load_bundle(
        bundle,
        expected_domain="fiqa_bge_same_encoder",
        expected_role="fit",
    )
    assert loaded["encoder_key"] == ("encoder", "revision", 2)
    np.save(bundle / "query_vectors.float32.npy", np.ones((2, 2), np.float32))
    with pytest.raises(ValueError, match="hash changed"):
        MODULE.load_bundle(
            bundle,
            expected_domain="fiqa_bge_same_encoder",
            expected_role="fit",
        )


def test_v16_writes_diagnostic_packet_and_per_query_arrays() -> None:
    for name in (
        "diagnostic_started.json",
        "diagnostic_result.json",
        "diagnostic_complete.json",
        'f"{safe_domain}__{method}__{metric}_at_"',
    ):
        assert name in SOURCE
