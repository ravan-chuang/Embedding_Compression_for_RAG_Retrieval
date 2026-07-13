from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_nq_pre_qrels_freeze.py"
PROTOCOL_PATH = ROOT / "protocols" / "beir_nq_rars_pca_confirmation_v1.json"
TEMPLATE_PATH = ROOT / "protocols" / "beir_nq_pre_qrels_manifest.template.json"

SPEC = importlib.util.spec_from_file_location("nq_freeze", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_manifest() -> dict:
    manifest = load(TEMPLATE_PATH)
    manifest["status"] = "frozen_before_test_qrels_access"
    manifest["protocol_sha256"] = MODULE.sha256_file(PROTOCOL_PATH)
    manifest["corpus_document_count"] = 2_680_000
    manifest["train_query_count"] = 50_000
    manifest["validation_query_count"] = 5_000
    manifest["selected_configs"]["pca"].update(alpha=0.75, top_b=40)
    manifest["selected_configs"]["rars"].update(alpha=0.75, top_b=40)
    for entry in manifest["files"].values():
        entry["path"] = "artifact.bin"
        entry["bytes"] = 1
        entry["sha256"] = "a" * 64
    return manifest


def test_real_protocol_and_draft_template_validate_without_qrels() -> None:
    protocol = load(PROTOCOL_PATH)
    draft = load(TEMPLATE_PATH)

    MODULE.validate_protocol(protocol)
    result = MODULE.validate_manifest(
        draft,
        protocol=protocol,
        protocol_path=PROTOCOL_PATH,
        allow_draft=True,
    )
    assert result["test_qrels_accessed"] is False
    assert result["test_outcomes_observed"] is False


def test_frozen_manifest_accepts_only_registered_artifacts_and_grid() -> None:
    protocol = load(PROTOCOL_PATH)
    result = MODULE.validate_manifest(
        frozen_manifest(),
        protocol=protocol,
        protocol_path=PROTOCOL_PATH,
    )
    assert result["artifact_count"] == len(MODULE.EXPECTED_FILE_LABELS)
    assert result["files_verified"] is False


def test_manifest_rejects_test_qrels_or_outcome_artifact() -> None:
    protocol = load(PROTOCOL_PATH)
    manifest = frozen_manifest()
    manifest["files"]["test_qrels"] = {
        "path": "qrels/test.tsv",
        "bytes": 1,
        "sha256": "b" * 64,
    }

    with pytest.raises(ValueError, match="extra=.*test_qrels"):
        MODULE.validate_manifest(
            manifest,
            protocol=protocol,
            protocol_path=PROTOCOL_PATH,
        )


def test_manifest_rejects_qrels_hidden_under_registered_label() -> None:
    protocol = load(PROTOCOL_PATH)
    manifest = frozen_manifest()
    manifest["files"]["dataset_archive"]["path"] = "qrels/test.tsv"

    with pytest.raises(ValueError, match="forbidden pre-qrels path"):
        MODULE.validate_manifest(
            manifest,
            protocol=protocol,
            protocol_path=PROTOCOL_PATH,
        )


def test_manifest_rejects_unregistered_outcome_field() -> None:
    protocol = load(PROTOCOL_PATH)
    manifest = frozen_manifest()
    manifest["metrics"] = {"recall@10": 1.0}

    with pytest.raises(ValueError, match="extra=.*metrics"):
        MODULE.validate_manifest(
            manifest,
            protocol=protocol,
            protocol_path=PROTOCOL_PATH,
        )


def test_manifest_rejects_any_test_access_flag() -> None:
    protocol = load(PROTOCOL_PATH)
    manifest = frozen_manifest()
    manifest["test_qrels_accessed"] = True

    with pytest.raises(ValueError, match="test_qrels_accessed"):
        MODULE.validate_manifest(
            manifest,
            protocol=protocol,
            protocol_path=PROTOCOL_PATH,
        )


def test_protocol_drift_is_rejected() -> None:
    protocol = copy.deepcopy(load(PROTOCOL_PATH))
    protocol["base_index"]["nprobe"] = 64

    with pytest.raises(ValueError, match="base_index.nprobe"):
        MODULE.validate_protocol(protocol)
