from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_rars_v17_setting_manifest.py"
SPEC = importlib.util.spec_from_file_location("freeze_rars_v17_setting_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PROTOCOL = json.loads(
    (ROOT / "protocols/rars_v17_million_scale_setting_transfer_v1.json").read_text(
        encoding="utf-8"
    )
)


def _bundle(
    root: Path,
    *,
    setting_id: str,
    role: str,
    query_count: int,
    document_count: int,
    nlist: int,
    nprobe: int,
    revision: str,
    query_namespace: str | None = None,
) -> Path:
    path = root / setting_id / role
    path.mkdir(parents=True)
    namespace = query_namespace or f"{setting_id}:{role}"
    query_ids = [f"{namespace}:{index}" for index in range(query_count)]
    (path / "query_ids.utf8.txt").write_text(
        "\n".join(query_ids) + "\n", encoding="utf-8"
    )
    payload = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "status": "RARS_V17_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS",
        "setting_id": setting_id,
        "domain_id": setting_id,
        "evidence_role": role,
        "encoder_id": "BAAI/bge-small-en-v1.5",
        "encoder_revision": revision,
        "dimension": 384,
        "query_count": query_count,
        "document_count": document_count,
        "opened_nq_test_evidence": setting_id == MODULE.NQ_SETTING,
        "index_contract": {
            "dimension": 384,
            "document_count": document_count,
            "nlist": nlist,
            "nprobe": nprobe,
            "subquantizers": 32,
            "bits_per_subquantizer": 8,
            "metric_type": 0,
        },
    }
    (path / "bundle_manifest.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    return path


def test_v17_freezer_accepts_preregistered_different_index_recipes(
    tmp_path: Path, monkeypatch
) -> None:
    msm_fit = _bundle(
        tmp_path,
        setting_id=MODULE.MSMARCO_SETTING,
        role="fit",
        query_count=1000,
        document_count=1_000_000,
        nlist=512,
        nprobe=16,
        revision="legacy",
    )
    msm_eval = _bundle(
        tmp_path,
        setting_id=MODULE.MSMARCO_SETTING,
        role="evaluation",
        query_count=800,
        document_count=1_000_000,
        nlist=512,
        nprobe=16,
        revision="legacy",
    )
    nq_fit = _bundle(
        tmp_path,
        setting_id=MODULE.NQ_SETTING,
        role="fit",
        query_count=2071,
        document_count=2_681_468,
        nlist=2048,
        nprobe=32,
        revision="pinned-nq",
    )
    nq_eval = _bundle(
        tmp_path,
        setting_id=MODULE.NQ_SETTING,
        role="evaluation",
        query_count=1381,
        document_count=2_681_468,
        nlist=2048,
        nprobe=32,
        revision="pinned-nq",
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_source",
        lambda _root, _commit: (PROTOCOL, {"test": {"sha256": "x"}}),
    )
    output = tmp_path / "setting_manifest.json"
    result = MODULE.freeze(
        argparse.Namespace(
            msmarco_fit=msm_fit,
            msmarco_evaluation=msm_eval,
            nq_fit=nq_fit,
            nq_evaluation=nq_eval,
            output=output,
            source_commit="0" * 40,
        )
    )
    assert result["status"] == "V17_DOMAIN_BUNDLES_FROZEN"
    assert result["minimum_document_count_verified"] is True
    assert result["nq_prior_opened_test_artifact_reused"] is True
    assert set(result["index_recipes_by_setting"]) == {
        MODULE.MSMARCO_SETTING,
        MODULE.NQ_SETTING,
    }
    assert result["index_recipes_by_setting"][MODULE.MSMARCO_SETTING] != (
        result["index_recipes_by_setting"][MODULE.NQ_SETTING]
    )
    assert result["exact_encoder_revision_match_claimed"] is False


def test_v17_freezer_cli_uses_msmarco_and_nq_roles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for argument in (
        "--msmarco-fit",
        "--msmarco-evaluation",
        "--nq-fit",
        "--nq-evaluation",
    ):
        assert f'parser.add_argument("{argument}"' in source
    assert "--fiqa-fit" not in source
    assert "--scifact-fit" not in source
    assert "fit/evaluation query IDs overlap" in source


def test_v17_bundle_validation_rejects_query_and_document_minima(
    tmp_path: Path,
) -> None:
    too_few_queries = _bundle(
        tmp_path / "queries",
        setting_id=MODULE.MSMARCO_SETTING,
        role="fit",
        query_count=999,
        document_count=1_000_000,
        nlist=512,
        nprobe=16,
        revision="legacy",
    )
    with pytest.raises(ValueError, match="requires 1000"):
        MODULE._load_bundle(
            too_few_queries,
            setting_id=MODULE.MSMARCO_SETTING,
            role="fit",
            protocol=PROTOCOL,
        )

    too_few_documents = _bundle(
        tmp_path / "documents",
        setting_id=MODULE.NQ_SETTING,
        role="evaluation",
        query_count=1381,
        document_count=1_999_999,
        nlist=2048,
        nprobe=32,
        revision="pinned-nq",
    )
    with pytest.raises(ValueError, match="requires 2000000"):
        MODULE._load_bundle(
            too_few_documents,
            setting_id=MODULE.NQ_SETTING,
            role="evaluation",
            protocol=PROTOCOL,
        )


def test_v17_freezer_rejects_fit_evaluation_query_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    msm_fit = _bundle(
        tmp_path,
        setting_id=MODULE.MSMARCO_SETTING,
        role="fit",
        query_count=1000,
        document_count=1_000_000,
        nlist=512,
        nprobe=16,
        revision="legacy",
        query_namespace="overlap",
    )
    msm_eval = _bundle(
        tmp_path,
        setting_id=MODULE.MSMARCO_SETTING,
        role="evaluation",
        query_count=800,
        document_count=1_000_000,
        nlist=512,
        nprobe=16,
        revision="legacy",
        query_namespace="overlap",
    )
    nq_fit = _bundle(
        tmp_path,
        setting_id=MODULE.NQ_SETTING,
        role="fit",
        query_count=2071,
        document_count=2_681_468,
        nlist=2048,
        nprobe=32,
        revision="pinned-nq",
    )
    nq_eval = _bundle(
        tmp_path,
        setting_id=MODULE.NQ_SETTING,
        role="evaluation",
        query_count=1381,
        document_count=2_681_468,
        nlist=2048,
        nprobe=32,
        revision="pinned-nq",
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_source",
        lambda _root, _commit: (PROTOCOL, {"test": {"sha256": "x"}}),
    )
    with pytest.raises(ValueError, match="fit/evaluation query IDs overlap"):
        MODULE.freeze(
            argparse.Namespace(
                msmarco_fit=msm_fit,
                msmarco_evaluation=msm_eval,
                nq_fit=nq_fit,
                nq_evaluation=nq_eval,
                output=tmp_path / "manifest.json",
                source_commit="0" * 40,
            )
        )
