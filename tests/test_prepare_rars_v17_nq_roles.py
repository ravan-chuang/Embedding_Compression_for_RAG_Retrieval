from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_rars_v17_nq_roles.py"
SPEC = importlib.util.spec_from_file_location("prepare_rars_v17_nq", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": MODULE.sha256_file(path),
    }


def _small_protocol() -> dict[str, object]:
    return {
        "data_policy": {
            "minimum_fit_queries_per_domain": 2,
            "minimum_evaluation_queries_per_domain": 2,
        },
        "index_policy": {
            "common": {
                "subquantizers": 32,
                "bits_per_subquantizer": 8,
            },
            MODULE.SETTING_ID: {
                "minimum_documents": 2_000_000,
                "nlist": 2048,
                "nprobe": 32,
            },
        },
    }


def test_deterministic_roles_are_order_independent_and_exact_60_40() -> None:
    qids = [f"q-{index:04d}" for index in range(3452)]
    first = MODULE.deterministic_roles(qids)
    second = MODULE.deterministic_roles(list(reversed(qids)))

    assert first == second
    assert len(first["fit"]) == 2071
    assert len(first["evaluation"]) == 1381
    assert set(first["fit"]).isdisjoint(first["evaluation"])
    assert set(first["fit"]) | set(first["evaluation"]) == set(qids)
    expected = sorted(
        qids,
        key=lambda qid: (
            hashlib.sha256(
                b"rars-v17-nq-role-v1\0" + qid.encode("utf-8")
            ).digest(),
            qid,
        ),
    )
    assert first["fit"] == expected[:2071]
    assert first["evaluation"] == expected[2071:]


def test_document_id_mapping_handles_fixed_width_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doc_ids.utf8.memmap"
    values = np.memmap(path, mode="w+", dtype="S8", shape=(5,))
    values[:] = np.asarray(
        [b"alpha", b"2", b"doc-3", b"unused", b"z"],
        dtype="S8",
    )
    values.flush()
    del values

    rows = MODULE.map_document_rows(
        path,
        document_count=5,
        width=8,
        required_ids={"alpha", "2", "doc-3"},
        batch_size=2,
    )

    assert rows == {"alpha": 0, "2": 1, "doc-3": 2}


def test_document_id_mapping_rejects_missing_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doc_ids.utf8.memmap"
    values = np.memmap(path, mode="w+", dtype="S4", shape=(3,))
    values[:] = np.asarray([b"x", b"x", b"z"], dtype="S4")
    values.flush()
    del values

    with pytest.raises(ValueError, match="duplicate document ID"):
        MODULE.map_document_rows(
            path,
            document_count=3,
            width=4,
            required_ids={"x"},
        )
    with pytest.raises(ValueError, match="absent from the frozen corpus"):
        MODULE.map_document_rows(
            path,
            document_count=3,
            width=4,
            required_ids={"y"},
        )


def _write_doc_id_reconciliation_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "artifact"
    doc_ids_path = artifact / "stage1/corpus/doc_ids.utf8.memmap"
    doc_ids_path.parent.mkdir(parents=True)
    doc_ids_path.write_bytes(b"doc-0\0\0\0doc-1\0\0\0")
    corpus_manifest_path = (
        artifact / "stage1/corpus/corpus_artifacts_manifest.json"
    )
    stale_record = {
        "path": str(doc_ids_path),
        "bytes": doc_ids_path.stat().st_size,
        "sha256": "0" * 64,
    }
    _write_json(
        corpus_manifest_path,
        {
            "protocol_id": MODULE.SOURCE_PROTOCOL_ID,
            "doc_ids": stale_record,
        },
    )
    pre_qrels_path = artifact / MODULE.PRE_QRELS_MANIFEST
    _write_json(
        pre_qrels_path,
        {
            "protocol_id": MODULE.SOURCE_PROTOCOL_ID,
            "status": "frozen_before_test_qrels_access",
            "test_qrels_accessed": False,
            "test_retrieval_performed": False,
            "test_outcomes_observed": False,
            "train_qrels_relevance_values_used": False,
            "files": {
                "corpus_manifest": _record(corpus_manifest_path),
                "doc_ids": _record(doc_ids_path),
            },
        },
    )
    audit_path = artifact / MODULE.ELIGIBLE_QUERY_AUDIT
    _write_json(
        audit_path,
        {
            "protocol_id": MODULE.SOURCE_PROTOCOL_ID,
            "status": "eligible_test_queries_frozen_before_retrieval",
            "pre_qrels_manifest_sha256": MODULE.sha256_file(
                pre_qrels_path
            ),
        },
    )
    return artifact, corpus_manifest_path, audit_path


def test_pre_qrels_doc_ids_reconcile_a_stale_stage1_digest(
    tmp_path: Path,
) -> None:
    artifact, corpus_manifest_path, _ = (
        _write_doc_id_reconciliation_fixture(tmp_path)
    )
    corpus = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))

    path, lineage = MODULE._verify_pre_qrels_doc_id_reconciliation(
        artifact,
        corpus_manifest_path=corpus_manifest_path,
        corpus_doc_ids_record=corpus["doc_ids"],
        expected_bytes=16,
    )

    assert path == artifact / "stage1/corpus/doc_ids.utf8.memmap"
    assert lineage["status"] == (
        "PRE_QRELS_DOC_IDS_VERIFIED_AFTER_STAGE1_HASH_MISMATCH"
    )
    assert lineage["historical_artifacts_modified"] is False
    assert (
        lineage["authoritative_pre_qrels_doc_ids"]["sha256"]
        == MODULE.sha256_file(path)
    )


def test_pre_qrels_doc_id_reconciliation_requires_audit_binding(
    tmp_path: Path,
) -> None:
    artifact, corpus_manifest_path, audit_path = (
        _write_doc_id_reconciliation_fixture(tmp_path)
    )
    corpus = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["pre_qrels_manifest_sha256"] = "f" * 64
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="does not bind"):
        MODULE._verify_pre_qrels_doc_id_reconciliation(
            artifact,
            corpus_manifest_path=corpus_manifest_path,
            corpus_doc_ids_record=corpus["doc_ids"],
            expected_bytes=16,
        )


def test_stage1_rejects_sub_two_million_corpus_before_opening_payloads(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    corpus_path = artifact / "stage1/corpus/corpus_artifacts_manifest.json"
    index_path = artifact / "stage1/index/index_manifest.json"
    _write_json(
        corpus_path,
        {
            "protocol_id": MODULE.SOURCE_PROTOCOL_ID,
            "test_qrels_accessed": False,
            "document_count": 1_999_999,
            "dimension": 384,
            "embedding_dtype": "float16",
            "doc_id_width_bytes": 8,
            "doc_id_dtype": "S8",
        },
    )
    _write_json(
        index_path,
        {
            "protocol_id": MODULE.SOURCE_PROTOCOL_ID,
            "test_qrels_accessed": False,
        },
    )

    with pytest.raises(ValueError, match="at least 2,000,000"):
        MODULE._validate_stage1(artifact, _small_protocol())


def test_prepare_preserves_audit_vector_alignment_and_writes_no_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "prepared"
    qids = ["q0", "q1", "q2", "q3", "q4"]
    vectors = np.repeat(
        np.arange(len(qids), dtype=np.float32)[:, None],
        MODULE.DIMENSION,
        axis=1,
    )
    source_files = {
        name: artifact / name
        for name in (
            "embeddings.bin",
            "doc_ids.bin",
            "index.bin",
            "corpus_manifest.json",
            "index_manifest.json",
            "audit.json",
            "summary.json",
            "complete.json",
            "vectors.npy",
            "per_query.csv",
            "qrels.tsv",
        )
    }
    embeddings = np.memmap(
        source_files["embeddings.bin"],
        mode="w+",
        dtype=np.float16,
        shape=(6, MODULE.DIMENSION),
    )
    embeddings[:] = 0
    embeddings.flush()
    del embeddings
    doc_ids = np.memmap(
        source_files["doc_ids.bin"],
        mode="w+",
        dtype="S4",
        shape=(6,),
    )
    doc_ids[:] = np.asarray(
        [b"d0", b"d1", b"d2", b"d3", b"d4", b"d5"],
        dtype="S4",
    )
    doc_ids.flush()
    del doc_ids
    source_files["index.bin"].write_bytes(b"index")
    for name in (
        "corpus_manifest.json",
        "index_manifest.json",
        "audit.json",
        "summary.json",
        "complete.json",
    ):
        source_files[name].write_text("{}\n", encoding="utf-8")
    _write_npy(source_files["vectors.npy"], vectors)
    source_files["per_query.csv"].write_text(
        "query_id,value\n"
        + "".join(f"{qid},0\n" for qid in qids),
        encoding="utf-8",
    )
    source_files["qrels.tsv"].write_text(
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"{qid}\td{index}\t1\n" for index, qid in enumerate(qids)),
        encoding="utf-8",
    )

    stage1 = {
        "document_count": 6,
        "dimension": MODULE.DIMENSION,
        "width": 4,
        "embeddings": source_files["embeddings.bin"],
        "doc_ids": source_files["doc_ids.bin"],
        "index": source_files["index.bin"],
        "corpus_manifest": source_files["corpus_manifest.json"],
        "index_manifest": source_files["index_manifest.json"],
        "corpus": {
            "embedding_model_snapshot_manifest": {
                "sha256": "a" * 64,
            },
        },
        "index_payload": {},
    }
    stage3 = {
        "qids": qids,
        "vectors": vectors,
        "vectors_path": source_files["vectors.npy"],
        "qrels_path": source_files["qrels.tsv"],
        "audit_path": source_files["audit.json"],
        "summary_path": source_files["summary.json"],
        "complete_path": source_files["complete.json"],
        "per_query_path": source_files["per_query.csv"],
    }
    monkeypatch.setattr(
        MODULE,
        "verify_source",
        lambda *args: (_small_protocol(), {}),
    )
    monkeypatch.setattr(MODULE, "_validate_stage1", lambda *args: stage1)
    monkeypatch.setattr(MODULE, "_validate_stage3", lambda *args: stage3)

    result = MODULE.prepare(
        argparse.Namespace(
            artifact_root=artifact,
            output_root=output,
            protocol=tmp_path / "protocol.json",
            source_commit="b" * 40,
        )
    )

    assert result["status"] == MODULE.STATUS
    assert result["metrics_computed"] is False
    assert result["sidecar_basis_fitted"] is False
    assert result["prior_confirmation_outcomes_known"] is True
    prepared = json.loads(
        (
            output
            / MODULE.SETTING_ID
            / "prepared_domain.json"
        ).read_text(encoding="utf-8")
    )
    for role in ("fit", "evaluation"):
        role_dir = output / MODULE.SETTING_ID / role
        role_qids = (
            role_dir / "query_ids.utf8.txt"
        ).read_text(encoding="utf-8").splitlines()
        role_vectors = np.load(
            role_dir / "query_vectors.float32.npy",
            allow_pickle=False,
        )
        assert np.array_equal(
            role_vectors[:, 0],
            np.asarray([qids.index(qid) for qid in role_qids]),
        )
        qrels_rows = json.loads(
            (role_dir / "qrels_rows.json").read_text(encoding="utf-8")
        )
        assert set(qrels_rows) == set(role_qids)
    assert prepared["retrieval_performed"] is False
    assert prepared["metrics_computed"] is False
    assert prepared["sidecar_basis_fitted"] is False


def test_resume_requires_every_registered_output_to_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build one valid small packet through the public preparation entry point.
    test_prepare_preserves_audit_vector_alignment_and_writes_no_metrics(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "prepared"
    fit_qids = (
        output / MODULE.SETTING_ID / "fit/query_ids.utf8.txt"
    )
    fit_qids.write_text(
        fit_qids.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="byte count changed|hash changed"):
        MODULE.verify_complete(output, source_commit="b" * 40)
