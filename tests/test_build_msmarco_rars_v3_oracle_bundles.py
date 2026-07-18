from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _load(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load("build_rars_v3", "build_msmarco_rars_v3_oracle_bundles.py")
MATERIALIZER = _load("materialize_rars_v3", "materialize_rars_v3_role_labels.py")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": MODULE.sha256_file(path),
    }


def _make_parent_bundle(tmp_path: Path) -> tuple[Path, Path, dict[str, object], list[str]]:
    parent = tmp_path / "parent"
    parent.mkdir()
    qids = [str(value) for value in range(1, 41)]
    rows = np.arange(len(qids), dtype=np.int64)
    candidate_count = 4
    ann_rows = np.asarray(
        [[index % 8, (index + 1) % 8, (index + 2) % 8, (index + 3) % 8]
         for index in range(len(qids))],
        dtype=np.int64,
    )
    unique_rows = np.arange(8, dtype=np.int64)
    arrays = {
        "query_vectors.float32.npy": np.arange(len(qids) * 3, dtype=np.float32).reshape(len(qids), 3),
        "ann_rows.int64.npy": ann_rows,
        "ann_scores.float32.npy": np.linspace(1.0, 0.0, len(qids) * candidate_count, dtype=np.float32).reshape(len(qids), candidate_count),
        "candidate_relevance.uint8.npy": ((ann_rows + rows[:, None]) % 3 == 0).astype(np.uint8),
        "relevant_counts.int32.npy": np.full(len(qids), 2, dtype=np.int32),
        "candidate_doc_rows.int64.npy": unique_rows,
        "ann_residual_rows.int64.npy": ann_rows.copy(),
        "pca_scores.float32.npy": np.linspace(1.5, 0.5, len(qids) * candidate_count, dtype=np.float32).reshape(len(qids), candidate_count),
        "rars_scores.float32.npy": np.zeros((len(qids), candidate_count), dtype=np.float32),
        "candidate_residuals.float32.npy": np.arange(8 * 3, dtype=np.float32).reshape(8, 3),
    }
    for filename, value in arrays.items():
        np.save(parent / filename, value)
    query_manifest_path = parent / "query_manifest.json"
    _write_json(
        query_manifest_path,
        {"role_id": "inner_train", "query_ids": qids, "query_rows": rows.tolist()},
    )
    files = {filename: _file_record(parent / filename) for filename in arrays}
    source_manifest_path = parent / "manifest.json"
    _write_json(
        source_manifest_path,
        {
            "protocol_id": "rars_v2_boundary_loss_feasibility_v1",
            "role_id": "inner_train",
            "files": files,
        },
    )
    parent_manifest_path = parent / "v2_2_manifest.json"
    parent_manifest = {
        "schema_version": 1,
        "protocol_id": "rars_v2_2_boundary_loss_development_v1",
        "source_commit": "b" * 40,
        "role_id": "inner_train",
        "query_count": len(qids),
        "candidate_count": candidate_count,
        "query_ids_sha256": MODULE.canonical_sha256(qids),
        "query_rows_sha256": MODULE.array_sha256(rows),
        "source_bundle_manifest_sha256": MODULE.sha256_file(source_manifest_path),
        "query_manifest": _file_record(query_manifest_path),
        "files": files,
    }
    _write_json(parent_manifest_path, parent_manifest)
    doc_ids_path = tmp_path / "doc_ids.int64.memmap"
    np.arange(100, 108, dtype=np.int64).tofile(doc_ids_path)

    role_indices = MODULE.split_development_qids(qids)
    roles = {}
    for role_id, indices in role_indices.items():
        role_qids = [qids[int(index)] for index in indices]
        roles[role_id] = {
            "query_count": len(role_qids),
            "source_order_newline_qid_sha256": MODULE._newline_sha256(role_qids),
            "numeric_sorted_newline_qid_sha256": MODULE._numeric_sorted_newline_sha256(role_qids),
        }
    protocol = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "status": "FROZEN_BEFORE_FIRST_ORACLE_RUN",
        "parent_lineage": {
            "parent_training_commit": "b" * 40,
            "parent_v2_2_protocol_id": "rars_v2_2_boundary_loss_development_v1",
            "parent_inner_train_manifest_sha256": MODULE.sha256_file(parent_manifest_path),
            "parent_inner_train_source_manifest_sha256": MODULE.sha256_file(source_manifest_path),
            "parent_inner_train_query_manifest_sha256": MODULE.sha256_file(query_manifest_path),
            "frozen_doc_ids_sha256": MODULE.sha256_file(doc_ids_path),
        },
        "data_policy": {
            "source_pool": {"query_count": len(qids)},
            "roles": roles,
        },
    }
    return parent, doc_ids_path, protocol, qids


def _build_toy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object], list[str], argparse.Namespace]:
    parent, doc_ids, protocol, qids = _make_parent_bundle(tmp_path)
    monkeypatch.setattr(MODULE, "_validate_canonical_repository", lambda *args: protocol)
    monkeypatch.setattr(
        MODULE,
        "_validate_closed_identities",
        lambda *args: (
            {
                "closed": (
                    ["1001", "1002"],
                    np.asarray([1001, 1002], dtype=np.int64),
                )
            },
            {"closed": {"sha256": "0" * 64, "bytes": 0, "path": "mock"}},
        ),
    )
    output = tmp_path / "v3"
    args = argparse.Namespace(
        parent_inner_train_bundle=parent,
        doc_ids=doc_ids,
        output_root=output,
        protocol=ROOT / MODULE.CANONICAL_PROTOCOL,
        source_commit="a" * 40,
        n_docs=8,
    )
    MODULE.build(args)
    return parent, output, protocol, qids, args


def test_registered_parent_query_identity_matches_protocol() -> None:
    protocol = json.loads(
        (ROOT / MODULE.CANONICAL_PROTOCOL).read_text(encoding="utf-8")
    )
    path = (
        ROOT
        / "results/rars_v2_2_fp32_replication/provenance/"
        "input-audit-00a0dee30767/inner_train/query_manifest.json"
    )
    assert MODULE.sha256_file(path) == protocol["parent_lineage"][
        "parent_inner_train_query_manifest_sha256"
    ]


def test_pairwise_overlap_audit_fails_closed() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MODULE._assert_disjoint(
            "left",
            ["1", "2"],
            np.asarray([1, 2]),
            "right",
            ["3", "2"],
            np.asarray([3, 4]),
        )


def test_source_commit_must_be_exact_lowercase_hex() -> None:
    MODULE._validate_exact_commit("a" * 40)
    for value in ("a" * 39, "A" * 40, "z" * 40):
        with pytest.raises(ValueError, match="exact lowercase"):
            MODULE._validate_exact_commit(value)


def test_candidate_freeze_subsets_parent_without_labels_retrieval_or_pca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output, _, qids, _ = _build_toy(tmp_path, monkeypatch)
    parent_rows = np.load(parent / "ann_rows.int64.npy")
    parent_scores = np.load(parent / "pca_scores.float32.npy")
    split = MODULE.split_development_qids(qids)
    for role_id in (MODULE.DESIGN_ROLE_ID, MODULE.AUDIT_ROLE_ID):
        role_dir = output / role_id
        indices = split[role_id]
        assert np.array_equal(np.load(role_dir / "ann_rows.int64.npy"), parent_rows[indices])
        assert np.array_equal(np.load(role_dir / "pca_scores.float32.npy"), parent_scores[indices])
        assert np.array_equal(
            np.load(role_dir / "candidate_doc_ids.int64.npy"),
            100 + parent_rows[indices],
        )
        unique_rows = np.load(role_dir / "candidate_doc_rows.int64.npy")
        lookup = np.load(role_dir / "ann_residual_rows.int64.npy")
        assert np.array_equal(unique_rows[lookup], parent_rows[indices])
        for forbidden in (
            "candidate_relevance.uint8.npy",
            "relevant_counts.int32.npy",
            "v3_role_labels_manifest.json",
        ):
            assert not (role_dir / forbidden).exists()
        manifest = json.loads((role_dir / "v3_candidate_manifest.json").read_text())
        assert manifest["data_access"]["qrels_opened_or_parsed"] is False
        assert manifest["data_access"]["faiss_imported_or_search_performed"] is False
        assert manifest["data_access"]["pca_fit_or_score_recomputation_performed"] is False
    assert sorted(path.name for path in (output / MODULE.FUTURE_ROLE_ID).iterdir()) == [
        "query_manifest.json",
        "v3_identity_manifest.json",
    ]
    summary = json.loads((output / "v3_oracle_bundle_freeze_summary.json").read_text())
    assert summary["status"] == "V3_QRELS_FREE_CANDIDATE_BUNDLES_FROZEN"
    assert summary["qrels_opened_or_parsed"] is False


def test_candidate_freeze_refuses_nonempty_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _, args = _build_toy(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="non-empty"):
        MODULE.build(args)


def test_parent_file_hash_mismatch_fails_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, doc_ids, protocol, _ = _make_parent_bundle(tmp_path)
    with (parent / "ann_scores.float32.npy").open("ab") as handle:
        handle.write(b"corrupt")
    monkeypatch.setattr(MODULE, "_validate_canonical_repository", lambda *args: protocol)
    monkeypatch.setattr(MODULE, "_validate_closed_identities", lambda *args: ({}, {}))
    output = tmp_path / "v3"
    args = argparse.Namespace(
        parent_inner_train_bundle=parent,
        doc_ids=doc_ids,
        output_root=output,
        protocol=ROOT / MODULE.CANONICAL_PROTOCOL,
        source_commit="a" * 40,
        n_docs=8,
    )
    with pytest.raises(ValueError, match="byte count changed"):
        MODULE.build(args)
    assert not output.exists()


def test_builder_source_has_no_qrels_faiss_search_or_pca_fit() -> None:
    source = (ROOT / "scripts/build_msmarco_rars_v3_oracle_bundles.py").read_text(
        encoding="utf-8"
    ).lower()
    assert "import faiss" not in source
    assert "load_qrels" not in source
    assert ".search(" not in source
    assert "fit_progressive_pca" not in source


def test_two_stage_materializer_keeps_audit_absent_until_verified_design_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output, protocol, qids, _ = _build_toy(tmp_path, monkeypatch)
    monkeypatch.setattr(
        MATERIALIZER, "_validate_canonical_repository", lambda *args: protocol
    )
    base_args = {
        "bundle_root": output,
        "parent_inner_train_bundle": parent,
        "source_commit": "a" * 40,
        "protocol": ROOT / MODULE.CANONICAL_PROTOCOL,
    }
    design = MATERIALIZER.materialize(
        argparse.Namespace(**base_args, role=MODULE.DESIGN_ROLE_ID, design_freeze=None)
    )
    design_dir = output / MODULE.DESIGN_ROLE_ID
    audit_dir = output / MODULE.AUDIT_ROLE_ID
    split = MODULE.split_development_qids(qids)
    parent_labels = np.load(parent / "candidate_relevance.uint8.npy")
    assert np.array_equal(
        np.load(design_dir / "candidate_relevance.uint8.npy"),
        parent_labels[split[MODULE.DESIGN_ROLE_ID]],
    )
    assert design["audit_release"]["design_freeze_required"] is False
    assert not (audit_dir / "candidate_relevance.uint8.npy").exists()
    with pytest.raises(ValueError, match="require --design-freeze"):
        MATERIALIZER.materialize(
            argparse.Namespace(**base_args, role=MODULE.AUDIT_ROLE_ID, design_freeze=None)
        )
    assert not (audit_dir / "v3_role_labels_started.json").exists()

    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    registered_path = oracle_dir / "design_artifact.npy"
    np.save(registered_path, np.asarray([1.0], dtype=np.float32))
    payload = {"protocol_id": MODULE.PROTOCOL_ID, "toy": True}
    fingerprint = MODULE.canonical_sha256(payload)
    _write_json(
        oracle_dir / "oracle_started.json",
        {
            "protocol_id": MODULE.PROTOCOL_ID,
            "source_commit": "a" * 40,
            "run_fingerprint": fingerprint,
            "fingerprint_payload": payload,
        },
    )
    freeze_path = oracle_dir / "design_freeze.json"
    _write_json(
        freeze_path,
        {
            "protocol_id": MODULE.PROTOCOL_ID,
            "source_commit": "a" * 40,
            "status": "DESIGN_ARTIFACTS_FROZEN_BEFORE_AUDIT_LOAD",
            "run_fingerprint": fingerprint,
            "source_hashes": {
                "protocol_sha256": MODULE.sha256_file(ROOT / MODULE.CANONICAL_PROTOCOL),
                "builder_sha256": MODULE.sha256_file(ROOT / "scripts/build_msmarco_rars_v3_oracle_bundles.py"),
                "label_materializer_sha256": MODULE.sha256_file(ROOT / "scripts/materialize_rars_v3_role_labels.py"),
                "core_sha256": MODULE.sha256_file(ROOT / "scripts/rars_v3_oracle_core.py"),
                "evaluator_sha256": MODULE.sha256_file(ROOT / "scripts/evaluate_rars_v3_oracle_first_feasibility.py"),
            },
            "registered_outputs": {registered_path.name: _file_record(registered_path)},
            "design_bundle_manifest": _file_record(design_dir / "v3_candidate_manifest.json"),
            "audit_bundle_manifest_registered_but_arrays_unloaded": _file_record(audit_dir / "v3_candidate_manifest.json"),
            "design_role_labels_manifest": _file_record(design_dir / "v3_role_labels_manifest.json"),
            "audit_bundle_loaded_before_this_freeze": False,
            "audit_role_labels_materialized_before_this_freeze": False,
        },
    )
    audit = MATERIALIZER.materialize(
        argparse.Namespace(
            **base_args,
            role=MODULE.AUDIT_ROLE_ID,
            design_freeze=freeze_path,
        )
    )
    assert audit["audit_release"]["design_freeze_verified"] is True
    assert np.array_equal(
        np.load(audit_dir / "candidate_relevance.uint8.npy"),
        parent_labels[split[MODULE.AUDIT_ROLE_ID]],
    )
    assert not any((output / MODULE.FUTURE_ROLE_ID).glob("*relevance*"))


def test_audit_materializer_rejects_tampered_registered_design_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output, protocol, _, _ = _build_toy(tmp_path, monkeypatch)
    monkeypatch.setattr(
        MATERIALIZER, "_validate_canonical_repository", lambda *args: protocol
    )
    base_args = {
        "bundle_root": output,
        "parent_inner_train_bundle": parent,
        "source_commit": "a" * 40,
        "protocol": ROOT / MODULE.CANONICAL_PROTOCOL,
    }
    MATERIALIZER.materialize(
        argparse.Namespace(**base_args, role=MODULE.DESIGN_ROLE_ID, design_freeze=None)
    )
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    payload = {"protocol_id": MODULE.PROTOCOL_ID}
    fingerprint = MODULE.canonical_sha256(payload)
    _write_json(
        oracle_dir / "oracle_started.json",
        {
            "protocol_id": MODULE.PROTOCOL_ID,
            "source_commit": "a" * 40,
            "run_fingerprint": fingerprint,
            "fingerprint_payload": payload,
        },
    )
    artifact = oracle_dir / "artifact.json"
    _write_json(artifact, {"ok": True})
    freeze = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "source_commit": "a" * 40,
        "status": "DESIGN_ARTIFACTS_FROZEN_BEFORE_AUDIT_LOAD",
        "run_fingerprint": fingerprint,
        "source_hashes": {
            "protocol_sha256": MODULE.sha256_file(ROOT / MODULE.CANONICAL_PROTOCOL),
            "builder_sha256": MODULE.sha256_file(ROOT / "scripts/build_msmarco_rars_v3_oracle_bundles.py"),
            "label_materializer_sha256": MODULE.sha256_file(ROOT / "scripts/materialize_rars_v3_role_labels.py"),
            "core_sha256": MODULE.sha256_file(ROOT / "scripts/rars_v3_oracle_core.py"),
            "evaluator_sha256": MODULE.sha256_file(ROOT / "scripts/evaluate_rars_v3_oracle_first_feasibility.py"),
        },
        "registered_outputs": {artifact.name: _file_record(artifact)},
        "design_bundle_manifest": _file_record(output / MODULE.DESIGN_ROLE_ID / "v3_candidate_manifest.json"),
        "audit_bundle_manifest_registered_but_arrays_unloaded": _file_record(output / MODULE.AUDIT_ROLE_ID / "v3_candidate_manifest.json"),
        "design_role_labels_manifest": _file_record(output / MODULE.DESIGN_ROLE_ID / "v3_role_labels_manifest.json"),
        "audit_bundle_loaded_before_this_freeze": False,
        "audit_role_labels_materialized_before_this_freeze": False,
    }
    freeze_path = oracle_dir / "design_freeze.json"
    _write_json(freeze_path, freeze)
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="design output artifact.json"):
        MATERIALIZER.materialize(
            argparse.Namespace(
                **base_args,
                role=MODULE.AUDIT_ROLE_ID,
                design_freeze=freeze_path,
            )
        )
    assert not (output / MODULE.AUDIT_ROLE_ID / "candidate_relevance.uint8.npy").exists()
