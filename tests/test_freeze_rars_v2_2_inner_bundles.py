from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "freeze_rars_v2_2_inner_bundles.py"
SPEC = importlib.util.spec_from_file_location("freeze_rars_v2_2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_split(path: Path, qids: list[str], rows: list[int]) -> None:
    path.write_text(json.dumps({"query_ids": qids, "query_rows": rows}))


def _write_source_bundle(
    bundle_dir: Path,
    query_vectors: np.ndarray,
    query_rows: np.ndarray,
) -> None:
    bundle_dir.mkdir(parents=True)
    query_count = len(query_rows)
    candidates = 3
    arrays = {
        "query_vectors.float32.npy": query_vectors[query_rows].astype(np.float32),
        "ann_rows.int64.npy": np.tile(np.arange(candidates), (query_count, 1)),
        "ann_scores.float32.npy": np.tile(
            np.asarray([0.3, 0.2, 0.1], np.float32), (query_count, 1)
        ),
        "candidate_relevance.uint8.npy": np.tile(
            np.asarray([1, 0, 0], np.uint8), (query_count, 1)
        ),
        "relevant_counts.int32.npy": np.ones(query_count, np.int32),
        "ann_residual_rows.int64.npy": np.tile(
            np.arange(candidates), (query_count, 1)
        ),
        "candidate_residuals.float32.npy": np.eye(candidates, 2, dtype=np.float32),
    }
    records = {}
    for filename, value in arrays.items():
        path = bundle_dir / filename
        np.save(path, value)
        records[filename] = MODULE.file_record(path)
    (bundle_dir / "manifest.json").write_text(json.dumps({
        "protocol_id": "rars_v2_boundary_loss_feasibility_v1",
        "files": records,
    }))


def _fixture(tmp_path: Path):
    train_qids = [f"q{value}" for value in range(30)]
    train_rows = list(range(30))
    outer_qids, outer_rows = ["outer"], [30]
    test_qids, test_rows = ["test"], [31]
    train_split = tmp_path / "train.json"
    outer_split = tmp_path / "outer.json"
    test_split = tmp_path / "test.json"
    _write_split(train_split, train_qids, train_rows)
    _write_split(outer_split, outer_qids, outer_rows)
    _write_split(test_split, test_qids, test_rows)
    query_vectors = np.arange(64, dtype=np.float32).reshape(32, 2)
    query_path = tmp_path / "queries.npy"
    np.save(query_path, query_vectors)
    fit, selection = MODULE.inner_partition(train_qids)
    bundle_root = tmp_path / "bundles"
    _write_source_bundle(bundle_root / "inner_train", query_vectors, fit)
    _write_source_bundle(bundle_root / "inner_validation", query_vectors, selection)
    args = argparse.Namespace(
        bundle_root=bundle_root,
        query_vectors=query_path,
        train_split=train_split,
        outer_validation_split=outer_split,
        clean_test_split=test_split,
        source_commit="0" * 40,
    )
    return args


def test_freezer_emits_role_specific_manifests_and_honest_access(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    result = MODULE.freeze(args)
    assert result["status"] == "INNER_BUNDLES_FROZEN"
    assert result["outer_validation_built_or_read_by_freezer"] is False
    for role in ("inner_train", "inner_validation"):
        manifest = json.loads(
            (args.bundle_root / role / "v2_2_manifest.json").read_text()
        )
        assert manifest["role_id"] == role
        assert manifest["data_access"][
            "closed_test_relevance_values_parsed_by_source_builder"
        ] is True
        assert manifest["data_access"][
            "closed_test_relevance_values_used"
        ] is False


def test_freezer_rejects_train_outer_overlap_before_bundle_use(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    train = json.loads(args.train_split.read_text())
    _write_split(
        args.outer_validation_split,
        [train["query_ids"][0]],
        [train["query_rows"][0]],
    )
    with pytest.raises(ValueError, match="overlap"):
        MODULE.freeze(args)
