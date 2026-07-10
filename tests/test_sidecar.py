import json
from pathlib import Path

import numpy as np

from app.sidecar import RARSSidecar


def write_test_artifact(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    scales = np.array([0.1, 0.2], dtype=np.float32)
    codes = np.array(
        [
            [10, 0],
            [0, 10],
            [5, 5],
            [-5, 5],
        ],
        dtype=np.int8,
    )
    doc_ids = np.array([100, 101, 102, 103], dtype=np.int64)

    np.save(path / "basis.npy", basis)
    np.save(path / "scales.npy", scales)
    np.save(path / "codes.int8.npy", codes)
    np.save(path / "doc_ids.npy", doc_ids)

    config = {
        "dim": 3,
        "rank": 2,
        "alpha": 1.0,
        "default_top_b": 2,
        "max_top_b": 4,
        "code_dtype": "int8",
        "doc_id_dtype": "int64",
        "basis_file": "basis.npy",
        "scales_file": "scales.npy",
        "codes_file": "codes.int8.npy",
        "doc_ids_file": "doc_ids.npy",
    }

    with open(path / "sidecar_config.json", "w") as f:
        json.dump(config, f)


def test_sidecar_loads_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "sidecar"
    write_test_artifact(artifact)

    sidecar = RARSSidecar(artifact)

    assert sidecar.num_docs == 4
    assert sidecar.dim == 3
    assert sidecar.rank == 2


def test_rows_from_doc_ids(tmp_path: Path) -> None:
    artifact = tmp_path / "sidecar"
    write_test_artifact(artifact)

    sidecar = RARSSidecar(artifact)

    rows = sidecar.rows_from_doc_ids([102, 100, 103])

    np.testing.assert_array_equal(rows, np.array([2, 0, 3], dtype=np.int64))


def test_compute_corrections(tmp_path: Path) -> None:
    artifact = tmp_path / "sidecar"
    write_test_artifact(artifact)

    sidecar = RARSSidecar(artifact)

    query = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    rows = np.array([0, 1, 2], dtype=np.int64)

    corrections = sidecar.compute_corrections(query, rows)

    # q @ basis = [1.5, 2.5]
    # row0 coeff = [1.0, 0.0] -> 1.5
    # row1 coeff = [0.0, 2.0] -> 5.0
    # row2 coeff = [0.5, 1.0] -> 3.25
    expected = np.array([1.5, 5.0, 3.25], dtype=np.float32)

    np.testing.assert_allclose(corrections, expected, rtol=1e-6, atol=1e-6)


def test_rerank_only_corrects_top_b(tmp_path: Path) -> None:
    artifact = tmp_path / "sidecar"
    write_test_artifact(artifact)

    sidecar = RARSSidecar(artifact)

    query = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    rows = np.array([0, 1, 2, 3], dtype=np.int64)
    ann_scores = np.array([10.0, 9.0, 8.0, 7.0], dtype=np.float32)

    result = sidecar.rerank(
        query_embedding=query,
        candidate_rows=rows,
        ann_scores=ann_scores,
        top_k=3,
        top_b=2,
        alpha=1.0,
    )

    # Only rows 0 and 1 are corrected:
    # row0: 10.0 + 1.5 = 11.5
    # row1: 9.0 + 5.0 = 14.0
    # row2 and row3 keep original scores.
    np.testing.assert_array_equal(
        result["candidate_rows"],
        np.array([1, 0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result["doc_ids"],
        np.array([101, 100, 102], dtype=np.int64),
    )
    np.testing.assert_allclose(
        result["corrected_scores"],
        np.array([14.0, 11.5, 8.0], dtype=np.float32),
    )

    assert result["top_b"] == 2
    assert result["actual_top_b"] == 2
    assert result["sidecar_enabled"] is True


def test_rerank_batch(tmp_path: Path) -> None:
    artifact = tmp_path / "sidecar"
    write_test_artifact(artifact)

    sidecar = RARSSidecar(artifact)

    query_embeddings = np.array(
        [
            [1.0, 2.0, 0.5],
            [0.5, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidate_rows = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    ann_scores = np.array(
        [
            [10.0, 9.0, 8.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    results = sidecar.rerank_batch(
        query_embeddings=query_embeddings,
        candidate_rows=candidate_rows,
        ann_scores=ann_scores,
        top_k=2,
        top_b=2,
    )

    assert len(results) == 2
    assert results[0]["candidate_rows"].shape == (2,)
    assert results[1]["candidate_rows"].shape == (2,)