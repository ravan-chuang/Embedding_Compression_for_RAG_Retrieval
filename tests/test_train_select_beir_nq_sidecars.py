from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_select_beir_nq_sidecars.py"
SPEC = importlib.util.spec_from_file_location("train_nq", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeIndex:
    def __init__(self, reconstructed: np.ndarray):
        self.reconstructed = np.asarray(reconstructed, dtype=np.float32)
        self.d = self.reconstructed.shape[1]

    def reconstruct_batch(self, rows: np.ndarray) -> np.ndarray:
        return self.reconstructed[np.asarray(rows, dtype=np.int64)]


def test_protocol_and_sampling_are_frozen() -> None:
    protocol = json.loads(MODULE.DEFAULT_PROTOCOL.read_text())
    MODULE.validate_protocol(protocol)
    first = MODULE.pca_sample_rows(100, 20, 42)
    second = MODULE.pca_sample_rows(100, 20, 42)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 20


def test_score_error_weights_and_weighted_draws_are_deterministic() -> None:
    ann_rows = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    ann = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    exact = np.asarray([[0.2, 0.0], [0.5, 0.1]], dtype=np.float32)
    weights = MODULE.aggregate_score_error_weights(ann_rows, ann, exact, 4)
    assert np.allclose(weights, [0.1, 0.4, 0.3, 0.0], atol=1e-6)

    rows_a, counts_a = MODULE.rars_weighted_draws(weights, 1000, 42)
    rows_b, counts_b = MODULE.rars_weighted_draws(weights, 1000, 42)
    assert np.array_equal(rows_a, rows_b)
    assert np.array_equal(counts_a, counts_b)
    assert counts_a.sum() == 1000
    assert 3 not in rows_a


def test_residual_covariance_basis_uses_residual_not_original(tmp_path: Path) -> None:
    original = np.asarray(
        [[3.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float16,
    )
    path = tmp_path / "embeddings.memmap"
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=original.shape)
    mm[:] = original
    mm.flush()
    reconstructed = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.9], [0.0, 0.9]],
        dtype=np.float32,
    )
    basis = MODULE.residual_covariance_basis(
        FakeIndex(reconstructed),
        mm,
        np.arange(4),
        rank=1,
        batch_size=2,
        compute_device="cpu",
    )
    assert basis.shape == (2, 1)
    assert basis[0, 0] > 0.99


def test_registered_selection_prefers_small_depth_then_alpha_tie_rule() -> None:
    rows = [
        {"alpha": 1.0, "top_b": 20, "overlap_gain": 0.10, "corrected_top10_overlap": 0.80},
        {"alpha": -0.5, "top_b": 10, "overlap_gain": 0.09, "corrected_top10_overlap": 0.79},
        {"alpha": 0.5, "top_b": 10, "overlap_gain": 0.09, "corrected_top10_overlap": 0.79},
    ]
    best = MODULE.select_registered_configuration(rows)
    assert best["top_b"] == 10
    assert best["alpha"] == -0.5


def test_stable_score_ties_preserve_candidate_order() -> None:
    scores = np.asarray([[1.0, 1.0, 0.5]], dtype=np.float32)
    assert MODULE.stable_descending_order(scores).tolist() == [[0, 1, 2]]


def test_sidecar_encoding_writes_resumable_matched_int8_artifacts(
    tmp_path: Path,
) -> None:
    original = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
        dtype=np.float16,
    )
    embeddings_path = tmp_path / "embeddings.memmap"
    embeddings = np.memmap(
        embeddings_path,
        dtype=np.float16,
        mode="w+",
        shape=original.shape,
    )
    embeddings[:] = original
    embeddings.flush()
    bases = {
        "pca": np.eye(2, dtype=np.float32),
        "rars": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    }

    first = MODULE.build_sidecar_codes(
        FakeIndex(np.zeros_like(original, dtype=np.float32)),
        embeddings,
        bases,
        tmp_path / "sidecars",
        n_docs=4,
        rank=2,
        batch_size=2,
        checkpoint_rows=2,
    )
    second = MODULE.build_sidecar_codes(
        FakeIndex(np.zeros_like(original, dtype=np.float32)),
        embeddings,
        bases,
        tmp_path / "sidecars",
        n_docs=4,
        rank=2,
        batch_size=2,
        checkpoint_rows=2,
    )

    for name in bases:
        assert first[name] == second[name]
        assert first[name]["codes"].stat().st_size == 4 * 2
        assert np.load(first[name]["scales"]).shape == (2,)
    assert not (tmp_path / "sidecars" / "sidecar_encoding_progress.json").exists()
