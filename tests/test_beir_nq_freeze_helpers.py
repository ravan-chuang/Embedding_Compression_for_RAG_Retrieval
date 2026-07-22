from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module(
    "nq_manifest_builder", "scripts/build_beir_nq_pre_qrels_manifest.py"
)
EVALUATOR = load_module("nq_evaluator", "scripts/evaluate_beir_nq_frozen.py")


def test_portable_manifest_paths_distinguish_drive_and_repo(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    repo = tmp_path / "repo"
    artifact.mkdir()
    repo.mkdir()
    drive_file = artifact / "stage1" / "value.bin"
    drive_file.parent.mkdir()
    drive_file.write_bytes(b"x")
    repo_file = repo / "scripts" / "runner.py"
    repo_file.parent.mkdir()
    repo_file.write_text("pass\n")

    assert BUILDER.portable_path(
        drive_file, artifact_root=artifact, repo=repo
    ) == "artifact://stage1/value.bin"
    assert BUILDER.portable_path(
        repo_file, artifact_root=artifact, repo=repo
    ) == "repo://scripts/runner.py"


def test_manifest_builder_rejects_true_test_access_flags() -> None:
    with pytest.raises(ValueError, match="Unsafe flag"):
        BUILDER.reject_unsafe_flags({"nested": {"test_qrels_accessed": True}})


def test_stage3_extracts_test_queries_and_qrels_from_test_archive_only(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "nq.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nq/corpus.jsonl", "must not be extracted\n")
        archive.writestr("nq/queries.jsonl", '{"_id":"q","text":"question"}\n')
        archive.writestr("nq/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq\td\t1\n")

    queries = EVALUATOR.extract_test_queries(
        archive_path, tmp_path / "stage3" / "queries.jsonl"
    )
    qrels = EVALUATOR.extract_test_qrels(
        archive_path, tmp_path / "stage3" / "qrels" / "test.tsv"
    )

    assert queries.read_text(encoding="utf-8").startswith('{"_id":"q"')
    assert "q\td\t1" in qrels.read_text(encoding="utf-8")
    assert not (tmp_path / "stage3" / "corpus.jsonl").exists()


def test_query_normalization_and_metrics() -> None:
    assert EVALUATOR.normalize_query_text("  Café\tQUERY ") == "café query"
    values = EVALUATOR.per_query_metrics(["d0", "d1", "d2"], {"d1", "other"})
    assert values["recall_at_10"] == 0.5
    assert values["success_at_10"] == 1.0
    assert values["mrr_at_10"] == 0.5
    assert 0 < values["ndcg_at_10"] < 1


def test_paired_bootstrap_is_deterministic_and_keeps_registered_primary() -> None:
    values = {
        system: {
            metric: np.asarray([offset, offset + 0.1, offset + 0.2])
            for metric in EVALUATOR.METRICS
        }
        for system, offset in [
            ("base_m32", 0.0),
            ("pca_r16_int8", 0.1),
            ("rars_r16_int8", 0.2),
        ]
    }
    first = EVALUATOR.paired_bootstrap(values, replicates=100, seed=7)
    second = EVALUATOR.paired_bootstrap(values, replicates=100, seed=7)
    assert first == second
    primary = next(
        row for row in first
        if row["contrast"] == "rars_minus_pca"
        and row["metric"] == "recall_at_10"
    )
    assert primary["mean_difference"] == pytest.approx(0.1)
    assert primary["ci_lower"] > 0


def test_sidecar_correction_can_flip_only_the_registered_candidate_pool(
    tmp_path: Path,
) -> None:
    code_path = tmp_path / "codes.memmap"
    codes = np.memmap(code_path, dtype=np.int8, mode="w+", shape=(3, 1))
    codes[:] = np.asarray([[0], [10], [0]], dtype=np.int8)
    codes.flush()
    rows = np.asarray([[0, 1, 2]], dtype=np.int64)
    scores = np.asarray([[1.0, 0.9, 0.8]], dtype=np.float32)
    ranking = EVALUATOR.corrected_ranking_rows(
        np.asarray([[1.0]], dtype=np.float32),
        rows,
        scores,
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([0.1], dtype=np.float32),
        codes,
        alpha=1.0,
        top_b=2,
    )
    assert ranking.tolist() == [[1, 0, 2]]
