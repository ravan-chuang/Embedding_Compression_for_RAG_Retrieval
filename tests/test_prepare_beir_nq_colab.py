from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_beir_nq_colab.py"
SPEC = importlib.util.spec_from_file_location("prepare_nq", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_registered_protocol_matches_colab_runner() -> None:
    protocol = MODULE.protocol_values(MODULE.DEFAULT_PROTOCOL)
    assert protocol["embedding"]["document_storage_dtype"] == "float16"
    assert protocol["base_index"]["nlist"] == 2048
    assert protocol["base_index"]["gpu_float16_lookup_tables"] is False


def test_split_archive_selection_never_selects_test_queries_or_qrels(
    tmp_path: Path,
) -> None:
    test_archive_path = tmp_path / "nq.zip"
    with zipfile.ZipFile(test_archive_path, "w") as archive:
        archive.writestr("nq/corpus.jsonl", "{}\n")
        archive.writestr("nq/queries.jsonl", "test query secret\n")
        archive.writestr("nq/qrels/test.tsv", "secret\td\t1\n")
    train_archive_path = tmp_path / "nq-train.zip"
    with zipfile.ZipFile(train_archive_path, "w") as archive:
        archive.writestr("nq-train/corpus.jsonl", "must not be selected\n")
        archive.writestr("nq-train/queries.jsonl", "{}\n")
        archive.writestr("nq-train/qrels/train.tsv", "q\td\t1\n")

    with zipfile.ZipFile(test_archive_path) as archive:
        test_selected = MODULE.choose_members(
            archive, {"corpus.jsonl": ("corpus.jsonl",)}
        )
    with zipfile.ZipFile(train_archive_path) as archive:
        train_selected = MODULE.choose_members(
            archive,
            {
                "queries.jsonl": ("queries.jsonl",),
                "qrels/train.tsv": ("qrels", "train.tsv"),
            },
        )

    assert set(test_selected) == {"corpus.jsonl"}
    assert set(train_selected) == {"queries.jsonl", "qrels/train.tsv"}
    selected_names = [
        info.filename for info in [*test_selected.values(), *train_selected.values()]
    ]
    assert all("test.tsv" not in name for name in selected_names)
    assert all("nq/queries.jsonl" != name for name in selected_names)


def test_scan_corpus_records_string_safe_ids_and_text_rule(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join([
            json.dumps({"_id": "doc-α", "title": "Title", "text": "Body"}) + "\n",
            json.dumps({"_id": "2", "title": "", "text": "Second"}) + "\n",
        ]),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"

    result = MODULE.scan_corpus(corpus, output)

    assert result["document_count"] == 2
    assert result["max_doc_id_utf8_bytes"] == len("doc-α".encode("utf-8"))
    assert result["test_qrels_accessed"] is False
    assert MODULE.corpus_text({"title": "T", "text": "X"}) == "T\nX"
    assert MODULE.corpus_text({"title": "Title only", "text": ""}) == "Title only\n"
    with pytest.raises(ValueError, match="empty title and text"):
        MODULE.corpus_text({"title": "", "text": ""})


def test_scan_corpus_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "same", "text": "one"})
        + "\n"
        + json.dumps({"_id": "same", "text": "two"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate corpus ID"):
        MODULE.scan_corpus(corpus, tmp_path / "manifest.json")


def test_unsafe_zip_member_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        MODULE.safe_member_name("../qrels/train.tsv")


def test_corpus_encoding_checkpoint_and_completion_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    corpus = artifact_root / "stage1" / "data" / "nq" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text(
        json.dumps({"_id": "a", "text": "alpha"})
        + "\n"
        + json.dumps({"_id": "long-id", "title": "T", "text": "beta"})
        + "\n",
        encoding="utf-8",
    )
    scan_path = artifact_root / "stage1" / "corpus" / "corpus_scan_manifest.json"
    MODULE.scan_corpus(corpus, scan_path)
    model_manifest = artifact_root / "stage1" / "model" / "snapshot_manifest.json"
    model_manifest.parent.mkdir(parents=True)
    model_manifest.write_text("{}\n", encoding="utf-8")

    class FakeModel:
        def encode(self, texts, **kwargs):
            result = np.zeros((len(texts), 384), dtype=np.float32)
            result[:, 0] = 1.0
            return result

    monkeypatch.setattr(MODULE, "load_or_create_model", lambda *args: FakeModel())

    first = MODULE.encode_corpus(
        artifact_root,
        MODULE.DEFAULT_PROTOCOL,
        batch_size=1,
        checkpoint_rows=1,
        device="cpu",
    )
    second = MODULE.encode_corpus(
        artifact_root,
        MODULE.DEFAULT_PROTOCOL,
        batch_size=1,
        checkpoint_rows=1,
        device="cpu",
    )

    assert first == second
    assert first["document_count"] == 2
    assert first["document_embeddings"]["bytes"] == 2 * 384 * 2
    assert first["doc_ids"]["bytes"] == 2 * len("long-id".encode("utf-8"))
    assert first["test_qrels_accessed"] is False
    assert not (artifact_root / "stage1" / "corpus" / "embedding_progress.json").exists()
