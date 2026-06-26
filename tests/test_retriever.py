from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from app.retriever import RetrievalService


class FakeSentenceTransformer:
    """Small deterministic encoder used to keep tests offline and fast."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        queries: list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        mapping = {
            "alpha": np.array([1.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0], dtype=np.float32),
            "mixed": np.array([0.6, 0.8], dtype=np.float32),
        }
        vectors = np.stack([mapping[q] for q in queries]).astype(np.float32)
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    index = faiss.IndexFlatIP(2)
    index.add(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.6, 0.8],
            ],
            dtype=np.float32,
        )
    )
    faiss.write_index(index, str(tmp_path / "index.faiss"))

    (tmp_path / "service_config.json").write_text(
        json.dumps(
            {
                "embedding_model": "fake/test-model",
                "index_type": "IndexFlatIP",
                "default_nprobe": 16,
            }
        ),
        encoding="utf-8",
    )

    docs = [
        {"doc_id": "doc-alpha", "title": "Alpha", "text": "Alpha document"},
        {"doc_id": "doc-beta", "title": "Beta", "text": "Beta document"},
        {"doc_id": "doc-mixed", "title": "Mixed", "text": "Mixed document"},
    ]
    with (tmp_path / "documents.jsonl").open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    return tmp_path


@pytest.fixture()
def loaded_service(monkeypatch: pytest.MonkeyPatch, artifact_dir: Path) -> RetrievalService:
    monkeypatch.setattr("app.retriever.SentenceTransformer", FakeSentenceTransformer)
    service = RetrievalService(artifact_dir)
    service.load()
    return service


def test_loads_artifact_and_returns_ranked_document(loaded_service: RetrievalService) -> None:
    result = loaded_service.search("alpha", top_k=2, nprobe=16)

    assert loaded_service.is_ready
    assert result["index_type"] == "IndexFlatIP"
    assert len(result["results"]) == 2
    assert result["results"][0]["doc_id"] == "doc-alpha"
    assert result["results"][0]["score"] == pytest.approx(1.0)
    assert result["latency_ms"] >= 0


def test_batch_search_uses_all_queries_and_preserves_order(loaded_service: RetrievalService) -> None:
    result = loaded_service.search_many(["alpha", "beta", "mixed"], top_k=1, nprobe=16)

    assert result["count"] == 3
    assert result["top_k"] == 1
    assert result["latency_ms_total"] >= 0
    assert result["latency_ms_per_query"] >= 0
    assert [item["query"] for item in result["items"]] == ["alpha", "beta", "mixed"]
    assert [item["results"][0]["doc_id"] for item in result["items"]] == [
        "doc-alpha",
        "doc-beta",
        "doc-mixed",
    ]


def test_load_rejects_index_document_count_mismatch(
    monkeypatch: pytest.MonkeyPatch, artifact_dir: Path
) -> None:
    monkeypatch.setattr("app.retriever.SentenceTransformer", FakeSentenceTransformer)
    documents_path = artifact_dir / "documents.jsonl"
    documents_path.write_text(
        json.dumps({"doc_id": "doc-alpha", "title": "Alpha", "text": "Alpha document"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Index/document mismatch"):
        RetrievalService(artifact_dir).load()
