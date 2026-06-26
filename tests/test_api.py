from __future__ import annotations

from app import main


class FakeRetriever:
    def __init__(self) -> None:
        self.is_ready = True
        self.config = {
            "index_type": "IndexIVFPQ",
            "embedding_model": "fake/test-model",
        }
        self.documents = [{"doc_id": "doc-1"}] * 3

    def search(self, query: str, top_k: int, nprobe: int | None) -> dict:
        return {
            "query": query,
            "top_k": top_k,
            "nprobe": nprobe,
            "latency_ms": 1.25,
            "index_type": "IndexIVFPQ",
            "results": [{"rank": 1, "doc_id": "doc-1", "score": 0.9, "title": "", "text": "x"}],
        }

    def search_many(self, queries: list[str], top_k: int, nprobe: int | None) -> dict:
        return {
            "count": len(queries),
            "top_k": top_k,
            "nprobe": nprobe,
            "latency_ms_total": 2.5,
            "latency_ms_per_query": 1.25,
            "index_type": "IndexIVFPQ",
            "items": [self.search(query, top_k, nprobe) for query in queries],
        }


def test_health_reports_service_metadata(monkeypatch) -> None:
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    payload = main.health()

    assert payload["status"] == "ok"
    assert payload["index_type"] == "IndexIVFPQ"
    assert payload["document_count"] == 3


def test_search_endpoint_function_delegates_to_retriever(monkeypatch) -> None:
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    payload = main.search(main.SearchRequest(query="test query", top_k=3, nprobe=8))

    assert payload["query"] == "test query"
    assert payload["top_k"] == 3
    assert payload["nprobe"] == 8
    assert payload["results"][0]["doc_id"] == "doc-1"


def test_batch_search_endpoint_function_delegates_to_retriever(monkeypatch) -> None:
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    payload = main.batch_search(
        main.BatchSearchRequest(queries=["first", "second"], top_k=2, nprobe=16)
    )

    assert payload["count"] == 2
    assert [item["query"] for item in payload["items"]] == ["first", "second"]
