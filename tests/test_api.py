from __future__ import annotations

from app import main


class FakeReranker:
    model_name = "fake/reranker"


class FakeRetriever:
    def __init__(self) -> None:
        self.is_ready = True
        self.config = {
            "index_type": "IndexIVFPQ",
            "embedding_model": "fake/test-model",
        }
        self.documents = [{"doc_id": "doc-1"}] * 3
        self.reranker = FakeReranker()
        self.sidecar = None
        self.sidecar_artifact_dir = None

    def search(
        self,
        query: str,
        top_k: int,
        nprobe: int | None,
        rerank: bool = False,
        candidate_k: int | None = None,
        sidecar: bool = False,
        sidecar_top_b: int | None = None,
    ) -> dict:
        return {
            "query": query,
            "top_k": top_k,
            "nprobe": nprobe,
            "rerank_enabled": rerank,
            "candidate_k": candidate_k or top_k,
            "sidecar_enabled": sidecar,
            "sidecar_top_b": sidecar_top_b,
            "latency_ms": 1.25,
            "index_type": "IndexIVFPQ",
            "results": [
                {
                    "rank": 1,
                    "doc_id": "doc-1",
                    "score": 0.9,
                    "title": "",
                    "text": "x",
                }
            ],
        }

    def search_many(
        self,
        queries: list[str],
        top_k: int,
        nprobe: int | None,
        rerank: bool = False,
        candidate_k: int | None = None,
        sidecar: bool = False,
        sidecar_top_b: int | None = None,
    ) -> dict:
        return {
            "count": len(queries),
            "top_k": top_k,
            "nprobe": nprobe,
            "rerank_enabled": rerank,
            "candidate_k": candidate_k or top_k,
            "sidecar_enabled": sidecar,
            "sidecar_top_b": sidecar_top_b,
            "latency_ms_total": 2.5,
            "latency_ms_per_query": 1.25,
            "index_type": "IndexIVFPQ",
            "items": [
                self.search(
                    query,
                    top_k,
                    nprobe,
                    rerank=rerank,
                    candidate_k=candidate_k,
                    sidecar=sidecar,
                    sidecar_top_b=sidecar_top_b,
                )
                for query in queries
            ],
        }


def test_health_reports_service_metadata(monkeypatch) -> None:
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    payload = main.health()

    assert payload["status"] == "ok"
    assert payload["index_type"] == "IndexIVFPQ"
    assert payload["document_count"] == 3
    assert payload["reranker_enabled"] is True
    assert payload["reranker_model"] == "fake/reranker"
    assert payload["sidecar_enabled"] is False
    assert payload["sidecar_artifact_dir"] is None
    assert payload["sidecar_default_top_b"] is None
    assert payload["sidecar_max_top_b"] is None


def test_search_endpoint_delegates_rerank_options(monkeypatch) -> None:
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    payload = main.search(
        main.SearchRequest(
            query="test query",
            top_k=3,
            nprobe=8,
            rerank=True,
            candidate_k=10,
            sidecar=True,
            sidecar_top_b=20,
        )
    )

    assert payload["query"] == "test query"
    assert payload["top_k"] == 3
    assert payload["nprobe"] == 8
    assert payload["rerank_enabled"] is True
    assert payload["candidate_k"] == 10
    assert payload["sidecar_enabled"] is True
    assert payload["sidecar_top_b"] == 20
    assert payload["results"][0]["doc_id"] == "doc-1"


def test_batch_search_endpoint_delegates_rerank_options(monkeypatch) -> None:
    monkeypatch.setattr(main, "retriever", FakeRetriever())

    payload = main.batch_search(
        main.BatchSearchRequest(
            queries=["first", "second"],
            top_k=2,
            nprobe=16,
            rerank=True,
            candidate_k=20,
            sidecar=True,
            sidecar_top_b=40,
        )
    )

    assert payload["count"] == 2
    assert payload["rerank_enabled"] is True
    assert payload["candidate_k"] == 20
    assert payload["sidecar_enabled"] is True
    assert payload["sidecar_top_b"] == 40
    assert [item["query"] for item in payload["items"]] == [
        "first",
        "second",
    ]
