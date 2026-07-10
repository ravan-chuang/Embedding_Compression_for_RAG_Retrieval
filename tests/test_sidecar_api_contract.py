from fastapi.testclient import TestClient

from app.main import app, retriever


def test_health_exposes_sidecar_metadata() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "sidecar_enabled" in data
    assert "sidecar_artifact_dir" in data
    assert "sidecar_default_top_b" in data
    assert "sidecar_max_top_b" in data


def test_search_schema_accepts_sidecar_fields_when_retriever_not_loaded() -> None:
    client = TestClient(app)

    response = client.post(
        "/search",
        json={
            "query": "What is a dividend stock?",
            "top_k": 5,
            "candidate_k": 100,
            "nprobe": 16,
            "sidecar": True,
            "sidecar_top_b": 20,
        },
    )

    # The app-level schema should accept sidecar fields.
    # In a unit-test context the global retriever is not loaded, so the endpoint
    # should fail at service readiness, not at request validation.
    assert response.status_code == 503
    assert "Retriever is not ready" in response.json()["detail"]


def test_search_rejects_invalid_sidecar_top_b_schema() -> None:
    client = TestClient(app)

    response = client.post(
        "/search",
        json={
            "query": "What is a dividend stock?",
            "top_k": 5,
            "sidecar": True,
            "sidecar_top_b": 101,
        },
    )

    assert response.status_code == 422


def test_batch_search_schema_accepts_sidecar_fields_when_retriever_not_loaded() -> None:
    client = TestClient(app)

    response = client.post(
        "/batch-search",
        json={
            "queries": [
                "What is a dividend stock?",
                "How does inflation affect bond prices?",
            ],
            "top_k": 5,
            "candidate_k": 100,
            "sidecar": True,
            "sidecar_top_b": 20,
        },
    )

    assert response.status_code == 503
    assert "Retriever is not ready" in response.json()["detail"]