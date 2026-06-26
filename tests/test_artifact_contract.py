from __future__ import annotations

import json
from pathlib import Path

import faiss


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "fiqa_ivfpq_m96"


def test_serialized_index_matches_tracked_doc_id_order() -> None:
    index = faiss.read_index(str(ARTIFACT_DIR / "index.faiss"))
    doc_ids = json.loads((ARTIFACT_DIR / "doc_ids.json").read_text(encoding="utf-8"))
    config = json.loads((ARTIFACT_DIR / "service_config.json").read_text(encoding="utf-8"))

    assert index.ntotal == len(doc_ids)
    assert index.ntotal == 57_638
    assert config["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert config["index_type"] == type(index).__name__
