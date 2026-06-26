# Retrieval API

This folder turns the benchmark into a small, CPU-portable retrieval service.

## Artifact contract

```text
artifacts/fiqa_ivfpq_m96/
├── index.faiss            # tracked in Git
├── service_config.json    # tracked in Git
├── doc_ids.json           # tracked in Git; preserves index-row order
└── documents.jsonl        # generated locally; ignored by Git
```

`documents.jsonl` is intentionally not stored in Git because it is a 45 MB
reconstructable copy of FiQA document metadata.

## One-time local setup

Create the metadata file from the official FiQA / BEIR corpus:

```bash
python scripts/prepare_fiqa_documents.py
```

The script downloads FiQA on first use, then writes
`artifacts/fiqa_ivfpq_m96/documents.jsonl` in the exact document order used by
the serialized Faiss index.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
python scripts/prepare_fiqa_documents.py
uvicorn app.main:app --reload
```

Open interactive API docs at:

```text
http://127.0.0.1:8000/docs
```

## Example requests

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a dividend stock?","top_k":5,"nprobe":16}'
```

## Docker

Before building the current Docker image, generate `documents.jsonl` locally:

```bash
python scripts/prepare_fiqa_documents.py
docker build -t embedding-retrieval-api .
docker run --rm -p 8000:8000 embedding-retrieval-api
```

## Scope

This is a retrieval component, not a complete generative RAG application. It
returns ranked FiQA source documents and scores. A later service layer can add
reranking, prompt construction, an LLM, observability, request batching, and a
Linux/NVIDIA GPU deployment path.
