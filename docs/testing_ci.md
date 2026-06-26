# Testing and Continuous Integration

The repository includes offline unit tests and a GitHub Actions workflow.

## Local test run

Create the Conda environment and install the development dependencies:

```bash
conda env create -f environment.yml
conda activate rag-api
pip install -r requirements-dev.txt
pytest -q
```

The tests are intentionally small and do not download FiQA or an embedding model.

## What is tested

- Retrieval artifact contract: the serialized Faiss index count matches the tracked
  `doc_ids.json` ordering.
- Retriever loading, ranking, batch matrix search, and index/document mismatch handling.
- FastAPI endpoint functions for health, single-query retrieval, and batch retrieval.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests. It:

1. creates the Conda environment from `environment.yml`,
2. installs application and test dependencies,
3. executes `pytest -q`.

The workflow validates the CPU retrieval code path only. GPU benchmark execution
remains in the Colab notebook because GitHub-hosted runners do not provide the
NVIDIA runtime used for the benchmark.
