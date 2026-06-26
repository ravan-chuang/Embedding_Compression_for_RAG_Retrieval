# v1.0.0 — Retrieval Engineering Release

## Highlights

- GPU Faiss IVF-PQ ADC and OPQ-IVF-PQ benchmark on FiQA / BEIR.
- Serialized 57,638-document `IndexIVFPQ` retrieval artifact.
- FastAPI service with `/health`, `/search`, and true matrix-batched `/batch-search`.
- Local API benchmark results for batch sizes 1, 8, and 32.
- Verified Docker Compose deployment.
- Offline unit tests plus GitHub Actions CI.

## Verified workflow

```text
GPU benchmark → serialized IVF-PQ artifact → FastAPI serving
→ local API benchmark → Docker Compose deployment → automated CI
```

## Notes

The GPU benchmark uses Google Colab with an NVIDIA Tesla T4. The API and Docker
deployment provide a local CPU-serving path for the serialized retrieval artifact.
