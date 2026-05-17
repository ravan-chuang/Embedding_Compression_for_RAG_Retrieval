# AI Embedding Compression for Efficient Semantic Search
**Key result:** Random Rotation + int4 achieved about **8x compression** while maintaining **Top-5 Recall above 0.90**.

A benchmark project for evaluating **AI embedding compression** methods in semantic retrieval systems.  
This project compares scalar quantization, vector quantization, product quantization, and random rotation on sentence embeddings.

## Why This Project Matters

Modern AI applications such as RAG, semantic search, recommendation systems, and vector databases store large numbers of high-dimensional embeddings.

Float32 embeddings are accurate but memory-intensive. This project studies whether lower-bit representations can reduce storage cost while preserving retrieval quality.

## Problem Statement

Given text embeddings stored as float32 vectors, evaluate whether compression methods can reduce storage size while maintaining semantic retrieval accuracy.

This project focuses on the trade-off between:

- Compression ratio
- Reconstruction error
- Cosine similarity
- Inner product distortion
- Top-5 retrieval recall

## Dataset and Setup

- Dataset: AG News subset
- Samples: 4,000 texts, 1,000 per class
- Train / Test split: 80% / 20%
- Test set size: 800
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Retrieval metric: Top-5 Recall

## Methods Compared

| Category | Methods |
|---|---|
| Baseline | Float32 embeddings |
| Scalar Quantization | int8, int4, int2 |
| Random Rotation | rotation + int8 / int4 / int2 |
| Vector Quantization | Single-codebook VQ |
| Product Quantization | PQ with m=8/16/32 and k=32/64 |
| PQ Enhancement | L2 normalization after PQ reconstruction |

## Key Results

| Method | Compression Ratio | Avg Cosine | Top-5 Recall | Notes |
|---|---:|---:|---:|---|
| Float32 Original | 1.00x | 1.0000 | 1.0000 | Baseline |
| Rotation + int8 | 4.00x | 0.9999 | 0.9958 | Highest retrieval quality among compressed methods |
| int8 | 4.00x | 0.9999 | 0.9938 | Near-lossless compression |
| Rotation + int4 | 8.00x | 0.9780 | 0.9058 | Best practical method under Recall ≥ 0.9 |
| int4 | 8.00x | 0.9720 | 0.9023 | Strong baseline |
| Rotation + PQ m32 k64 | 10.46x | 0.6890 | 0.6305 | Best PQ result |
| PQ m32 k32 | 18.86x | 0.6509 | 0.5785 | Higher compression, lower recall |

## Main Finding

The best practical setting is:

> **Random Rotation + int4 Scalar Quantization**

It achieved approximately **8x compression** while maintaining **Top-5 Recall above 0.90**.

This means that, for this dataset and embedding model, int4 scalar quantization provides the best practical trade-off between storage reduction and retrieval quality.

## PQ Analysis

Product Quantization improved after increasing the dataset size, using `m=32`, testing `k=32/64`, and applying L2 normalization after reconstruction.

Best PQ result:

- Method: `rotation_pq_m32_k64_l2norm`
- Compression ratio: 10.46x
- Top-5 Recall: 0.6305

However, basic PQ still underperformed int4 scalar quantization in retrieval quality. This suggests that stronger methods such as OPQ, Residual PQ, or learned quantization would be needed to make PQ more competitive.

## Visual Results

### Compression Ratio vs Retrieval Quality

![Compression Ratio vs Retrieval Quality](compression_vs_retrieval.png)

### Top-5 Recall by Method

![Top-5 Recall](top5_recall.png)

### Compression Ratio by Method

![Compression Ratio](compression_ratio.png)

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook ai_embedding_compression_experiment.ipynb
```

Or run the Python script:

```bash
python ai_embedding_compression_experiment.py
```

## Project Files

```text
Ai-embedding-compression/
├── README.md
├── requirements.txt
├── ai_embedding_compression_experiment.ipynb
├── ai_embedding_compression_experiment.py
├── metrics_summary.csv
├── practical_methods_recall_ge_0_9.csv
├── pq_ranking.csv
├── compression_vs_retrieval.png
├── top5_recall.png
├── compression_ratio.png
├── avg_cosine.png
├── resume_bullets.md
├── interview_pitch.md
├── project_summary_zh.md
└── portfolio_card.md
```

## Engineering Takeaways

- int8 is nearly lossless but only provides about 4x compression.
- int4 offers the best practical trade-off, achieving about 8x compression with Top-5 Recall above 0.90.
- int2 is too aggressive for semantic retrieval.
- Single-codebook VQ is too coarse for high-dimensional sentence embeddings.
- PQ benefits from larger `m`, larger `k`, more training data, and L2 normalization, but basic PQ still trails int4.
- Random rotation improves scalar quantization and connects this project to ideas used in modern AI compression methods such as TurboQuant.

## Limitations

- The experiment uses AG News instead of real RAG documents.
- The current implementation uses brute-force retrieval instead of an ANN index.
- PQ is implemented as a baseline and does not include OPQ, Residual PQ, or learned quantization.
- Search latency and memory usage are estimated through compression ratio, not benchmarked with a production vector database.

## Future Work

- Integrate Faiss `IndexPQ`, `IndexIVFPQ`, or OPQ.
- Evaluate search latency and index size on 100K+ embeddings.
- Add Residual PQ or learned quantization.
- Test with real RAG documents instead of AG News.
- Compare against production vector databases such as OpenSearch, Qdrant, Milvus, or Faiss.

## Portfolio Description

This project demonstrates practical skills in:

- Python experiment design
- Embedding-based retrieval
- Quantization and compression
- Evaluation metrics
- Data-driven engineering decisions
- AI system optimization
