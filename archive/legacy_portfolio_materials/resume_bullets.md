# Resume Bullets

## English Version

**AI Embedding Compression for Efficient Semantic Search**  
Python, NumPy, scikit-learn, sentence-transformers, AG News

- Built a benchmark pipeline to compare scalar quantization, vector quantization, product quantization, and random rotation for sentence embedding compression.
- Evaluated compression ratio, MSE, cosine similarity, inner product error, and Top-5 retrieval recall on 4,000 AG News samples.
- Found that random-rotation int4 quantization achieved about 8.0x compression while maintaining Top-5 Recall above 0.91.
- Improved PQ experiments with larger codebooks, m=32 sub-vector partitioning, and L2-normalized reconstruction; analyzed why basic PQ still underperforms int4 for retrieval quality.
- Connected the experiment to practical vector search and RAG storage optimization scenarios.

## 中文版本

**AI Embedding 壓縮與語意搜尋效能分析**  
Python, NumPy, scikit-learn, sentence-transformers, AG News

- 實作 scalar quantization、VQ、PQ 與 random rotation，比較不同 embedding 壓縮方法。
- 使用 4,000 筆 AG News 文本轉換為 384 維 sentence embeddings，評估壓縮率、MSE、cosine similarity、inner product error 與 Top-5 Recall。
- 實驗顯示 random rotation + int4 可達約 8.0x 壓縮，且 Top-5 Recall 維持 0.91 以上。
- 透過 m=32、k=32/64 與 L2-normalized reconstruction 改善 PQ 實驗，並分析 basic PQ 仍不如 int4 的原因。
- 將資料壓縮方法連結到 RAG / semantic search / vector database 的儲存成本最佳化問題。
