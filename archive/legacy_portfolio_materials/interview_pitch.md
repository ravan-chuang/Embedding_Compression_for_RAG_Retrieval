# Interview Pitch

## 30-second version

I built an embedding compression benchmark for semantic search. The project compares scalar quantization, VQ, PQ, and random rotation on AG News sentence embeddings. I evaluated compression ratio, MSE, cosine similarity, inner product error, and Top-5 retrieval recall. The key finding was that random-rotation int4 quantization achieved about 8.0x compression while keeping Top-5 Recall above 0.91, making it the best practical trade-off in this experiment.

## 60-second version

This project studies how to reduce the storage cost of embeddings in semantic search systems. I used 4,000 AG News samples, converted them into 384-dimensional sentence embeddings, and compared several compression methods: scalar int8/int4/int2 quantization, single-codebook VQ, product quantization, and random rotation.

Instead of only measuring compression ratio, I also evaluated reconstruction error, average cosine similarity, inner product distortion, and Top-5 retrieval recall. The most practical result was random rotation plus int4 scalar quantization, which achieved around 8.0x compression while maintaining Top-5 Recall above 0.91. I also tested stronger PQ settings with m=32 and k=64 plus L2-normalized reconstruction. PQ improved, but still did not match int4 retrieval quality.

The main engineering takeaway is that embedding compression should be selected based on a quality threshold, not compression ratio alone.

## 中文 60 秒版本

我做的是一個 AI embedding 壓縮與語意搜尋效能分析專案。這個專案的動機是：RAG、語意搜尋或向量資料庫會儲存大量 float32 embeddings，資料量一大會造成記憶體與儲存成本上升。

我使用 4,000 筆 AG News 文本，把它們轉成 384 維 sentence embeddings，然後比較 scalar quantization、VQ、PQ 和 random rotation。評估指標不是只看壓縮率，也包含 MSE、cosine similarity、inner product error 和 Top-5 Recall。

最後發現 random rotation + int4 可以達到約 8.0x 壓縮，並維持 Top-5 Recall 超過 0.91，是這次實驗中最佳的實用設定。PQ 在 m=32、k=64 和 L2 normalization 後有改善，但 retrieval quality 仍低於 int4。

這個專案讓我學到，壓縮不是壓得越小越好，而是要根據搜尋品質門檻做工程取捨。
