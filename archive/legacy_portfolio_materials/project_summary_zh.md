# 作品集專案摘要：AI Embedding 壓縮與語意搜尋效能分析

## 專案定位

這是一個針對語意搜尋系統的 embedding 壓縮實驗專案。  
專案比較 scalar quantization、VQ、PQ 與 random rotation 等方法，分析不同壓縮策略對儲存成本與 Top-5 retrieval recall 的影響。

## 實驗設定

- Dataset：AG News
- 總資料量：4,000 筆文本
- Train / Test：80% / 20%
- Embedding model：all-MiniLM-L6-v2
- Embedding 維度：384
- 評估指標：compression ratio、MSE、avg cosine、inner product error、Top-5 Recall

## 主要結果

| 方法 | 壓縮率 | Avg Cosine | Top-5 Recall |
|---|---:|---:|---:|
| Rotation + int8 | 4.00x | 0.9999 | 0.9958 |
| int8 | 4.00x | 0.9999 | 0.9938 |
| Rotation + int4 | 8.00x | 0.9780 | 0.9058 |
| int4 | 8.00x | 0.9720 | 0.9022 |
| Rotation + PQ m32 k64 | 10.46x | 0.6889 | 0.6320 |

## 核心結論

若要求 Top-5 Recall ≥ 0.9，最佳實用方法是 **Rotation + int4 Scalar Quantization**。

它可以達到約 **8x 壓縮率**，同時維持 **Top-5 Recall 超過 0.90**。  
PQ 經過資料量增加、m=32/k=64 與 L2 normalization 後有改善，但 retrieval quality 仍低於 int4。

## 適合放在履歷的重點

這個專案展示了：

- embedding retrieval 系統的壓縮評估能力
- 量化與資料壓縮概念
- Python / NumPy / scikit-learn / sentence-transformers 實作能力
- 以指標做工程決策的能力
- AI 系統儲存成本與搜尋品質 trade-off 分析能力
