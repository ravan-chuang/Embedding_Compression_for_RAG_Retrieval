# RARS 全實驗審查、失敗歸因與 V9 後續計畫

## 結論先行

截至 2026-07-22，最可靠的研究結論不是「RARS 已普遍勝過 PQ/PCA」，而是：

1. 在**既有 IVF-PQ 索引不可重建**的限制下，16 B/doc 的 document-residual sidecar 確實能恢復一部分 Top-k loss；
2. V8 是目前第一次同時以相同儲存、相同 int8 scorer、相同 Top-B，顯著勝過 Base 與 PCA 的方法版本；
3. V8 證據仍是 2,307-query outcome-informed five-fold OOF development，不是獨立確認；
4. 下一個唯一主實驗應是凍結 V8 的一次性 V9 confirmation，不是再換 loss、rank、seed 或 adapter；
5. 就算 V9 通過，仍需新的外部 collection/model 與完整效率實驗，才足以形成有競爭力的 SIGIR short paper 證據鏈。

完整數值登錄於
[`rars_experiment_evidence_registry_v1.json`](rars_experiment_evidence_registry_v1.json)，
V9 的機器可讀契約在
[`rars_v9_locked_confirmation_v1.json`](../protocols/rars_v9_locked_confirmation_v1.json)。

## 實驗結果總表

| 階段 | 主要結果 | 正式判定 | 證據能支持什麼 |
|---|---|---|---|
| Fixed-budget residual PQ | 部分 residual 方案勝過 M32，但可重建時 uniform M48 更強 | 不作主方法 | RARS 必須定位為 frozen-index retrofit，而不是全域碼率最優 |
| V1 clean split | R@10 `0.68333→0.70733`，`+0.02400`，CI `[+0.01050,+0.03783]` | 正向但有污染稽核 | Sidecar 對 Base 有效；尚未證明勝過 PCA |
| FiQA transfer | BGE `+0.01311`、MiniLM `+0.00962`，兩個 CI 均跨 0 | directional | 有轉移方向，沒有確認證據 |
| TREC DL 2019 restricted | RARS−PCA `−0.01812`，CI `[−0.07351,+0.01685]`，42 queries | primary unsupported | 不能主張外部優勢；coverage 僅約 12.24% |
| BEIR NQ | RARS−PCA 約 `−0.00041`，CI `[−0.00599,+0.00497]`，3,452 queries | unsupported | exact Top40 有 headroom，但舊 proxy 與 relevance 相關僅約 0.15 |
| V2 | R@10 崩至 `0.38450` | failure | train objective 與 deploy scorer 不一致會直接失效 |
| V2.1 | int8 boundary `0.67867` vs PCA `0.70433` | `NO_GO_OR_REVISE` | deployable 不等於優於 PCA |
| V2.2 | held-out seed mean `+0.00769` vs PCA；seed44 僅 10 個改善、門檻 11 | `UNSTABLE_NO_QAT` | 平均值不能取代 query-level breadth |
| V3 oracle | fixed-access comparator 未留下足夠 allocator headroom | `STOP_NO_HEADROOM` | 停止 adaptive storage allocator 主線 |
| V5 PQ-aware 100K | R@10 `−0.00206`；R@100 `+0.00275`；僅 24 flip pairs、2 queries 改善 | `STOP_PQ_AWARE_100K_PILOT` | 100K/M32 太接近天花板，adapter 易過擬合 |
| V6 1M diagnostic | Base R@100 `.84731`，same-IVF exact `.89395`，full exact `.97298` | diagnostic GO | PQ gap `.04663` 真實存在，但僅佔總 gap 37.11%，routing 更大 |
| V7 query adapter | R@100 `+0.00758`，CI 跨 0；query cosine `.94682` | `STOP_V7_QUERY_ADAPTER_PILOT` | query-side 補償 document PQ error 造成過大 drift |
| V8 cutoff sidecar | Base `.67992`、PCA `.69264`、RARS `.70282`；RARS−PCA `+.01019`，CI 全正 | `GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL` | 第一次有演算法級 development signal；尚未 confirmation |

## V8 真正新增的證據

V8 的價值不是單純再得到一個正數，而是把先前失敗逐一變成設計約束：

- **直接在 int8 deploy scorer 中評估**，避免 V2 的 FP32/train-deploy mismatch；
- **以 Top-10 promotion/protection pair 為主**，避免全域重建 MSE 與 retrieval outcome 脫節；
- **role mass 及 query mass 平衡**，避免 V2.2 的少數 query 支配平均；
- **只修正 Base Top40、保留 frozen candidates/index**，維持 retrofit 主張；
- **與同 rank、同 int8、同 alpha、同 Top-B 的 PCA 比較**，不再只挑弱 Base；
- **五折 OOF development**，比同資料直接 fit/evaluate 更嚴格。

結果為 Base `.679923`、PCA `.692638`、RARS `.702825`、same-candidate exact
`.773768`。RARS 相對 Base 提升 `+.022901`，95% CI
`[+.014666,+.031426]`，80 queries 改善、25 受損；相對 PCA 提升
`+.010186`，CI `[+.003468,+.017122]`，45 改善、22 受損。候選層 gap
recovery 為 24.40%。完整 sidecar 為 16.025024 B/doc，0 saturation，原 M32
index hash 前後一致。

這些數字足以進入確認，卻不允許把 OOF 稱為獨立結果。OOF 限制模型對單一
query 的直接擬合，但 pair 定義、objective、gate 與整體研究方向仍由同一
`oracle_design` outcome-informed 開發形成。

## 已知失敗的共同根因

### 1. 問題空間與信號密度不匹配

V5 的 Base R@100 已達 `.97527`，只有 24 個 PQ-induced flip pairs。這不是
「seed 不夠多」，而是可學習的量化錯誤太少。V6 擴到 1M 後，Base R@100
降至 `.84731` 且得到 4,413 個 flips，才證明問題有足夠 headroom。

### 2. 最佳化 proxy 與最終排名不一致

重建 MSE、score-error regression、FP32 margin 與 Recall@10 並非同一目標。
NQ 的 exact Top40 headroom 達 `+.08379`，但 RARS proxy 與 relevance Pearson
僅 `.1502`；這解釋了「理論上可修、實際方法沒修到」。V8 改成 cutoff-aware
pair 是正確方向，但仍是 margin regression，不是真正的 listwise Recall。

### 3. 修錯誤發生的位置

PQ residual 在 document/index side。V7 只移動 query，雖有小幅 point gain，
但平均 cosine 降至 `.94682`，遠低於 `.995` guardrail。這代表全域 query
幾何被破壞，且不能改變已儲存的 document reconstruction error。

### 4. Candidate routing 與 PQ ranking 被混為一談

V6 顯示 PQ-specific gap 只佔 Base-to-full-exact gap 的 37.11%。RARS 只能重排
已進入 probed Top100 的文件，不能召回未探測文件。因此 V9 必須同時報
same-candidate exact、`nprobe=32/64` 與 M48，不能把 routing loss 說成 sidecar
能解決。

### 5. 平均提升掩蓋稀疏、脆弱效果

V2.2 的 mean 與 CI 都不差，仍因 seed44 只改善 10 queries 而停止。V5 更只有
2 queries 改善。往後所有 gate 必須同時包含 effect size、CI、improved/harmed
與 net support，禁止只用均值挑成功故事。

### 6. 外部結果與開發結果不一致

TREC 的 primary direction 為負，NQ 接近零。這表示早期 RARS 方法沒有建立
跨 corpus 的演算法優勢。V8 尚未外部驗證，所以「V8 解決泛化」目前仍是假說。

### 7. V8 optimizer 記錄異常

V8 完整資料的 recorded surrogate loss 從 `.0756863` 升到 `.0876917`，五折皆
同方向。梯度公式與 `basis -= AdamStep` 表面正確，但舊日誌只記 update 前
objective，無法辨別 Adam proposal 與 QR retraction 的責任。故目前可以主張
empirical artifact 有效，不能主張 loss 收斂或 optimizer 正確最小化。

## 改良後的主實驗：V9 locked confirmation

V9 **不是新演算法版本**；它是第一次真正把 V8 方法、artifact、統計與停止
條件一起鎖死的確認程序。

### 資料邊界

使用 803-query `future_method_holdout`。該 role 在 V3--V8 均維持 identity-only，
所以相對 V8 是 prospective；但它源自早期 v2 `inner_train` 母池，因此證據名稱
只能是 `WITHIN_PROGRAM_PROSPECTIVE_HOLDOUT_NOT_INDEPENDENT`。

### 一次性執行順序

1. 驗證乾淨、精確的 source commit 與 canonical protocol；
2. 驗證 V8 `method_freeze`、PCA/RARS basis/scales/codes 的 SHA-256；
3. 由 frozen train split 與 global query vectors 建立、驗證完全 qrels-free 的
   future identity packet；不重走會解析 qrels 的舊 V2 candidate builder；
4. 驗證 byte-identical M32 與 qrels-free rebuilt M48；
5. durable 寫入 `input_freeze.json` 與 `confirmation_started.json`，此時
   `outcome_opened=false`；
6. 第一次且只一次讀 qrels，計算 Base/PCA/RARS/exact/nprobe/M48；
7. 寫 per-query arrays、統計、正式 decision 與 completion marker；
8. 無論成敗，不在該 role 上重跑、改 threshold 或救結果。

Primary endpoint 只有 `Recall@10(RARS-v8)-Recall@10(PCA)`。Algorithm tier 要求
至少 `+.005`、paired-bootstrap lower > 0、one-sided paired randomization
`p≤.025`、至少 15 queries 改善且 net +8；同時 RARS−Base 至少 `+.01`、gap
recovery 至少 15%、改善 breadth 過門檻。若 RARS 未勝 PCA，但 PCA 與 RARS
都證明 generic sidecar 對 Base 有效，只能落在 generic-sidecar tier。

## Loss-direction audit：不污染 V9 的平行診斷

已為 projected Adam 增加不改變 update 的記錄欄位：

- pre-update pair、anchor、total objective；
- Adam proposal 後 objective；
- QR retraction 後 objective；
- proposal、retraction、full-step 三段 loss change；
- gradient norm。

診斷只在 development data 進行，另加 finite-difference gradient check、每個
checkpoint 的 Recall/query support/subspace distance。若只是欄位命名問題，修正
文件即可；若是 gradient 或 QR 問題，現有 V8 仍保留為 empirical artifact，修正版
必須叫 V8.1/V10 並重新走 development protocol，不能覆寫 V8 confirmation。

## Confirmation 之後的方法分支

### 若 V9 algorithm tier 通過

優先只開一條新方法線：**IVF-cell-group-conditioned cutoff-aware sidecars**。
原因是 V8 已支持 cutoff-aware objective，下一個可檢驗瓶頸才是單一 global
rank-16 basis 的容量。第一個 pilot 應用 8 個 deterministic cell groups、每份
rank-16 basis、document code仍 16 B/doc；group 由既有 IVF list 決定，不新增
learned query router。開發 gate 建議為相對 global V8 `+0.003` R@10、CI lower>0、
至少 6/8 groups 非負、amortized metadata ≤0.5 B/doc。

先做 local subspace，不同時再改 listwise loss與 pair mining，否則無法歸因。

### 若 V9 失敗

不要立刻增加 basis 數或 rank。先診斷：

- development 與 future 的 cutoff density、positive count、teacher/base flip 分布；
- promotion/protection support shift；
- unjudged challenger 的 false-negative 風險；
- V8 basis 維度的 principal angle、query energy 與 pair alignment；
- harmful query 的 rank movement severity。

下一版應優先處理 confidence-weighted unjudged challengers、multi-positive
set-level mining 與 distribution shift；容量增加只有在確認 global basis 明顯
欠擬合時才合理。

## 論文仍缺的兩組實驗

1. **External confirmation**：至少一個新 corpus/same model，加一個新
   corpus/different model；方法、rank、alpha、Top-B 不可在該資料上選。
2. **System Pareto**：同硬體、同 batch 測 M32、M32+PCA、M32+RARS、higher
   nprobe、M48；報 correction-only 與 end-to-end P50/P95/P99、QPS、load time、
   peak memory、完整 artifact bytes。Python prototype 不得稱 fused production
   kernel。

## 建議排程與停止規則

| 階段 | 產物 | 成功條件 | 失敗後行動 |
|---|---|---|---|
| A. 程式與輸入凍結 | V9 protocol、M48 qrels-free manifest、evaluator CI | 所有 contract test、SHA、identity audit 過關 | 不開 qrels，修工程問題 |
| B. 一次性 V9 | complete packet + per-query arrays | algorithm 或 generic tier | 不重跑；進 failure diagnosis |
| C. mechanism audit | loss-direction、basis/principal-angle、harm analysis | 能解釋主要 gain/harm 來源 | 不開新容量線 |
| D. 外部確認 | 新資料/模型 locked results | 主要方向可複製、CI/支持充分 | 收窄論文主張或轉 workshop |
| E. 系統評估 | accuracy-storage-latency Pareto | 在 frozen-index限制下有清楚非支配點 | 不作 production claim |
| F. 下一代方法 | local subspace 或 noise-aware mining，擇一 | 對 global V8 有預註冊增益 | 停止疊版本 |

## 最終研究定位

目前最合理的論文問題是：

> 在無法重訓 encoder、重建 IVF-PQ index 或重寫 PQ codes 的部署限制下，能否以
> 16 B/doc、cutoff-aware 的 residual sidecar，穩定恢復 Top-k ranking loss？

這個定位比「全面解決 PQ recall loss」更精確，也更能解釋為什麼要使用 sidecar
而不是 uniform M48、full reranking 或 end-to-end QAT。V9 的任務就是判斷這個
主張能否從 promising development result 升級為可寫入論文主結果的證據。
