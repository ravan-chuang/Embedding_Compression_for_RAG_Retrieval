# FiQA BGE-small RARS Transfer Validation

This experiment validates retrieval-aware residual sidecar correction on FiQA with `BAAI/bge-small-en-v1.5`.

## Setting

- Dataset: FiQA / BEIR test queries
- Documents: 57,638
- Queries: 648
- Embedding model: `BAAI/bge-small-en-v1.5`
- Frozen index: IVF-PQ, M=32, nlist=256, nprobe=16
- Candidate pool: Top-100 IVF-PQ candidates
- Sidecar rank: 16

## Qrels-based final metrics

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| IVF-PQ base | 0.2935 | 0.4784 | 0.2964 | 0.2373 |
| Score-error RARS Top10 | 0.2967 | 0.4892 | 0.3101 | 0.2435 |
| Score-error RARS Top20 | 0.3184 | 0.5062 | 0.3180 | 0.2559 |
| Score-error RARS Top40 | 0.3235 | 0.5201 | **0.3220** | 0.2587 |
| Score-error RARS Top100 | 0.3232 | 0.5185 | 0.3213 | 0.2583 |
| PCA-current Top20 | 0.3203 | 0.5123 | 0.3137 | 0.2557 |
| PCA-current Top40 | **0.3282** | **0.5231** | 0.3181 | **0.2600** |

## Interpretation

Both PCA-current and score-error-weighted RARS substantially improve over the frozen IVF-PQ baseline on FiQA.

Score-error RARS Top40 gives strong qrels gains over the frozen IVF-PQ baseline:

- Recall@10: 0.2935 -> 0.3235
- Success@10: 0.4784 -> 0.5201
- MRR@10: 0.2964 -> 0.3220
- nDCG@10: 0.2373 -> 0.2587

However, PCA-current Top40 remains a strong baseline and slightly leads Recall@10, Success@10, and nDCG@10 in this setting.

Conservative conclusion:

> On FiQA BGE-small, residual sidecar correction improves frozen IVF-PQ retrieval. Score-error-weighted RARS improves proxy score-error alignment and achieves the strongest MRR@10, while current-setting PCA remains competitive and slightly leads some qrels metrics.
