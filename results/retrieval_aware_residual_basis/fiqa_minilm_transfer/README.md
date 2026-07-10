# FiQA MiniLM RARS Transfer Validation

This experiment validates retrieval-aware residual sidecar correction on FiQA with `sentence-transformers/all-MiniLM-L6-v2`.

## Setting

- Dataset: FiQA / BEIR test queries
- Documents: 57,638
- Queries: 648
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Frozen index: IVF-PQ, M=32, nlist=256, nprobe=16
- Candidate pool: Top-100 IVF-PQ candidates
- Sidecar rank: 16

## Proxy diagnostics

| Basis | Pearson corr. with exact-minus-ANN | Sign agreement | Best alpha MSE |
|---|---:|---:|---:|
| PCA-current | 0.4926 | 0.6909 | 0.8738 |
| Score-error weighted | **0.5342** | **0.7054** | **0.8396** |
| Top10-boundary weighted | 0.5220 | 0.7045 | 0.8961 |

Score-error weighted RARS gives the strongest proxy alignment with exact-minus-ANN score error.

## Qrels-based final metrics

### Fixed alpha = 1.0

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| IVF-PQ base | 0.3446 | 0.5494 | **0.3603** | 0.2862 |
| Score-error RARS Top10 | 0.3333 | 0.5386 | 0.3495 | 0.2757 |
| Score-error RARS Top20 | 0.3428 | 0.5478 | 0.3490 | 0.2789 |
| Score-error RARS Top40 | 0.3440 | 0.5494 | 0.3489 | 0.2795 |
| Score-error RARS Top100 | 0.3500 | **0.5586** | 0.3513 | 0.2822 |
| PCA-current Top20 | 0.3370 | 0.5417 | 0.3461 | 0.2782 |
| PCA-current Top40 | 0.3389 | 0.5432 | 0.3469 | 0.2795 |

With fixed alpha=1.0, qrels metrics are mixed. Score-error RARS Top100 improves Recall@10 and Success@10, but MRR@10 and nDCG@10 remain below the frozen IVF-PQ baseline.

### Proxy-selected alpha = 0.75

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| IVF-PQ base | 0.3446 | 0.5494 | **0.3603** | 0.2862 |
| PCA-current Top40 | 0.3484 | 0.5525 | 0.3550 | 0.2857 |
| Score-error RARS Top40 | 0.3502 | 0.5540 | 0.3577 | 0.2857 |
| Top10-boundary RARS Top40 | **0.3506** | **0.5540** | 0.3595 | **0.2880** |

Using the proxy-selected alpha=0.75 improves the transfer result. The best qrels point is `top10_boundary_weighted_bestproxy_alpha0.75_top40`, which gives modest Recall@10, Success@10, and nDCG@10 gains over the frozen IVF-PQ baseline, while MRR@10 remains essentially flat/slightly lower.

## Interpretation

FiQA MiniLM is a mixed transfer setting:

- Proxy score-error alignment improves clearly.
- Fixed alpha=1.0 is not robust for MiniLM.
- Proxy-selected alpha=0.75 gives small qrels gains in Recall@10, Success@10, and nDCG@10.
- MRR@10 does not improve over the frozen IVF-PQ baseline.
- The result is useful as evidence that RARS transfer is model-sensitive rather than universally positive.

Conservative conclusion:

> On FiQA MiniLM, retrieval-aware correction improves proxy score-error alignment strongly, while qrels gains are small and alpha-sensitive. The best proxy-selected boundary basis gives modest Recall@10 and nDCG@10 gains, but not a clean across-metric win.
