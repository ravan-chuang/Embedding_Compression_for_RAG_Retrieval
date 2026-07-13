# External Query-Level Diagnostics

This directory contains post-hoc diagnostics for the frozen 42-query
TREC DL 2019 MS MARCO 1M corpus-restricted evaluation.

No retrieval, fitting, selection, or retuning was performed.

## RARS versus PCA query outcomes

| Metric | Win | Tie | Loss | Mean difference |
|---|---:|---:|---:|---:|
| Recall@10 | 3 | 37 | 2 | -0.0181 |
| Success@10 | 0 | 41 | 1 | -0.0238 |
| MRR@10 | 2 | 39 | 1 | +0.0248 |
| nDCG@10 | 11 | 20 | 11 | +0.0066 |

## Recall influence

Only five queries had different Recall@10 values between RARS and PCA.

The largest influence came from query `962179`:

- Query: `when was the salvation army founded`
- Indexed positive judgments: 1
- PCA Recall@10: 1.0
- RARS Recall@10: 0.0
- Per-query difference: -1.0

Removing this single query changes the mean RARS-minus-PCA Recall@10
difference from -0.0181 to +0.0058.

This is an influence diagnostic only. The query remains part of the frozen
primary evaluation, and the preregistered result is unchanged.

## Aggregate metric heterogeneity

No query simultaneously showed lower RARS Recall and higher RARS MRR or nDCG
relative to PCA. Therefore, the lower aggregate Recall and higher aggregate
MRR/nDCG arise from different query subsets rather than a direct within-query
coverage versus early-rank trade-off.

## Limitations

The analysis contains only 42 eligible queries and is conditional on the frozen
MS MARCO 1M corpus subset. Results are not official full-corpus TREC DL 2019
benchmark scores.
