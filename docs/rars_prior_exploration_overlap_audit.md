# RARS Prior-Exploration Overlap Audit

## Status correction

The deterministic `4,980 / 1,000 / 1,000` split correctly separates fitting,
validation selection, and held-out evaluation inside the clean pipeline.
However, the split generator did not exclude the 1,000 queries used by an
earlier exploratory study.

A query-ID comparison found:

| Destination in clean split | Earlier exploratory queries |
|---|---:|
| Train | 729 |
| Validation | 134 |
| Held-out test | 137 |

Therefore, the 1,000-query result is **held out from the clean train/validation
pipeline**, but it is not fully untouched across the complete project history.

## Contamination-excluded sensitivity analysis

The 137 previously explored test queries were removed using query IDs only.
No outcome-dependent filtering or retuning was performed. Metrics were then
recomputed on the remaining 863 prior-unseen queries.

| Metric | Frozen M32 | Frozen RARS | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Recall@10 | 0.6956 | 0.7124 | +0.0168 | [+0.0029, +0.0303] |
| Success@10 | 0.7034 | 0.7231 | +0.0197 | [+0.0058, +0.0336] |
| MRR@10 | 0.4809 | 0.4915 | +0.0106 | [-0.0003, +0.0218] |
| nDCG@10 | 0.5299 | 0.5422 | +0.0123 | [+0.0030, +0.0218] |

Recall@10, Success@10, and nDCG@10 retain confidence intervals above zero.
MRR@10 remains directionally positive but its interval narrowly crosses zero.

## Interpretation boundary

This audit supports the claim that the observed improvement is not entirely
driven by the 137 previously explored queries. It does **not** create a new
project-history-level untouched test set, because all 6,980 MS MARCO development
queries have now influenced fitting, selection, evaluation, or subsequent
analysis.

The project should therefore use these terms:

- **held-out test** for the 1,000-query clean-pipeline test;
- **prior-exploration-excluded sensitivity subset** for the 863-query audit;
- **external confirmation set** for a future pre-registered evaluation on new
  queries such as TREC Deep Learning passage queries.

## Approved headline

> On a 1,000-query held-out split, frozen RARS improves Recall@10 from 0.6833
> to 0.7073. A query-ID-only sensitivity audit excluding 137 queries used in
> earlier exploratory work retains a Recall@10 gain of +0.0168 on the remaining
> 863 prior-unseen queries, with a 95% paired-bootstrap interval of
> [+0.0029, +0.0303].

## Prohibited headline

Do not describe the current 1,000-query result as fully untouched across the
entire project history.
