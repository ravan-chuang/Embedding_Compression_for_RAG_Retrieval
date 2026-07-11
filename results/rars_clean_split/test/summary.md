# Clean-Split RARS Untouched Test Results

## Protocol

- Dataset: MS MARCO deterministic 1M passage subset
- Train queries: 4,980
- Validation queries: 1,000
- Untouched test queries: 1,000
- Frozen base index: IVF-PQ M=32, nlist=512, nprobe=16
- Frozen RARS sidecar: rank 16, int8
- Selected basis: score-error weighted
- Alpha: 0.75
- Correction depth: Top-40
- Bootstrap replicates: 10,000

## Main results

| System | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| IVF-PQ M32 | 0.6833 | 0.6910 | 0.4722 | 0.5204 |
| Frozen RARS Top40 | 0.7073 | 0.7180 | 0.4851 | 0.5360 |

## Paired bootstrap differences

| Metric | Difference | 95% CI |
|---|---:|---:|
| Recall@10 | +0.0240 | [+0.0105, +0.0378] |
| Success@10 | +0.0270 | [+0.0130, +0.0410] |
| MRR@10 | +0.0129 | [+0.0030, +0.0229] |
| nDCG@10 | +0.0156 | [+0.0068, +0.0244] |

All four 95% confidence intervals are strictly above zero.
