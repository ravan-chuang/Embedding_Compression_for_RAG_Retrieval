# TREC DL 2019 MS MARCO 1M Restricted External Results

This is a corpus-restricted external sensitivity analysis over 42 eligible
TREC DL 2019 passage queries.

It is not an official full-corpus TREC DL 2019 benchmark result.

## Aggregate metrics

| System | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Base IVF-PQ M32 | 0.3507 | 0.8095 | 0.6939 | 0.4405 |
| PCA rank-16 int8 | 0.3445 | 0.8095 | 0.7093 | 0.4558 |
| RARS rank-16 int8 | 0.3264 | 0.7857 | 0.7341 | 0.4624 |

## Preregistered primary contrast

RARS minus PCA Recall@10:

- Difference: -0.0181
- 95% paired-bootstrap CI: [-0.0735, 0.0168]
- Probability difference > 0: 0.2972
- Bootstrap replicates: 20,000

The preregistered primary hypothesis was not supported.

## Interpretation

RARS showed lower Recall@10 than PCA and Base, while achieving higher point
estimates for MRR@10 and nDCG@10. This suggests a possible trade-off between
relevant-document coverage and early-rank quality.

No fitting, selection, or retuning was performed during external evaluation.

## Limitations

Only 502 of the original 4,102 positive TREC DL 2019 passage judgments were
present in the frozen MS MARCO 1M subset. Results are conditional on that corpus
subset and must not be reported as official full-corpus benchmark scores.
