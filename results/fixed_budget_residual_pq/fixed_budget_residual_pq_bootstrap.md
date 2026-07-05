# Paired Bootstrap Significance: Fixed-Budget Residual-PQ

All comparisons use the held-out FiQA query set.
Delta is defined as method_a minus method_b.

| method_a                                   | method_b                                   | metric       |   heldout_query_count |   bootstrap_resamples |   point_delta |    ci_low_95 |   ci_high_95 | ci_excludes_zero   |
|:-------------------------------------------|:-------------------------------------------|:-------------|----------------------:|----------------------:|--------------:|-------------:|-------------:|:-------------------|
| residual_pq_8b_reconstruction_error_top50  | base_ivfpq_m32                             | recall_at_10 |                   528 |                 10000 |   0.0261981   |  0.0142595   |    0.0389064 | True               |
| residual_pq_8b_reconstruction_error_top50  | base_ivfpq_m32                             | mrr_at_10    |                   528 |                 10000 |   0.0281739   |  0.0139279   |    0.0428902 | True               |
| residual_pq_8b_reconstruction_error_top50  | base_ivfpq_m32                             | ndcg_at_10   |                   528 |                 10000 |   0.02513     |  0.0162384   |    0.0345124 | True               |
| residual_pq_16b_reconstruction_error_top50 | base_ivfpq_m32                             | recall_at_10 |                   528 |                 10000 |   0.0287834   |  0.0182358   |    0.0401519 | True               |
| residual_pq_16b_reconstruction_error_top50 | base_ivfpq_m32                             | mrr_at_10    |                   528 |                 10000 |   0.0281896   |  0.0153345   |    0.042295  | True               |
| residual_pq_16b_reconstruction_error_top50 | base_ivfpq_m32                             | ndcg_at_10   |                   528 |                 10000 |   0.0282405   |  0.0187593   |    0.0381661 | True               |
| residual_pq_32b_reconstruction_error_top50 | base_ivfpq_m32                             | recall_at_10 |                   528 |                 10000 |   0.00762536  |  0.000288488 |    0.0156163 | True               |
| residual_pq_32b_reconstruction_error_top50 | base_ivfpq_m32                             | mrr_at_10    |                   528 |                 10000 |   0.0126353   |  0.0013977   |    0.0246987 | True               |
| residual_pq_32b_reconstruction_error_top50 | base_ivfpq_m32                             | ndcg_at_10   |                   528 |                 10000 |   0.0110693   |  0.00346646  |    0.0193953 | True               |
| residual_pq_16b_reconstruction_error_top50 | residual_pq_8b_reconstruction_error_top50  | recall_at_10 |                   528 |                 10000 |   0.00258531  | -0.0105438   |    0.0156294 | False              |
| residual_pq_16b_reconstruction_error_top50 | residual_pq_8b_reconstruction_error_top50  | mrr_at_10    |                   528 |                 10000 |   1.57828e-05 | -0.0155507   |    0.0151613 | False              |
| residual_pq_16b_reconstruction_error_top50 | residual_pq_8b_reconstruction_error_top50  | ndcg_at_10   |                   528 |                 10000 |   0.00311059  | -0.00714185  |    0.013404  | False              |
| residual_pq_16b_reconstruction_error_top50 | residual_pq_32b_reconstruction_error_top50 | recall_at_10 |                   528 |                 10000 |   0.021158    |  0.0101099   |    0.0327159 | True               |
| residual_pq_16b_reconstruction_error_top50 | residual_pq_32b_reconstruction_error_top50 | mrr_at_10    |                   528 |                 10000 |   0.0155544   |  0.00108208  |    0.0308187 | True               |
| residual_pq_16b_reconstruction_error_top50 | residual_pq_32b_reconstruction_error_top50 | ndcg_at_10   |                   528 |                 10000 |   0.0171712   |  0.00708127  |    0.0275889 | True               |
| uniform_ivfpq_m48                          | residual_pq_16b_reconstruction_error_top50 | recall_at_10 |                   528 |                 10000 |   0.0260786   |  0.00713175  |    0.0460016 | True               |
| uniform_ivfpq_m48                          | residual_pq_16b_reconstruction_error_top50 | mrr_at_10    |                   528 |                 10000 |   0.0316664   |  0.00752721  |    0.0551872 | True               |
| uniform_ivfpq_m48                          | residual_pq_16b_reconstruction_error_top50 | ndcg_at_10   |                   528 |                 10000 |   0.0264555   |  0.0105173   |    0.0424957 | True               |

Interpretation:
- A CI excluding zero indicates a directional difference under this paired bootstrap procedure.
- The test validates the current held-out split; it is not cross-dataset evidence.