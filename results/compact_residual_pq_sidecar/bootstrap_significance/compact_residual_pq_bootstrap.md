# Paired Bootstrap: Compact Residual-PQ Sidecars

All tests use the held-out FiQA split.
Delta is defined as method_a minus method_b.

| method_a               | method_b                     | metric       |   heldout_query_count |   bootstrap_resamples |   point_delta |   ci_low_95 |   ci_high_95 | ci_excludes_zero   |
|:-----------------------|:-----------------------------|:-------------|----------------------:|----------------------:|--------------:|------------:|-------------:|:-------------------|
| compact_8bit_m16_top50 | base_ivfpq_m32               | recall_at_10 |                   528 |                 10000 |    0.044779   |  0.0310418  |   0.0597572  | True               |
| compact_8bit_m16_top50 | base_ivfpq_m32               | mrr_at_10    |                   528 |                 10000 |    0.044615   |  0.0255942  |   0.0640128  | True               |
| compact_8bit_m16_top50 | base_ivfpq_m32               | ndcg_at_10   |                   528 |                 10000 |    0.0439521  |  0.0316282  |   0.0568445  | True               |
| compact_8bit_m16_top50 | legacy_residual_pq_16b_top50 | recall_at_10 |                   528 |                 10000 |    0.0159956  |  0.00365459 |   0.0289425  | True               |
| compact_8bit_m16_top50 | legacy_residual_pq_16b_top50 | mrr_at_10    |                   528 |                 10000 |    0.0164254  | -0.00186658 |   0.0348733  | False              |
| compact_8bit_m16_top50 | legacy_residual_pq_16b_top50 | ndcg_at_10   |                   528 |                 10000 |    0.0157116  |  0.00440162 |   0.0272774  | True               |
| uniform_ivfpq_m48      | compact_8bit_m16_top50       | recall_at_10 |                   528 |                 10000 |    0.010083   | -0.00814501 |   0.0286235  | False              |
| uniform_ivfpq_m48      | compact_8bit_m16_top50       | mrr_at_10    |                   528 |                 10000 |    0.015241   | -0.00801343 |   0.0384262  | False              |
| uniform_ivfpq_m48      | compact_8bit_m16_top50       | ndcg_at_10   |                   528 |                 10000 |    0.0107439  | -0.00371663 |   0.0258904  | False              |
| compact_4bit_m32_top50 | base_ivfpq_m32               | recall_at_10 |                   528 |                 10000 |    0.0420477  |  0.0260573  |   0.0586042  | True               |
| compact_4bit_m32_top50 | base_ivfpq_m32               | mrr_at_10    |                   528 |                 10000 |    0.0498745  |  0.0321052  |   0.0682243  | True               |
| compact_4bit_m32_top50 | base_ivfpq_m32               | ndcg_at_10   |                   528 |                 10000 |    0.0408963  |  0.0296346  |   0.0529534  | True               |
| compact_4bit_m32_top50 | compact_8bit_m16_top50       | recall_at_10 |                   528 |                 10000 |   -0.00273125 | -0.0176679  |   0.0122641  | False              |
| compact_4bit_m32_top50 | compact_8bit_m16_top50       | mrr_at_10    |                   528 |                 10000 |    0.00525944 | -0.0137411  |   0.0245901  | False              |
| compact_4bit_m32_top50 | compact_8bit_m16_top50       | ndcg_at_10   |                   528 |                 10000 |   -0.00305587 | -0.0151871  |   0.00947838 | False              |
| uniform_ivfpq_m48      | compact_4bit_m32_top50       | recall_at_10 |                   528 |                 10000 |    0.0128142  | -0.005957   |   0.0319914  | False              |
| uniform_ivfpq_m48      | compact_4bit_m32_top50       | mrr_at_10    |                   528 |                 10000 |    0.00998151 | -0.0134096  |   0.0324287  | False              |
| uniform_ivfpq_m48      | compact_4bit_m32_top50       | ndcg_at_10   |                   528 |                 10000 |    0.0137998  | -0.00151801 |   0.028818   | False              |

Interpretation:
- A confidence interval excluding zero supports a directional difference on this held-out split.
- This is not cross-dataset evidence.