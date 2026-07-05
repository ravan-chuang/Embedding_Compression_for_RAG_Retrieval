# Oracle Exact Candidate Rescoring Ceiling

Within each M=32 compressed Top-L candidate pool, all candidates are
rescored with exact Float32 document-query inner products.

This is an upper bound for candidate-side refinement, not a deployable
method or a storage-matched comparison.

## Held-out oracle result

| method                         | policy                    |   candidate_depth |   recall_at_10 |   mrr_at_10 |   ndcg_at_10 |   mean_top10_overlap_with_exact |   heldout_recovered_drop_events |   heldout_relevant_drop_recovery_rate |   refinement_latency_ms_per_query |
|:-------------------------------|:--------------------------|------------------:|---------------:|------------:|-------------:|--------------------------------:|--------------------------------:|--------------------------------------:|----------------------------------:|
| oracle_exact_candidate_rescore | all_candidates_exact_fp32 |                20 |       0.34237  |    0.392738 |     0.313152 |                        0.582576 |                              64 |                              0.31068  |                         0.061062  |
| oracle_exact_candidate_rescore | all_candidates_exact_fp32 |                50 |       0.381007 |    0.415546 |     0.338609 |                        0.763258 |                             130 |                              0.631068 |                         0.0502665 |
| oracle_exact_candidate_rescore | all_candidates_exact_fp32 |               100 |       0.396438 |    0.419797 |     0.346508 |                        0.864962 |                             156 |                              0.757282 |                         0.10305   |

## Comparison with base, candidate-distortion, and uniform M=48

| method                         | policy                                   |   candidate_depth |   recall_at_10 |   mrr_at_10 |   ndcg_at_10 |   mean_top10_overlap_with_exact |   heldout_recovered_drop_events |   heldout_relevant_drop_recovery_rate |   refinement_latency_ms_per_query |
|:-------------------------------|:-----------------------------------------|------------------:|---------------:|------------:|-------------:|--------------------------------:|--------------------------------:|--------------------------------------:|----------------------------------:|
| base_ivfpq_m32                 | none                                     |                 0 |       0.290683 |    0.303215 |     0.239512 |                        0.429356 |                               0 |                             0         |                         0         |
| oracle_exact_candidate_rescore | all_candidates_exact_fp32                |                20 |       0.34237  |    0.392738 |     0.313152 |                        0.582576 |                              64 |                             0.31068   |                         0.061062  |
| oracle_exact_candidate_rescore | all_candidates_exact_fp32                |                50 |       0.381007 |    0.415546 |     0.338609 |                        0.763258 |                             130 |                             0.631068  |                         0.0502665 |
| oracle_exact_candidate_rescore | all_candidates_exact_fp32                |               100 |       0.396438 |    0.419797 |     0.346508 |                        0.864962 |                             156 |                             0.757282  |                         0.10305   |
| selective_residual_m32         | calibration_candidate_distortion_sidecar |                20 |       0.29676  |    0.306844 |     0.243827 |                        0.438826 |                               7 |                             0.0339806 |                         0.0818077 |
| selective_residual_m32         | calibration_candidate_distortion_sidecar |                50 |       0.29496  |    0.303141 |     0.241464 |                        0.438826 |                               7 |                             0.0339806 |                         0.120699  |
| selective_residual_m32         | calibration_candidate_distortion_sidecar |               100 |       0.294645 |    0.301202 |     0.240449 |                        0.435606 |                               7 |                             0.0339806 |                         0.136885  |
| uniform_ivfpq_m48              | uniform_code_budget                      |               100 |       0.345545 |    0.363071 |     0.294208 |                        0.541856 |                              97 |                             0.470874  |                         0.0171802 |

Interpretation:
- If oracle Top-L remains far below uniform M=48, candidate-side residual refinement has limited headroom.
- If oracle Top-L approaches or exceeds uniform M=48, a more storage-efficient residual sidecar remains worth pursuing.
- Oracle cannot recover documents absent from the compressed Top-L pool.