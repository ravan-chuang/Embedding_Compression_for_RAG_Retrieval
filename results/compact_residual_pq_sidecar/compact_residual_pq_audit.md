# Compact Residual-PQ Sidecar Experiment

The compact layout replaces per-selected-document IDs with a bitmap and block-level rank-prefix index keyed by Faiss internal document IDs.

The stored residual-PQ centroid table is FP16. At serving-time it is restored to FP32 only for decoding.

The recorded refinement latency uses an evaluation-only dense document-index-to-slot accelerator. It is excluded from strict deployable storage accounting; bitmap/rank-prefix equivalence is validated separately.

## Strict storage accounting

| layout_name                    |   residual_pq_m |   residual_pq_nbits |   selected_document_count |   selected_fraction |   base_pq_total_bytes |   residual_codebook_raw_bytes |   residual_codebook_serialized_bytes |   selection_bitmap_bytes |   selection_rank_prefix_bytes |   selection_index_serialized_bytes |   residual_code_payload_bytes_per_selected_doc |   residual_code_payload_bytes |   total_bytes_per_vector |   storage_budget_target_bytes_per_vector |
|:-------------------------------|----------------:|--------------------:|--------------------------:|--------------------:|----------------------:|------------------------------:|-------------------------------------:|-------------------------:|------------------------------:|-----------------------------------:|-----------------------------------------------:|------------------------------:|-------------------------:|-----------------------------------------:|
| compact_4bit_m32_fp16_codebook |              32 |                   4 |                     56354 |            0.977723 |               1844416 |                         12288 |                                12352 |                     7205 |                           908 |                               8192 |                                             16 |                        901664 |                       48 |                                       48 |
| compact_8bit_m16_fp16_codebook |              16 |                   8 |                     44834 |            0.777855 |               1844416 |                        196608 |                               196672 |                     7205 |                           908 |                               8192 |                                             16 |                        717344 |                       48 |                                       48 |

## Held-out FiQA results

| layout_name                    |   residual_pq_m |   residual_pq_nbits |   selected_document_count |   selected_fraction |   total_bytes_per_vector |   recall_at_10 |   mrr_at_10 |   ndcg_at_10 |   heldout_recovered_drop_events |   heldout_relevant_drop_recovery_rate |   selected_residual_mse |   selected_residual_mean_cosine | lookup_equivalence   |
|:-------------------------------|----------------:|--------------------:|--------------------------:|--------------------:|-------------------------:|---------------:|------------:|-------------:|--------------------------------:|--------------------------------------:|------------------------:|--------------------------------:|:---------------------|
| compact_4bit_m32_fp16_codebook |              32 |                   4 |                     56354 |            0.977723 |                       48 |       0.332731 |     0.35309 |     0.280408 |                              67 |                              0.325243 |             0.000250867 |                        0.520107 | pass                 |
| compact_8bit_m16_fp16_codebook |              16 |                   8 |                     44834 |            0.777855 |                       48 |       0.335462 |     0.34783 |     0.283464 |                              68 |                              0.330097 |             0.000254047 |                        0.558002 | pass                 |