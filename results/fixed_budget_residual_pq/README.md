# Fixed-Budget Residual-PQ Result Package

This directory contains reproducible artifacts for the FiQA held-out
fixed-budget Residual-PQ experiment.

## Best observed selective configuration

- Residual-PQ code size: 16 bytes
- Allocation: reconstruction-error sidecar
- Candidate refinement depth: Top-50
- Recall@10: 0.3195
- MRR@10: 0.3314
- nDCG@10: 0.2678

## Included files

- `fixed_budget_residual_pq_heldout_results.csv`
- `fixed_budget_residual_pq_storage_config.csv`
- `fixed_budget_residual_pq_bootstrap.csv`
- `oracle_candidate_rescoring_heldout_results.csv`
- `residual_pq_sweep_heldout_results.csv`

Storage accounting includes base PQ codes, residual-code payload,
document-ID metadata, and amortized Residual-PQ codebook storage.
