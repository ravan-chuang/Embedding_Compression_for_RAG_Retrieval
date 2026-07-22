# RARS-v13 Signed Score-Distilled RPQ Closure

This directory preserves the audited one-shot V13 fresh-query development
result produced from source commit
`d8cb761c289fe17ea2c2bfb92059e8b5553cfd74`.

## Formal outcome

The run completed successfully and the independently recomputed decision is:

```text
STOP_SIGNED_SCORE_RPQ_NO_STABLE_ADVANTAGE
```

On 5,000 previously unused MS MARCO train queries restricted to the frozen 1M
corpus, primary-seed Recall@10 is:

| Method | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Frozen Base M32 IVF-PQ | 0.1404667 | 0.0624877 | 0.0804607 |
| PCA16-int8 | 0.1555667 | 0.0710190 | 0.0905243 |
| Unsupervised PCA64 RPQ16x8 | 0.1892667 | 0.0901543 | 0.1132554 |
| Signed score-distilled RPQ | 0.1898667 | 0.0899975 | 0.1132763 |
| Same-candidate exact | 0.2456667 | 0.1285013 | 0.1560502 |

The challenger-minus-unsupervised Recall@10 difference is `+0.0006`, with
95% CI `[-0.0006, +0.0018]`, one-sided paired-randomization `p=0.255017`, and
query support `6 improved / 3 harmed`. All three aggregate seed gains equal
`+0.0006`, but their changed-query sets are disjoint. Six preregistered
statistical or support gates fail.

The negative result is method evidence, not an execution failure. Every one of
the 15 fold-seed fits reduced its signed-score surrogate, all assignments
remained unchanged, the exact 16-byte payload was materialized, and the
source, environment, query isolation, folds, arrays, and output hashes passed
audit. The result therefore closes fixed-assignment centroid score regression;
it does not close the much stronger unsupervised rank-64 RPQ representation.

## Data boundary

- Exactly 5,000 query ids were selected before candidate retrieval.
- The selected qids have zero overlap with the historical 6,980-query registry
  and the 2,500-query V12 development role.
- Fold counts are `982 / 1008 / 1039 / 995 / 976`.
- The task is restricted to the existing 1M-document corpus and is not an
  official full-corpus MS MARCO score.
- No V9, V10, V11, V12, or old holdout packet was opened by the trainer.

## Portable closure contents

The committed closure includes all 5,000-query Recall, MRR, and nDCG arrays
for Base, PCA16, same-candidate exact, every unsupervised seed, and every
challenger seed; qids and folds; diagnostics; final bases/codebooks; lineage
manifests; and the executed notebook.

The original `(1,000,000, 16)` uint8 assignment matrix is intentionally not
duplicated in Git. It was verified before import at exactly `16,000,000` bytes
with SHA-256
`3896c1fcbba37997c7881c723af8e6b38dbf619f36aa0153ddc78e97cd28e664`.
All 256 centroids are occupied in every block. Its record remains in
`development_result.json`, `development_complete.json`, and
`artifact_audit.json`.

Run the portable closure audit with:

```bash
python scripts/verify_rars_v13_committed_closure.py
```
