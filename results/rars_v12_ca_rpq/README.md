# RARS-v12 Anchored Cutoff-Aware RPQ Closure

This directory preserves the audited one-shot V12 fresh-query development
result produced from source commit
`07b0fe09b82babb3b06ffd1649266a656dd07df1`.

## Formal outcome

The run completed successfully and the independently recomputed decision is:

```text
STOP_CA_RPQ_NO_STABLE_ADVANTAGE
```

On 2,500 previously unused MS MARCO train queries restricted to the frozen 1M
corpus, primary-seed Recall@10 is:

| Method | Recall@10 |
|---|---:|
| Frozen Base M32 IVF-PQ | 0.1296 |
| Unsupervised PCA64 RPQ16x8 | 0.1872 |
| Anchored cutoff-aware RPQ | 0.1874 |
| Same-candidate exact | 0.2374 |

The challenger-minus-unsupervised difference is `+0.0002`, with 95% CI
`[-0.0026, +0.0032]`, one-sided paired-randomization `p=0.496555`, and query
support `8 improved / 8 harmed`. Seed gains are
`[+0.0002, +0.0012, -0.0010]`; three of five primary-seed folds are negative.
Eight preregistered statistical or stability gates fail.

The negative result is method evidence, not an execution failure. Source,
environment, query isolation, fold assignment, objective monotonicity,
centroid drift, per-query arrays, codebook occupancy, and payload records all
passed audit.

## Data boundary

- Exactly 2,500 query ids were selected before candidate retrieval.
- The selected qids have zero overlap with the historical 6,980-query MS MARCO
  registry.
- The five fold counts are `498 / 451 / 506 / 535 / 510`.
- The task is restricted to the existing 1M-document corpus and is not an
  official full-corpus MS MARCO score.
- No V9, V10, V11, or old holdout packet was opened.

## Portable closure contents

The committed closure includes all 2,500-query Recall, MRR, and nDCG arrays for
Base, same-candidate exact, every unsupervised seed, and every challenger seed;
qids and folds; diagnostics; final basis/codebooks; lineage manifests; and the
executed notebook.

The original `(1,000,000, 16)` uint8 code matrix is intentionally not duplicated
in Git. It was verified before import at exactly `16,000,000` bytes with SHA-256
`0e54d760ad1f67cf787ef5214412a3c242e0b2dd7736f162b2a77b985b2aad20`.
All 256 centroids are occupied in every block. Its record remains in
`development_result.json`, `development_complete.json`, and
`artifact_audit.json`.

Run the portable closure audit with:

```bash
python scripts/verify_rars_v12_committed_closure.py
```

