# RARS-v14 Query-Whitened Anisotropic Rate-RPQ Closure

This directory preserves the audited V14 architecture diagnostic produced
from source commit
`a3895e8d2ba298b155ac7f866453af134fd3c222`.

## Formal outcome

The run completed successfully and the independently recomputed decision is:

```text
STOP_V14_NO_ANISOTROPIC_RATE_SIGNAL
```

V14 reuses the already opened 5,000-query V13 development role. It is an
outcome-informed architecture diagnostic, not fresh confirmation evidence.

| Method | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Frozen Base M32 IVF-PQ | 0.1404667 | 0.0624877 | 0.0804607 |
| PCA16-int8 | 0.1555667 | 0.0710190 | 0.0905243 |
| V13 uniform PCA64 RPQ16x8 | 0.1892667 | 0.0901543 | 0.1132554 |
| Uniform query-whitened RPQ | 0.1878667 | 0.0885567 | 0.1117019 |
| V14 anisotropic-rate RPQ | 0.1889667 | 0.0887476 | 0.1120846 |
| Same-candidate exact | 0.2456667 | 0.1285013 | 0.1560502 |

The primary V14-minus-V13-uniform Recall@10 difference is `-0.0003`, with
95% CI `[-0.0044, +0.0038]`, one-sided paired-randomization `p=0.574364`, and
support `54 improved / 56 harmed`. Seed gains are
`[-0.0003, +0.0003, +0.0027]`; three of five primary-seed fold gains are
negative. Eight preregistered gates fail.

All fold and final fits select the same exact 128-bit allocation:

```text
[9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7]
```

The allocation is stable and the exact 16-byte payload is valid, but it does
not improve over the unwhitened uniform RPQ comparator. The positive
`+0.0011` point estimate over the uniform-whitened ablation only recovers part
of the quality lost by whitening. V14 therefore closes this particular
query-whitened anisotropic-rate design; it does not close the stronger
uniform rank-64 RPQ representation.

## Portable closure contents

The committed closure includes all per-query Recall, MRR, and nDCG arrays;
qids and folds; fold/seed diagnostics; final basis, transforms, variable-rate
codebooks and allocation; decision markers; and the executed notebook.

The original `(1,000,000, 16)` packed uint8 payload is intentionally not
duplicated in Git. It was verified before import at exactly `16,000,000` bytes
with SHA-256
`fd2dc092f38bfdecbac9881c9dc17dd0a900ee53b431de0665ca23e6b12ca6ad`.
Its record and complete code histograms remain in `diagnostic_result.json`,
`diagnostic_complete.json`, and `artifact_audit.json`.

Run the portable closure audit with:

```bash
python scripts/verify_rars_v14_committed_closure.py
```
