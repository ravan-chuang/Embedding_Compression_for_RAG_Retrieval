# RARS-v8 cutoff-aware sidecar closure packet

This directory closes the one-shot development and qrels-free artifact stages
of `rars_v8_cutoff_sidecar_v1`. The frozen development decision is
`GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL`: RARS-v8 clears every registered
development gate against both frozen IVF-PQ and the storage-matched PCA
sidecar. This is authorization to design a separate prospective confirmation,
not a confirmed general-performance claim.

## Outcome

All quality numbers below use five-fold out-of-fold predictions for RARS on
the 2,307-query `oracle_design` role. The PCA basis is label-free and fitted to
the development candidate residual union.

| Method | Known-positive Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Frozen M32 nprobe16 IVF-PQ | 0.679923 | 0.463378 | 0.512141 |
| Rank-16 int8 PCA sidecar | 0.692638 | 0.483620 | 0.530897 |
| Rank-16 int8 RARS-v8 sidecar | **0.702825** | **0.493528** | **0.540213** |
| Same-candidate FP32 oracle | 0.773768 | 0.580966 | 0.624358 |

- RARS minus Base Recall@10: `+0.022901`, paired-query 95% bootstrap
  `[+0.014666, +0.031426]`; improved / harmed / unchanged queries:
  `80 / 25 / 2,202`.
- RARS minus PCA Recall@10: `+0.010186`, paired-query 95% bootstrap
  `[+0.003468, +0.017122]`; improved / harmed / unchanged queries:
  `45 / 22 / 2,240`.
- PCA minus Base Recall@10: `+0.012715`, paired-query 95% bootstrap
  `[+0.005346, +0.020373]`.
- Same-candidate gap recovered by RARS: `24.403%`.
- Mined support: `7,534` pairs, including `962` promotion pairs from `261`
  queries and `6,572` protection pairs from `1,595` queries. Each role has
  exactly half of the aggregate training weight.
- All ten frozen development gates pass. No future or audit role was opened.

The paired bootstrap uses 20,000 query resamples and frozen seed `20260722`.
The committed OOF vectors allow the primary Recall comparisons, confidence
intervals, support counts, and decision to be recomputed exactly.

## Full-corpus artifacts

After the GO decision, the separate no-label builder encoded both methods over
all 1,000,000 corpus rows. Each representation contains a 384×16 FP32 basis,
16 FP32 scales, and a 1,000,000×16 int8 code array:

- code payload: `16 B/document`;
- basis + scales + NPY code representation: `16.025024 B/document`;
- saturated coefficients reported by the builder: `0`;
- external document-ID bytes duplicated: `0`;
- frozen IVF-PQ index before and after: `41,183,924` bytes and identical
  SHA-256 `863637a68933a33f3d4c32920f492cab4e52ba83b12863d6f5854e1cc937f0e8`.

The two 16 MB code arrays are intentionally not duplicated in Git. Their exact
byte counts and SHA-256 hashes remain registered in the method manifests and
`artifact_audit.json`. The audit loaded both full arrays and verified shape,
dtype, range, storage accounting, and every upstream artifact registration.

## Important diagnostic

The recorded surrogate loss rises in every fold and in the final fit; the
final full-data trace goes from `0.075686` to `0.087692`. The implementation
records the objective immediately before each update, so the last value is not
a post-update terminal evaluation, but the trajectory still does not support
a convergence claim. This does not change the frozen OOF ranking result or its
gate decision: training loss did not select a checkpoint, hyperparameters were
frozen before execution, and the Recall vectors independently reproduce the
reported gain. The paper must describe this as an optimizer/surrogate warning
and must not claim that the margin objective converged.

## Evidence boundary

This is outcome-informed development evidence, not an untouched confirmation,
official MS MARCO evaluation, or external-domain result. Zero-labelled
challengers are unjudged candidates, not explicitly judged non-relevant
documents. The `same_candidate_exact` oracle only rescored the frozen
candidate pool and is not a deployable baseline. A broader same-IVF exact
search therefore has a different ceiling.

The result does not authorize opening the existing future-method holdout from
the development notebook. The next valid step is a separately frozen,
prospective evaluator with no tuning. That evaluation must also disclose
higher-nprobe and rebuilt M48 baselines because those are strong alternatives
when routing cost or index rebuilds are allowed.

## Contents

- `development/`: exact start, output, freeze, completion, fold, pair-support,
  basis/scale, and four OOF Recall artifacts from the Drive export.
- `sidecars/`: exact result/completion records and the small PCA/RARS basis,
  scale, configuration, and manifest artifacts. Full code arrays remain in the
  durable external export.
- `executed_notebook/`: executed Colab notebook; its cell sources match the
  clean committed notebook, all nine code cells ran once in order, and there
  are no error outputs.
- `artifact_audit.json`: ZIP identity, full-export verification result, code
  array statistics, and exact external code registrations.
- `closure_manifest.json`: repository-local byte counts and SHA-256 hashes for
  every packet file except the manifest itself.

## Verification

From the repository root:

```bash
python scripts/verify_rars_v8_cutoff_sidecar_packet.py \
  --packet-root results/rars_v8_cutoff_sidecar
```

The verifier checks source lineage, every repository-local artifact hash,
array shapes/dtypes/finiteness, orthonormal bases, exact bootstrap intervals,
all frozen gates, fold accounting, sidecar storage, external code
registrations, and notebook source parity. A verification failure invalidates
the local packet; it is not a new scientific result.
