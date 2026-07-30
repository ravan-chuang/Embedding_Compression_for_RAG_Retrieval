# RARS-v9 locked confirmation protocol

## Decision first

V8 has earned exactly one next step: a one-shot evaluation of the already
frozen artifact.  It has **not** earned another round of loss, rank, alpha,
Top-B, seed, or threshold selection on the same evidence.

The confirmation role contains 803 queries that V3--V8 never opened.  It is
prospective relative to V8, but it is not an independent test set: those
queries came from the historical v2 `inner_train` pool.  The strongest valid
description is therefore **within-program prospective confirmation**.

The machine-readable authority is
[`protocols/rars_v9_locked_confirmation_v1.json`](../protocols/rars_v9_locked_confirmation_v1.json).

## Frozen question

Does the exact RARS-v8 rank-16 int8 sidecar improve known-positive Recall@10
over both the frozen M32 IVF-PQ base and the storage-matched frozen PCA
sidecar, without refitting or selecting anything from the confirmation
outcomes?

The sole primary endpoint is the paired per-query difference
`Recall@10(RARS-v8) - Recall@10(PCA)`.  RARS-versus-Base, Success@10, MRR@10,
nDCG@10, gap recovery, query support, increased `nprobe`, and rebuilt M48 are
secondary or limitation analyses.

## What is immutable

- The V8 source commit is `c9d95f15d55e7700db069da69567157f2eed469e`.
- Both sidecars remain rank 16, int8, `alpha=0.75`, and `Top-B=40`.
- PCA and RARS use the exact full-corpus basis, scales, and codes already
  registered by SHA-256.
- The M32 IVF-PQ index remains byte-identical (`nlist=512`, `M=32`, 8 bits,
  `nprobe=16`).
- The query order contains exactly 803 registered identities.
- The 803 identity/query-vector packet is rebuilt directly from the frozen
  training split and global query-vector matrix.  The old V2 candidate-bundle
  builder is not used because it parses qrels.
- The statistical tests, thresholds, and decision labels are fixed before
  the first qrels access.

The evaluator must verify every input, write and fsync
`confirmation_started.json` with `outcome_opened=false`, and only then parse
qrels.  A non-empty output directory is not silently reused.

## Comparators

| Method | Index mutation | Added document payload | Purpose |
|---|---:|---:|---|
| Frozen M32, `nprobe=16` | no | 0 B/doc | deployed base |
| PCA rank-16 int8 | no | 16 B/doc | storage-matched generic sidecar |
| RARS-v8 rank-16 int8 | no | 16 B/doc | frozen algorithm under test |
| Same-candidate exact | no | FP32 oracle only | candidate-level ceiling |
| M32, `nprobe=32/64` | no | 0 B/doc | routing-compute alternatives |
| Rebuilt M48, `nprobe=16` | yes | 16 extra code bytes/doc | same-code-budget rebuild limitation |

The M48 index must be built without qrels and hash-frozen before the start
marker.  It is a limitation baseline, not part of primary method selection.

## Registered decision

Algorithm confirmation requires all of the following:

- RARS-minus-Base Recall@10 at least `+0.01`, with bootstrap lower bound above
  zero;
- RARS-minus-PCA Recall@10 at least `+0.005`, with bootstrap lower bound above
  zero and one-sided paired randomization `p <= 0.025`;
- candidate-gap recovery at least 15%;
- at least 20 improved queries and net +10 over Base;
- at least 15 improved queries and net +8 over PCA.

If the common sidecar gates pass and PCA itself gains at least `+0.01` over
Base with a positive lower bound, but the RARS-over-PCA gates fail, the result
supports only the generic frozen-sidecar claim.  Otherwise V8 stops.

No outcome under this protocol authorizes changing the thresholds or rerunning
V8.  Any modified method is a new development protocol.

## Loss-direction anomaly

The V8 development packet records a full-data surrogate loss increase from
`0.0756863` to `0.0876917`, with the same direction in every fold.  This does
not invalidate the already-built artifact, but it blocks a convergence claim.

The confirmation evaluator therefore does not retrain.  A separate
development-only audit must log the objective before the update, after the
Adam proposal, and after QR retraction; compare the analytical gradient with
finite differences; separate the pair and anchor terms; and align each
checkpoint with retrieval metrics.  A logging error changes documentation.  A
gradient or retraction defect defines a new V8.1 method and cannot overwrite
V8.

## Interpretation

Passing this protocol would materially strengthen the claim that a lightweight
cutoff-aware frozen-index sidecar is useful inside this research program.  It
would still leave two publication-critical tasks: external validation on a
new collection or model, and a controlled end-to-end efficiency study.  A
failed confirmation redirects research toward pair-label noise and
distribution shift, not toward arbitrary capacity increases.
