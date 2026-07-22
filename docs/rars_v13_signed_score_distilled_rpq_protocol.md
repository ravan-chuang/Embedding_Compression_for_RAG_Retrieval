# RARS-v13 Signed Score-Distilled Residual PQ

## Motivation

V12 successfully reduced its weighted coefficient-reconstruction objective but
did not improve ranking over unsupervised RPQ. Its primary Recall@10 gain was
only `+0.0002`, with CI `[-0.0026, +0.0032]`, seed gains
`[+0.0002, +0.0012, -0.0010]`, and three negative folds. This is a
surrogate-ranking mismatch: assigning non-negative reconstruction weights to
both a positive and challenger does not require their query-conditioned score
margin to improve.

V13 replaces that objective rather than tuning V12. It keeps the successful
V11/V12 architecture—rank-64 PCA and a `16 B/doc` RPQ16x8 sidecar—but learns
each four-dimensional codeword from signed query-residual score targets.

## New fresh-query boundary

V13 selects 5,000 new MS MARCO train queries with at least one positive in the
frozen 1M corpus. It excludes both the historical 6,980-query registry and all
2,500 V12 qids before encoding, candidate retrieval, or metric computation.
Selection and five-fold assignment use new fixed SHA-256 salts. The experiment
remains a restricted-corpus development task, not an official MS MARCO score.

## Closed-form signed score distillation

For residual coefficient block `z_dm`, projected query block `q_qm`, and the
fixed unsupervised product-code assignment `a(d,m)`, the teacher target is the
signed score contribution

```text
y_qdm = q_qm^T z_dm.
```

For every training query and Base Top-40 candidate, V13 fits codeword `c_mk`
by weighted ridge regression:

```text
min_c  sum_(q,d:a(d,m)=k) w_qd (q_qm^T c - y_qdm)^2
       + (c - c0_mk)^T A_anchor_mk (c - c0_mk).
```

The observation weight is largest near the Base rank-10 score boundary:

```text
w_qd = [1 + 4 exp(-|s_base(q,d) - s_base_rank10(q)| / 0.02)]
       × [2 if d is a known positive else 1].
```

All Top-40 candidates contribute, so supervision is dense and unjudged
candidates are not mislabeled as non-relevant. The anchor uses the average
projected-query covariance at the centroid's own observation mass. Each solve
is only `4 × 4`, deterministic after the unsupervised initializer, and clipped
to `0.15` training-block RMS.

Crucially, post-update assignments remain byte-identical to the unsupervised
RPQ assignments. V13 changes only the global 65,536-byte codebook, not the
`16 B/doc` payload. This isolates score calibration from code reassignment and
prevents the seed-dependent assignment churn diagnosed after V12.

## Comparators and gates

Every fold fits all methods using only the other four folds:

1. frozen Base M32 IVF-PQ;
2. storage-matched rank-16 int8 PCA;
3. unsupervised rank-64 RPQ16x8;
4. signed score-distilled rank-64 RPQ16x8; and
5. same-candidate FP32 exact scoring.

All five folds run seeds `20261001`, `20261002`, and `20261003`; the first is
primary and seed selection is forbidden. A GO requires, among other gates,
Recall@10 gains of at least `+0.003` over both unsupervised RPQ and PCA16, a
positive bootstrap lower bound, one-sided randomization `p <= 0.05`, at least
30 improved and net 15 improved queries, non-negative gains in every seed and
primary fold, no MRR/nDCG regression beyond `-0.002`, non-increasing training
objectives, drift within `0.15`, zero assignment changes, and a real 1M × 16
byte code payload.

Any failure yields `STOP_SIGNED_SCORE_RPQ_NO_STABLE_ADVANTAGE`. A GO authorizes
only writing a separate independent-confirmation protocol; it does not reopen
an old holdout or establish a publishable claim by itself.

The machine-readable source of truth is
[`rars_v13_signed_score_distilled_rpq_v1.json`](../protocols/rars_v13_signed_score_distilled_rpq_v1.json).
