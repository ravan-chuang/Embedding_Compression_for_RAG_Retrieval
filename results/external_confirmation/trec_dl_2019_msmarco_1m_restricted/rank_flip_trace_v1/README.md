# Frozen External Rank-Flip Trace

This directory contains a post-hoc rank-flip trace for the five queries whose
Recall@10 differed between frozen RARS and PCA systems.

The trace reruns the frozen scoring pipeline using:

- candidate_k: 100
- nprobe: 16
- Top-B: 40
- alpha: 0.75
- residual rank: 16

No fitting, parameter selection, or retuning was performed.

## Dominant influential query

For query `962179`, `when was the salvation army founded`, the restricted
qrels contain one judged-positive passage, document `3705165`.

| System | Rank | Correction | Final score |
|---|---:|---:|---:|
| Base IVF-PQ | 3 | — | 0.684523 |
| PCA | 8 | -0.016577 | 0.667946 |
| RARS | 13 | -0.027517 | 0.657007 |

Relative to PCA, RARS applied an additional score change of approximately
-0.01094 to the judged-positive passage, moving it outside the Top-10.

At the same time, passages `5374952`, `4831573`, and `724895`, which had no
positive judgment in the restricted qrels, moved from PCA ranks 12, 15, and 11
to RARS ranks 5, 7, and 9.

The dominant external Recall loss therefore resulted from a sparse Top-10
boundary flip involving both a stronger negative RARS correction to the sole
judged-positive passage and more favorable corrections to several competing
passages.

## Judgment terminology

A passage marked non-positive in this analysis has no positive relevance
judgment in the restricted qrels. It must not automatically be interpreted as
confirmed non-relevant, because the judgment set is incomplete.

## Interpretation

The external Recall difference was driven by a small number of boundary
exchanges rather than broad ranking degradation. This trace is explanatory
post-hoc analysis and does not alter the frozen preregistered result.
