# RARS-v14 Query-Whitened Anisotropic Rate-RPQ Diagnostic

## Why V14 exists

V13 is a valid negative result: its signed-score surrogate decreased in every
fit, yet its primary Recall@10 gain over unsupervised RPQ was only `+0.0006`
with a confidence interval crossing zero. The next experiment therefore does
not add another supervised loss. It tests a different representation question:
whether 128 residual bits should be distributed uniformly across 16 PCA blocks.

V14 is an outcome-informed architecture diagnostic on the already opened V13
development queries. It cannot confirm a paper claim. A successful result only
permits freezing a new experiment on disjoint queries.

## Frozen-index method

For each cross-validation training split, V14 fits a rank-64 uncentred PCA
basis. Each four-dimensional block receives a positive-definite score metric
estimated from Base Top-40 projected queries, with extra weight near the Base
rank-10 cutoff. Relevance labels do not enter PCA, metric estimation, rate
allocation, or codebook training.

If `G_m = L_m L_m^T` is the regularized, trace-normalized query metric for
block `m`, residual coefficient `z_m` is transformed as `y_m = z_m L_m`.
Euclidean k-means in `y_m` therefore minimizes query-distribution-weighted
score distortion rather than ordinary coefficient reconstruction error. The
dual query coordinate is `q'_m = q_m L_m^{-T}`, whose covariance is whitened;
this is the precise sense in which the method is query-whitened.

The rate proxy `A_m` is the weighted transformed residual energy. The exact
integer bit allocation is solved by dynamic programming:

```text
minimize  sum_m A_m * 2^(-2 b_m / 4)
subject to b_m in {6, 7, 8, 9, 10} and sum_m b_m = 128.
```

Each block uses `2^b_m` centroids. Variable-width codes are packed into one
little-endian 128-bit stream, so the deployable payload remains exactly
16 bytes per document. The IVF-PQ index, document embeddings, encoder, and
candidate generator remain immutable.

## Comparisons and decision

The primary comparator is the committed V13 unsupervised RPQ16x8 OOF array for
the same seed. A uniform eight-bit query-whitened ablation separates metric
learning from rate allocation. Base, PCA16, and same-candidate exact arrays are
carried forward from the audited V13 closure.

The preregistered GO requires a statistically supported `+0.003` Recall@10
gain over V13 uniform RPQ, at least `+0.001` over the uniform-whitened ablation,
nonnegative gain in every seed and fold, multi-seed query-level support, MRR
and nDCG guardrails, a non-uniform allocation, and an exact 16-byte payload.
Any failed gate yields `STOP_V14_NO_ANISOTROPIC_RATE_SIGNAL`.

The canonical machine-readable contract is
`protocols/rars_v14_query_whitened_anisotropic_rate_rpq_diagnostic_v1.json`.
