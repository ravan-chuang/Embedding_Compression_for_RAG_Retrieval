# RARS-v12 Anchored Cutoff-Aware Residual Product Quantization

## Why V12 exists

V10 showed that rank-16 scalar quantization was not the bottleneck. V11 then
showed that a wider rank-64 residual representation has substantial headroom
and that a 16-byte RPQ code retains `87.28%` of the rank-64 FP32 gain over the
rank-16 PCA sidecar. The preregistered V11 decision was
`GO_TO_SEPARATE_CA_RPQ_CUTOFF_PROTOCOL`.

V12 tests one new algorithm, not another open-ended sweep. Its baseline is the
strong V11 architecture: rank-64 PCA coefficients, 16 four-dimensional
product blocks, and one 8-bit index per block. The challenger keeps the basis,
payload, query path, candidate depth, alpha, and product partition fixed. It
changes only the RPQ centroids through one conservative cutoff-weighted update.

## Fresh data boundary

All 6,980 cached MS MARCO development queries have already participated in
earlier RARS work. V12 therefore uses passage-ranking **training** queries. A
pre-metric freezer:

1. parses the official `queries.train.tsv` and `qrels.train.tsv` sources;
2. keeps only queries with at least one positive passage in the exact frozen
   1M document-id array;
3. excludes all qids in the historical train/validation/test split files;
4. sorts eligible qids by the frozen SHA-256 selection key; and
5. freezes the first 2,500 queries before candidate retrieval or metrics.

This is a corpus-restricted development task, not the official 8.8M-passage
MS MARCO evaluation. Relevant-count denominators include all positive qrels
for a selected query that occur in the frozen 1M corpus.

Five deterministic folds are derived from a separate SHA-256 key. For every
fold, PCA, unsupervised RPQ, cutoff pairs, block weights, and updated centroids
are fit using only the other four folds. Concatenated held-out predictions are
the sole primary endpoint.

## Algorithm

Let `z_d = B^T r_d` be the fixed rank-64 PCA residual coefficient vector,
partitioned into 16 four-dimensional blocks. V12 first trains an ordinary
unsupervised RPQ codebook `C0` on the training folds.

Static promotion and protection pairs are mined at the Top-10 boundary. For a
pair query `q`, its mass across blocks is proportional to

```text
||B_block^T q||² / sum_j ||B_j^T q||².
```

Both the known-positive and unjudged-challenger residual rows receive that
mass. Protection pairs receive the frozen multiplier `2.0`. Each block is
normalized to mean additive boost `8.0` on active residual rows and clipped at
total weight `25.0`.

Within each **frozen initial Voronoi cell**, the updated centroid has the
closed form

```text
c* = (sum_i w_i z_i + 32 c0) / (sum_i w_i + 32).
```

Centroid movement is clipped to `0.25` times the training block RMS. Empty
cells remain byte-identical to `C0`. After this single update, documents are
reassigned to their nearest updated centroids. There is no STE, soft
assignment, gradient descent, encoder update, learned basis, or metric-based
checkpoint selection.

The deployment payload remains exactly

```text
16 product blocks × one uint8 index = 16 B/document.
```

The final all-development model is export-only and cannot alter the OOF
decision. It must materialize a real `(1,000,000, 16)` uint8 sidecar rather
than extrapolating storage from the candidate union.

## Stability and decision

All five folds run three preregistered Faiss seeds. Seed `20260901` is primary;
the other seeds are stability checks, never selectable alternatives.

`GO_TO_FRESH_CA_RPQ_CONFIRMATION_PROTOCOL` requires all of the following:

- primary OOF Recall@10 gain over unsupervised RPQ at least `+0.003`;
- paired-bootstrap 95% lower bound above zero and one-sided paired
  randomization `p <= 0.05`;
- at least 20 improved queries and net support at least 10;
- primary gain over Base at least `+0.01`;
- every seed and every primary-seed fold non-negative;
- median seed gain at least `+0.002`;
- at least 20% same-candidate gap recovery;
- MRR@10 and nDCG@10 changes no worse than `-0.002`;
- every closed-form objective non-increasing and every centroid within the
  drift limit; and
- a verified 16-byte full-corpus code matrix.

Any failure yields `STOP_CA_RPQ_NO_STABLE_ADVANTAGE`. A GO permits only writing
a new independent-confirmation protocol. It does not open an old holdout and
does not establish a publishable algorithm claim by itself.

## Audit improvement over V11

The V12 packet includes qids, fold ids, and per-query Recall, MRR, and nDCG for
Base, same-candidate exact, unsupervised RPQ, and CA-RPQ for every seed. The
packet verifier can therefore reconstruct the primary statistics, fold/seed
gates, secondary guardrails, payload contract, and formal decision without
trusting aggregate JSON fields.

The machine-readable source of truth is
[`rars_v12_anchored_cutoff_rpq_v1.json`](../protocols/rars_v12_anchored_cutoff_rpq_v1.json).
