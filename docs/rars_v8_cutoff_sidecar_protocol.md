# RARS-v8 Cutoff-Aware Frozen-Index Sidecar

## Revised research claim

RARS-v8 tests a narrower, deployment-grounded claim:

> A small post-hoc residual sidecar can correct part of the Top-k retrieval
> degradation of an already deployed IVF-PQ index without retraining an
> encoder, changing routing, rewriting PQ codes, or rebuilding the index.

This is not a new product quantizer and it is not claimed to be the best use of
48 bytes per document when rebuilding is allowed.  A sidecar can only reorder
documents returned by the frozen IVF probes; it cannot recover a routing miss.
The machine-readable development contract is
[`protocols/rars_v8_cutoff_sidecar_v1.json`](../protocols/rars_v8_cutoff_sidecar_v1.json).

## What was wrong with the earlier experiments

The previous line mixed several different questions and repeatedly spent
compute before checking whether the proposed mechanism matched the error.

| Earlier problem | V8 correction |
|---|---|
| Global reconstruction loss treats harmless dimensions like cutoff-changing errors. | Fit directly to exact residual margins for Top-10 promotion and protection pairs. |
| Free query/document projections can gain scale, drift, and overfit. | Use one symmetric rank-16 basis, enforce orthonormality after every update, and anchor it to residual PCA. |
| Pair counts were mistaken for independent evidence. | Give every represented query equal total mass inside each pair role, then split mass 50/50 between promotion and protection. |
| Sparse MS MARCO qrels were described as binary relevance. | Call non-positive candidates **unjudged challengers**; only positive qrels are explicit relevance labels. |
| FP32 development could hide int8 deployment loss. | Fit scales, encode int8, and calculate every selection/reporting metric through the deployed int8 scorer. |
| Query-only V7 tried to compensate document-side PQ error by moving queries. | Keep original query routing and attach correction coefficients to document residuals, where the approximation error occurs. |
| A method could be tuned and tested on the same role. | Use five-fold out-of-fold development, close a hash-bound method packet, build the 1M artifact without qrels, then require a new locked evaluator. |
| RARS was compared only with a weak frozen base. | Compare with storage-matched residual PCA, same-candidate exact oracle, larger `nprobe`, exact reranking, and same-storage M48 when the relevant stage is authorized. |

These changes respond directly to the observed evidence.  V6 found distributed
PQ-specific headroom on 2,307 development queries: Base-PQ Recall@100 was
`0.84731`, same-route FP32 Recall@100 was `0.89395`, and 4,413 uncapped flips
spanned 189 queries.  V7 nevertheless stopped: its query-only adapter changed
Recall@100 from `0.84957` to `0.85714`, but the paired 95% interval crossed
zero, recovered only 13.46% of the gap, and reduced mean query cosine to
`0.94682`.  V8 therefore removes query adaptation rather than increasing its
capacity.

## Deployed scorer

For document residual `r_d = d_fp32 - reconstruct_IVFPQ(d)`, V8 stores:

```text
B       : one 384 x 16 orthonormal FP32 basis
scale   : 16 FP32 max-absolute int8 scales
code_d  : round((r_d B) / scale), clipped to [-127, 127]
```

The frozen Base-PQ result supplies the candidate set and original score.  Only
its Top-40 candidates are corrected:

```text
s_v8(q,d) = s_pq(q,d) + 0.75 * (q B) dot (code_d * scale)
```

The representation payload is 16 bytes per document plus one small shared
basis and scale vector.  External document IDs are not duplicated: sidecar row
`i` is defined to match corpus/Faiss row `i`.

## Cutoff-aware pair construction

V8 mines static pairs from the Base-PQ order on `oracle_design`.

- **Promotion:** an explicit positive is outside Base Top-10 but within
  Top-40; an unjudged challenger is in Base Top-10; same-route FP32 must prefer
  the positive.
- **Protection:** an explicit positive is already in Base Top-10; an unjudged
  challenger is at Base rank 11--26; same-route FP32 must prefer the positive.

At most four challengers are retained per positive.  A teacher filter prevents
training the sidecar to imitate a ranking reversal that FP32 does not support.
The target is not embedding MSE:

```text
target(q,p,n) = q dot (r_p - r_n)
prediction    = (q B) dot ((r_p - r_n) B)
```

The objective is query-balanced smooth-L1 target error plus a PCA subspace
anchor.  Full-batch Adam runs for exactly 160 steps with a reduced-QR
retraction after each update.  There is no seed search, early stopping, or
training-loss checkpoint selection.

## Development separation

Only the 2,307-query, already outcome-informed `oracle_design` role is opened.
The exact five V3 folds (`463 / 468 / 470 / 436 / 470`) produce out-of-fold
RARS metrics.  Unlabelled candidate residuals may initialize PCA and calibrate
int8 scales in every fold; relevance pairs never cross the validation fold.

The following remain forbidden:

- `oracle_audit`, the 803-query `future_method_holdout`, outer validation,
  clean test, TREC, NQ, and FiQA outcomes;
- changing rank, alpha, Top-B, loss weights, or thresholds after the first V8
  metric;
- mutating the encoder, embeddings, IVF lists, PQ codebooks, or PQ codes;
- reporting these development metrics as independent confirmation.

## Three-stage execution

1. **Development:**
   [`scripts/train_rars_v8_cutoff_sidecar.py`](../scripts/train_rars_v8_cutoff_sidecar.py)
   verifies the durable V6 packet, runs five-fold OOF comparison, and emits a
   hash-bound `method_freeze.json`.  It never imports or opens the Faiss index.
   The runnable entry point is the
   [commit-pinned Colab notebook](../notebooks/MSMARCO_RARS_v8_Cutoff_Sidecar_Development.ipynb).
2. **Qrels-free artifact build:** only after a GO decision,
   [`scripts/build_rars_v8_frozen_sidecars.py`](../scripts/build_rars_v8_frozen_sidecars.py)
   makes a direct map in memory, performs two streamed passes over the 1M
   residuals, writes PCA and RARS int8 sidecars, and verifies that the serialized
   index hash is unchanged.  Its CLI accepts no qrels, query, or holdout path.
3. **Locked evaluation:** a separate confirmation protocol must bind the exact
   method-freeze and full-corpus artifact hashes before any withheld outcomes
   are opened.  V8 development never authorizes future access by itself.

The third stage is intentionally not frozen before the first-stage claim tier
is known.  If RARS does not beat PCA, the later primary claim must be a generic
frozen residual-sidecar claim; the hypothesis may not be rewritten after
seeing holdout results.

## Development decision

The algorithm tier requires all of the following on OOF Recall@10:

- RARS gain over Base at least `+0.010`, with paired-bootstrap lower bound > 0;
- RARS gain over PCA at least `+0.002`, with lower bound > 0;
- at least 15% same-candidate exact gap recovery;
- at least 30 improved queries and at least 10 net improved queries;
- promotion pairs spanning at least 120 queries and protection pairs spanning
  at least 500 queries.

Passing produces `GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL`.  If the common
sidecar evidence passes but RARS-over-PCA does not, the result is explicitly
demoted to `GO_TO_GENERIC_SIDECAR_CONFIRMATION_PROTOCOL`.  Otherwise it is
`STOP_V8_CUTOFF_SIDECAR`.  Every decision keeps future access set to false.

## Required later baselines and measurements

A locked evaluation must report Base M32/nprobe16, storage-matched PCA, V8,
same-candidate exact scoring, larger `nprobe`, and a rebuilt M48 same-storage
baseline.  Exact-vector reranking is an oracle/high-storage baseline, not the
main competitor.  Quality reporting includes Recall@10, MRR@10, nDCG@10,
query-paired confidence intervals, improved/harmed counts, and candidate-gap
recovery.

Efficiency must include correction-only and end-to-end P50/P95/P99 latency,
QPS at batch sizes 1/32/64, complete artifact bytes, and peak host/device
memory.  A Python/NumPy prototype must not be described as a fused production
kernel.  A C++/SIMD kernel is useful only after the algorithm survives the
quality gate.

## Interpretation limits

Even a positive V8 result supports a retrofit claim under a fixed deployed
index.  It does not establish superiority to ScaNN, JPQ, OPQ, or a rebuilt
higher-rate PQ system.  Cross-dataset evidence and a genuinely independent
confirmation set remain required before a broad or SIGIR-ready claim.
