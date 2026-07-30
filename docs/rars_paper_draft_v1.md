# RARS Paper Draft v1

> Historical draft. It predates the storage-matched PCA comparator and the
> preregistered external result. Use
> [`rars_paper_draft_v2.md`](rars_paper_draft_v2.md) for the current framing.

## Working title

**Retrieval-Aware Residual Sidecars for Frozen IVF-PQ Indexes**

## Abstract

Approximate nearest-neighbor indexes used in retrieval-augmented generation are often expensive to rebuild after deployment. We study whether retrieval quality can be recovered by attaching a compact post-hoc residual sidecar to a frozen IVF-PQ index without retraining its quantizers or rewriting existing PQ codes. Our method, Retrieval-Aware Residual Subspace (RARS), learns a rank-16 residual basis weighted by candidate score error and stores 16 int8 coefficients per document. At query time, RARS applies a low-rank score correction to the top candidates returned by the unchanged base index.

We evaluate RARS on a deterministic one-million-passage MS MARCO benchmark using a strict query-level protocol: 4,980 queries for basis fitting, 1,000 validation queries for configuration selection, and an untouched 1,000-query test split used once after freezing the selected configuration and evaluator. The frozen configuration uses a score-error-weighted basis, alpha 0.75, and Top-40 correction. On the untouched test split, RARS improves Recall@10 from 0.6833 to 0.7073, Success@10 from 0.6910 to 0.7180, MRR@10 from 0.4722 to 0.4851, and nDCG@10 from 0.5204 to 0.5360. Paired-bootstrap 95% confidence intervals are strictly above zero for all four metrics; the Recall@10 difference is +0.0240 with a 95% confidence interval of [+0.0105, +0.0378].

The residual representation requires 16.03 bytes per document and leaves the deployed IVF-PQ index unchanged. These results position RARS as a statistically validated retrofit for recovering ranking quality when rebuilding and re-encoding an existing index is operationally undesirable.

## Contributions

1. **Frozen-index residual retrofit.** We formulate a post-hoc sidecar that improves an existing IVF-PQ index without retraining the coarse quantizer, rewriting PQ codes, or changing the base candidate generator.

2. **Retrieval-aware residual basis.** We introduce a score-error-weighted rank-16 residual subspace that prioritizes residual directions associated with candidate ranking error rather than reconstruction variance alone.

3. **Leakage-resistant evaluation protocol.** We separate basis fitting, hyperparameter selection, and final reporting into deterministic, non-overlapping 4,980 / 1,000 / 1,000 query splits, and freeze the selected configuration before untouched-test evaluation.

4. **Statistically positive million-scale result.** On an untouched 1,000-query MS MARCO test split, RARS improves Recall@10 by +0.0240, with a paired-bootstrap 95% confidence interval of [+0.0105, +0.0378]. Success@10, MRR@10, and nDCG@10 also improve with confidence intervals above zero.

5. **Deployment-aware accounting.** We report representation size, complete artifact size, candidate correction depth, and live-Faiss correction cost separately, and retain the negative result that a higher-rate rebuilt IVF-PQ index remains stronger when re-encoding is allowed.

## Core claim

> A rank-16 int8 retrieval-aware residual sidecar can significantly improve a frozen low-rate IVF-PQ index on unseen queries without rebuilding or rewriting the base index.

## Claim boundary

The paper should not claim that RARS is globally storage-optimal, universally superior to PCA, or stronger than a newly rebuilt higher-rate IVF-PQ index. Its validated role is a frozen-index retrofit under operational rebuild constraints.
