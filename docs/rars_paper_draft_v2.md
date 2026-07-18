# RARS Paper Draft v2

## Working title

**Residual Sidecars for Frozen IVF-PQ Retrieval: Development Gains, Sparse Query Support, and External Fragility**

## Current submission status

This version replaces the method-superiority framing in v1. The committed evidence supports a frozen-index retrofit result, but it does not support a general claim that retrieval-aware basis learning is better than ordinary residual PCA.

The larger preregistered [BEIR Natural Questions confirmation](beir_nq_confirmation_protocol.md) is now complete. Its 3,452-query one-shot test result does not support RARS superiority over PCA, and neither sidecar improves Recall@10 over the frozen base. All three existing evaluation sets are closed to further method selection.

A later three-seed v2.2 FP32 development replication is also complete. Its two previously unseen optimizer seeds have a positive mean effect over Base and direct PCA, but seed 44 fails the preregistered positive-query-support condition. The formal decision is `UNSTABLE_NO_QAT`; this is development-only evidence and does not reopen any closed evaluation set.

## Abstract

Rebuilding a deployed approximate-nearest-neighbor index can be costly even when its compressed document representation has lost retrieval quality. We study a post-hoc residual sidecar for a frozen IVF-PQ index. Each document stores 16 int8 coefficients in a shared rank-16 residual basis, and only the top 40 of 100 retrieved candidates receive a low-rank score correction. We compare an ordinary unweighted PCA basis with Retrieval-Aware Residual Subspace (RARS), which weights residual directions by train-query candidate score error.

On a deterministic one-million-passage MS MARCO benchmark, RARS is fitted on 4,980 queries, selected on 1,000 validation queries, and evaluated on a disjoint 1,000-query split after freezing the configuration and evaluator. RARS improves Recall@10 over the frozen IVF-PQ base from 0.6833 to 0.7073, a paired difference of +0.0240 with a 95% bootstrap confidence interval of [+0.0105, +0.0378]. A project-history audit later finds 137 queries that appeared in earlier exploratory work; excluding them leaves a positive Recall@10 difference of +0.0168 on 863 prior-unseen queries.

We then preregister a storage-matched RARS-versus-PCA comparison on an external TREC DL 2019 query set restricted to the same frozen one-million-passage corpus. Across 42 eligible queries, RARS minus PCA Recall@10 is -0.0181 with a 95% confidence interval of [-0.0735, +0.0168], so the primary external hypothesis is not supported. RARS has higher MRR@10 and nDCG@10 point estimates, but their intervals also cross zero. Query-level tracing shows that the Recall result depends on five boundary-changing queries and is especially sensitive to one sparsely judged query.

Finally, a newly fitted full-corpus BEIR NQ experiment freezes Base, PCA, RARS, configuration selection, and evaluator before opening official test qrels. Across 3,452 eligible test queries, RARS minus PCA Recall@10 is -0.000410 with a 95% confidence interval of [-0.005987, +0.004972]. Base, PCA, and RARS obtain Recall@10 of 0.37973, 0.37811, and 0.37770, respectively. The primary hypothesis is again unsupported, and no post-result retuning is performed.

We then evaluate the separately frozen RARS-v2.2 FP32 boundary-loss objective on the same 1,019 inner-validation queries. Previously unseen optimizer seeds 43 and 44 obtain mean Recall@10 0.714426, improving Base by +0.021099 and direct PCA by +0.007687; both paired-query bootstrap lower bounds are above zero. However, seed 44 improves only 10 queries over PCA, below the preregistered requirement of 11. The formal decision is `UNSTABLE_NO_QAT`. Held-out here means optimizer seeds, not queries or datasets.

These results show that compact residual correction can improve a frozen index in a controlled in-distribution pipeline, while neither score-error weighting nor the completed boundary-loss study establishes reliable superiority over strong PCA under independent confirmation. The boundary-loss objective raises the development mean but does not broaden query-level support enough to pass its frozen robustness gate.

## Research questions

1. Can a compact residual sidecar improve a frozen low-rate IVF-PQ index without rewriting its PQ codes?
2. Under matched storage and serving constraints, does RARS outperform ordinary residual PCA on independent queries?
3. What storage and online correction cost is required, and when is rebuilding a higher-rate index still preferable?
4. Can a relevance-boundary FP32 objective broaden query-level gains enough to justify a later quantized stage?

## Evidence hierarchy

| Evidence | Role | Result | Allowed claim |
|---|---|---|---|
| MS MARCO 4,980 / 1,000 / 1,000 split | Clean-pipeline base comparison | RARS minus base Recall@10 `+0.0240`, CI `[+0.0105, +0.0378]` | Positive within the frozen clean pipeline |
| 863-query overlap-excluded subset | Post-hoc sensitivity | RARS minus base Recall@10 `+0.0168`, CI `[+0.0029, +0.0303]` | Robustness to known prior-query overlap, not a new test |
| TREC DL 2019 / frozen 1M restriction | Preregistered external comparator | RARS minus PCA Recall@10 `-0.0181`, CI `[-0.0735, +0.0168]` | Primary RARS-over-PCA hypothesis unsupported |
| BEIR NQ / full 2.68M corpus | Preregistered one-shot comparator | RARS minus PCA Recall@10 `-0.000410`, CI `[-0.005987, +0.004972]` | Large independent confirmation does not support superiority |
| BEIR NQ locked post-hoc diagnosis | Exploratory failure analysis only | Exact Top-40 minus Base `+0.08379`; RARS proxy/relevance Pearson `0.150` | Headroom exists, but NQ retuning remains prohibited |
| RARS-v2.2 / 1,019 development selection queries | Held-out optimizer-seed replication, development only | Held-out mean `+0.007687` vs PCA; seed 44 support `10 < 11` | Mean effect passes, robustness condition fails; no QAT or external claim |
| Earlier same-query MS MARCO and FiQA runs | Developmental diagnostics | Mixed, often positive versus base; PCA remains strong | Ablations and motivation only |

## Contributions

1. **Frozen-index retrofit.** A rank-16 int8 residual sidecar improves scores for a fixed Top-100 candidate set without changing the coarse quantizer, inverted lists, codebooks, or PQ codes.
2. **Storage-matched basis comparison.** RARS and unweighted PCA share the same document payload, quantizer, candidate pool, correction path, and validation selection rule.
3. **Auditable evaluation.** The repository records deterministic query splits, pre-evaluation freezes, artifact hashes, per-query metrics, and paired-bootstrap outputs, including a project-history overlap audit.
4. **Negative external evidence.** The preregistered TREC and BEIR NQ external results are retained even though they fail the primary hypothesis, with diagnostics that do not change either frozen conclusion.
5. **Development robustness boundary.** The v2.2 optimizer-seed mean effect is positive, but its preregistered query-support gate fails and blocks QAT, showing why mean effects and positive confidence intervals are insufficient on their own.
6. **Deployment boundary.** The v1 sidecar costs `16.025 B/document` for the residual representation (`24.028 B/document` including external IDs), while a rebuilt M48 index remains stronger when re-encoding is allowed. The v2.2 result is FP32-only.

## Main results to report

### Clean-pipeline MS MARCO comparison

| System | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen IVF-PQ M32 | 0.6833 | 0.6910 | 0.4722 | 0.5204 |
| Frozen RARS Top40 | 0.7073 | 0.7180 | 0.4851 | 0.5360 |
| Difference | +0.0240 | +0.0270 | +0.0129 | +0.0156 |

### Frozen external comparison

| System | Recall@10 | Success@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen IVF-PQ M32 | 0.3507 | 0.8095 | 0.6939 | 0.4405 |
| PCA rank-16 int8 | 0.3445 | 0.8095 | 0.7093 | 0.4558 |
| RARS rank-16 int8 | 0.3264 | 0.7857 | 0.7341 | 0.4624 |

The primary paired contrast is RARS minus PCA Recall@10: `-0.0181`, 95% CI `[-0.0735, +0.0168]`. This result must appear in the abstract, results, limitations, and conclusion.

### Full-corpus BEIR NQ comparison

| System | Recall@10 |
|---|---:|
| Frozen IVF-PQ M32 | 0.37973 |
| PCA rank-16 int8 | 0.37811 |
| RARS rank-16 int8 | 0.37770 |

The primary paired contrast is RARS minus PCA Recall@10: `-0.000410`, 95% CI `[-0.005987, +0.004972]`. This result is independent confirmation evidence but does not support superiority.

### RARS-v2.2 FP32 development replication

| System / seed | Recall@10 | Gain vs Base | Gain vs direct PCA | Improved / harmed vs PCA |
|---|---:|---:|---:|---:|
| Frozen Base | 0.693327 | -- | -- | -- |
| Direct PCA FP32 | 0.706739 | +0.013412 | -- | -- |
| Seed 42, observed anchor | 0.713935 | +0.020608 | +0.007197 | 10 / 2 |
| Seed 43, held-out optimizer seed | 0.714917 | +0.021590 | +0.008178 | 12 / 3 |
| Seed 44, held-out optimizer seed | 0.713935 | +0.020608 | +0.007197 | 10 / 2 |
| Held-out seed mean | **0.714426** | **+0.021099** | **+0.007687** | -- |

The registered requirement is at least 11 improved queries for every held-out seed and contrast. Seed 44 has 10 versus PCA, so the formal decision is `UNSTABLE_NO_QAT`. This table is development-only and must remain separate from the two external comparisons.

## Claim boundary

The paper may claim that:

- low-rank int8 residual correction is a feasible retrofit for a frozen IVF-PQ index;
- RARS improves the frozen base in the clean MS MARCO pipeline;
- known prior-query overlap does not fully explain the clean-pipeline Recall gain;
- the external experiment does not establish RARS superiority over PCA;
- sparse judgments and a small external query count make that external estimate unstable.
- the BEIR NQ result also does not support RARS superiority over PCA or Base;
- v2.2 has a positive held-out-optimizer-seed mean over PCA but fails its frozen query-support condition, so QAT is not authorized.

The paper must not claim that:

- the 1,000-query split is untouched across the complete project history;
- RARS universally beats PCA or the frozen base;
- the restricted 42-query result is an official full-corpus TREC benchmark;
- post-hoc removal of the influential query reverses the preregistered result;
- the sidecar is storage-optimal when rebuilding a higher-rate index is possible.
- v2.2 is an independent-query, test-set, cross-dataset, or external confirmation;
- the v2.2 support gate passed, or its query bootstrap includes optimizer-seed uncertainty;
- v2.2 provides int8, latency, storage, serving, or deployment evidence.

## Four-page short-paper structure

1. **Introduction and deployment setting (0.5 page).** Frozen-index upgrade constraint, question, and mixed headline result.
2. **Method and matched comparator (0.75 page).** Residual correction equation, PCA and RARS basis construction, identical storage and serving path.
3. **Protocol (0.75 page).** Clean split, project-history audit, TREC/NQ one-shot rules, v2.2 optimizer-seed/support gate, paired statistics.
4. **Results (1.25 pages).** Clean base comparison, TREC and NQ external comparisons, one compact v2.2 sparse-support table, storage/latency.
5. **Limitations and conclusion (0.75 page).** Corpus restriction, external null/negative results, development-only v2.2 evidence, rebuild boundary, required new confirmation.

## Post-v2.2 research gate

BEIR NQ confirmation and v2.2 replication are complete. The MS MARCO clean
split and inner-validation pool, TREC set, and NQ test are closed. Under v2.2,
no added seed, threshold change, method revision, or QAT is authorized.

Any new method must be separately versioned and start with an oracle study on
development-only data. The oracle must test whether counterfactual Recall gain
per byte can expand positive-query support under matched storage and accessed-
byte budgets. If the matched-byte oracle cannot materially exceed fixed-rate
PCA/uniform correction, or the recoverable gain remains concentrated in about
1% of queries, the adaptive-allocation line should stop before learned routing,
QAT, or a new external corpus is opened.

Only after the new method, budgets, comparators, evaluator, support metrics,
and stop rules are frozen may a new independent dataset be opened once. Until
that happens, the strongest manuscript is a mixed-evidence empirical study of
frozen-index residual sidecars, sparse boundary gains, and failure to establish
general superiority over ordinary PCA.
