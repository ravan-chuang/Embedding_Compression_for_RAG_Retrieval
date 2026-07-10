# Query-Adaptive RARS Correction Diagnostics

This package contains preliminary query-adaptive correction diagnostics for the
RARS-Score residual sidecar on the frozen MS MARCO 1M IVF-PQ M32 index.

The goal is to evaluate whether the residual sidecar must always be applied to
a fixed number of candidates, or whether query-level uncertainty can reduce
correction cost.

## Fixed-depth RARS-Score operating points

All methods use the same frozen IVF-PQ M32 base index and the same RARS-Score
rank-16 int8 residual sidecar with alpha = 0.75.

| Method | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|---|---:|---:|---:|---:|---:|
| Frozen IVF-PQ M32 / Top0 | 0.6628 | 0.6740 | 0.4659 | 0.5099 | 0 |
| RARS-Score Top20 | 0.6989 | 0.7090 | 0.4845 | 0.5324 | 20 |
| RARS-Score Top40 | 0.6999 | 0.7100 | 0.4845 | 0.5325 | 40 |

Top20 recovers almost all of the Top40 Recall@10 gain while halving the
correction depth.

- Top20 gain over frozen base: 0.0362 Recall@10
- Top40 gain over frozen base: 0.0372 Recall@10
- Top20 retains approximately 97.3% of the Top40 Recall@10 gain.

## Gate 0: same-split oracle and single-feature diagnostics

Gate 0 evaluates query-adaptive correction on the same 1,000-query split.

| Router | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|---|---:|---:|---:|---:|---:|
| Always Top40 | 0.6999 | 0.7100 | 0.4845 | 0.5325 | 40.0 |
| Oracle Top0/Top20/Top40 | 0.7103 | 0.7200 | 0.5113 | 0.5551 | 3.7 |
| Best single-feature gate | 0.7004 | 0.7100 | 0.4840 | 0.5321 | 34.0 |
| Cheapest gate within 0.001 Recall | 0.6994 | 0.7090 | 0.4829 | 0.5312 | 28.0 |

The oracle result shows substantial headroom for query-adaptive activation:
most queries do not require correction, while a small subset benefits strongly.

## Gate 1: sorted train/test split diagnostic

A deterministic sorted 500/500 split was tested as an initial train/test
diagnostic. This split showed a large distribution shift:

| Split | Top0 Recall@10 | Top20 Recall@10 | Top40 Recall@10 |
|---|---:|---:|---:|
| Train | 0.6927 | - | 0.7170 |
| Test | 0.6328 | 0.6768 | 0.6828 |

Because the sorted split creates strong distribution shift, these results are
used only as a diagnostic and not as the main adaptive-routing conclusion.

## Gate 1b: random 5-fold threshold routing

Gate 1b uses random 5-fold cross-validation. For each fold, thresholds are
selected on 800 training queries and evaluated on 200 held-out test queries.

| Strategy | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|---|---:|---:|---:|---:|---:|
| Always Top0 | 0.6628 | 0.6740 | 0.4659 | 0.5099 | 0.0 |
| Always Top20 | 0.6989 | 0.7090 | 0.4845 | 0.5324 | 20.0 |
| Always Top40 | 0.6999 | 0.7100 | 0.4845 | 0.5325 | 40.0 |
| Best train-recall gate, target Top20 | 0.6979 | 0.7080 | 0.4840 | 0.5317 | 19.38 |
| Best train-recall gate, target Top40 | 0.6989 | 0.7090 | 0.4839 | 0.5318 | 38.24 |
| Cheapest within 0.001 train Recall, target Top20 | 0.6944 | 0.7040 | 0.4815 | 0.5289 | 15.32 |
| Cheapest within 0.001 train Recall, target Top40 | 0.6949 | 0.7050 | 0.4819 | 0.5292 | 27.56 |

Simple one-dimensional threshold gates do not robustly outperform fixed-depth
activation. In particular, when the threshold gate saves substantial correction
cost, held-out Recall@10 drops more than the fixed Top20 operating point.

## 5-fold oracle upper bound

| Oracle metric | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Top0 rate | Top20 rate | Top40 rate | Avg corrected candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Recall@10 oracle | 0.7103 | 0.7200 | 0.5113 | 0.5551 | 0.820 | 0.175 | 0.005 | 3.7 |
| nDCG@10 oracle | 0.7097 | 0.7200 | 0.5118 | 0.5551 | 0.821 | 0.174 | 0.005 | 3.68 |

The oracle router indicates that query-adaptive activation has substantial
headroom, but the current single-feature gates are not strong enough to realize
that headroom robustly.

## Interpretation

The strongest current cost-aware operating point is fixed Top20 RARS-Score:

- It reaches Recall@10 0.6989.
- It is within 0.0010 Recall@10 of fixed Top40.
- It halves correction depth from 40 to 20 candidates/query.

The adaptive-routing diagnostics support future learned routing work:

- Sidecar gains are sparse across queries.
- Oracle routing selects Top0 for roughly 82% of queries.
- Oracle routing selects Top20 for roughly 17.5% of queries.
- Oracle routing rarely requires Top40.
- Simple score-margin or score-variance gates are not yet robust enough.

## Files

Key diagnostic artifacts:

- `query_adaptive_oracle_router_summary.csv`
- `query_adaptive_oracle_router_per_query.csv`
- `query_uncertainty_features.csv`
- `query_feature_gain_analysis.csv`
- `query_adaptive_margin_gate_sweep.csv`
- `query_adaptive_single_feature_gate_sweep.csv`
- `query_adaptive_gate0_summary.csv`
- `gate1_binary_threshold_train_sweep.csv`
- `gate1_binary_threshold_train_test_eval.csv`
- `gate1_train_test_summary.csv`
- `gate1_test_oracle_router_summary.csv`
- `gate1b_5fold_train_sweep.csv`
- `gate1b_5fold_train_test_eval.csv`
- `gate1b_5fold_cv_summary.csv`
- `gate1b_5fold_strategy_summary.csv`
- `gate1b_5fold_quality_cost_pareto.csv`
- `gate1b_5fold_oracle_router.csv`
- `gate1b_5fold_oracle_router_summary.csv`

## Conservative claim

Fixed Top20 correction is a strong cost-aware deployment setting. It retains
nearly all of the Top40 RARS-Score quality gain at half correction depth.
Single-feature query-adaptive gates are not yet robust, but oracle routing
shows strong headroom for future learned adaptive activation.
