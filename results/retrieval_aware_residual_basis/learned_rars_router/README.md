# Learned RARS Router Diagnostics

This experiment evaluates whether a learned query-level router can select the RARS correction depth per query.

The router chooses among:

- Top0: no sidecar correction
- Top20: correct the top 20 ANN candidates
- Top40: correct the top 40 ANN candidates

The goal is to approach fixed Top20 / Top40 RARS quality while correcting far fewer candidates per query.

## Fixed-depth baselines

| Strategy | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|---|---:|---:|---:|---:|---:|
| Always Top0 | 0.6628 | 0.6740 | 0.4659 | 0.5099 | 0.0 |
| Always Top20 | 0.6989 | 0.7090 | **0.4845** | 0.5324 | 20.0 |
| Always Top40 | 0.6999 | 0.7100 | 0.4845 | **0.5325** | 40.0 |

## Oracle routing

The oracle chooses the cheapest depth among Top0 / Top20 / Top40 that attains the best per-query Recall@10.

| Depth | Queries | Fraction |
|---:|---:|---:|
| 0 | 950 | 0.950 |
| 20 | 45 | 0.045 |
| 40 | 5 | 0.005 |

The oracle reaches Recall@10 `0.7103` with only `1.1` corrected candidates per query on average. This shows that substantial query-adaptive headroom exists in principle.

## Learned router results

| Strategy | Recall@10 | Success@10 | MRR@10 | nDCG@10 | Avg corrected candidates |
|---|---:|---:|---:|---:|---:|
| Offline exact-proxy features + logistic regression | 0.6813 | 0.6920 | 0.4718 | 0.5185 | 4.56 |
| Deployable features + logistic regression | 0.6774 | 0.6870 | 0.4760 | 0.5214 | 5.94 |
| Deployable features + random forest | 0.6628 | 0.6740 | 0.4671 | 0.5109 | 0.70 |
| Deployable features + histogram gradient boosting | 0.6628 | 0.6740 | 0.4659 | 0.5099 | 0.02 |

The learned routers do not approach fixed Top20 / Top40 quality. The strongest learned result reaches Recall@10 `0.6813`, below fixed Top20 at `0.6989`.

## Interpretation

This is a negative diagnostic result.

Although oracle Top0/Top20/Top40 routing reaches Recall@10 `0.7103` with only `1.1` corrected candidates per query, the 5-fold learned routers fail to recover that headroom. The label distribution is highly imbalanced: 95.0% of queries need no correction under the oracle, 4.5% prefer Top20, and only 0.5% prefer Top40. Most learned models either collapse toward Top0 or over-correct without enough retrieval gain.

The current handcrafted ANN-score, correction-magnitude, and query-vector features are therefore insufficient for robust learned routing. Fixed Top20 remains the strongest deployable cost-aware RARS operating point currently validated.

## Files

- `fixed_depth_metrics.csv`
- `oracle_label_distribution.csv`
- `router_5fold_summary.csv`
- `router_strategy_comparison.csv`
- `router_feature_importance.csv`
