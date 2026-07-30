# RARS-v2.2 three-seed FP32 replication result

## Formal outcome

The frozen `rars_v2_2_fp32_replication_v1` protocol is complete. The formal
decision is **`UNSTABLE_NO_QAT`**. QAT protocol definition is not authorized.

This classification does not mean that the seed-level means fluctuate
substantially. All three seeds pass the registered Base and direct-PCA
effect-size thresholds, and the held-out-seed query-bootstrap intervals are
strictly above zero. The failed condition is narrower: seed 44 improves only
10 of 1,019 queries over direct PCA, while the frozen minimum is 11.

## Frozen lineage

| Item | Frozen value |
|---|---|
| Parent development protocol | `rars_v2_2_boundary_loss_development_v1` |
| Replication protocol | `rars_v2_2_fp32_replication_v1` |
| Training source commit | `bb9b106e69b9a453756fd800665f701614ce67b3` |
| Replication control commit | `00a0dee30767b04b8c650c28d63f4f662ef61517` |
| Queries | 1,019 inner-validation queries |
| Observed anchor | Seed 42 |
| Held-out optimizer seeds | 43 and 44 |
| Bootstrap | 20,000 paired query resamples; seed `20260717` |
| Environment | Python 3.12.13, NumPy 1.26.4, Torch 2.11.0+cu128, CUDA 12.8, Tesla T4 |

The held-out batch ran seeds 43 and 44 sequentially with return code zero and
recorded `interim_metrics_revealed: false`. The committed packet verifier
rechecks all aggregate, runner, and trainer records against their registered
byte counts and SHA-256 hashes.

## Results

| System / seed | Recall@10 | Gain vs Base | Gain vs direct PCA | Improved / harmed queries vs PCA |
|---|---:|---:|---:|---:|
| Frozen Base | 0.693326791 | -- | -- | -- |
| Direct PCA FP32 | 0.706738633 | +0.013411842 | -- | -- |
| Seed 42, observed anchor | 0.713935231 | +0.020608440 | +0.007196598 | 10 / 2 |
| Seed 43, held-out | 0.714916585 | +0.021589794 | +0.008177952 | 12 / 3 |
| Seed 44, held-out | 0.713935231 | +0.020608440 | +0.007196598 | 10 / 2 |
| Held-out mean, seeds 43/44 | **0.714425908** | **+0.021099117** | **+0.007687275** | -- |
| Descriptive all-seed mean | 0.714262349 | +0.020935558 | +0.007523716 | -- |

The all-seed Recall@10 sample standard deviation is `0.000566585`. The
held-out-seed paired-query bootstrap gives:

| Contrast | Mean difference | 95% CI | P(difference > 0) |
|---|---:|---:|---:|
| v2.2 minus Base | +0.021099117 | [+0.009322866, +0.033366045] | 0.9997 |
| v2.2 minus direct PCA | +0.007687275 | [+0.001472031, +0.014556755] | 0.9912 |

## Decision audit

All registered conditions pass except one:

| Condition | Result |
|---|---|
| All three seeds pass both effect-size thresholds | Pass |
| Both held-out seeds are positive versus Base and PCA | Pass |
| Held-out means meet both thresholds | Pass |
| All-seed means meet both thresholds | Pass |
| Both held-out query-bootstrap lower bounds are above zero | Pass |
| Every held-out seed has at least 11 improved queries per contrast | **Fail** |

Seed 43 has 12 improved queries over direct PCA. Seed 44 has 10. The protocol
therefore requires `UNSTABLE_NO_QAT`, even though the aggregate effect is
positive and numerically consistent across optimizer seeds.

## Claim boundary

This result supports the following limited statement:

> On the same 1,019-query development selection split, the boundary-loss FP32
> sidecar reproduces a positive mean gain over Base and direct PCA across two
> previously unseen optimizer seeds, but the improvement remains too sparse
> across queries to pass the preregistered robustness gate.

It does **not** support any of the following claims:

- independent-query, test-set, cross-dataset, or external confirmation;
- a successful three-seed outcome-blind preregistration, because seed 42 was
  already observed when the replication protocol was frozen;
- an int8, QAT, storage, latency, or deployment improvement;
- a bootstrap interval that includes optimizer-seed or dataset uncertainty;
- permission to add seed 45, remove seed 44, change the 1% support rule, or
  proceed to QAT under v2.2.

## Artifact packet

The immutable result packet is committed under
[`results/rars_v2_2_fp32_replication/`](../results/rars_v2_2_fp32_replication/README.md).
The source protocols remain unchanged. Any new loss, allocation mechanism,
support threshold, quantization stage, or dataset must use a separately
versioned protocol.
