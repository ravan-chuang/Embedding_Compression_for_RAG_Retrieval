# RARS-v2.2 three-seed FP32 replication protocol

This document freezes the development-only optimizer-seed replication that
follows the seed-42 Stage-A result. It does not modify the original
`rars_v2_2_boundary_loss_development_v1` protocol. The machine-readable source
of truth is
`protocols/rars_v2_2_fp32_replication_v1.json`.

The outcome for seed 42 was already known when this protocol was frozen. Seed
42 is retained as audited lineage evidence and as one member of the registered
three-seed summary, but the primary replication gate is the still-unseen pair
of seeds 43 and 44. Therefore, this is an honest held-out-seed stability check,
not a fully outcome-blind three-seed preregistration.

## Metric and claim boundary

Every metric in this protocol is **Recall@10**. MRR is not used. Aggregate
Recall@10 is the arithmetic mean of the 1,019 per-query Recall@10 values and is
computed in float64 before display rounding.

The three seeds reuse the same inner-validation queries, and that same split
selects the best epoch inside each run. Here, “held-out” means unseen optimizer
seeds, not held-out queries. A successful result establishes development-set
optimizer stability only. It is not external, test-set, or confirmatory
evidence.

## Frozen execution lineage

Training must run from the exact clean detached commit:

```text
bb9b106e69b9a453756fd800665f701614ce67b3
```

The registered source hashes are:

| Input | SHA-256 |
|---|---|
| `scripts/aggregate_rars_v2_2_fp32_replication.py` | `64dab39bf6f5b7905552ced2994fa8247265df5ffc7df6c93a44f6a021c3741d` |
| `scripts/train_boundary_loss_sidecar.py` (metric helper) | `b547ff5961a86917da1ed01ec7aceeab04e4140f14be626a595550fb1bede294` |
| `scripts/build_msmarco_rars_v2_boundary_bundles.py` | `148470ddb115bcb9f9f6101924897293bc395b47297f00840450dbbe5dd6b175` |
| `scripts/freeze_rars_v2_2_inner_bundles.py` | `d416d769f005efe62bbcef8308beef25379c2f3a4c30ead5d42dda5e2b7357f2` |
| parent Stage-A protocol | `dbe5914e7cf7c6cbc6c811bcf8846e74dc4bee41ff9878371a3f193c9bced23e` |
| `scripts/train_boundary_loss_sidecar_v2_2.py` | `61eb3c9cc10ef7032246f28671fdb59bea81ffc2fc29e866c53526fe4e13eae0` |
| `scripts/rars_v2_2_core.py` | `1ee96d77dde59d365613b1d5b8726a56b05312500584e28cd49e41f7dbe61299` |
| inner-train manifest | `3508ea77cc0b89344a2290c45f703a8eb13c08223c863e4745951a1ebdb42b0e` |
| inner-validation manifest | `5daecd55de04c81e4b80b3307aea2ba1975ee3a77588b710b352f5a95accd26d` |
| PCA rank-16 FP32 basis | `ffad7e9a65d87045edef7e8d96e5fb90f2a5cc2e213038db72392e65c9ed8fec` |
| selected PCA configuration | `019170f80a085e1d3c57311b7fb504450351312be4ea7bb41e7f8c1de60f44d4` |

The split-audit SHA-256 is
`8bb13030c2808f5036bb6395a0408bc105daec1e5db5254d8f011f8ba1c8df4f`.
The frozen configuration is the one recorded in the JSON protocol. Across
seeds 42, 43, and 44, the only permitted configuration difference is `seed`.

Before either held-out seed starts, the runtime must match the environment
recorded by seed 42 for the full Python version string, NumPy, Torch, CUDA,
and device. The registered runtime is Python 3.12.13, NumPy 1.26.4, Torch
2.11.0+cu128, CUDA 12.8, and `cuda`; hardware must be an NVIDIA T4 with
compute capability 7.5. Deterministic algorithms must be enabled and
`cudnn_benchmark` must be false. `CUBLAS_WORKSPACE_CONFIG` is recorded, but
the runner must not introduce a setting that was absent from seed 42. Any
mismatch is `INVALID` before seeds 43 or 44 are trained.

The seed-42 notebook retained the exact manifest and split-audit hashes but
left the bundle payloads on ephemeral Colab disk. Before either held-out seed
starts, the runner may therefore materialize the two inner bundles exactly
once, with the same `bb9b106` builder/freezer, the same absolute Colab paths,
and `--inner-only`. The resulting inner-train manifest, inner-validation
manifest, and split audit must reproduce the three registered SHA-256 values
byte for byte. Any mismatch returns `INVALID`; it cannot be accepted as a new
bundle version.

This materialization honestly reopens and parses the inherited shared
6,980-query qrels container, including relevance values outside the two inner
roles. Only inner-train and inner-validation values may enter labels, loss,
selection, comparators, or metrics. The freezer also reads outer and clean-test
split identities for the disjointness audit, but no outer/test relevance or
outcome may enter the replication. BEIR NQ and TREC remain forbidden. The
audit packet must preserve the build summary, split audit, freeze summary,
query manifests, and role manifests with their hashes.

## Seeds and sealed execution

The registered seeds are exactly 42, 43, and 44. No extra seed, replacement
seed, or best-seed selection is allowed. Python, NumPy, Torch CPU, and Torch
CUDA RNGs all receive the registered seed.

Seeds 43 and 44 must run as a sealed pair. Neither run's aggregate or per-query
metric may be inspected until both runs have emitted `TRAINING_COMPLETE` and
all required artifacts have been hashed. An exogenous interruption may be
retried with the same seed only when no valid completion artifact was
produced; the failed attempt and its logs must remain in the audit packet.
NaN, numerical divergence, deterministic-runtime failure, or a completed poor
result is not replaceable or retryable.

## Exact decision variables

For seed `s`, define:

```text
B_s = Base Recall@10
P_s = PCA FP32 Recall@10
R_s = RARS-v2.2 FP32 Recall@10
delta_base_s = R_s - B_s
delta_pca_s  = R_s - P_s
joint_pass_s = delta_base_s >= 0.01135 and delta_pca_s >= 0.005
```

The primary held-out means are:

```text
heldout_base = mean(delta_base_43, delta_base_44)
heldout_pca  = mean(delta_pca_43,  delta_pca_44)
```

The all-three means use seeds 42, 43, and 44 in the same way.

Positive support is a strict per-query Recall@10 improvement. For each of
seeds 43 and 44 separately, at least `ceil(0.01 * 1019) = 11` queries must
improve over Base and at least 11 queries must improve over PCA.

## Held-out paired joint-query bootstrap

The bootstrap is calculated only after both held-out runs are complete and
unsealed. Per query `i`, define:

```text
g_base_i = mean(RARS(43,i) - Base(43,i), RARS(44,i) - Base(44,i))
g_pca_i  = mean(RARS(43,i) - PCA(43,i),  RARS(44,i) - PCA(44,i))
```

The Base, PCA, and RARS arrays must share the same frozen 1,019-query identity
and order. Use NumPy `Generator(PCG64)` with seed `20260717`. For each of
20,000 replicates, sample 1,019 query positions with replacement and use that
same index vector for both held-out seeds and both comparator calculations.
Record the mean sampled `g_base_i` and `g_pca_i` values. The lower endpoints
are `np.quantile(statistics, 0.025, method="linear")`. Both lower endpoints
must be strictly greater than zero.

This is a paired query bootstrap over the fixed development queries. It is not
a bootstrap over independent datasets or independent seeds.

## Decision order

Validation happens before scientific classification:

1. Verify the exact source commit, all registered source hashes, complete run
   fingerprints, frozen configuration, query identity, environment records,
   and every artifact byte count and SHA-256 for all three seeds. Any failure
   returns `INVALID`; no success or failure conclusion may be computed.
2. Compute `heldout_base` and `heldout_pca`. If `heldout_base < 0.01135` or
   `heldout_pca < 0.005`, return `STOP`.
3. If both held-out thresholds pass, return `STABLE_GO_TO_QAT` only when every
   condition below also passes:

   - seeds 43 and 44 each have positive gain over both Base and PCA;
   - both held-out means meet `0.01135` and `0.005`;
   - both all-three means meet `0.01135` and `0.005`;
   - all three seeds pass both thresholds jointly, so a strong held-out seed
     cannot average away a weak held-out seed;
   - both held-out paired-bootstrap 95% lower bounds are greater than zero;
   - each held-out seed has at least 11 positive-support queries over each
     comparator.

4. When both held-out thresholds pass but any remaining condition fails,
   return `UNSTABLE_NO_QAT`.

The meanings are intentionally distinct:

- `STABLE_GO_TO_QAT`: permits freezing a separate QAT protocol. It does not
  itself permit an int8 or deployment claim.
- `STOP`: the unseen-seed mean effect misses a registered effect-size gate.
- `UNSTABLE_NO_QAT`: the mean gate passes, but the evidence is not robust
  enough to authorize QAT.
- `INVALID`: provenance or execution is not auditable. It is neither a
  scientific success nor a scientific failure.

No decision permits adding seed 45, dropping a failed seed, changing the loss,
or revising these thresholds after seeing seeds 43 or 44.

## Required artifacts

Each seed must retain the eight trainer outputs plus its completion record:

```text
training_started.json
query_projection.float32.npy
document_projection.float32.npy
selection_base_per_query.float64.npy
selection_v2_2_per_query.float64.npy
training_history.json
selection_metrics.json
training_summary.json
training_complete.json
```

The replication evaluator must additionally retain a PCA FP32 per-query
Recall@10 array in the identical query order, a query-identity manifest, both
held-out seeds' Base/PCA support counts, the 20,000 paired-bootstrap statistics
or a content-addressed lossless equivalent, and a final replication manifest.
That manifest must link the three run IDs, input and output hashes,
environments, metrics, confidence intervals, condition flags, and decision.

The current trainer's `training_complete.json` hashes eight outputs. The
replication decision must not be emitted merely from notebook console text or
rounded aggregate metrics.

## Audited seed-42 anchor

Seed 42 is fixed to run ID:

```text
0956f2f51c2183020a9d68ecb6a89987f7bcef99e0587ceb507a05f303a03b83
```

It selected epoch 9 and produced:

| Metric | Recall@10 |
|---|---:|
| Base | 0.6933267909715407 |
| PCA FP32 | 0.7067386326463853 |
| bounded PCA-parameter warm start | 0.7062479555119398 |
| RARS-v2.2 FP32 | 0.7139352306182531 |
| gain over Base | 0.02060843964671244 |
| gain over PCA FP32 | 0.007196597971867891 |

Relative to Base, 36 queries improved, 12 were harmed, and 971 were unchanged.
The eight audited output records are:

| Output | Bytes | SHA-256 |
|---|---:|---|
| `training_started.json` | 1,904 | `25d23f02f175542a2779fe53a29a52c325eb5dcaa13c573f47310f9cef80cb44` |
| `query_projection.float32.npy` | 24,704 | `44f4c5f3f895eb3d84c159d4025a97bac4b7913085f272945f9d176d93777bb3` |
| `document_projection.float32.npy` | 24,704 | `d6c0f610163845418213239833213b3096a0f8d07c58265e15d0aa2e4c4c4f5a` |
| `selection_base_per_query.float64.npy` | 8,280 | `31dfd394a2f446848c649a7a3357664a10ff841975be9f31b1e83c0d96b803e8` |
| `selection_v2_2_per_query.float64.npy` | 8,280 | `64e80ae359b46dbd77555a5e32269341eecf149fad7d97fac53a21c8ae226059` |
| `training_history.json` | 5,449 | `2d0d8648b860135c8d95ed54317a45f0e83b7947316e0415a46946949c66fac9` |
| `selection_metrics.json` | 629 | `eea4323e46354cf18f6413643e7c93fcb6a158b12773eb4d8cd06270f366a83a` |
| `training_summary.json` | 3,776 | `737798792297c3c042c12c869e646eb1864ee6240bb47d20599545299c346800` |

These records were recovered read-only from the completed Google Drive run and
verified by byte count and SHA-256. The `training_started.json` record binds
the run fingerprint and recorded environment; the JSON protocol binds the
full metrics and source lineage used by the replication decision.
