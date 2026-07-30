# RARS-v11 Pre-Metric Faiss Compatibility Repair

## Scope

The first V11 execution attempt used source commit
`08bf7184acc7f91816a9d5303468c8fa3f40b607`. It completed environment, source,
input-manifest, and V6-lineage validation and wrote only
`diagnostic_started.json`. It did not write a per-query metric array,
`rank_rate_result.json`, `diagnostic_freeze.json`, or
`diagnostic_complete.json`.

An isolated, no-metric stage probe localized the failure to rank-32 RPQ
training after PCA, rank-16 int8, rank-64 FP32, and packed rank-32 int4 stages
had completed. The exception was:

```text
AttributeError: 'ProductQuantizer' object has no attribute 'is_trained'
```

The pinned Colab package `faiss-gpu-cu12==1.12.0` exposes trained PQ centroids
and `compute_codes` but does not expose the `ProductQuantizer.is_trained`
property available in the local Faiss binding used during development.

## Repair

The compatibility-only repair removes the binding-specific property access and
replaces it with version-independent postconditions:

- exact centroid-table element count;
- finite centroid values;
- exact `(document_count, 16)` uint8 code shape;
- decoded coefficient shape equality;
- finite coefficient reconstruction MSE.

The protocol, ranks, PCA basis construction, product partition, codebook size,
iterations, seed, alpha, Top-B, inference seeds, thresholds, and formal
decisions are unchanged. No metric or label-dependent output existed before
the repair.

The notebook runner now saves evaluator stdout and stderr under the
commit-specific `runner-logs` directory before raising a subprocess error.
This logging change does not affect the diagnostic computation.

## Rerun policy

The incomplete `08bf7184acc7/diagnostic-once` directory is preserved as failed
pre-metric provenance and must not be overwritten. A fresh exact commit creates
a different commit-keyed output directory and may execute the unchanged frozen
V11 protocol once. This is an implementation-compatibility rerun, not a method
or threshold revision.
