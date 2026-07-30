# RARS-v5 Pre-outcome Notebook Repair

## Status

This is an implementation-only notebook repair made before the v5 100K bundle
was built and before any v5 training or selection outcome was observed. The
frozen v5 method, loss, data roles, hyperparameters, gates, and implementation
commit remain unchanged.

## Failure

The first notebook attempt rebuilt the v3 candidate roles successfully but
stopped before materializing the already-observed `oracle_audit` labels. The v3
materializer rejected the rebuilt design candidate manifest:

```text
ValueError: Registered design_bundle_manifest hash changed
```

No audit-label artifact was created, the v5 builder had not run, and no v5
metric was available.

## Cause

The completed v3 design freeze registered manifests whose recursive lineage
contains absolute file paths. Its original notebook cloned v3 to:

```text
/content/Embedding_Compression_for_RAG_Retrieval_rars_v3_oracle
```

The initial v5 notebook used:

```text
/content/Embedding_Compression_for_RAG_Retrieval_rars_v3
```

Although both clones checked out the same commit, the changed absolute path
altered the rebuilt lineage hash. The audit gate correctly refused to release
the audit labels.

## Repair

The v5 notebook now restores the exact historical v3 clone path. No hash check
is disabled, no manifest is edited, and no output is copied around the gate. A
notebook contract test prevents the path from drifting again.

The repaired notebook must be rerun from a fresh Colab runtime so all local
v2.2/v3 bundle artifacts are rebuilt under the registered paths. The expected
behavior is that the original v3 design freeze verifies recursively before the
audit labels are materialized.
