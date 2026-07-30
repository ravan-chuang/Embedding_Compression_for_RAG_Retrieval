# RARS-v2.2 FP32 replication closure packet

This directory is the committed closure packet for
`rars_v2_2_fp32_replication_v1`. The formal decision is
`UNSTABLE_NO_QAT`: the registered mean-effect and confidence-interval gates
pass, but the positive-query-support condition does not.

## Outcome

- Base Recall@10: `0.693326790972`
- Direct PCA FP32 Recall@10: `0.706738632646`
- Held-out optimizer seeds 43/44 mean Recall@10: `0.714425907753`
- Held-out mean gain over Base: `+0.021099116781`
- Held-out mean gain over direct PCA: `+0.007687275106`
- Held-out paired-query 95% CI versus Base:
  `[+0.009322865554, +0.033366045142]`
- Held-out paired-query 95% CI versus direct PCA:
  `[+0.001472031403, +0.014556754989]`
- Seed 43 positive support versus PCA: `12 / 1,019` queries
- Seed 44 positive support versus PCA: `10 / 1,019` queries
- Registered support requirement: `11` queries per held-out seed and contrast

Seed 44 is one positive-support query below the frozen requirement. QAT is
therefore not authorized. No seed may be added, no threshold may be changed,
and this packet must not be reclassified after the fact.

## Evidence boundary

The same 1,019 inner-validation queries select the best epoch inside every
run. “Held-out” refers only to previously unseen optimizer seeds 43 and 44.
This is development-only optimizer-seed evidence, not held-out-query,
test-set, external, or confirmatory evidence. The paired bootstrap resamples
queries; it does not estimate uncertainty across seeds or datasets.

## Contents

- `aggregate-00a0dee30767/`: all 19 registered aggregate outputs plus the
  `replication_complete.json` marker.
- `seeds/`: the eight trainer outputs and completion marker for seeds 42, 43,
  and 44.
- `provenance/`: environment, dependency freeze, input audit, batch record,
  runner manifest, and logs.
- `executed_notebook/`: the executed Colab notebook. Its Markdown/code sources
  exactly match the committed clean notebook; only outputs and execution
  metadata differ.
- `closure_manifest.json`: repository-local byte counts and SHA-256 hashes for
  every immutable evidence file in this packet.

## Verification

From the repository root:

```bash
python scripts/verify_rars_v2_2_replication_packet.py
```

The verifier checks aggregate hashes, runner hashes, all three seed packets,
NPY shapes/dtypes, CSV row counts, the formal decision, and executed-notebook
source parity. Treat any verification failure as an invalid local packet, not
as a new scientific result.
