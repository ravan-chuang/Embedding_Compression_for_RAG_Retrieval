# RARS-v5 PQ-aware 100K pilot closure packet

This directory closes `rars_v5_pq_aware_100k_pilot_v1`. The formal decision is
`STOP_PQ_AWARE_100K_PILOT`: the rank-8 query/document adapters produce a small,
sparse Recall@100 gain, but five of the six preregistered gates fail. Seeds 43
and 44, a RARS combination, a 1M rebuild, external evaluation, and the
803-query future role are not authorized by this result.

## Outcome

- Selection queries: `728`
- Training pairs: `8,244`
- PQ-induced flip pairs: `24` (`0.291%`)
- Base hard-IVF-PQ known-positive Recall@10: `0.942536630037`
- Selected adapter hard-IVF-PQ Recall@10: `0.940476190476`
- Recall@10 change: `-0.002060439560`
- Base hard-IVF-PQ known-positive Recall@100: `0.975274725275`
- Selected adapter hard-IVF-PQ Recall@100: `0.978021978022`
- Recall@100 gain: `+0.002747252747`
- Paired-bootstrap 95% interval: `[0, +0.006868131868]`
- Exact-teacher gap recovery: `11.111%`
- Improved / harmed / unchanged queries: `2 / 0 / 726`
- Adapted FP32 Recall@100: `1.0`
- Selected epoch: `3`

The two improved queries each move from per-query Recall@100 `0` to `1`. The
base-to-exact gap is 18 query-equivalents; the adapter recovers two and leaves
16. The registered minimum was a `+0.005` mean gain, a strictly positive
bootstrap lower bound, 15% gap recovery, at least 20 improved queries, and no
more than `0.002` Recall@10 loss. Only the FP32 guardrail passes.

## Training diagnosis

Training loss falls from `0.145817` at epoch 1 to `0.091059` at epoch 8, a
`37.6%` reduction. Retrieval quality does not track that optimization:
Recall@10 peaks at epoch 1, Recall@100 peaks at epoch 3, and both finish below
the identity baseline at epoch 8. This is evidence that the frozen surrogate
is poorly aligned with hard-PQ ranking outcomes in this near-ceiling setting,
not evidence that the optimizer failed to reduce its objective.

No alternate post-hoc checkpoint rescues the primary decision. Epochs 1 and 2
have better Recall@10 than epoch 3 but fail the registered Recall@100 magnitude,
uncertainty, recovery, and support requirements.

## Evidence boundary

This is an outcome-informed 100K development pilot. Relevance denominators
contain only positives previously observed in the v3 design/audit Top-100
roles. The 728-query selection role is previously observed diagnostic-audit
data, not an untouched test set. Zero-labelled candidates are unjudged mined
negatives, not explicit non-relevant judgments. These numbers are not official
MS MARCO Recall and do not establish performance on the frozen 1M deployment
index or an external collection.

## Contents

- `training_started.json`, `training_history.json`, `pilot_result.json`, and
  `training_complete.json`: exact exported JSON artifacts.
- `per_query_recall_at_100.json`: repository-local base, adapter, and delta
  vectors in frozen selection order, derived from the two registered NPY files.
- `artifact_audit.json`: full-export byte/hash, shape, dtype, finiteness,
  adapter-norm, Top-100-range, and uniqueness checks.
- `executed_notebook/`: the executed Colab notebook. Its Markdown/code sources
  match the clean repaired notebook; all eight code cells ran once in order
  without an error output.
- `closure_manifest.json`: repository-local hashes for every immutable packet
  file except the manifest itself.

This is intentionally a thin packet. The eight binary adapter, Top-100, and
per-query NPY payloads remain in the durable Drive export and are registered by
exact byte count and SHA-256 in `training_complete.json`. The full export was
audited before this packet was written. The two Recall@100 vectors needed to
recompute the primary gain and support are preserved as JSON here.

## Verification

From the repository root:

```bash
python scripts/verify_rars_v5_pq_aware_100k_packet.py
```

The verifier checks repository-local hashes, run/source identity, JSON
registrations, epoch selection, per-query Recall@100 reconstruction, all six
formal gates, external binary registrations, and executed-notebook source
parity. Any verification failure invalidates the local packet; it is not a new
scientific outcome.
