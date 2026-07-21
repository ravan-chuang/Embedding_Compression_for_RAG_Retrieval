# RARS-v7 Frozen-Index Query Adapter Pilot

## Decision being tested

V6 established that the frozen 1M M32x8 index has measurable, distributed
PQ-specific headroom: Base-PQ Recall@100 was `0.84731`, same-route FP32
Recall@100 was `0.89395`, and 4,413 PQ-induced flip triplets spanned 189
queries. That is a signal to test learnability, not evidence that an adapter
will work.

V7 asks whether a **query-only 48 KiB adapter** can recover part of that gap
while the million-document index remains byte-identical. The machine-readable
contract is
[`protocols/rars_v7_query_adapter_pilot_v1.json`](../protocols/rars_v7_query_adapter_pilot_v1.json).
It is frozen before the first v7 training or selection metric.

## Deployment constraint

The document encoder, cached document embeddings, IVF centroids, inverted-list
membership, PQ codebooks, and all PQ codes are immutable. The original query
chooses the same 16 IVF lists as the deployed baseline. Only scoring inside
those lists uses:

```text
q' = normalize(q + 0.1 U(Vq))
score(q, d) = q'ᵀ reconstruct_PQ(d)
```

`V` is `16 x 384`, `U` is `384 x 16`, and `U` begins at zero. The 12,288
FP32 parameters occupy 49,152 bytes. Epoch zero must reproduce the verified
V6 Base-PQ and same-route FP32 Recall arrays exactly before training can
continue.

This is deliberately different from document-side QAT: applying the method
does not re-encode the corpus or rebuild the index. It is also different from
RARS: no document sidecar is fitted in this stage.

## Training evidence and separation

Only the already outcome-informed 2,307-query `oracle_design` identity is
used. Query IDs are ordered by a salted SHA-256 digest before labels are read,
producing an exact, query-disjoint split:

| Role | Queries | Use |
|---|---:|---|
| Training | 1,845 | pair mining and gradient updates |
| Selection | 462 | epoch metrics, guardrails, checkpoint selection, final pilot gate |

`oracle_audit` and the 803-query `future_method_holdout` are not opened. The
future role remains identity-only. Because the source design role has already
been inspected in prior research, even a V7 GO remains development evidence,
not independent confirmation.

## Top-k-aware objective

Two pair types are mined from a fixed union of Base-PQ Top-200, same-route
FP32 Top-200, and routed explicit positives:

1. **Top-100 promotion:** an explicit positive is above an unjudged boundary
   challenger under FP32 scoring but at or below it under fixed PQ scoring.
2. **Top-10 protection:** an explicit positive already in Base-PQ Top-10 is
   compared with an unjudged Base-PQ rank 11--26 challenger, provided the FP32
   teacher prefers the positive.

The pairwise loss simultaneously promotes the positive and suppresses the
challenger relative to that positive. MS MARCO's absent judgment is not an
explicit negative label, so the implementation consistently calls it an
**unjudged challenger**.

```text
L = L_promote100
  + 2.0 L_protect10
  + 0.5 L_margin_distillation
  + 0.05 L_query_drift
```

Weights are normalized so that, within each pair type, every represented
query contributes the same total weight. This prevents the 4,413 observed
pairs from being mistaken for 4,413 independent queries.

## Selection and early stopping

Training uses one frozen seed (`42`) for at most 12 epochs. Epoch zero is an
eligible identity checkpoint. An epoch is eligible only if:

- hard-PQ Recall@10 drops by at most `0.0025`; and
- adapted same-route FP32 Recall@100 drops by at most `0.0025`.

Eligible checkpoints are ordered by hard-PQ Recall@100, hard-PQ Recall@10,
adapted same-route FP32 Recall@100, query cosine, then earlier epoch. Training
loss never selects a checkpoint. Training stops after three consecutive
epochs without a new eligible best once at least three epochs have run.

## Frozen pilot gate

All conditions must pass:

1. a non-identity epoch is selected;
2. hard-PQ Recall@100 improves by at least `+0.005`;
3. the paired-bootstrap 95% lower bound is above zero;
4. at least 15% of the selection split's same-route FP32/PQ gap is recovered;
5. at least five selection queries improve and the net improved count is at
   least three;
6. the Recall@10 and same-route FP32 guardrails remain satisfied; and
7. mean adapted/original query cosine is at least `0.995`.

A pass yields `GO_TO_V7_DEVELOPMENT_AUDIT`. It authorizes only a separately
frozen audit of the exact checkpoint. A failure yields
`STOP_V7_QUERY_ADAPTER_PILOT`. Neither result authorizes opening the future
holdout or combining the adapter with RARS.

## Required audit trail

Before V7 starts,
[`scripts/verify_rars_v6_1m_headroom_packet.py`](../scripts/verify_rars_v6_1m_headroom_packet.py)
must verify both V6 JSON files and every registered array byte/hash, recompute
the reported means and Recall@100 decomposition, and reproduce all six V6
gates. V7 records the exact V6 result hash, split identity, source blobs,
environment, pair support, per-epoch metrics, selected arrays, adapter bytes,
and before/after frozen-index hashes.

## What a positive result would not prove

This pilot does not measure production QPS, official MS MARCO metrics,
cross-dataset generalization, comparison with OPQ/ScaNN/JPQ, three-seed
stability, or adapter-plus-RARS complementarity. Those are later experiments
only after a guarded standalone adapter result survives a separate audit.
