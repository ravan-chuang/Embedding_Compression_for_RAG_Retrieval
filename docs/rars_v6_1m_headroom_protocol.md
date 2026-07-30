# RARS-v6 1M PQ-Specific Headroom Protocol

## Goal

This outcome-informed development diagnostic asks one question before any new
PQ-aware loss is implemented:

> Does the exact frozen 1M M32x8 IVF-PQ index contain enough *PQ-specific*,
> query-distributed Recall@100 loss to justify another training experiment?

The machine-readable contract is
[`protocols/rars_v6_1m_headroom_v1.json`](../protocols/rars_v6_1m_headroom_v1.json).
It is frozen before the first v6 metric is observed. Passing is not a method
result. It authorizes only a separately preregistered loss-development stage.

## Why This Stage Exists

The closed v5 pilot improved only two of 728 queries. Its 24 selected flip
pairs were sparse, but that count alone cannot establish why the experiment
stopped. The v5 support gate also required 20 improved queries even though its
binary Recall@100 baseline left only 18 improvable queries. V6 therefore
measures attainable PQ-specific headroom before defining another training
objective or support gate.

This protocol does not require Base Recall@100 to fall below an arbitrary
threshold. A low baseline can be caused by IVF routing, corpus coverage, or PQ
ranking. Only the last component is the intended signal.

## Data Boundary

The diagnostic uses the already-observed v3 `oracle_design` identity and query
vectors for 2,307 queries. Positive judgments for exactly those query IDs are
selected from the historical shared MS MARCO qrels container and mapped to the
frozen 1M corpus. The evaluator may parse that shared container, but it must
not compute or emit any non-design outcome.

The evaluator must not open or score:

- v3 `oracle_audit`;
- the 803-query `future_method_holdout`;
- outer validation, clean test, or an external collection.

A document absent from positive qrels is an **unjudged mined challenger**, not
an explicit non-relevant judgment. Results are 1M development diagnostics,
not official MS MARCO metrics or independent confirmation.

## Frozen Retrieval Stack

| Item | Value |
|---|---:|
| Corpus | 1,000,000 passages |
| Embedding | normalized 384-d FP16 cache, scored in FP32 |
| Index | frozen IVF-PQ, inner product |
| IVF lists / probes | 512 / 16 |
| PQ | M32, 8 bits per subquantizer |
| Analysis depth | 200 |
| Reported cutoffs | 10 and 100 |

Neither the index nor the cached embeddings are rewritten. No adapter,
encoder, codebook, PQ code, or sidecar is fitted.

## Headroom Decomposition

For each query, the evaluator computes three retrieval results:

1. **Full exact:** FP32 inner-product search over all 1M documents.
2. **IVF exact:** FP32 scoring restricted to the exact same 16 IVF lists
   selected for that query.
3. **Base PQ:** canonical CPU search through the frozen Faiss IVF-PQ index.

The CPU coarse-quantizer probe IDs are shared by Base PQ and IVF exact. This
prevents CPU/GPU coarse-scoring differences from being misreported as PQ loss.

The diagnostic reports:

```text
routing gap     = Recall(full exact) - Recall(IVF exact)
PQ-specific gap = Recall(IVF exact) - Recall(base PQ)
total gap       = Recall(full exact) - Recall(base PQ)
```

Only the PQ-specific gap is considered recoverable evidence for the next
PQ-aware stage. A routing miss is not relabelled as quantization headroom.

## Flip-Support Diagnostic

Around the base rank-100 boundary, the evaluator forms explicit-positive versus
unjudged-challenger triplets. A PQ-induced flip requires a positive FP32
same-probe margin and a non-positive reconstructed-PQ margin. It reports both
the uncapped population and the training-like cap of four challengers per
positive.

Support is assessed at multiple grains:

- uncapped triplet count;
- distinct flip-bearing queries and positive documents;
- flip-weight effective sample size (ESS);
- maximum share of flip weight contributed by one query.

The `500`-triplet condition is an engineering gate, not a significance test.
It cannot pass without query breadth and ESS, and it cannot replace a later
query-paired confidence interval.

## Frozen Decision

All conditions must pass:

1. every selected positive qrel maps into the frozen 1M corpus;
2. PQ-specific Recall@100 headroom is at least `+0.005`;
3. at least 500 uncapped flip triplets exist;
4. at least 100 distinct queries contain a flip;
5. flip-weight ESS is at least 250;
6. no query contributes more than 2% of total flip weight.

Passing yields `GO_TO_V6_LOSS_IMPLEMENTATION`. It does not authorize training
under this protocol. Any signal failure yields
`STOP_NO_DISTRIBUTED_PQ_HEADROOM`; an incomplete resource smoke yields
`STOP_RESOURCE_SMOKE_FAILED`.

## Resource Smoke

The Colab run records stage wall time, host peak RSS, CUDA peak allocated and
reserved memory, and local free disk before and after execution. Corpus tensors
are temporary and must not be exported. The run requires at least 8 GB free in
`/content` before it starts.

The exact computation is deliberately chunked. An out-of-memory result is a
failed resource smoke, not evidence for or against the research hypothesis.

## Next Stage, Only After GO

A later protocol may implement explicit Top-10 protection, Top-100 repair,
dynamic boundary-pair refresh, per-query weight control, and constrained
checkpoint selection. RARS basis refitting and adapter-plus-RARS evaluation
remain later stages after an adapter itself succeeds. This diagnostic does not
authorize either action.
