# RARS-v15 Cross-Fitted Selective RPQ Gate Protocol

## Decision being tested

V11--V14 isolate a consistent pattern: rank-64 uniform residual PQ provides
real capacity at `16 B/doc`, while learned centroid, signed-score, whitening,
and anisotropic-rate changes do not stably improve it. V14 also shows that an
always-on residual sidecar still harms a small subset of queries.

V15 therefore freezes the document representation instead of modifying it. It
tests one query-time decision:

> Should the system return the always-on uniform-RPQ ranking for this query, or
> fall back to the untouched Base ranking?

The primary endpoint is out-of-fold Recall@10 versus the identical always-on
uniform RPQ sidecar. The machine-readable contract is
[`protocols/rars_v15_cross_fitted_selective_rpq_gate_v1.json`](../protocols/rars_v15_cross_fitted_selective_rpq_gate_v1.json).

## Evidence boundary

This is outcome-informed development on the already opened 5,000-query V13
role. It is not fresh evidence or confirmation. It may only decide whether a
separate disjoint-query protocol should be written.

The following remain immutable:

- the 1M-document M32 IVF-PQ index and all original PQ structures;
- the rank-64 PCA basis and uniform RPQ16x8 representation recipe;
- `alpha=0.75`, `Top-B=40`, candidate pool 100, and final cutoff 10;
- the exact 16-byte document payload;
- all features, weights, folds, seeds, calibration rules, thresholds, and
  statistical gates.

V15 adds zero bytes per document. Its only new artifact is a global linear
gate of at most 4 KiB.

## Cross-fitted method

For outer fold `f`:

1. Fit rank-64 PCA and uniform RPQ16x8 without relevance labels on the four
   non-held-out folds, exactly reproducing the V13 uniform comparator.
2. Compute twelve label-free features from Base and uniform-sidecar scores.
3. Use fold `(f + 1) mod 5` only to calibrate the gate threshold.
4. Fit a weighted ridge utility model on the remaining three folds. Its target
   is per-query `Recall@10(uniform RPQ) - Recall@10(Base)`.
5. Select among the frozen 21 score quantiles on the calibration fold. A
   candidate must have coverage in `[0.20, 0.95]`, positive Recall improvement
   over always-on RPQ, and nonnegative MRR/nDCG changes. If none qualifies, the
   fold uses the always-on sidecar.
6. Use the outer held-out labels only after the model and threshold are fixed.

The complete ranking is selected per query. Candidate-level mixing is not
allowed: the output is either the untouched Base ranking or the complete
uniform-RPQ ranking.

The packet also stores the cross-fit feature tensor and per-query sidecar
metrics for every fold and seed. The independent verifier uses those arrays to
refit each ridge model, repeat calibration-threshold selection, reproduce every
held-out gate mask, and rebuild the formal GO/STOP decision. These audit arrays
are research evidence only; they add no document-side deployment storage.

The regression uses one fixed configuration:

- ridge `1.0`;
- neutral-query weight `0.05`;
- harm-query weight `2.0`;
- minimum feature scale `1e-6`;
- seeds `20261001`, `20261002`, and `20261003`;
- primary seed `20261001`.

## Why this addresses the observed failures

- V12/V13 changed document codebooks but produced sparse or non-repeatable
  ranking changes. V15 does not touch document codes.
- V14 produced stable geometry but balanced `54/56` improved/harmed support.
  V15 learns the sign of query-level utility instead of another global
  reconstruction proxy.
- V7 changed query embeddings and violated its drift guardrail. V15 never
  changes the query vector used by the ANN index.
- The fallback makes every calibration failure explicit and prevents an
  automatically selected harmful threshold from being deployed.

## Formal gate

The result advances only if every frozen requirement passes, including:

- Recall@10 gain over always-on uniform RPQ at least `+0.003`;
- paired-bootstrap lower bound above zero and one-sided randomization
  `p <= 0.05`;
- at least 30 improved, at most 15 harmed, and at least 15 net-improved queries;
- all three seed gains nonnegative, median seed gain at least `+0.002`, and no
  negative primary-seed fold;
- MRR and nDCG changes no worse than `-0.001`;
- primary coverage between 20% and 95%;
- exactly the existing 16-byte parent payload, zero additional document bytes,
  and at most 4 KiB of global gate parameters.

The only decisions are:

```text
GO_TO_FRESH_SELECTIVE_RPQ_GATE_PROTOCOL
STOP_V15_NO_SELECTIVE_GATE_SIGNAL
```

Even GO authorizes only writing a fresh-query protocol. It is not method
confirmation.
