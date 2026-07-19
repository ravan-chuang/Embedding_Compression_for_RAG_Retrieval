# RARS-v4 tri-state post-PQ feasibility protocol

## Purpose

RARS-v4 Phase-0 tests a necessary condition before another loss is trained:
does the available supervision and the already-frozen post-PQ action space
support all three intended operations?

1. **Protection:** retain judged-relevant documents in the current Top-10.
2. **Promotion:** move judged-relevant Top-100 candidates into the Top-10.
3. **Penalty:** move an explicitly judged non-relevant Top-10 document out.

This is not a trainer and is not a claim that the three operations constitute a
new loss.  V2.2 already used positive/label-0 boundary pairs.  The only
scientifically defensible new direction would be an explicit-negative mask plus
compression-consistent supervision, followed by matched ablations.

The machine-readable contract is
`protocols/rars_v4_tristate_action_feasibility_v1.json`.

## Why the label schema is the first gate

The historical MS MARCO loader retained only grades greater than zero.  It then
wrote every other Top-100 candidate as binary zero.  Consequently, the old
`candidate_relevance.uint8.npy` cannot distinguish these two cases:

- a source judgment row exists and says the document is non-relevant;
- no source judgment row exists.

RARS-v4 therefore rematerializes labels from the source judgment JSON and uses
an `int8` state:

| Value | Meaning |
|---:|---|
| `+1` | source row exists and grade is positive |
| `-1` | source row exists and grade is non-positive |
| `0` | no source row exists; unjudged |

Positive-only list entries, binary candidate labels, and missing rows are never
accepted as primary negatives.  An unjudged-as-negative calculation may be
reported later as sensitivity only; it cannot unlock training.

For the currently cached positive-only MS MARCO qrels, the expected correct
Phase-0 result is `STOP_NO_EXPLICIT_NEGATIVE_SEMANTICS`.  That is useful: it
shows that adding a nominal negative penalty would reuse the same pseudo-negative
signal as v2.2 rather than add verified supervision.

## Evidence status and roles

This protocol was motivated after the v2.2 and v3 outcomes were observed.  It
is an outcome-informed development diagnostic, not an independent
preregistration.

- `v4_design_observed`: the 2,307-query v3 design role.
- `v4_diagnostic_audit`: the 851-query v3 audit role, opened only if the new
  design gate passes.  Its v3 positive-label outcomes are already known, so it
  is only a diagnostic replication.
- `future_fp32_holdout`: the 803-query v3 future role.  Phase-0 must not create
  its candidate arrays, labels, support counts, or metrics.

All MS MARCO roles are historical development data because the parent v2 work
used the full inner-train pool.  No external or untouched-confirmation claim is
permitted.

## Frozen score actions

The v3 representation is reused without fitting or tuning:

- one design-frozen uncentered rank-32 residual SVD basis;
- one global int8 scale per component;
- prefix actions of 0, 8, 16, or 32 accessed bytes;
- base ranks 1–40 are variable, ranks 41–100 remain at the base score;
- total progressive access is at most 640 bytes per query;
- ranking ties are score descending, document ID ascending.

A second optimistic diagnostic permits either the Base or exact residual score
for each of ranks 1–40.  Its cost is an action count, not bytes, and it is not a
deployable representation.

Both exact solvers optimize lexicographically:

1. maximize judged-positive Top-10 count;
2. among equal positive counts, minimize explicit-negative Top-10 count;
3. minimize action cost;
4. use deterministic threshold and document-ID traversal.

Thus a lower explicit-negative count can never compensate for lost Recall.

## Label-support measurements

Every eligible query remains in every coverage denominator, including queries
with no explicit negative or no possible swap.  The design and audit summaries
report:

- explicit-negative coverage in Top-100;
- explicit-negative coverage in Base Top-10;
- relevant-outside/negative-inside promotion-pair coverage;
- relevant-inside/negative-outside protection-pair coverage;
- queries where a unary penalty would have an explicit negative and a judged
  relevant replacement;
- label-only Recall ceiling
  `min(actionable outside positives, Top-10 explicit negatives) / total relevant`;
- judged and unjudged counts in Top-10, ranks 11–40, and ranks 41–100;
- Wilson intervals, paired bootstrap interval, effective support, and
  concentration of positive mass.

The registered design floors include 25% Top-100 explicit-negative coverage,
10% Top-10 penalty coverage, 5% promotion and protection pair coverage, at
least 1 point mean label-only Recall headroom, at least 5% positive-query
support, effective support of 30, and no more than 40% of positive mass in the
largest 1% of queries.

## Action-space measurements

Only after the label/schema gates pass does the evaluator build score tensors.
It reports downward, upward, and joint swap reachability, then solves both the
selective-exact and progressive action oracles.  The progressive gate requires:

- at least +0.010 mean Recall over the frozen v3 matched-access comparator;
- paired-bootstrap lower bound greater than +0.005;
- broad positive support;
- at least 35% recovery of the label-only ceiling;
- at least 60% of positive gain mass attributable to an explicit-negative-out
  and judged-positive-in event;
- at least 5% joint swap reachability;
- no query above 640 accessed bytes;
- positive direction in at least four of five deterministic folds, with the
  worst fold no lower than -0.002.

The action oracle is stronger than a deployable global trainer.  Passing it is
necessary but not sufficient.

## Chronology

1. Recreate and verify the exact frozen v2.2 parent bundle.
2. Recreate the v3 qrels-free design/audit candidate bundles; the 803-query
   future role remains identity-only.
3. Recursively verify the completed v3 output, including all registered hashes,
   its design freeze, run fingerprint, selected comparator, and disclosed
   metrics.
4. Materialize only the v4 design tri-state labels.
5. Run the design label gate.  Expensive action oracles are skipped if the
   label schema or support fails.
6. Write a durable design freeze.
7. Only a design GO allows a second process to materialize the already-observed
   diagnostic-audit tri-state labels and run the one-shot audit.

Partial output is never reused.  Complete reuse requires the same source commit,
fingerprint, and recursively verified output hashes.

## Formal interpretation

- `STOP_NO_EXPLICIT_NEGATIVE_SEMANTICS`: the source cannot distinguish explicit
  negatives from unjudged candidates.  Do not build the proposed loss from this
  cache.
- `STOP_NEGATIVE_SUPPORT`: correct semantics exist, but too few candidates or
  queries carry explicit-negative judgments.
- `STOP_NO_NOVEL_ACTION_SUPPORT`: the new unary term has too little independent
  actionable support beyond v2.2.
- `STOP_SWAP_SUPPORT`: label-supported Recall headroom is too small, sparse, or
  concentrated.
- `STOP_NO_COMPRESSION_CONSISTENT_HEADROOM`: labels support swaps, but the frozen
  post-PQ actions do not recover them broadly enough.
- `DESIGN_GO_TO_DIAGNOSTIC_AUDIT`: only the already-observed 851-query diagnostic
  role may be opened.
- `GO_FREEZE_FP32_DEVELOPMENT_PROTOCOL`: only a separate, hash-frozen FP32 study
  with matched baselines and ablations may be designed.
- `INVALID_LINEAGE_OR_SOLVER`: numerical outputs must not be interpreted.

No Phase-0 outcome authorizes training, QAT, future-holdout access, external
evaluation, storage claims, deployment claims, or a SIGIR success claim.
