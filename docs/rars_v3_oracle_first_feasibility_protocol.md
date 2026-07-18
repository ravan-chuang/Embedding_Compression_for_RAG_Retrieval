# RARS-v3 Oracle-First Feasibility Protocol

## Decision this experiment makes

This experiment asks whether a **non-deployable query-label oracle** can spend
0/8/16/32 bytes across the frozen Top-40 candidates and recover broad,
compression-consistent Recall@10 over the best design-frozen 640-byte
matched-access comparator. The eligible set includes a strong fixed 16-byte
sidecar and four registered alternatives.

It does not train RARS-v3. A pass authorizes only a new static-storage oracle
protocol. It does not authorize a learned allocator, QAT, external evaluation,
deployment, or a storage-compression claim.

The machine-readable contract is
`protocols/rars_v3_oracle_first_feasibility_v1.json`. This document and that
JSON are frozen before the first oracle result is inspected.

## Data boundary

The v2.2 1,019-query `inner_validation` role is closed and forbidden. V3 uses
only the old 3,961-query `inner_train` development role, then applies:

```text
u64_be(SHA256("rars_v3_split_v1\0" || qid)[0:8]) mod 10
```

- buckets 0--5: `oracle_design`, 2,307 queries;
- buckets 6--7: `oracle_audit`, 851 queries; and
- buckets 8--9: `future_method_holdout`, 803 queries.

The future role is identity-audited only. Its candidates, labels, and metrics
must not be read in this gate. The 851-query audit role is withheld from v3
design, but it is still historical v2 development data and is not independent
confirmation.

The design role has five fixed diagnostic folds from the analogous
`rars_v3_fold_v1\0` hash. V3 never reruns retrieval and never reopens the shared
qrels JSON. The candidate builder verifies both exact parent manifests and
every qrels-free candidate payload, but deliberately defers reading label
payload bytes. It then deterministically subsets the exact frozen Top-100
arrays. It is label-free and cannot create an audit label artifact.

Because the original payload was ephemeral, the exact frozen v2.2 builder may
first re-materialize it; that inherited prerequisite necessarily reopens the
historical 6,980-query qrels JSON and must reproduce the preregistered parent
hashes byte for byte. V3 builder and evaluator code do not open qrels. They
reuse only the parent's hash-registered `candidate_relevance` and
`relevant_counts` arrays. A separate V3 process materializes design rows first.
Only after the basis, scales, singular-value diagnostics, all eligible baseline
per-query vectors, selected comparator, solver preflight, environment, source
hashes, budgets, and decision thresholds have been written and verified in
`design_freeze.json` may a second process materialize audit rows. The parent
label source is procedurally, not cryptographically, sealed; this is a
development audit, not independent confirmation. Future-role identity rows are
written only for split auditing; future candidate and label payload rows are
never selected or written by V3.

The enforced lifecycle is therefore:

1. freeze qrels- and label-free design/audit candidates plus the future identity;
2. materialize design labels and run the design phase;
3. verify the durable design freeze, then materialize audit labels; and
4. run the audit phase once and emit a complete or formally invalid artifact.

## Frozen retrieval and score references

- Corpus: MS MARCO passage 1M development corpus.
- Embeddings: 384 dimensions.
- Base: frozen IVF-PQ M32, nlist 512, nprobe 16 candidate arrays inherited
  exactly from v2.2.
- Candidate set: deterministic subsets of the hash-registered v2.2 base
  Top-100; V3 retrieval and candidate reordering are forbidden.
- Correctable candidates: frozen base ranks 1--40.
- Final cutoff: Top-10.
- Tie-break: score descending, document ID ascending.

`Exact40` adds the exact FP32 query-residual dot product only at base ranks
1--40. `Exact100` does so throughout the frozen Top-100. These are
compression-consistent score-rescore references, not qrels-optimal ceilings;
removing quantization can occasionally lower measured Recall.

`QrelCandidate100` packs judged-relevant candidates first and is reported only
as a candidate-coverage ceiling.

## One progressive residual representation

A single label-free, **uncentered second-moment truncated-SVD** rank-32 residual
basis is fitted on design candidate residuals. Its column signs are
canonicalized. One max-absolute int8 scale per coefficient is fitted on the
same design residuals. The singular values and adjacent gaps are frozen so a
near-degenerate component ordering is visible rather than silently treated as
portable PCA.

For tier `r` in `{0, 8, 16, 32}`:

\[
s_r(q,d)=s_{ANN}(q,d)+0.75\sum_{j<r}
(q^\top u_j)\,\mathrm{scale}_j\,\mathrm{code}_{dj}.
\]

The nonzero tiers are prefixes of the same ordered rank-32 code. Separate
bases, scales, or codebooks per tier are forbidden. Tier zero performs no
sidecar read and adds exactly zero correction.

## Exact matched-access oracle

For every query, the oracle assigns one tier to each frozen Top-40 candidate:

\[
r_{qd}\in\{0,8,16,32\},\qquad
\sum_{d\in C_q^{40}} r_{qd}\le 640.
\]

Candidates 41--100 remain at tier zero but still compete for Top-10. The
objective is the judged-relevant count in the final Top-10. Since a query has a
fixed relevance denominator, this exactly maximizes its Recall@10.

The implementation enumerates every stable `(score, document_id)` threshold
and solves a dynamic program over selected-count and 8-byte budget states. A
brute-force equivalence test on small problems is mandatory. Equal-recall
solutions use fewer bytes first and a deterministic document-ID traversal.

The complete required curve is 0, 320, 640, and 1,280 bytes/query, corresponding
to Oracle0, Oracle8, Oracle16, and Oracle32. Oracle0 must reproduce Base exactly;
Oracle32 is still a label oracle and is not interchangeable with uniform
rank-32. The oracle may read only the current phase's materialized labels and
is permanently labeled `QUERY_LABEL_ORACLE_NON_DEPLOYABLE`. It assumes that
rank-32 codes could be stored and therefore cannot support a persistent-storage
claim.

## Registered matched baselines

The 640-byte candidates are:

1. frozen PCA rank-16 int8;
2. uniform progressive rank-16 int8;
3. rank-32 for the 20 largest candidate residual norms, zero otherwise;
4. rank-32 for the 20 candidates nearest the base rank-10 boundary, zero
   otherwise; and
5. rank-32 for the 20 candidates with the highest design candidate exposure,
   zero otherwise.

The highest design Recall@10 becomes the primary comparator. Ties use fewer
actual accessed bytes and then lexical method name. Every eligible method's
per-query vector and hash, not just the winner, is frozen before audit. Audit
cannot reselect it.

Ten fixed-seed random 0/32 assignments and uniform 8/32 are secondary
diagnostics, not eligible primary comparators.

## Counterfactual recovery

Let `Comparator`, `Exact40`, and `OracleB` provide per-query recall vectors.
Primary compression-recoverable mass is:

\[
H_C=\sum_q\max(R_q(Exact40)-R_q(Comparator),0).
\]

Recovered mass is capped query-wise by that positive mass and additionally
requires the oracle Top-10 to be closer to `Exact40` than the comparator under
symmetric set distance. `CFR_C=C_C/H_C`; `H_C=0` is an immediate score-headroom
failure. The former Base-relative CFR remains a secondary diagnostic, not a
substitute for comparator-relative recovery.

Required diagnostics include relevant-drop recovery, non-relevant intrusion
removal, comparator-relevant harm, a per-document influence table, the full
gain-per-accessed-byte curve, and the fraction of comparator-relative oracle
positive mass that moves toward `Exact40`.

## Frozen audit gates

All of the following must pass on the 851-query audit role:

- `Exact40 - primary comparator >= +0.010` Recall@10;
- `Oracle16 - primary comparator >= +0.010`;
- paired 20,000-resample 95% bootstrap lower bound `> +0.005`;
- positive support at least `max(40, ceil(0.05 * 851)) = 43` queries;
- negative gain mass / positive gain mass at most `1/3`;
- top 1% of queries at most 40% of positive gain mass;
- effective positive support at least 30;
- comparator-relative `CFR8 >= 0.20` and `CFR16 >= 0.35`;
- at least 70% of comparator-relative positive oracle gain mass moves Top-10
  membership closer to `Exact40`;
- at least four of five design-fold gains are positive; and
- worst design-fold gain is at least `-0.002`.

The paired bootstrap uses NumPy PCG64 seed `20260719`; identical resample
indices must be used for all compared methods.

The run requires Python 3.12.13 and NumPy 1.26.4. It records the complete NumPy
and BLAS configuration, uses the explicit NumPy `linear` quantile method, and
does not invoke Faiss in either the v3 builder or evaluator. A fixed exhaustive
and tie-heavy brute-force equivalence preflight is saved before audit access.

## Formal outcomes

- `INVALID`: a lineage, split, hash, exact-solver, byte, artifact, or access
  contract fails.
- `KILL_NO_SCORE_HEADROOM`: `Exact40` lacks the preregistered residual headroom
  over the selected comparator or positive comparator-relative recovery mass.
- `STOP_NO_HEADROOM`: any primary access-oracle effect, uncertainty, support,
  concentration, fold, harm, or recovery gate fails.
- `GO_TO_STATIC_STORAGE_ORACLE`: every access gate passes.

The GO outcome authorizes only a separately frozen static document-allocation
oracle with actual serialized-byte accounting. Only a later design-to-audit
static allocator can determine whether a variable-rate storage method is worth
building.
