# RARS Clean Query-Split Protocol

## Objective

Replace same-query adaptation with a strict train / validation / test protocol.

## Primary split

MS MARCO dev queries: 6,980 total.

- Train: 4,980 queries
- Validation: 1,000 queries
- Test: 1,000 queries
- Fixed seed: 20260712

Commit these files:

```text
splits/msmarco_rars_train_qids.json
splits/msmarco_rars_validation_qids.json
splits/msmarco_rars_test_qids.json
splits/msmarco_rars_split_manifest.json
```

## Data-use contract

### Train only

Use train queries to:

- generate ANN candidates;
- compute exact-minus-ANN score errors;
- fit RARS-Score / RARS-Boundary bases;
- train any learned router.

### Validation only

Use validation queries to choose:

- basis variant;
- alpha;
- Top-B;
- rank, if not fixed;
- router thresholds or model configuration.

### Test only

Run test after all code, artifacts, hashes, and hyperparameters are frozen.
Do not sweep or retune after observing test metrics.

## Fair comparison

PCA and RARS must use the same:

- frozen IVF-PQ base index;
- candidate pool;
- rank;
- int8 coefficient format;
- document order;
- final cutoff;
- test queries.

Report:

- Frozen IVF-PQ M=32
- PCA sidecar
- RARS sidecar
- IVF-PQ M=48 rebuild baseline
- exact Top-100 candidate oracle

M=48 must be described as a rebuild baseline, not a retrofit baseline.

## Freeze requirement

Before test execution, commit:

```text
splits/
configs/rars_clean_split_selected.json
artifacts/<clean-split-artifact>/
scripts/<training-and-evaluation-code>
docs/rars_query_split_protocol.md
```

Suggested commit:

```text
experiment: freeze RARS clean split protocol and selected configuration
```

## Final test metrics

On untouched test queries, report:

- Recall@10
- Success@10
- MRR@10
- nDCG@10
- paired-bootstrap 95% CI
- bootstrap fraction with gain <= 0

Use at least 20,000 resamples and a fixed seed.

## Claim rules

Allowed:

- “RARS improves the frozen base on unseen queries” when the point estimate is positive.
- “RARS significantly improves…” only when the paired CI excludes zero.
- “RARS has a positive point estimate over PCA…” when the CI crosses zero.

Not allowed:

- calling validation results held-out test results;
- calling per-setting retraining zero-shot transfer;
- claiming global storage optimality;
- retuning after test evaluation.

## FiQA

Use 5-fold outer cross-validation. For each outer fold:

1. fit the basis using only non-test queries;
2. choose alpha and Top-B using an inner validation split;
3. evaluate once on the outer test fold;
4. concatenate out-of-fold predictions;
5. bootstrap the concatenated per-query metrics.

MS MARCO clean-split results remain the primary paper evidence.

## Audit checklist

- [ ] Every query ID appears exactly once across train/validation/test.
- [ ] RARS fitting reads train IDs only.
- [ ] Alpha and Top-B selection read validation IDs only.
- [ ] Test script performs no sweep.
- [ ] Artifact hashes are recorded before test execution.
- [ ] Freeze commit exists before test metrics are produced.
- [ ] README no longer calls same-query-adapted results untouched held-out results.
