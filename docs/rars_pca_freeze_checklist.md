# RARS vs PCA Freeze Checklist

## Implementation
- [ ] PCA basis uses ordinary unweighted residual SVD only.
- [ ] PCA and RARS share rank, int8 quantizer, candidate pool, correction code, and storage accounting.
- [ ] PCA fitting reads no qrels or held-out data.
- [ ] RARS method is unchanged.
- [ ] Tests cover deterministic PCA fitting and artifact loading.

## Validation freeze
- [ ] Alpha and Top-B search spaces match the protocol.
- [ ] Cost-aware selection rule is implemented exactly.
- [ ] PCA config is selected using validation only.
- [ ] RARS config is selected using validation only.
- [ ] Validation table, configs, hashes, and evaluator are committed.

## External dataset freeze
- [ ] Query IDs, sources, qrels, embeddings, and checksums are committed.
- [ ] Prior query-ID and normalized-text overlap audits are committed.
- [ ] Primary metric is Recall@10.
- [ ] Primary contrast is RARS minus PCA.
- [ ] Bootstrap uses 20,000 resamples and seed 20260712.

## Evaluation
- [ ] Base, PCA, and RARS run together once.
- [ ] M48 and oracle are contextual only.
- [ ] Per-query metrics and paired differences are saved.
- [ ] No post-result tuning occurs.

## Reporting
- [ ] Historical held-out and sensitivity results remain labeled developmental.
- [ ] External confirmation is separate.
- [ ] Negative or non-significant RARS-minus-PCA results are reported.
- [ ] Representation and full artifact storage are both reported.
