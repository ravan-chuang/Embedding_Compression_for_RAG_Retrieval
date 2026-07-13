# External confirmation dataset preparation

## Critical corpus-compatibility rule

TREC Deep Learning passage qrels are defined against the corresponding MS MARCO
passage corpus. This repository's current frozen RARS index contains a
deterministic 1,000,000-passage subset rather than the full passage collection.

Before freezing TREC DL 2019 or 2020 as an external confirmation set, audit
every positive qrel against the index's committed document IDs.

The default acceptable condition is:

```text
positive_qrels_missing_from_indexed_corpus = 0
```

If this fails, the strongest option is to rebuild Base, PCA, and RARS-compatible
artifacts over the full compatible corpus. A corpus-restricted qrels analysis is
allowed only when it is explicitly preregistered before retrieval outcomes are
observed and is reported as subset-conditional external evidence.

## Preparation order

1. obtain topics and qrels without running retrieval;
2. prepare prior query manifests containing IDs and query text;
3. run `prepare_external_confirmation_set.py`;
4. inspect only overlap and corpus-coverage audits;
5. choose and document the corpus policy;
6. change manifest status from `DRAFT_READY_FOR_REVIEW` to
   `frozen_before_qrels_evaluation`;
7. commit the manifest and evaluator hashes;
8. merge the freeze PR;
9. run the one-shot evaluator exactly once.

## Non-outcome audits

The following are permitted before freeze:

- file integrity;
- query count and duplicate checks;
- query-ID overlap;
- normalized-query-text overlap;
- qrels availability;
- positive-qrel coverage in the indexed corpus.

Base, PCA, and RARS retrieval metrics are prohibited before freeze.
