# RARS-v6 Pre-Metric Faiss Downcast Repair

The first attempted execution of the frozen RARS-v6 1M headroom diagnostic
stopped during immutable-index validation, before any search or v6 metric was
computed. Faiss 1.12 returned a generic `IndexIVF` Python wrapper from
`extract_index_ivf()`. The underlying C++ object remained an `IndexIVFPQ`, but
the generic wrapper did not expose `.pq`; the validator consequently observed
`subquantizers=-1` and rejected the registered M32 index.

The repair applies `faiss.downcast_index()` to the extracted IVF object before
reading `pq.M` and `pq.nbits`. A regression test reproduces the generic-wrapper
case and requires M32/8-bit validation after downcasting.

This repair does not change the corpus, queries, qrels, index bytes, retrieval
configuration, metric definitions, flip miner, thresholds, formal decision, or
data-access boundary. No v6 outcome was available when it was made. The failed
attempt therefore remains a pre-metric implementation failure rather than a
scientific result.
