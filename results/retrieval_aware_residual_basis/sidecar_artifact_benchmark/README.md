# RARS Sidecar Artifact Benchmark

Artifact-backed evaluation on the held-out MS MARCO 1M split.

Reported quality metrics use `qrels_subset.json`. Candidate exact-Top10 overlap
and score MSE use cached exact scores inside the same Top-100 ANN candidate pool.

Latency is sidecar-only and excludes query encoding, Faiss retrieval, HTTP,
serialization, and document lookup.
