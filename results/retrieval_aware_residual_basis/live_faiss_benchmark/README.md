# Live Faiss + RARS Benchmark

Measures live Faiss IVF-PQ search and RARS correction on the aligned
MS MARCO 1M artifacts.

Two correction implementations are compared:

- `loop`: per-query Python loop
- `vectorized`: batched NumPy projection and `einsum`

`estimated_combined_*` adds separately measured search and correction samples.
`end_to_end_*` measures search and correction inside the same invocation.

For each method, baseline and method end-to-end runs are measured in alternating
order. `paired_end_to_end_overhead_pct` is the preferred overhead statistic.
`vs_separate_search_mean_pct` is retained only as a diagnostic and should not be
used as the main reported overhead.

The benchmark also verifies that loop and vectorized correction produce
numerically equivalent score matrices.

The benchmark excludes query encoding, HTTP, response serialization,
and document lookup.
