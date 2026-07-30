# RARS-v16 termination note

`rars_v16_causal_generalization_diagnostic_v1` is stopped before any retrieval
metric, basis fit, or sidecar evaluation. Preparation exposed two design
problems: both corpora are below one million documents, and SciFact contains
only 5,183 corpus vectors, which is below Faiss's recommended training support
for 256-centroid PQ subquantizers.

The prepared FiQA/SciFact files are retained only as engineering evidence.
They must not be promoted to the primary generalization experiment. V17
replaces the experiment with MS MARCO 1M and full BEIR NQ (2,681,468
documents). V16 ended before a retrieval metric, basis fit, model comparison,
or decision-bearing statistic was produced, so no V16 outcome is being hidden
or overwritten.
