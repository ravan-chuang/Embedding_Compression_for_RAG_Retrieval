# RARS-v6 Pre-Metric Torch Row-Gather Repair

The first execution after the Faiss-wrapper repair passed immutable-index
validation and completed full-exact scoring, but stopped during same-IVF exact
Top-k construction before metrics or a formal gate were computed.

The block Top-k helper correctly produced candidate-column positions with
shape `[batch, k]`, but applied them as `rows[positions]`. Torch interpreted
those values as dimension-0 indices even though `rows` had shape `[1, n]`,
triggering a CUDA device-side out-of-bounds assertion. The repair uses
`torch.gather(rows, 1, positions)`, matching the already-correct merge path.
A CPU-only regression double now checks both the first block and a subsequent
merge. Read-only memmap query slices are also copied before Torch conversion to
remove a warning without changing their values.

This repair does not change inputs, search spaces, scores, metrics, flip
definitions, thresholds, decisions, or data access. No v6 metric or gate was
available when it was made. Because a CUDA device-side assertion corrupts the
active CUDA context, the repaired run requires a fresh Colab runtime and a new
commit-specific durable output directory.
