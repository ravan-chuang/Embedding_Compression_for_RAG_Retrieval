# RARS-v11 Rank–Rate Capacity Diagnostic

## Purpose

V11 is a fixed architecture screen, not a new learned method. V10 showed that
replacing rank-16 int8 coefficients with identical-basis FP32 coefficients
improves Recall@10 by only `0.000433`. The remaining same-candidate gap is
therefore dominated by representation rank/model capacity rather than scalar
quantization precision.

V11 asks two preregistered questions:

1. Does a fixed PCA rank-64 FP32 residual sidecar expose at least `+0.005`
   Recall@10 headroom over the rank-16 int8 PCA sidecar?
2. Can a 16-byte rank-64 residual product code retain at least half of that
   headroom and improve PCA by at least `+0.003`?

Failure of the first gate closes the global linear rank-expansion route.
Failure of the second closes this 16-byte RPQ encoding. Passing both gates only
permits writing a separate cutoff-aware CA-RPQ development protocol.

## Frozen deployment boundary

The document/query embeddings and original M32 IVF-PQ index remain unchanged.
All sidecars use Base candidates, `alpha=0.75`, Top-B `40`, and final cutoff
`10`. The primary deployable candidate uses a fixed rank-64 PCA basis and 16
product subquantizers:

```text
64 projected residual dimensions
→ 16 blocks × 4 dimensions
→ one 256-entry codebook per block
→ 16 uint8 indices/document
→ exactly 16 B/document
```

At query time, 16 lookup tables are built from the projected query. Each
candidate correction is the sum of 16 table lookups. The globally shared
rank-64 basis and 16 codebooks are reported separately from per-document
payload bytes.

## Fixed screen

V11 reports every registered row; no winner is selected after execution:

- frozen M32 Base;
- PCA rank-16 int8;
- PCA rank-16, rank-32, and rank-64 FP32 ceilings;
- packed PCA rank-32 int4 (`16 B/document`);
- PCA rank-32 RPQ 16×8 (`16 B/document`);
- PCA rank-64 RPQ 16×8 (`16 B/document`, primary deployable candidate);
- same-candidate exact ceiling.

The rank-64 basis is fit once by deterministic uncentred PCA. Rank-16 and
rank-32 use its leading nested columns. RPQ is trained without qrels, cutoff
pairs, a query adapter, or basis optimization. Labels are used only to compute
the already opened `oracle_design` development metrics.

## Evidence boundary

The 2,307-query role has already informed V8 and V10. V11 results are therefore
diagnostic only and cannot establish independent generalization. The evaluator
accepts no V9/V10 packet path and cannot open `future_method_holdout`,
`oracle_audit`, or an external dataset.

The three possible formal decisions are:

- `STOP_LINEAR_RANK_EXPANSION_NO_HEADROOM`;
- `STOP_RPQ_16B_CANNOT_RETAIN_HEADROOM`;
- `GO_TO_SEPARATE_CA_RPQ_CUTOFF_PROTOCOL`.

Even a GO authorizes only protocol writing and fresh development-data
construction. It does not authorize opening an old holdout or claiming that a
cutoff-aware algorithm has succeeded.

## Execution

Use
[MSMARCO_RARS_v11_Rank_Rate_Diagnostic.ipynb](../notebooks/MSMARCO_RARS_v11_Rank_Rate_Diagnostic.ipynb)
once in a fresh T4 Colab runtime. The machine-readable source of truth is
[rars_v11_rank_rate_diagnostic_v1.json](../protocols/rars_v11_rank_rate_diagnostic_v1.json).
Return both the executed notebook and downloaded ZIP without editing or
rerunning after metrics appear.
