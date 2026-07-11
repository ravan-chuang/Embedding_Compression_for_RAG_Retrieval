# Clean Query-Level Evaluation Protocol

```mermaid
flowchart LR
    A[6,980 MS MARCO dev queries] --> B[Train: 4,980]
    A --> C[Validation: 1,000]
    A --> D[Untouched test: 1,000]

    B --> B1[Build ANN candidate cache]
    B1 --> B2[Compute exact candidate score error]
    B2 --> B3[Fit weighted rank-16 residual bases]
    B3 --> B4[Encode 1M-document int8 sidecars]

    C --> C1[Evaluate candidate-level proxy]
    C1 --> C2[Select basis / alpha / Top-B]
    C2 --> C3[Freeze selected_config.json]
    C3 --> C4[Commit freeze manifest and hashes]

    D --> D1[Selection-free evaluator]
    C4 --> D1
    D1 --> D2[One-shot qrels metrics]
    D2 --> D3[10,000 paired bootstrap resamples]
    D3 --> D4[Audit manifest and per-query outputs]
```

## Frozen configuration

| Parameter | Value |
|---|---|
| Basis | Score-error weighted |
| Rank | 16 |
| Coefficients | Per-dimension int8 |
| Alpha | 0.75 |
| Candidate pool | Top-100 |
| Corrected candidates | Top-40 |
| Final cutoff | Top-10 |
| IVF-PQ | M=32, nlist=512, nprobe=16, nbits=8 |

## Protocol safeguards

- Split query IDs and query-vector row mappings are deterministic.
- Train, validation, and test intersections are empty.
- The test evaluator does not contain SVD, fitting, basis selection, alpha sweep, Top-B sweep, or validation loading.
- The selected configuration and evaluator were committed before the untouched test was run.
- SHA-256 hashes record the frozen configuration, basis, scales, codes, index, query vectors, document IDs, qrels, evaluator, test metrics, and per-query outputs.
- The test split must not be used for future model or hyperparameter selection.
