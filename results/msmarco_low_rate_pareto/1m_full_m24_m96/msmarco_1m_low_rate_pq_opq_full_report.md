# MS MARCO 1M Low-Rate PQ / OPQ Full Sweep

## Scope

- Corpus: 1,000,000 MS MARCO passages
- Embedding model: `BAAI/bge-small-en-v1.5`
- Embedding dimension: 384
- IVF nlist: 4096
- PQ nbits: 8
- M values: 24, 32, 48, 64, 96
- nprobe values: 4, 16, 32, 64
- Methods: GPU IVF-PQ and Native Faiss OPQMatrix + GPU IVF-PQ

## Merge Validation

- Metadata consistency check: passed
- Combined benchmark points: 40
- Combined artifacts: 10

## nprobe=64: OPQ Incremental Value

| M | Plain Recall@10 | OPQ Recall@10 | Δ Recall@10 | Plain build (s) | OPQ build (s) | Build multiplier |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.6494 | 0.6880 | +0.0386 | 9.9 | 550.9 | 55.8× |
| 32 | 0.6957 | 0.7226 | +0.0269 | 10.3 | 863.0 | 83.6× |
| 48 | 0.7489 | 0.7609 | +0.0120 | 12.6 | 1162.7 | 92.6× |
| 64 | 0.7729 | 0.7752 | +0.0023 | 12.9 | 1575.3 | 122.2× |
| 96 | 0.7887 | 0.7895 | +0.0008 | 17.8 | 2168.7 | 122.1× |

## Interpretation

Within this fixed configuration, OPQ provides the clearest Recall@10 recovery at lower code rates and its incremental benefit contracts as M increases. The storage overhead of the OPQ transform is small, while its offline training/build cost rises substantially. These observations are specific to this corpus, embedding model, index configuration, and evaluation workload.

## Files

- `msmarco_1m_low_rate_pq_opq_full_summary.csv`
- `msmarco_1m_low_rate_pq_opq_full_artifacts.csv`
- `msmarco_1m_low_rate_pq_opq_full_metadata.json`
- `quality_storage_pareto.png`
- `quality_qps_pareto.png`
