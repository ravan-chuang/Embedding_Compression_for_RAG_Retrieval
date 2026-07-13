# RARS vs PCA Comparator Protocol

## Purpose

This protocol pre-registers a storage-matched comparison between:

1. the frozen IVF-PQ M32 base index;
2. a rank-16 int8 PCA residual sidecar;
3. a rank-16 int8 Retrieval-Aware Residual Subspace (RARS) sidecar.

The primary methodological question is:

> Does retrieval-aware residual-basis learning improve retrieval quality beyond an ordinary PCA residual sidecar under the same frozen-index, storage, candidate-pool, and serving constraints?

This protocol must be committed before any new comparator evaluation is run.

## Scope

The current MS MARCO 6,980-query development pool has already influenced method development, fitting, validation, evaluation, or later analysis. It may be used only for implementation, train-only fitting, validation-only selection, debugging, ablations, and sensitivity analysis. It must not be used to manufacture a new project-history-level untouched test claim.

The final confirmatory comparison must use a new external query set under a frozen protocol.

## Registered systems

| System ID | Description | Base index changed | Added representation |
|---|---|---:|---:|
| `base_m32` | Frozen IVF-PQ M32 | No | 0 B/document |
| `pca_r16_int8` | Ordinary PCA residual sidecar | No | 16 int8 coefficients/document |
| `rars_r16_int8` | Score-error-weighted RARS sidecar | No | 16 int8 coefficients/document |
| `m48_rebuild` | Rebuilt IVF-PQ M48 reference | Yes | 48 PQ bytes/document |
| `candidate_oracle` | Exact rescoring within frozen Top-100 candidates | No | Evaluation ceiling only |

The primary inferential comparison is `rars_r16_int8` versus `pca_r16_int8`.

## Shared fixed configuration

| Parameter | Frozen value |
|---|---:|
| Corpus | Deterministic MS MARCO 1M passage subset |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Dimension | 384 |
| Base index | IVF-PQ M32 |
| `nlist` | 512 |
| `nprobe` | 16 |
| PQ bits | 8 |
| Candidate pool | Top-100 |
| Final cutoff | Top-10 |
| Sidecar rank | 16 |
| Sidecar coefficients | Per-dimension int8 |
| Bootstrap resamples | 20,000 |
| Bootstrap seed | 20260712 |
| Primary metric | Recall@10 |

The frozen IVF-PQ index, coarse quantizer, codebooks, inverted lists, and PQ codes must remain unchanged.

## Basis definitions

### PCA comparator

For residual

```text
r(x) = x - x_hat_PQ
```

the PCA basis is the top 16 right singular vectors of an unweighted residual sample.

The sampling procedure is fixed:

- sample source: document residual rows;
- maximum sample count: 300,000;
- random seed: 42;
- no query labels, qrels, ANN scores, exact candidate scores, or ranking-error weights;
- deterministic basis orientation when needed.

### RARS

RARS uses the existing score-error-weighted residual-basis procedure trained from train-query candidate-score errors only. No RARS method revision is permitted during the comparator experiment.

## Storage-matched encoding

PCA and RARS must use the same encoder:

1. project document residuals into a rank-16 basis;
2. compute one symmetric scale per coefficient dimension using the same rule;
3. round and clip to int8;
4. store exactly 16 int8 coefficients per document;
5. use identical file formats and metadata accounting.

Shared basis/scales and external document IDs must be reported separately.

## Split usage

### Train

Permitted:

- fit PCA basis;
- fit RARS basis;
- build train candidate-score caches for RARS;
- derive int8 scales and codes;
- verify deterministic generation.

Prohibited:

- select alpha;
- select Top-B;
- report confirmatory qrels results.

### Validation

Permitted:

- select alpha;
- select Top-B;
- apply the registered cost-aware rule;
- deterministic tie-breaking.

Prohibited:

- refit either basis;
- revise quantization;
- add new method features;
- inspect external confirmation labels.

## Validation search space

```text
alpha ∈ {-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1,
          0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}

Top-B ∈ {10, 20, 40, 100}
```

PCA and RARS may select different alpha and Top-B values from the same search space.

## Selection rule

For each method:

1. compute validation candidate-overlap gain over the frozen base;
2. identify the maximum gain across alpha and Top-B;
3. select the smallest Top-B retaining at least 90% of that maximum gain;
4. within that Top-B, select the alpha with the highest overlap;
5. break exact ties by smaller absolute alpha, then smaller numeric alpha.

## Freeze requirements

Before confirmatory evaluation, commit:

- PCA basis and scales;
- RARS basis and scales;
- selected PCA config;
- selected RARS config;
- validation selection table;
- evaluator source and SHA-256;
- artifact manifests;
- external dataset/query manifest;
- statistical-analysis plan.

Large 1M-document code files may stay outside Git only when SHA-256, byte size, shape, dtype, and generation command are committed.

## Primary hypothesis

```text
H1: mean_q [Recall@10_RARS(q) - Recall@10_PCA(q)] > 0
```

The primary result is the paired mean Recall@10 difference with a two-sided 95% percentile bootstrap confidence interval.

Support requires:

```text
95% CI lower bound > 0
```

## Secondary outcomes

- RARS versus PCA on Success@10;
- RARS versus PCA on MRR@10;
- RARS versus PCA on nDCG@10;
- PCA versus base;
- RARS versus base.

These are secondary and must be labeled accordingly.

## External confirmation set

Preferred new query sets compatible with the same passage corpus:

- TREC Deep Learning 2019 passage queries;
- TREC Deep Learning 2020 passage queries.

Before loading qrels or running the evaluator, commit:

- exact query IDs;
- query source and checksum;
- qrels source and checksum;
- query embedding checksum;
- duplicate-ID and normalized-text overlap audits;
- exclusion rules;
- frozen PCA/RARS configs;
- primary metric and contrast;
- bootstrap count and seed.

Any overlap exclusion must use query identity only and be defined before outcome inspection.

## One-shot rule

After freeze:

```text
load frozen index
load frozen PCA artifact
load frozen RARS artifact
run all systems once
write per-query metrics
run paired bootstrap
write audit manifest
stop
```

No alpha, Top-B, basis, quantization, candidate depth, nprobe, or metric definition may change after results are observed.

## Required outputs

```text
results/rars_pca_comparator/
├── selected_pca_config.json
├── selected_rars_config.json
├── validation_selection.csv
├── freeze_manifest.json
├── external/
│   ├── dataset_manifest.json
│   ├── metrics.json
│   ├── per_query_metrics.csv
│   ├── paired_bootstrap.json
│   ├── audit_manifest.json
│   └── summary.md
└── README.md
```

## Decision matrix

### RARS significantly beats PCA

> Retrieval-aware residual-basis learning provides value beyond an ordinary storage-matched PCA sidecar.

### Both beat base, but RARS does not significantly beat PCA

> Low-rank residual sidecars are effective frozen-index retrofits, while this experiment does not establish additional benefit from retrieval-aware weighting.

### Neither reliably beats base

> The current sidecar result does not generalize to the external confirmation set.

### PCA beats RARS

> Ordinary variance-preserving residual structure generalizes better than the current score-error-weighted basis in this external setting.

All outcomes must be reported.

## Prohibited actions

- selecting PCA settings from the current 1,000-query held-out result;
- selecting settings from the 863-query sensitivity subset;
- calling a resplit of the current 6,980 queries untouched;
- changing the primary metric after observing results;
- reporting only the favorable external dataset;
- hiding a non-significant or negative RARS-versus-PCA result;
- mixing exploratory and confirmatory rows without labels.

## Protocol version

```text
protocol_id: rars_pca_comparator_v1
status: preregistration
bootstrap_replicates: 20000
bootstrap_seed: 20260712
primary_metric: Recall@10
primary_contrast: RARS - PCA
```
