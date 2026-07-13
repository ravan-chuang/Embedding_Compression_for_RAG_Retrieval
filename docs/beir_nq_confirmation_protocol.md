# BEIR Natural Questions Independent Confirmation Protocol

## Status

The design is ready to freeze in Git. No BEIR NQ test qrels were downloaded,
opened, parsed, summarized, or evaluated while preparing this protocol.

Machine-readable specification:
[`protocols/beir_nq_rars_pca_confirmation_v1.json`](../protocols/beir_nq_rars_pca_confirmation_v1.json)

Colab execution guide:
[`docs/beir_nq_colab_runbook.md`](beir_nq_colab_runbook.md)

## Why Natural Questions

The official BEIR dataset table describes Natural Questions as a public
train/test retrieval dataset with approximately 2.68 million corpus documents
and 3,452 test queries. This is a useful middle ground:

- substantially more independent queries than the closed 42-query restricted
  TREC DL 2019 analysis;
- a complete benchmark corpus instead of a 1M subset with low qrels coverage;
- independent train queries for basis fitting and validation;
- materially cheaper document embedding than the 8.84M-passage full MS MARCO
  corpus;
- large enough to remain a meaningful ANN and sidecar experiment.

Sources:

- [BEIR repository and dataset table](https://github.com/beir-cellar/beir#available-datasets)
- [BEIR NQ archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip)
- [DPR Natural Questions data documentation](https://github.com/facebookresearch/DPR#retriever-input-data-format)

The registered archive MD5 is
`d4d3d2e48787a744b6f6e691ff534307`, as published by BEIR.

## Scientific question

> When PCA and RARS are newly fitted on the same independent corpus using only
> NQ train queries, does RARS improve Recall@10 over a storage-matched PCA
> sidecar on the untouched official NQ test queries?

This evaluates the RARS algorithm, not transfer of the MS MARCO basis. Both
sidecars are corpus-specific because they encode residuals of the new frozen
NQ IVF-PQ index.

## Frozen systems

| Item | Value |
|---|---|
| Corpus | Full BEIR NQ corpus |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Dimension | 384, normalized |
| Embedding cache | FP16 on Drive; FP32 for Faiss/residual computation |
| Model identity | One Stage-1 local snapshot, file-hashed before test access |
| Query encoding | Raw BEIR query text, no added instruction prefix |
| Base index | Inner-product IVF-PQ M32 |
| IVF configuration | `nlist=2048`, `nprobe=32`, `nbits=8` |
| Candidate pool | Top-100 |
| Final cutoff | Top-10 |
| Sidecars | PCA rank-16 int8 and RARS rank-16 int8 |
| Quantizer | Shared per-coefficient symmetric max-absolute int8 |
| Primary metric | Recall@10 |
| Primary contrast | RARS minus PCA |
| Bootstrap | 20,000 paired resamples, seed `20260720` |

The `nlist` and `nprobe` values are fixed now and may not be changed after test
access. Index construction and search use one NVIDIA T4 through Faiss with
FP32 lookup tables; coarse and PQ training each use 25 iterations and seed 42.
Operational batch sizes may change for memory safety because they do not alter
scores or rankings.

The IVF setting is an ex-ante scale adjustment rather than a result-driven
sweep: `nlist` increases from 512 on the 1M corpus to 2,048 on the roughly
2.68M-document NQ corpus, while `nprobe=32` keeps the approximate number of
scanned codes in the same order of magnitude. No NQ test query is used to
choose this operating point.

## Query-use contract

### NQ train split

Train-split query IDs are deterministically divided using:

```text
h = uint64(first 16 hex digits of
    sha256("beir_nq_rars_pca_confirmation_v1:" + qid))

validation: h mod 10 == 0
fit:        h mod 10 in {1, ..., 9}
```

Fit queries may construct candidate score errors and fit the RARS basis.
Validation queries may select alpha and Top-B using exact candidate-overlap
proxy only. Train qrels relevance values are prohibited.

### NQ test split

All official NQ test queries remaining after identity-only prior-overlap
exclusions form the one-shot test. At least 3,000 queries must remain; otherwise
the run stops and the protocol is reported infeasible rather than weakened
post hoc.

Test query IDs, qrels, retrieval outputs, and metrics may not influence:

- IVF-PQ parameters;
- PCA or RARS basis fitting;
- quantization;
- rank;
- alpha or Top-B;
- candidate depth or final cutoff;
- metric definitions or evaluator code.

## Four-stage gate

### Stage 0 — design freeze

Commit before downloading or opening test qrels:

- the machine-readable protocol;
- this document;
- the pre-qrels manifest contract;
- the validator and its tests.

### Stage 1 — qrels-free artifact construction

Permitted:

- download the official archive and verify its MD5;
- extract and encode the full corpus;
- read the NQ train split and query texts;
- build the base index;
- create deterministic fit/validation query manifests;
- fit PCA and RARS;
- select each sidecar using the validation proxy;
- implement and test the selection-free evaluator on synthetic fixtures.

The Stage-1 extractor materializes only `corpus.jsonl`, `queries.jsonl`, and
`qrels/train.tsv`. It deliberately leaves `qrels/test.tsv` inside the verified
source archive until the Stage-2 Git freeze is complete.

Prohibited:

- opening, parsing, or summarizing `qrels/test.tsv`;
- identifying test outcomes;
- test-query retrieval or metrics.

The registered train-only split is generated with:

```bash
python scripts/create_beir_nq_train_validation_splits.py \
  --queries <nq-dir>/queries.jsonl \
  --train-qrels <nq-dir>/qrels/train.tsv \
  --output-dir <artifact-dir>/query_splits
```

This command uses only the first column of `qrels/train.tsv` as official train
membership, refuses any file not named `train.tsv`, and records that relevance
values and test qrels were not used.

### Stage 2 — method-artifact freeze

Fill
[`protocols/beir_nq_pre_qrels_manifest.template.json`](../protocols/beir_nq_pre_qrels_manifest.template.json),
set its status to `frozen_before_test_qrels_access`, and verify every artifact:

```bash
python scripts/validate_nq_pre_qrels_freeze.py \
  --manifest <filled-pre-qrels-manifest.json> \
  --artifact-root <google-drive-artifact-root> \
  --repo-root <repository-root> \
  --verify-files
```

The validator deliberately has no qrels argument. The filled manifest, selected
PCA/RARS configs, evaluator, and hashes must be committed before Stage 3.

### Stage 3 — identity audit and one-shot evaluation

Only after the Stage-2 commit:

1. read the official test qrels;
2. verify every positive qrel document exists in the full indexed corpus;
3. audit query-ID and normalized-text overlap against every previous project
   query manifest;
4. freeze identity-only exclusions, eligible query IDs, and qrels hash in a
   second Git commit;
5. run Base, PCA, and RARS together once;
6. write per-query metrics and the registered paired bootstrap;
7. stop without retuning, regardless of outcome.

## Pre-qrels validator

The draft contract can be checked now without any artifacts or qrels:

```bash
python scripts/validate_nq_pre_qrels_freeze.py \
  --manifest protocols/beir_nq_pre_qrels_manifest.template.json \
  --allow-draft
```

The validator rejects:

- an unexpected `test_qrels` or outcome file;
- any true test-access flag;
- alpha or Top-B outside the frozen grid;
- drift in `nlist`, `nprobe`, rank, bootstrap, primary metric, or contrast;
- missing or incorrect hashes in a completed freeze;
- an unexpectedly small corpus.

The prior-query identity registry is also frozen before test access. Query IDs
are compared only inside their declared dataset namespace, while Unicode-NFKC,
case-folded, whitespace-collapsed query text is compared across namespaces.

## Storage planning

Approximate working requirements before compression and checkpoints:

| Artifact | Estimate |
|---|---:|
| FP16 document embedding cache | 2.06 GB |
| One rank-16 int8 code matrix | 0.043 GB |
| Two sidecar code matrices | 0.086 GB |
| Int64 document IDs | about 0.022 GB |
| Recommended working storage | at least 20 GB |

The implementation streams residual batches and does not materialize the full
4.12 GB FP32 residual matrix. Runtime is intentionally not promised before a
hardware-specific pilot; batch size may change, scientific parameters may not.

## Reporting rule

- If the primary 95% CI lower bound is above zero, report a qualified
  independent RARS advantage over PCA.
- If the interval crosses zero, report that RARS superiority is not
  established.
- If the interval is fully below zero, report that ordinary residual PCA is
  stronger in this setting.

All three outcomes are publishable project results. None authorizes tuning on
the closed NQ test set.
