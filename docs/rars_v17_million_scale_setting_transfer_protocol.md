# RARS-v17 Million-Scale Setting-Transfer Diagnostic

## Decision

V16 is terminated before any retrieval metric or sidecar fit. FiQA (57,638
documents) and SciFact (5,183 documents) are useful smoke tests but are too
small for the requested million-scale evidence. SciFact also supplies too few
vectors to train every 256-way PQ subquantizer reliably.

V17 replaces them with two existing, fully materialized frozen indexes. Both
settings satisfy the new lower bound of one million indexed documents:

| Setting | Documents | Queries used | IVF-PQ |
|---|---:|---:|---|
| MS MARCO deterministic subset | 1,000,000 | 3,961 fit / 1,019 evaluation | M32×8, nlist 512, nprobe 16 |
| Full BEIR Natural Questions | 2,681,468 | deterministic 60/40 split of 3,452 opened queries | M32×8, nlist 2,048, nprobe 32 |

The common constraints are a 384-dimensional BGE-small family embedding,
inner-product retrieval, M32×8 PQ, a frozen Top-100 candidate pool, Top-40
correction, and a 16-byte rank-16 sidecar. The legacy MS MARCO cache predates
exact model-revision registration, so V17 checks the encoder family and vector
dimension but does not claim exact snapshot identity across settings.

## Evidence boundary

This is an outcome-informed mechanism diagnostic, not a new confirmation.
The NQ official test was already evaluated in the earlier one-shot
confirmation. V17 may reuse its vectors, candidates, and qrels only after
marking the entire NQ role as opened development evidence. Its deterministic
fit/evaluation split reduces direct fitting leakage inside V17, but cannot
erase prior knowledge of the aggregate NQ outcome.

The two indexes deliberately retain their ex-ante scale-specific IVF recipes.
Consequently, the cross-setting contrast combines corpus and index-setting
changes. It must not be called a pure domain-shift causal effect. Each setting
is reported separately; the equal-setting average is descriptive.

## Mechanism matrix

V17 keeps the V16 decomposition:

1. same-candidate exact headroom over Base;
2. PCA rank-64 FP32 over PCA rank-16 FP32;
3. PCA rank-16 FP32 over rank-16 int8;
4. locally fitted cutoff-aware rank-16 int8 over local PCA rank-16 int8;
5. local cutoff-aware basis over the MS MARCO source-setting basis;
6. pooled-fit basis over the source-setting basis.

All hyperparameters remain frozen at alpha 0.75, Top-B 40, rank 16, five
folds, and the registered cutoff-pair objective. Negative results are retained.

## Resource plan

The notebook reuses existing document embeddings and indexes. It rematerializes
only MS MARCO development candidates and NQ candidate-residual unions. Expected
temporary storage is 4–8 GB, not the full FP32 residual matrix. The evaluator
opens only hashed bundles and never reads either full corpus. The NQ source is
the public [BEIR Natural Questions corpus](https://github.com/beir-cellar/beir);
the notebook expects the already completed local artifact root
`/content/drive/MyDrive/rars-beir-nq-confirmation-v2`.

The historical NQ packet contains one explicitly reconciled lineage
inconsistency: the Stage-1 corpus manifest retains an earlier document-ID
digest, while the later pre-qrels freeze records the document-ID file actually
used by the frozen one-shot evaluation. V17 does not edit either historical
artifact. It accepts the later record only after verifying the pre-qrels
status and safety flags, its frozen corpus-manifest record, the exact
document-ID bytes and digest, and the Stage-3 query-audit binding to the
pre-qrels manifest. The reconciliation is written into the prepared-domain
manifest and is not presented as independent confirmation evidence.

## Interpretation

- Stable local improvement on both settings supports setting-specific
  lightweight correction, not universal transfer.
- A local-over-source gap supports a corpus/index-setting interaction.
- Pooled non-inferiority with broad query support motivates a later, genuinely
  independent multi-setting confirmation.
- Failure on NQ is still informative because it occurs at realistic corpus
  scale rather than in a 5K-document PQ regime.
