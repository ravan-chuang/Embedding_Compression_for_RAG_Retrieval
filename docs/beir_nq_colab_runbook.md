# BEIR NQ Confirmation on Colab T4 + Google Drive

This runbook executes the frozen BEIR NQ confirmation protocol without opening
test qrels during method construction. The companion notebook is
[`notebooks/BEIR_NQ_RARS_PCA_Confirmation_Colab.ipynb`](../notebooks/BEIR_NQ_RARS_PCA_Confirmation_Colab.ipynb).

## Non-negotiable gates

There are three separate Git freezes:

1. **Design freeze:** commit protocol, builders, trainers, registry, evaluator,
   and tests before downloading NQ.
2. **Method-artifact freeze:** after validation-only selection, commit the
   filled pre-qrels manifest and selected PCA/RARS configuration copies before
   opening `qrels/test.tsv`.
3. **Identity-audit freeze:** after the qrels coverage and prior-query identity
   audit, commit the eligible-query audit before any test retrieval.

The scripts require full 40-character commit hashes and a clean checkout.
They do not accept abbreviated hashes. No script automatically commits or
pushes.

## Colab setup

Use a GPU runtime and confirm that Colab assigned an NVIDIA T4. The protocol
does not allow silently switching to an A100, L4, or CPU because Faiss GPU
training/search is part of the frozen implementation.

Use at least 25 GB of free Drive space. The main persistent artifacts are:

| Artifact | Approximate size |
|---|---:|
| Source archive and selective extraction | dataset-dependent |
| FP16 document embeddings | 2.06 GB |
| Frozen IVF-PQ index | roughly 0.1 GB plus metadata |
| Fit/validation candidate caches | query-count-dependent |
| Two rank-16 int8 sidecars | about 0.086 GB |

The implementation streams residuals and does not store a full 4.12 GB FP32
residual matrix.

The notebook uses this default Drive root:

```text
/content/drive/MyDrive/rars-beir-nq-confirmation-v1
```

Re-running a completed cell is safe. Corpus encoding, IVF-PQ addition, ANN
candidate search, exact candidate scoring, max-absolute calibration, and
sidecar encoding use completion manifests or checkpoint files. Do not delete
`.part`, progress, or checkpoint files after an interrupted session; reconnect
the same Drive and rerun the same cell.

## Stage 0 — commit the design

Before running the notebook's first dataset cell, create a normal Git commit
containing the complete design package. Record its full hash as
`DESIGN_FREEZE_COMMIT` in the notebook. The `init` command verifies:

- `HEAD` equals that hash;
- the checkout is clean;
- every required protocol, builder, trainer, evaluator, registry, and validator
  exists in that commit;
- no test-access flag is true.

If this check fails, stop. Do not bypass it by editing the runner in Colab.

## Stage 1 — qrels-free artifacts

The notebook runs these operations in order:

1. resume-download `nq.zip` and verify the registered MD5;
2. selectively extract `corpus.jsonl`, `queries.jsonl`, and
   `qrels/train.tsv` only;
3. scan the full corpus, validate unique string document IDs, and record the
   exact count and SHA-256;
4. derive the deterministic fit/validation split from train membership;
5. save one local BGE-small model snapshot and hash every file;
6. encode all documents to a Drive-backed FP16 memmap;
7. encode fit then validation queries to one FP32 NumPy array;
8. train and incrementally add the full corpus to the frozen IVF-PQ index.

The splitter scans the shared BEIR query JSONL but retains and emits text only
for official train-membership IDs. It reads only the first column of
`qrels/train.tsv`; relevance values are not parsed.

## Stage 2 — PCA/RARS fitting and validation selection

The sidecar trainer:

- searches Top-100 fit and validation candidate pools with the frozen index;
- computes exact dense scores only inside those frozen candidate pools;
- fits unweighted residual PCA on 300,000 uniformly sampled document rows;
- fits RARS using 300,000 score-error-weighted draws from fit-query candidates;
- streams both rank-16 bases over the corpus to produce matched int8 codes;
- independently selects alpha and Top-B for PCA and RARS using validation
  exact-candidate Top-10 overlap only;
- records that train relevance values, validation qrels, test qrels, and test
  retrieval were not used.

The manifest builder copies the two small selected configs into
`results/beir_nq_confirmation/pre_qrels/`, writes
`protocols/beir_nq_pre_qrels_manifest.json`, and verifies every Drive/repository
file's bytes and SHA-256. Inspect these files, then commit them. Record the full
hash as `METHOD_FREEZE_COMMIT`.

Do not run the next section until that commit exists and the checkout is clean.

## Stage 3A — identity and coverage audit

After the method-artifact freeze only:

```bash
python scripts/evaluate_beir_nq_frozen.py audit \
  --artifact-root /content/drive/MyDrive/rars-beir-nq-confirmation-v1 \
  --repo /content/Embedding_Compression_for_RAG_Retrieval \
  --method-freeze-commit <40-hex-method-commit> \
  --output-dir /content/drive/MyDrive/rars-beir-nq-confirmation-v1/stage3/audit
```

This is the first command allowed to extract and parse `qrels/test.tsv`. It:

- verifies every pre-qrels artifact again;
- checks every positive qrel document against the full string-ID corpus map;
- compares same-namespace IDs and normalized query text against the frozen
  prior-query registry;
- requires at least 3,000 eligible queries;
- writes no retrieval results or metrics.

Copy only `eligible_test_query_audit.json` into a repository results directory,
inspect it, and commit it. Keep the qrels file on Drive rather than in Git.
Record this full hash as `AUDIT_FREEZE_COMMIT`.

## Stage 3B — one-shot evaluation

With the audit JSON tracked in the clean audit-freeze commit:

```bash
python scripts/evaluate_beir_nq_frozen.py evaluate \
  --artifact-root /content/drive/MyDrive/rars-beir-nq-confirmation-v1 \
  --repo /content/Embedding_Compression_for_RAG_Retrieval \
  --method-freeze-commit <40-hex-method-commit> \
  --audit-freeze-commit <40-hex-audit-commit> \
  --audit <repo-path-to-eligible_test_query_audit.json> \
  --output-dir /content/drive/MyDrive/rars-beir-nq-confirmation-v1/stage3/evaluation
```

The evaluator loads the same hashed model snapshot, runs Base/PCA/RARS
together, writes all per-query metrics, and performs 20,000 paired bootstrap
resamples with seed `20260720`. A start sentinel prevents an accidental second
run. `--resume` is allowed only after interruption and only when all frozen
input hashes are unchanged.

After completion, stop. A positive, null, or negative RARS-minus-PCA result does
not authorize retuning on NQ.

## Failure policy

- **Wrong archive MD5:** stop; retain the partial file for diagnosis.
- **Non-T4 GPU:** request a new Colab runtime; do not change the protocol.
- **Drive disconnect:** remount the same Drive and rerun the interrupted cell.
- **Artifact/hash mismatch:** stop and identify the changed input; do not update
  hashes merely to make validation pass.
- **Fewer than 3,000 eligible test queries:** report the protocol infeasible.
- **CUDA out of memory:** reduce an operational batch size only. Do not change
  `nlist`, `nprobe`, M, rank, candidate depth, alpha grid, or Top-B grid.
