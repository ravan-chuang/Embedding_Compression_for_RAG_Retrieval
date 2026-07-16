# RARS-v2.2 FP32 Boundary-Loss Development Protocol

## Decision this experiment makes

RARS-v2.2 Stage A answers one narrow question: does a rank-16 learned residual
scorer have enough representation and objective signal to beat a directly
computed FP32 PCA sidecar before int8 quantization is introduced?

The first frozen seed-42 run has two required gates on the fixed inner
validation split:

- at least `+0.01135` Recall@10 over Base; and
- at least `+0.005` Recall@10 over FP32 PCA.

Both gates must pass. A failure stops this rank-16 learned-sidecar line. A pass
authorizes only frozen FP32 replication with seeds 42, 43, and 44; it does not
yet authorize QAT, outer evaluation, or a deployment claim.

The machine-readable protocol is
`protocols/rars_v2_2_boundary_loss_development_v1.json`. Method changes require
a new protocol ID; this file and that JSON are not edited to accommodate an
outcome.

## Evidence boundary

Only the deterministic `inner_train` and `inner_validation` partitions of the
original 4,980-query MS MARCO train split are accepted by the trainer.

The historical 1,000-query MS MARCO outer validation set is **burned development
data**. Run-0 outcomes were inspected before v2.1 was revised, so this set can
reproduce historical results but cannot select v2.2, support a confirmatory
claim, or be relabeled as sealed. The status is recorded in
`protocols/rars_v2_data_access_registry_v1.json`.

The legacy bundle builder opens and parses a shared 6,980-query qrels JSON. The
v2.2 freezer therefore records the following facts separately:

- the shared qrels container was opened;
- clean-test query IDs and relevance values were parsed by the source builder;
- no clean-test relevance value entered an inner bundle label, loss, selection
  metric, or result; and
- no outer or clean-test outcome is read by the v2.2 trainer.

This is honest development evidence, not untouched evidence. BEIR NQ and TREC
DL 2019 have already informed the project and cannot be reused as independent
v2 confirmation. A new external collection must begin in `SEALED_UNOPENED`
state after the complete method and evaluator are frozen.

## Frozen Stage-A method

The scorer is

\[
s'(q,d)=s_{ANN}(q,d)+0.05\tanh\left(
\frac{(qW_q)^\top(r_dW_d)}{0.05}
\right)
\]

for documents in frozen ANN positions 1--40. Positions 41--100 receive exactly
zero learned correction.

Initialization uses the frozen PCA rank-16 basis `B` and its selected `alpha`:

\[
W_q=\alpha B, \qquad W_d=B.
\]

Before the tanh bound, this exactly equals the PCA residual correction. With
the bound, it is a PCA parameter warm-start rather than score parity. Epoch 0
is evaluated and can remain the selected checkpoint if training degrades it.

Stage A has:

- no query-only gate;
- no fake quantization, learned scale, or int8 code;
- no random projection initialization;
- no outer-validation argument; and
- no static pair cache.

At the start of each epoch, pairs are re-mined from the current FP32 ranking.
Promotion pairs place a correctable judged-relevant candidate currently below
Top-10 against an unjudged Top-10 candidate. Protection pairs place a current
Top-10 judged-relevant candidate against the strongest correctable unjudged
candidate outside Top-10. As in the source MS MARCO setup, unjudged candidates
are explicitly treated as `unjudged-as-negative`, not known non-relevant.

The loss gives 80% total mass to promotion and 20% to protection. Inside each
type, every contributing query receives equal total weight regardless of how
many judged positives or mined pairs it has. The target margin is the larger of
`0.0001` and the current local rank-10/rank-11 score gap.

## Bundle freeze and lineage

Build only the two inner source bundles:

```bash
python scripts/build_msmarco_rars_v2_boundary_bundles.py \
  --inner-only \
  --embeddings /path/to/embeddings.fp16.memmap \
  --doc-ids /path/to/doc_ids.int64.memmap \
  --query-vectors /path/to/query_vectors.fp32.npy \
  --index /path/to/frozen_ivfpq_m32_nlist512.index \
  --qrels /path/to/qrels_subset.json \
  --train-split splits/msmarco_rars_train_split.json \
  --validation-split splits/msmarco_rars_validation_split.json \
  --output-root /path/to/v2.2/bundles \
  ...
```

Then freeze role identities and split lineage:

```bash
python scripts/freeze_rars_v2_2_inner_bundles.py \
  --bundle-root /path/to/v2.2/bundles \
  --query-vectors /path/to/query_vectors.fp32.npy \
  --train-split splits/msmarco_rars_train_split.json \
  --outer-validation-split splits/msmarco_rars_validation_split.json \
  --clean-test-split splits/msmarco_rars_test_split.json
```

The freezer verifies:

- qid and query-row uniqueness;
- no qid or row overlap across inner train, inner validation, burned outer, and
  clean test;
- exact query-vector content and order for each inner role;
- every source-bundle file byte count and SHA-256; and
- role-specific `v2_2_manifest.json` files.

The trainer accepts `role_id=inner_train` for fitting and
`role_id=inner_validation` for epoch selection. Renaming an outer directory
cannot bypass this check.

## Stage-A training

Run only from an exact, clean 40-character commit:

```bash
python scripts/train_boundary_loss_sidecar_v2_2.py \
  --bundle-dir /path/to/v2.2/bundles/inner_train \
  --selection-bundle-dir /path/to/v2.2/bundles/inner_validation \
  --pca-basis results/rars_pca_comparator/bases/pca_unweighted_rank16.float32.npy \
  --pca-config results/rars_pca_comparator/selected_pca_config.json \
  --output-dir /path/to/v2.2/seed42-fp32 \
  --source-commit FULL_40_CHARACTER_COMMIT \
  --epochs 10 \
  --seed 42 \
  --device cuda
```

The run fingerprint covers the exact source commit, trainer and core source
hashes, both frozen bundle manifests, PCA basis/config hashes, seed, optimizer,
mining, score, and stop-rule parameters. A partial directory is never reused.
A complete directory is reused only with `--reuse-complete` when its fingerprint
and every output hash match.

Stage A writes only:

- FP32 query and document projections;
- training history including epoch-0 selection;
- Base and v2.2 per-query selection metrics;
- selection summary and the registered decision; and
- started/complete lineage manifests.

It must not write a gate, quantization scale, or document code.

## Interpretation and later gates

`STOP_RANK16_LEARNED_SIDECAR` means that substantial relevance-packing oracle
headroom is not enough: this frozen rank-16 representation/objective failed to
beat the actual FP32 PCA comparator. Do not tune on the burned outer split.

`GO_TO_THREE_SEED_FP32_REPLICATION` is provisional. Seeds 42, 43, and 44 must
show the same direction without relying on fewer than 1% of queries. Only then
may a separately versioned QAT stage be defined. Int8 must retain at least 70%
of the replicated FP32 gain.

Before any method claim, a selection-free evaluator must report Base, direct
dot-product exact Top-40, PCA FP32/int8, RARS-v1 FP32/int8, and boundary
FP32/int8 on identical candidates. Before any deployment claim, full-corpus
codes, total storage including shared parameters, latency, and serving parity
are mandatory.
