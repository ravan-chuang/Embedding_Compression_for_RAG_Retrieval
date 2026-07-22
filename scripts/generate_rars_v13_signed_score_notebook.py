#!/usr/bin/env python3
"""Generate the source-hash-pinned RARS-v13 fresh development notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/MSMARCO_RARS_v13_Signed_Score_RPQ_Development.ipynb"
PINNED_SOURCES = (
    "protocols/rars_v13_signed_score_distilled_rpq_v1.json",
    "scripts/rars_v13_signed_score_core.py",
    "scripts/freeze_rars_v13_fresh_queries.py",
    "scripts/build_rars_v13_fresh_bundle.py",
    "scripts/train_rars_v13_signed_score_rpq.py",
    "scripts/verify_rars_v13_signed_score_rpq_packet.py",
    "scripts/rars_v11_rank_rate_core.py",
    "scripts/rars_v8_cutoff_sidecar_core.py",
    "scripts/train_rars_v8_cutoff_sidecar.py",
    "scripts/evaluate_rars_v6_1m_headroom.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True) or [""]


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


def build() -> dict:
    source_hashes = {name: sha256(ROOT / name) for name in PINNED_SOURCES}
    cells = [
        markdown(
            """# MS MARCO RARS-v13 Signed Score-Distilled RPQ Development

## TL;DR

V12 showed that unsigned cutoff-weighted residual reconstruction was not a
stable improvement over rank-64 RPQ. V13 corrects that objective mismatch: it
fits the **signed query–residual score** inside each frozen RPQ cell using one
anchored 4×4 ridge solve. The PCA basis, assignments, frozen M32 index,
16-byte payload, alpha `0.75`, Top-B `40`, and product partition do not change.

This notebook first freezes 5,000 genuinely disjoint MS MARCO **training**
queries whose positives occur in the frozen 1M corpus. It then runs five-fold
OOF evaluation for three fixed seeds and materializes a real 1M × 16-byte
sidecar. Run once in a fresh T4 runtime. Do not edit, rerun, or reuse partial
outputs after metrics appear.
"""
        ),
        markdown(
            """## Evidence and safety boundary

The official MS MARCO train query/qrels sources are used only for a
corpus-restricted development task. Every historical 6,980 qid and all 2,500
V12 development qids are excluded before encoding. Query selection and fold
assignment are fixed by SHA-256 rules before candidate retrieval.

Passing all gates authorizes only a new independent-confirmation protocol. It
does not open an old RARS holdout, establish official MS MARCO performance,
implement ScaNN, or demonstrate production latency.
"""
        ),
        code(
            f"""import hashlib, json, os, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

REPO_URL = 'https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval.git'
SOURCE_BRANCH = 'codex/rars-v8-cutoff-sidecar'
V13_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v13')
ENV_ROOT = Path('/content/rars-v13-env')
if ENV_ROOT.exists():
    shutil.rmtree(ENV_ROOT)
venv = subprocess.run([
    sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages',
    str(ENV_ROOT),
], text=True, capture_output=True)
if venv.returncode != 0:
    print(venv.stdout)
    print(venv.stderr, file=sys.stderr)
    venv.check_returncode()
EXPERIMENT_PYTHON = str(ENV_ROOT / 'bin/python')
subprocess.run([
    sys.executable, '-m', 'pip', '--python', EXPERIMENT_PYTHON, 'install', '-q',
    'numpy==1.26.4', 'faiss-gpu-cu12==1.12.0', 'pytest>=8,<9',
    'sentence-transformers==3.4.1',
], check=True)
EXPERIMENT_ENV = os.environ.copy()
EXPERIMENT_ENV['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
versions = subprocess.check_output([
    EXPERIMENT_PYTHON, '-c',
    'import importlib.metadata, numpy, torch; '
    'print(numpy.__version__); print(torch.__version__); print(torch.version.cuda); '
    'print(importlib.metadata.version("sentence-transformers"))',
], text=True, env=EXPERIMENT_ENV).splitlines()
assert versions == ['1.26.4', '2.11.0+cu128', '12.8', '3.4.1'], versions

def clone_exact(destination, commit):
    if destination.exists():
        shutil.rmtree(destination)
    subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True).strip()
    assert actual == commit and not dirty, (actual, commit, dirty)

resolved = subprocess.check_output([
    'git', 'ls-remote', REPO_URL, f'refs/heads/{{SOURCE_BRANCH}}'
], text=True).split()[0]
clone_exact(V13_REPO, resolved)
V13_IMPLEMENTATION_COMMIT = resolved

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

SOURCE_HASHES = {json.dumps(source_hashes, indent=4)}
for relative, expected in SOURCE_HASHES.items():
    actual = sha256_file(V13_REPO / relative)
    assert actual == expected, (relative, actual, expected)
print('Exact V13 run commit:', V13_IMPLEMENTATION_COMMIT)
print('All V13 source hashes verified:', versions)
"""
        ),
        code(
            """PROTOCOL = V13_REPO / 'protocols/rars_v13_signed_score_distilled_rpq_v1.json'
protocol = json.loads(PROTOCOL.read_text())
assert protocol['status'] == 'FROZEN_BEFORE_FIRST_V13_FRESH_DEVELOPMENT_RUN'
assert protocol['fresh_query_freeze']['target_query_count'] == 5000
assert protocol['method']['rank'] == 64
assert protocol['method']['payload_bytes_per_document'] == 16
assert protocol['score_distillation']['updates'] == 1
assert protocol['score_distillation']['post_update_assignments'] == 'byte-identical to unsupervised assignments'
assert protocol['rpq_training']['all_seeds_run_in_all_folds'] is True
assert protocol['development_gate']['old_holdout_reuse_authorized'] is False
subprocess.run([
    EXPERIMENT_PYTHON, '-m', 'pytest', '-q',
    'tests/test_rars_v13_signed_score_core.py',
    'tests/test_rars_v13_protocol_contract.py',
    'tests/test_rars_v13_pipeline_contract.py',
    'tests/test_rars_v11_rank_rate_core.py',
    'tests/test_rars_v8_cutoff_sidecar_core.py',
], cwd=V13_REPO, env=EXPERIMENT_ENV, check=True)
print('V13 numerical, data-isolation, payload, and decision contracts passed.')
"""
        ),
        markdown(
            """## Frozen corpus and official fresh-query sources

The existing document embeddings and M32 IVF-PQ index stay unchanged. Only
the small official query/qrels files are downloaded. The freezer records their
exact byte counts and SHA-256 hashes before selecting or encoding queries.
"""
        ),
        code(
            """DRIVE = Path('/content/drive/MyDrive/rag-pq-checkpoints')
CACHE = DRIVE / 'msmarco_basis_gate0_cache'
INDEX = DRIVE / 'msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index'
V13_ROOT = DRIVE / 'rars-v13-signed-score-rpq' / V13_IMPLEMENTATION_COMMIT[:12]
QUERY_FREEZE = V13_ROOT / 'fresh-query-freeze'
BUNDLE = V13_ROOT / 'fresh-development-bundle'
DEVELOPMENT = V13_ROOT / 'development-once'
RUNNER_LOGS = V13_ROOT / 'runner-logs'
RUNNER_LOGS.mkdir(parents=True, exist_ok=True)
required = [
    CACHE / 'embeddings.fp16.memmap',
    CACHE / 'doc_ids.int64.memmap',
    INDEX,
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, {'missing_artifacts': missing}
for path in (QUERY_FREEZE, BUNDLE, DEVELOPMENT):
    assert not path.exists() or not any(path.iterdir()), (
        f'{path} is non-empty. Do not overwrite or reuse a partial V13 run.'
    )
assert shutil.disk_usage('/content').free >= 8_000_000_000, 'Need 8 GB local disk'
assert INDEX.stat().st_size == protocol['frozen_index_contract']['index_bytes']
assert sha256_file(INDEX) == protocol['frozen_index_contract']['index_sha256']
print('Frozen 1M corpus/index verified and V13 outputs are empty.')
"""
        ),
        code(
            """SOURCE_DATA = Path('/content/rars-v13-msmarco-train-sources')
if SOURCE_DATA.exists():
    shutil.rmtree(SOURCE_DATA)
SOURCE_DATA.mkdir(parents=True)
queries_archive = SOURCE_DATA / 'queries.tar.gz'
qrels_train = SOURCE_DATA / 'qrels.train.tsv'
queries_train = SOURCE_DATA / 'queries.train.tsv'
urllib.request.urlretrieve(
    protocol['fresh_query_freeze']['official_queries_archive_url'],
    queries_archive,
)
urllib.request.urlretrieve(
    protocol['fresh_query_freeze']['official_qrels_url'],
    qrels_train,
)
with tarfile.open(queries_archive, 'r:gz') as archive:
    members = [
        member for member in archive.getmembers()
        if Path(member.name).name == protocol['fresh_query_freeze']['queries_member']
    ]
    assert len(members) == 1, [member.name for member in members]
    source = archive.extractfile(members[0])
    assert source is not None
    queries_train.write_bytes(source.read())
assert sum(1 for _ in queries_train.open()) > 500000
assert sum(1 for _ in qrels_train.open()) > 500000
print(json.dumps({
    'queries_train_bytes': queries_train.stat().st_size,
    'queries_train_sha256': sha256_file(queries_train),
    'qrels_train_bytes': qrels_train.stat().st_size,
    'qrels_train_sha256': sha256_file(qrels_train),
}, indent=2))
"""
        ),
        markdown(
            """## Pre-candidate fresh-query freeze

This is the only query-selection step. It excludes all 6,980 historical qids
and all 2,500 V12 qids, applies the fixed corpus-coverage and SHA-256 rule, encodes exactly 5,000
queries with the pinned MiniLM revision, and freezes five folds. No search or
metric is performed here.
"""
        ),
        code(
            """QUERY_FREEZE.mkdir(parents=True, exist_ok=True)
freezer = subprocess.run([
    EXPERIMENT_PYTHON, str(V13_REPO / 'scripts/freeze_rars_v13_fresh_queries.py'),
    '--queries-train', str(queries_train),
    '--qrels-train', str(qrels_train),
    '--doc-ids', str(CACHE / 'doc_ids.int64.memmap'),
    '--prior-qids', str(V13_REPO / 'splits/msmarco_rars_train_qids.json'),
    '--prior-qids', str(V13_REPO / 'splits/msmarco_rars_validation_qids.json'),
    '--prior-qids', str(V13_REPO / 'splits/msmarco_rars_test_qids.json'),
    '--prior-qids', str(V13_REPO / 'results/rars_v12_ca_rpq/development/query_ids.utf8.txt'),
    '--output-dir', str(QUERY_FREEZE),
    '--protocol', str(PROTOCOL),
    '--source-commit', V13_IMPLEMENTATION_COMMIT,
    '--target-query-count', '5000',
    '--device', 'cuda',
], text=True, capture_output=True, cwd=V13_REPO, env=EXPERIMENT_ENV)
(RUNNER_LOGS / 'query_freeze_stdout.log').write_text(freezer.stdout)
(RUNNER_LOGS / 'query_freeze_stderr.log').write_text(freezer.stderr)
if freezer.returncode != 0:
    print(freezer.stdout[-12000:])
    print(freezer.stderr[-12000:])
    freezer.check_returncode()
query_freeze = json.loads((QUERY_FREEZE / 'fresh_query_freeze.json').read_text())
query_manifest = json.loads((QUERY_FREEZE / 'fresh_query_manifest.json').read_text())
assert query_freeze['status'] == 'RARS_V13_FRESH_QUERY_FREEZE_COMPLETE'
assert query_freeze['selection']['candidate_retrieval_performed'] is False
assert query_manifest['query_count'] == 5000
assert query_manifest['historical_qid_overlap'] == []
assert min(query_manifest['fold_counts']) >= 800
print(json.dumps({
    'selected_qid_hash': query_freeze['selection']['selected_qid_hash'],
    'fold_counts': query_manifest['fold_counts'],
    'positive_qrels_in_frozen_corpus': query_manifest['positive_qrels_in_frozen_corpus'],
}, indent=2))
"""
        ),
        markdown(
            """## Candidate/residual bundle

Only after the query freeze is durable, retrieve the frozen M32 Top-100,
materialize labels for the corpus-restricted qrels, and reconstruct residuals
for the candidate union. This stage still computes no ranking metric.
"""
        ),
        code(
            """BUNDLE.mkdir(parents=True, exist_ok=True)
bundler = subprocess.run([
    EXPERIMENT_PYTHON, str(V13_REPO / 'scripts/build_rars_v13_fresh_bundle.py'),
    '--query-freeze-root', str(QUERY_FREEZE),
    '--embeddings', str(CACHE / 'embeddings.fp16.memmap'),
    '--doc-ids', str(CACHE / 'doc_ids.int64.memmap'),
    '--index', str(INDEX),
    '--output-dir', str(BUNDLE),
    '--protocol', str(PROTOCOL),
    '--source-commit', V13_IMPLEMENTATION_COMMIT,
], text=True, capture_output=True, cwd=V13_REPO, env=EXPERIMENT_ENV)
(RUNNER_LOGS / 'bundle_stdout.log').write_text(bundler.stdout)
(RUNNER_LOGS / 'bundle_stderr.log').write_text(bundler.stderr)
if bundler.returncode != 0:
    print(bundler.stdout[-12000:])
    print(bundler.stderr[-12000:])
    bundler.check_returncode()
bundle_manifest = json.loads((BUNDLE / 'fresh_bundle_manifest.json').read_text())
assert bundle_manifest['status'] == 'RARS_V13_FRESH_DEVELOPMENT_BUNDLE_FROZEN'
assert bundle_manifest['metrics_computed'] is False
assert bundle_manifest['old_rars_holdout_opened'] is False
print(json.dumps({
    'query_count': bundle_manifest['query_count'],
    'candidate_residual_count': bundle_manifest['candidate_residual_count'],
    'positive_candidate_hits': bundle_manifest['positive_candidate_hits'],
    'fold_counts': bundle_manifest['fold_counts'],
}, indent=2))
"""
        ),
        markdown(
            """## Five-fold, three-seed V13 development

Every fold refits the storage-matched PCA16 comparator, PCA64, and all three
unsupervised RPQ initializers using the other four folds. The challenger keeps
the initializer assignments byte-identical and receives one signed-score
closed-form codebook update. After all OOF arrays are complete, the
export-only fit writes a real 16,000,000-byte full-corpus assignment payload.
This cell can take tens of minutes; do not interrupt it.
"""
        ),
        code(
            """DEVELOPMENT.mkdir(parents=True, exist_ok=True)
trainer = subprocess.run([
    EXPERIMENT_PYTHON, str(V13_REPO / 'scripts/train_rars_v13_signed_score_rpq.py'),
    '--bundle-root', str(BUNDLE),
    '--embeddings', str(CACHE / 'embeddings.fp16.memmap'),
    '--index', str(INDEX),
    '--output-dir', str(DEVELOPMENT),
    '--protocol', str(PROTOCOL),
    '--source-commit', V13_IMPLEMENTATION_COMMIT,
    '--full-corpus-batch-size', '10000',
], text=True, capture_output=True, cwd=V13_REPO, env=EXPERIMENT_ENV)
(RUNNER_LOGS / 'development_stdout.log').write_text(trainer.stdout)
(RUNNER_LOGS / 'development_stderr.log').write_text(trainer.stderr)
if trainer.returncode != 0:
    print('V13 trainer return code:', trainer.returncode)
    print('===== STDOUT =====')
    print(trainer.stdout[-16000:])
    print('===== STDERR =====')
    print(trainer.stderr[-16000:])
    trainer.check_returncode()
print('V13 fresh-query OOF development and full-corpus sidecar completed.')
"""
        ),
        code(
            """verification = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V13_REPO / 'scripts/verify_rars_v13_signed_score_rpq_packet.py'),
    '--packet-root', str(DEVELOPMENT),
    '--repo-root', str(V13_REPO),
], text=True, cwd=V13_REPO, env=EXPERIMENT_ENV))
result = json.loads((DEVELOPMENT / 'development_result.json').read_text())
complete = json.loads((DEVELOPMENT / 'development_complete.json').read_text())
assert verification['status'] == 'RARS_V13_PACKET_VERIFIED'
assert verification['formal_decision'] == result['formal_decision'] == complete['formal_decision']
assert result['v9_packet_opened'] is False
assert result['v10_packet_opened'] is False
assert result['v11_packet_opened'] is False
assert result['v12_packet_opened'] is False
assert result['old_holdout_opened'] is False
assert result['fresh_confirmation_access_authorized'] is False
assert (DEVELOPMENT / 'full_corpus_signed_score_assignments.uint8.memmap').stat().st_size == 16000000
print(json.dumps({
    'formal_decision': result['formal_decision'],
    'metrics': result['metrics'],
    'comparisons': result['comparisons'],
    'seed_gains': result['seed_gains'],
    'fold_gains': result['fold_gains'],
    'candidate_gap_recovery_fraction': result['candidate_gap_recovery_fraction'],
    'maximum_centroid_drift_fraction': result['maximum_centroid_drift_fraction'],
    'failed_gates': result['decision']['failed_gates'],
    'packet_verification': verification,
}, indent=2, allow_nan=False))
"""
        ),
        markdown(
            """## Handoff

Accept the printed decision unchanged. Copy the completed development packet
and the small lineage manifests into a download archive. Return that ZIP and
this executed notebook for independent audit. Do not rerun after inspecting
the result.
"""
        ),
        code(
            """PACKET = Path('/content') / f'rars-v13-signed-score-rpq-{V13_IMPLEMENTATION_COMMIT[:12]}'
if PACKET.exists():
    shutil.rmtree(PACKET)
(PACKET / 'development').mkdir(parents=True)
(PACKET / 'lineage').mkdir(parents=True)
for path in DEVELOPMENT.iterdir():
    if path.is_file():
        shutil.copy2(path, PACKET / 'development' / path.name)
for path in (
    QUERY_FREEZE / 'fresh_query_freeze.json',
    QUERY_FREEZE / 'fresh_query_manifest.json',
    QUERY_FREEZE / 'fresh_qrels.json',
    BUNDLE / 'fresh_bundle_manifest.json',
    BUNDLE / 'fresh_bundle_complete.json',
):
    shutil.copy2(path, PACKET / 'lineage' / path.name)
archive = Path(shutil.make_archive(str(PACKET), 'zip', root_dir=PACKET))
print('Download:', archive, archive.stat().st_size, 'bytes')
from google.colab import files
files.download(str(archive))
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
