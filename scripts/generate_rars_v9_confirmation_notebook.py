#!/usr/bin/env python3
"""Generate the clean source-hash-pinned RARS-v9 Colab notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/MSMARCO_RARS_v9_Locked_Confirmation.ipynb"
PINNED_SOURCES = (
    "protocols/rars_v9_locked_confirmation_v1.json",
    "scripts/build_rars_v9_future_identity.py",
    "scripts/build_rars_v9_m48_baseline.py",
    "scripts/evaluate_rars_v9_locked_confirmation.py",
    "scripts/rars_v9_confirmation_core.py",
    "scripts/rars_v8_cutoff_sidecar_core.py",
    "scripts/evaluate_rars_v6_1m_headroom.py",
    "scripts/rars_v6_headroom_core.py",
    "splits/msmarco_rars_train_split.json",
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
            """# MS MARCO RARS-v9 Locked Within-Program Confirmation

## Goal

Evaluate the **already frozen** RARS-v8 rank-16 int8 sidecar exactly once on
the 803-query `future_method_holdout`. The sole primary endpoint is paired
known-positive `Recall@10(RARS-v8) - Recall@10(PCA)`.

This role is prospective relative to V8, but it came from the historical v2
development pool. The result is therefore **within-program prospective
confirmation, not independent or official MS MARCO evidence**.
"""
        ),
        markdown(
            """## Irreversible boundary

Cells through the preflight are qrels-free. They build only query identities,
query vectors, and a uniform M48 limitation index; verify the V8 artifacts;
and freeze every input. The confirmation cell then writes a durable start
marker before it parses qrels.

Do not edit rank, alpha, Top-B, hashes, metrics, thresholds, or comparator
settings. Do not rerun the final cell after any outcome is shown. Return the
packet for audit whether the result confirms or stops V8.
"""
        ),
        code(
            f"""import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

REPO_URL = 'https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval.git'
SOURCE_BRANCH = 'codex/rars-v8-cutoff-sidecar'
REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v9')
ENV_ROOT = Path('/content/rars-v9-env')
if ENV_ROOT.exists():
    shutil.rmtree(ENV_ROOT)
subprocess.run([sys.executable, '-m', 'venv', str(ENV_ROOT)], check=True)
EXPERIMENT_PYTHON = str(ENV_ROOT / 'bin/python')
subprocess.run([
    EXPERIMENT_PYTHON, '-m', 'pip', 'install', '-q',
    'numpy==1.26.4', 'faiss-gpu-cu12==1.12.0', 'pytest>=8,<9',
], check=True)
EXPERIMENT_ENV = os.environ.copy()
EXPERIMENT_ENV['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
assert subprocess.check_output([
    EXPERIMENT_PYTHON, '-c', 'import numpy; print(numpy.__version__)'
], text=True, env=EXPERIMENT_ENV).strip() == '1.26.4'

if REPO.exists():
    shutil.rmtree(REPO)
resolved = subprocess.check_output([
    'git', 'ls-remote', REPO_URL, f'refs/heads/{{SOURCE_BRANCH}}'
], text=True).split()[0]
subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', resolved], check=True)
V9_IMPLEMENTATION_COMMIT = subprocess.check_output([
    'git', '-C', str(REPO), 'rev-parse', 'HEAD'
], text=True).strip()
assert V9_IMPLEMENTATION_COMMIT == resolved
assert not subprocess.check_output([
    'git', '-C', str(REPO), 'status', '--porcelain'
], text=True).strip()

SOURCE_HASHES = {json.dumps(source_hashes, indent=4)}
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

for relative, expected in SOURCE_HASHES.items():
    actual = sha256_file(REPO / relative)
    assert actual == expected, (relative, actual, expected)
print('Exact run commit:', V9_IMPLEMENTATION_COMMIT)
print('All registered source hashes verified before any outcome access.')
"""
        ),
        code(
            """PROTOCOL = REPO / 'protocols/rars_v9_locked_confirmation_v1.json'
subprocess.run([
    EXPERIMENT_PYTHON, '-m', 'pytest', '-q',
    'tests/test_rars_v8_cutoff_sidecar_core.py',
    'tests/test_rars_v9_confirmation_core.py',
    'tests/test_rars_v9_confirmation_protocol_contract.py',
    'tests/test_build_rars_v9_future_identity_contract.py',
    'tests/test_build_rars_v9_m48_baseline_contract.py',
    'tests/test_evaluate_rars_v9_locked_confirmation_contract.py',
], cwd=REPO, check=True, env=EXPERIMENT_ENV)
protocol = json.loads(PROTOCOL.read_text())
assert protocol['status'] == 'FROZEN_BEFORE_FIRST_V9_OUTCOME_ACCESS'
assert protocol['claim_boundary']['evidence_tier'] == (
    'WITHIN_PROGRAM_PROSPECTIVE_HOLDOUT_NOT_INDEPENDENT'
)
assert protocol['confirmation_gate']['method_or_threshold_tuning_authorized'] is False
print('Protocol and implementation contract tests passed.')
"""
        ),
        code(
            """DRIVE = Path('/content/drive/MyDrive/rag-pq-checkpoints')
CACHE = DRIVE / 'msmarco_basis_gate0_cache'
M32_INDEX = DRIVE / 'msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index'
V8_ROOT = DRIVE / 'rars-v8-cutoff-sidecar' / 'c9d95f15d55e'
V8_DEVELOPMENT = V8_ROOT / 'development'
V8_SIDECARS = V8_ROOT / 'sidecars'
V9_ROOT = DRIVE / 'rars-v9-locked-confirmation' / V9_IMPLEMENTATION_COMMIT[:12]
FUTURE_IDENTITY = V9_ROOT / 'future-identity-qrels-free'
M48_ROOT = V9_ROOT / 'm48-qrels-free'
CONFIRMATION = V9_ROOT / 'confirmation-once'

required = [
    CACHE / 'embeddings.fp16.memmap',
    CACHE / 'doc_ids.int64.memmap',
    CACHE / 'query_vectors.fp32.npy',
    CACHE / 'qrels_subset.json',
    M32_INDEX,
    V8_DEVELOPMENT / 'method_freeze.json',
    V8_DEVELOPMENT / 'development_result.json',
    V8_SIDECARS / 'sidecars_complete.json',
    V8_SIDECARS / 'pca/manifest.json',
    V8_SIDECARS / 'pca/codes.int8.npy',
    V8_SIDECARS / 'rars/manifest.json',
    V8_SIDECARS / 'rars/codes.int8.npy',
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, {'missing': missing}
assert shutil.disk_usage('/content').free >= 8_000_000_000, 'Need at least 8 GB local disk'
for output in (FUTURE_IDENTITY, M48_ROOT, CONFIRMATION):
    assert not output.exists() or not any(output.iterdir()), (
        f'Refusing to overwrite or reuse {output}'
    )
V9_ROOT.mkdir(parents=True, exist_ok=True)
print('Frozen artifacts found; all V9 outputs are empty.')
"""
        ),
        markdown(
            """## Stage 1 — qrels-free future identity

This reproduces the registered v2.1 inner partition and V3 split from the
frozen training split, then copies only the 803 query vectors. It does not use
the historical candidate builder and accepts no qrels argument.
"""
        ),
        code(
            """subprocess.run([
    EXPERIMENT_PYTHON, str(REPO / 'scripts/build_rars_v9_future_identity.py'),
    '--train-split', str(REPO / 'splits/msmarco_rars_train_split.json'),
    '--query-vectors', str(CACHE / 'query_vectors.fp32.npy'),
    '--output-dir', str(FUTURE_IDENTITY),
    '--protocol', str(PROTOCOL),
    '--source-commit', V9_IMPLEMENTATION_COMMIT,
], check=True, cwd=REPO, env=EXPERIMENT_ENV)
identity = json.loads((FUTURE_IDENTITY / 'v9_identity_complete.json').read_text())
assert identity['status'] == 'RARS_V9_QRELS_FREE_FUTURE_IDENTITY_COMPLETE'
assert identity['query_count'] == 803
assert identity['qrels_opened'] is False
assert identity['candidate_arrays_created'] is False
assert identity['labels_materialized'] is False
print('Qrels-free future identity frozen:', identity['outputs'])
"""
        ),
        markdown(
            """## Stage 2 — qrels-free M48 rebuild

M48 is the same-code-budget limitation baseline for M32 plus a 16-byte
sidecar. It is trained and added without qrels, then hash-frozen before the
confirmation start marker. On a T4 this is the longest preflight stage.
"""
        ),
        code(
            """subprocess.run([
    EXPERIMENT_PYTHON, str(REPO / 'scripts/build_rars_v9_m48_baseline.py'),
    '--embeddings', str(CACHE / 'embeddings.fp16.memmap'),
    '--output-dir', str(M48_ROOT),
    '--protocol', str(PROTOCOL),
    '--source-commit', V9_IMPLEMENTATION_COMMIT,
    '--training-rows', '200000',
    '--training-seed', '20260723',
    '--batch-size', '32768',
    '--use-gpu',
], check=True, cwd=REPO, env=EXPERIMENT_ENV)
m48 = json.loads((M48_ROOT / 'm48_build_complete.json').read_text())
assert m48['status'] == 'RARS_V9_QRELS_FREE_M48_BASELINE_COMPLETE'
assert m48['qrels_opened'] is False
assert m48['outcome_metric_computed'] is False
print('Qrels-free M48 frozen:', m48['index'])
"""
        ),
        markdown(
            """## STOP AND VERIFY BEFORE OUTCOME ACCESS

The next cell is the one-shot confirmation. Running it will create
`confirmation_started.json`, then open qrels and produce the final decision.

Before continuing, verify that this is the intended branch/commit and that
both preceding packets say `qrels_opened: false`. Do not interrupt the cell.
If it errors after the start marker, return the directory and traceback for
audit; do not delete the directory and rerun.
"""
        ),
        code(
            """assert identity['qrels_opened'] is False
assert m48['qrels_opened'] is False
assert not CONFIRMATION.exists() or not any(CONFIRMATION.iterdir())
print(json.dumps({
    'source_commit': V9_IMPLEMENTATION_COMMIT,
    'evidence_tier': protocol['claim_boundary']['evidence_tier'],
    'query_count': identity['query_count'],
    'future_identity_sha256': sha256_file(FUTURE_IDENTITY / 'v9_identity_complete.json'),
    'm48_index_sha256': m48['index']['sha256'],
    'v8_method_freeze_sha256': sha256_file(V8_DEVELOPMENT / 'method_freeze.json'),
    'rars_codes_sha256': sha256_file(V8_SIDECARS / 'rars/codes.int8.npy'),
    'pca_codes_sha256': sha256_file(V8_SIDECARS / 'pca/codes.int8.npy'),
    'next_action': 'ONE-SHOT OUTCOME ACCESS; NO RETUNING OR RERUN',
}, indent=2))
"""
        ),
        code(
            """subprocess.run([
    EXPERIMENT_PYTHON, str(REPO / 'scripts/evaluate_rars_v9_locked_confirmation.py'),
    '--future-role-dir', str(FUTURE_IDENTITY),
    '--v8-development-packet', str(V8_DEVELOPMENT),
    '--sidecar-root', str(V8_SIDECARS),
    '--embeddings', str(CACHE / 'embeddings.fp16.memmap'),
    '--doc-ids', str(CACHE / 'doc_ids.int64.memmap'),
    '--qrels', str(CACHE / 'qrels_subset.json'),
    '--m32-index', str(M32_INDEX),
    '--m48-index', str(M48_ROOT / 'ivfpq_m48_nlist512_nprobe16.index'),
    '--m48-complete', str(M48_ROOT / 'm48_build_complete.json'),
    '--output-dir', str(CONFIRMATION),
    '--protocol', str(PROTOCOL),
    '--source-commit', V9_IMPLEMENTATION_COMMIT,
], check=True, cwd=REPO, env=EXPERIMENT_ENV)
complete = json.loads((CONFIRMATION / 'confirmation_complete.json').read_text())
result = json.loads((CONFIRMATION / 'confirmation_result.json').read_text())
assert complete['status'] == 'RARS_V9_LOCKED_CONFIRMATION_COMPLETE'
assert complete['method_or_threshold_tuning_authorized'] is False
assert result['independent_confirmation_claim_allowed'] is False
print(json.dumps({
    'formal_decision': complete['formal_decision'],
    'evidence_tier': complete['evidence_tier'],
    'metrics': result['metrics'],
    'comparisons': result['comparisons'],
    'candidate_gap_recovery_fraction': result['candidate_gap_recovery_fraction'],
    'decision': result['decision'],
}, indent=2, allow_nan=False))
"""
        ),
        markdown(
            """## Return the immutable packet

Download the small confirmation directory and send it back for audit. The M48
index and full sidecars remain in Drive and are referenced by SHA-256; they are
not duplicated in the zip.
"""
        ),
        code(
            """from google.colab import files
archive_base = Path('/content') / f'rars-v9-confirmation-{V9_IMPLEMENTATION_COMMIT[:12]}'
archive_path = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=CONFIRMATION))
print('Download:', archive_path, archive_path.stat().st_size, 'bytes')
files.download(str(archive_path))
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
    payload = build()
    OUTPUT.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
