#!/usr/bin/env python3
"""Generate the source-hash-pinned V14 anisotropic rate-RPQ notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/MSMARCO_RARS_v14_Anisotropic_Rate_RPQ_Diagnostic.ipynb"
PINNED_SOURCES = (
    "protocols/rars_v14_query_whitened_anisotropic_rate_rpq_diagnostic_v1.json",
    "scripts/rars_v14_anisotropic_rate_core.py",
    "scripts/evaluate_rars_v14_anisotropic_rate_rpq.py",
    "scripts/verify_rars_v14_anisotropic_rate_rpq_packet.py",
    "scripts/verify_rars_v13_committed_closure.py",
    "scripts/verify_rars_v13_signed_score_rpq_packet.py",
    "scripts/train_rars_v13_signed_score_rpq.py",
    "scripts/rars_v13_signed_score_core.py",
    "scripts/rars_v11_rank_rate_core.py",
    "scripts/rars_v8_cutoff_sidecar_core.py",
    "scripts/train_rars_v8_cutoff_sidecar.py",
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
            """# MS MARCO RARS-v14 Query-Whitened Anisotropic Rate-RPQ

## Purpose

V13 proved that lowering a signed-score surrogate is not enough: its stable
aggregate seed gain came from disjoint changed queries and the confidence
interval crossed zero. V14 therefore changes representation capacity rather
than adding another loss. It asks whether a fixed 128-bit residual payload
should allocate different rates to query-sensitive PCA blocks.

This is an **outcome-informed architecture diagnostic** on the already opened
V13 development queries. It is not fresh evidence or confirmation. A GO only
authorizes writing a new protocol on disjoint queries.
"""
        ),
        markdown(
            """## Fixed safety boundary

- The original 1M-document M32 IVF-PQ index and embeddings are immutable.
- PCA, query metrics, rates, and codebooks are fit inside four training folds.
- Relevance labels are used only after scoring, never to learn the representation.
- The primary comparison is the audited V13 unsupervised RPQ16x8 OOF result.
- A uniform eight-bit query-whitened ablation isolates rate allocation.
- Every deployable code is packed into exactly 16 bytes per document.
"""
        ),
        code(
            f"""import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

REPO_URL = 'https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval.git'
SOURCE_BRANCH = 'codex/rars-v8-cutoff-sidecar'
V14_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v14')
ENV_ROOT = Path('/content/rars-v14-env')
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
], check=True)
EXPERIMENT_ENV = os.environ.copy()
EXPERIMENT_ENV['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
versions = subprocess.check_output([
    EXPERIMENT_PYTHON, '-c',
    'import numpy, torch, faiss; '
    'print(numpy.__version__); print(torch.__version__); '
    'print(torch.version.cuda); print(faiss.__version__)',
], text=True, env=EXPERIMENT_ENV).splitlines()
assert versions == ['1.26.4', '2.11.0+cu128', '12.8', '1.12.0'], versions

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
clone_exact(V14_REPO, resolved)
V14_IMPLEMENTATION_COMMIT = resolved

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

SOURCE_HASHES = {json.dumps(source_hashes, indent=4)}
for relative, expected in SOURCE_HASHES.items():
    actual = sha256_file(V14_REPO / relative)
    assert actual == expected, (relative, actual, expected)
print('Exact V14 source:', V14_IMPLEMENTATION_COMMIT)
print('Pinned environment:', versions)
"""
        ),
        code(
            """PROTOCOL = V14_REPO / 'protocols/rars_v14_query_whitened_anisotropic_rate_rpq_diagnostic_v1.json'
protocol = json.loads(PROTOCOL.read_text())
assert protocol['status'] == 'FROZEN_BEFORE_FIRST_V14_DIAGNOSTIC_RUN'
assert protocol['evidence_boundary']['tier'] == 'POST_OUTCOME_ARCHITECTURE_DIAGNOSTIC_ON_V13_DEVELOPMENT_QUERIES'
assert protocol['input_contract']['labels_used_for_metric_or_rate_learning'] is False
assert protocol['method']['total_bits_per_document'] == 128
assert protocol['method']['payload_bytes_per_document'] == 16
assert protocol['method']['minimum_bits_per_block'] == 6
assert protocol['method']['maximum_bits_per_block'] == 10
subprocess.run([
    EXPERIMENT_PYTHON, '-m', 'pytest', '-q',
    'tests/test_rars_v14_anisotropic_rate_core.py',
    'tests/test_rars_v14_protocol_contract.py',
    'tests/test_rars_v14_pipeline_contract.py',
    'tests/test_rars_v14_notebook_contract.py',
    'tests/test_verify_rars_v13_committed_closure.py',
], cwd=V14_REPO, env=EXPERIMENT_ENV, check=True)
closure = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V14_REPO / 'scripts/verify_rars_v13_committed_closure.py'),
    '--repo-root', str(V14_REPO),
], text=True, cwd=V14_REPO, env=EXPERIMENT_ENV))
assert closure['status'] == 'RARS_V13_COMMITTED_CLOSURE_VERIFIED'
assert closure['formal_decision'] == 'STOP_SIGNED_SCORE_RPQ_NO_STABLE_ADVANTAGE'
print('V13 parent closure and all V14 contracts verified.')
"""
        ),
        markdown(
            """## Input and output preflight

V14 reuses the exact frozen V13 bundle and completed Drive packet. The V13
packet verifier runs again before V14 reads the OOF comparator arrays. The V14
output directory must be absent or empty; partial results are never resumed.
"""
        ),
        code(
            """DRIVE = Path('/content/drive/MyDrive/rag-pq-checkpoints')
CACHE = DRIVE / 'msmarco_basis_gate0_cache'
INDEX = DRIVE / 'msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index'
V13_ROOT = DRIVE / 'rars-v13-signed-score-rpq' / 'd8cb761c289f'
V13_BUNDLE = V13_ROOT / 'fresh-development-bundle'
V13_PACKET = V13_ROOT / 'development-once'
V14_ROOT = DRIVE / 'rars-v14-anisotropic-rate-rpq' / V14_IMPLEMENTATION_COMMIT[:12]
OUTPUT = V14_ROOT / 'diagnostic-once'
RUNNER_LOGS = V14_ROOT / 'runner-logs'
RUNNER_LOGS.mkdir(parents=True, exist_ok=True)
required = [
    CACHE / 'embeddings.fp16.memmap',
    INDEX,
    V13_BUNDLE / 'fresh_bundle_complete.json',
    V13_PACKET / 'development_complete.json',
    V13_PACKET / 'full_corpus_signed_score_assignments.uint8.memmap',
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, {'missing_artifacts': missing}
assert not OUTPUT.exists() or not any(OUTPUT.iterdir()), (
    f'{OUTPUT} is non-empty. Delete only this V14 output or use a fresh runtime.'
)
assert shutil.disk_usage('/content').free >= 6_000_000_000, 'Need 6 GB local disk'
assert INDEX.stat().st_size == protocol['frozen_index_contract']['index_bytes']
assert sha256_file(INDEX) == protocol['frozen_index_contract']['index_sha256']
print('V13 bundle/packet and frozen index are present; V14 output is empty.')
"""
        ),
        markdown(
            """## Five-fold diagnostic and full 16-byte materialization

The evaluator caches PCA and query-metric geometry once per fold, trains three
anisotropic seeds, trains the uniform-whitened ablation for the primary seed,
then performs one export-only full-development fit. The final 1M-document
payload is real, not an extrapolated byte count. A T4 run can take tens of
minutes; do not interrupt or edit the output directory.
"""
        ),
        code(
            """OUTPUT.mkdir(parents=True, exist_ok=True)
runner = subprocess.run([
    EXPERIMENT_PYTHON, str(V14_REPO / 'scripts/evaluate_rars_v14_anisotropic_rate_rpq.py'),
    '--v13-bundle-root', str(V13_BUNDLE),
    '--v13-packet-root', str(V13_PACKET),
    '--embeddings', str(CACHE / 'embeddings.fp16.memmap'),
    '--index', str(INDEX),
    '--output-dir', str(OUTPUT),
    '--protocol', str(PROTOCOL),
    '--source-commit', V14_IMPLEMENTATION_COMMIT,
    '--full-corpus-batch-size', '2048',
], text=True, capture_output=True, cwd=V14_REPO, env=EXPERIMENT_ENV)
(RUNNER_LOGS / 'diagnostic_stdout.log').write_text(runner.stdout)
(RUNNER_LOGS / 'diagnostic_stderr.log').write_text(runner.stderr)
if runner.returncode != 0:
    print('V14 return code:', runner.returncode)
    print('===== STDOUT =====')
    print(runner.stdout[-20000:])
    print('===== STDERR =====')
    print(runner.stderr[-20000:])
    runner.check_returncode()
print('V14 diagnostic and full-corpus payload completed.')
"""
        ),
        code(
            """verification = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V14_REPO / 'scripts/verify_rars_v14_anisotropic_rate_rpq_packet.py'),
    '--packet-root', str(OUTPUT),
    '--repo-root', str(V14_REPO),
], text=True, cwd=V14_REPO, env=EXPERIMENT_ENV))
result = json.loads((OUTPUT / 'diagnostic_result.json').read_text())
complete = json.loads((OUTPUT / 'diagnostic_complete.json').read_text())
assert verification['status'] == 'RARS_V14_PACKET_VERIFIED'
assert verification['formal_decision'] == result['formal_decision'] == complete['formal_decision']
assert result['labels_used_for_representation_learning'] is False
assert result['future_method_holdout_opened'] is False
assert result['old_rars_holdout_opened'] is False
assert result['fresh_query_access_authorized'] is False
assert (OUTPUT / 'full_corpus_qw_ar_rpq_codes.uint8.memmap').stat().st_size == 16000000
print(json.dumps({
    'formal_decision': result['formal_decision'],
    'metrics': result['metrics'],
    'comparisons': result['comparisons'],
    'seed_gains': result['seed_gains'],
    'fold_gains': result['fold_gains'],
    'multi_seed_consensus': result['multi_seed_consensus'],
    'allocations': result['allocations'],
    'failed_gates': result['decision']['failed_gates'],
    'packet_verification': verification,
}, indent=2, allow_nan=False))
"""
        ),
        markdown(
            """## Handoff

Accept the printed decision unchanged. Download the ZIP and this executed
notebook. Do not rerun after inspecting metrics. Even a GO remains diagnostic
and cannot be presented as independent evidence.
"""
        ),
        code(
            """PACKET = Path('/content') / f'rars-v14-anisotropic-rate-rpq-{V14_IMPLEMENTATION_COMMIT[:12]}'
if PACKET.exists():
    shutil.rmtree(PACKET)
PACKET.mkdir(parents=True)
for path in OUTPUT.iterdir():
    if path.is_file():
        shutil.copy2(path, PACKET / path.name)
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
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
