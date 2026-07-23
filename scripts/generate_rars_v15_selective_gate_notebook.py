#!/usr/bin/env python3
"""Generate the source-hash-pinned V15 selective-gate development notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/MSMARCO_RARS_v15_Selective_RPQ_Gate_Development.ipynb"
PINNED_SOURCES = (
    "protocols/rars_v15_cross_fitted_selective_rpq_gate_v1.json",
    "scripts/rars_v15_selective_gate_core.py",
    "scripts/evaluate_rars_v15_selective_gate.py",
    "scripts/verify_rars_v15_selective_gate_packet.py",
    "scripts/verify_rars_v14_committed_closure.py",
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
            """# MS MARCO RARS-v15 Cross-Fitted Selective RPQ Gate

## Purpose

V11--V14 show that the strongest stable 16-byte representation is the uniform
PCA64 RPQ16x8 sidecar, while changing centroids, signed-score objectives,
whitening, or rate allocation does not yield a stable advantage. V15 leaves
that document representation untouched and tests whether a tiny query-level
gate can safely fall back to Base on predicted-harm queries.

This is **outcome-informed development on the already opened V13 query role**.
It is not fresh evidence or confirmation. Even a GO only authorizes writing a
new disjoint-query protocol.
"""
        ),
        markdown(
            """## Fixed safety boundary

- The 1M-document IVF-PQ index and V13 uniform 16-byte payload are immutable.
- Representation fitting is label-free and remains inside the outer folds.
- Gate fitting, calibration, and held-out scoring use disjoint fold roles.
- Each query returns either the complete uniform-sidecar ranking or untouched
  Base ranking; candidate-level outcome mixing is forbidden.
- No new document bytes are materialized. The exported global gate is at most
  4 KiB.
"""
        ),
        code(
            f"""import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

REPO_URL = 'https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval.git'
SOURCE_BRANCH = 'codex/rars-v8-cutoff-sidecar'
V15_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v15')
ENV_ROOT = Path('/content/rars-v15-env')
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
clone_exact(V15_REPO, resolved)
V15_IMPLEMENTATION_COMMIT = resolved

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

SOURCE_HASHES = {json.dumps(source_hashes, indent=4)}
for relative, expected in SOURCE_HASHES.items():
    actual = sha256_file(V15_REPO / relative)
    assert actual == expected, (relative, actual, expected)
print('Exact V15 source:', V15_IMPLEMENTATION_COMMIT)
print('Pinned environment:', versions)
"""
        ),
        code(
            """PROTOCOL = V15_REPO / 'protocols/rars_v15_cross_fitted_selective_rpq_gate_v1.json'
protocol = json.loads(PROTOCOL.read_text())
assert protocol['status'] == 'FROZEN_BEFORE_FIRST_V15_DEVELOPMENT_RUN'
assert protocol['evidence_boundary']['tier'] == 'POST_V14_OUTCOME_INFORMED_QUERY_GATING_DEVELOPMENT_ON_V13_QUERIES'
assert protocol['uniform_sidecar']['payload_bytes_per_document'] == 16
assert protocol['storage_contract']['additional_document_bytes'] == 0
assert protocol['gate']['feature_count'] == 12
subprocess.run([
    EXPERIMENT_PYTHON, '-m', 'pytest', '-q',
    'tests/test_rars_v15_selective_gate_core.py',
    'tests/test_rars_v15_protocol_contract.py',
    'tests/test_rars_v15_pipeline_contract.py',
    'tests/test_rars_v15_notebook_contract.py',
    'tests/test_verify_rars_v14_committed_closure.py',
    'tests/test_verify_rars_v13_committed_closure.py',
], cwd=V15_REPO, env=EXPERIMENT_ENV, check=True)
v13_closure = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V15_REPO / 'scripts/verify_rars_v13_committed_closure.py'),
    '--repo-root', str(V15_REPO),
], text=True, cwd=V15_REPO, env=EXPERIMENT_ENV))
v14_closure = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V15_REPO / 'scripts/verify_rars_v14_committed_closure.py'),
    '--repo-root', str(V15_REPO),
], text=True, cwd=V15_REPO, env=EXPERIMENT_ENV))
assert v13_closure['formal_decision'] == 'STOP_SIGNED_SCORE_RPQ_NO_STABLE_ADVANTAGE'
assert v14_closure['formal_decision'] == 'STOP_V14_NO_ANISOTROPIC_RATE_SIGNAL'
print('V13 and V14 committed parent closures and all V15 contracts verified.')
"""
        ),
        markdown(
            """## Input and output preflight

V15 reuses the exact V13 frozen bundle and audited Drive packet. The V13 full
16-byte payload is verified but not copied or rewritten. The V15 output must be
absent or empty; partial outputs are never resumed.
"""
        ),
        code(
            """DRIVE = Path('/content/drive/MyDrive/rag-pq-checkpoints')
V13_ROOT = DRIVE / 'rars-v13-signed-score-rpq' / 'd8cb761c289f'
V13_BUNDLE = V13_ROOT / 'fresh-development-bundle'
V13_PACKET = V13_ROOT / 'development-once'
V15_ROOT = DRIVE / 'rars-v15-selective-rpq-gate' / V15_IMPLEMENTATION_COMMIT[:12]
OUTPUT = V15_ROOT / 'development-once'
RUNNER_LOGS = V15_ROOT / 'runner-logs'
RUNNER_LOGS.mkdir(parents=True, exist_ok=True)
required = [
    V13_BUNDLE / 'fresh_bundle_complete.json',
    V13_PACKET / 'development_complete.json',
    V13_PACKET / 'full_corpus_signed_score_assignments.uint8.memmap',
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, {'missing_artifacts': missing}
assert not OUTPUT.exists() or not any(OUTPUT.iterdir()), (
    f'{OUTPUT} is non-empty. Delete only this V15 output or use a fresh runtime.'
)
assert shutil.disk_usage('/content').free >= 3_000_000_000, 'Need 3 GB local disk'
parent_payload = V13_PACKET / 'full_corpus_signed_score_assignments.uint8.memmap'
assert parent_payload.stat().st_size == 16_000_000
print('V13 bundle/packet are present; parent payload is unchanged; V15 output is empty.')
"""
        ),
        markdown(
            """## Five-fold cross-fitted development

For each outer fold and each of three fixed seeds, the evaluator reproduces
the V13 uniform sidecar, fits the gate on three folds, calibrates on one fold,
and scores only the remaining held-out fold. It then exports one small global
gate without modifying the OOF endpoint. On a T4 this can take tens of minutes;
do not interrupt or edit the output directory.
"""
        ),
        code(
            """OUTPUT.mkdir(parents=True, exist_ok=True)
runner = subprocess.run([
    EXPERIMENT_PYTHON, str(V15_REPO / 'scripts/evaluate_rars_v15_selective_gate.py'),
    '--v13-bundle-root', str(V13_BUNDLE),
    '--v13-packet-root', str(V13_PACKET),
    '--output-dir', str(OUTPUT),
    '--protocol', str(PROTOCOL),
    '--source-commit', V15_IMPLEMENTATION_COMMIT,
], text=True, capture_output=True, cwd=V15_REPO, env=EXPERIMENT_ENV)
(RUNNER_LOGS / 'development_stdout.log').write_text(runner.stdout)
(RUNNER_LOGS / 'development_stderr.log').write_text(runner.stderr)
if runner.returncode != 0:
    print('V15 return code:', runner.returncode)
    print('===== STDOUT =====')
    print(runner.stdout[-20000:])
    print('===== STDERR =====')
    print(runner.stderr[-20000:])
    runner.check_returncode()
print('V15 cross-fitted development completed.')
"""
        ),
        code(
            """verification = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V15_REPO / 'scripts/verify_rars_v15_selective_gate_packet.py'),
    '--packet-root', str(OUTPUT),
    '--v13-packet-root', str(V13_PACKET),
    '--repo-root', str(V15_REPO),
], text=True, cwd=V15_REPO, env=EXPERIMENT_ENV))
result = json.loads((OUTPUT / 'development_result.json').read_text())
complete = json.loads((OUTPUT / 'development_complete.json').read_text())
assert verification['status'] == 'RARS_V15_SELECTIVE_GATE_PACKET_VERIFIED'
assert verification['formal_decision'] == result['formal_decision'] == complete['formal_decision']
assert result['labels_used_for_gate_fitting'] is True
assert result['labels_used_for_representation_learning'] is False
assert result['future_method_holdout_opened'] is False
assert result['old_rars_holdout_opened'] is False
assert result['fresh_query_access_authorized'] is False
assert result['final_export']['additional_document_bytes'] == 0
assert result['final_export']['global_model_bytes'] <= 4096
print(json.dumps({
    'formal_decision': result['formal_decision'],
    'metrics': result['metrics'],
    'comparisons': result['comparisons'],
    'seed_gains': result['seed_gains'],
    'fold_gains': result['fold_gains'],
    'applied_coverages': result['applied_coverages'],
    'failed_gates': result['decision']['failed_gates'],
    'packet_verification': verification,
}, indent=2, allow_nan=False))
"""
        ),
        markdown(
            """## Handoff

Accept the printed decision unchanged. Download the ZIP and this executed
notebook. Do not tune or rerun after inspecting metrics. A GO remains
development evidence and cannot be presented as independent confirmation.
"""
        ),
        code(
            """archive_base = Path('/content') / f'rars-v15-selective-gate-{V15_IMPLEMENTATION_COMMIT[:12]}'
archive_path = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=OUTPUT))
assert archive_path.is_file()
print('Archive:', archive_path, archive_path.stat().st_size, sha256_file(archive_path))

from google.colab import files
files.download(str(archive_path))
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name},
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
