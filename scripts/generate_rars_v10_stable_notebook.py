#!/usr/bin/env python3
"""Generate the clean source-hash-pinned RARS-v10 development notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/MSMARCO_RARS_v10_Stable_Sidecar_Development.ipynb"
PINNED_SOURCES = (
    "protocols/rars_v10_pca_anchored_harm_constrained_v1.json",
    "scripts/rars_v10_stable_core.py",
    "scripts/train_rars_v10_stable_sidecar.py",
    "scripts/rars_v8_cutoff_sidecar_core.py",
    "scripts/train_rars_v8_cutoff_sidecar.py",
    "scripts/rars_v3_oracle_core.py",
    "scripts/verify_rars_v6_1m_headroom_packet.py",
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
            """# MS MARCO RARS-v10 Stable Sidecar Development

## Goal

Run one post-confirmation development configuration: a PCA-anchored,
query-harm-constrained rank-16 int8 residual sidecar. The method keeps the
frozen M32 IVF-PQ index, original queries, alpha `0.75`, Top-B `40`, and one
`16 B/document` sidecar.

This is **outcome-informed development on the historically opened
`oracle_design` role**. V9 motivated the stability question, but this notebook
does not read any V9 file and can never reuse the 803-query role.
"""
        ),
        markdown(
            """## Frozen decision boundary

There is one configuration and two outcomes:

- `GO_TO_FRESH_EXTERNAL_V10_PROTOCOL`: every PCA-superiority, query-breadth,
  worst-fold, metric-guardrail, gradient-audit, and monotonicity gate passes.
- `STOP_V10_NO_STABLE_PCA_ADVANTAGE`: any gate fails.

Neither result authorizes opening an old holdout. A GO only permits writing a
new protocol for a genuinely fresh dataset/model. Do not edit objective
weights, rank, angle, seeds, steps, threshold, or checkpoint after execution.
"""
        ),
        code(
            f"""import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

REPO_URL = 'https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval.git'
SOURCE_BRANCH = 'codex/rars-v8-cutoff-sidecar'
TRAINING_COMMIT = 'bb9b106e69b9a453756fd800665f701614ce67b3'
V3_IMPLEMENTATION_COMMIT = '05c2ae43b7d11783460822d10c590240dab1a399'
V6_IMPLEMENTATION_COMMIT = '26a7717b964eed979b3bf7a3149d0d24e9bce3f1'

ENV_ROOT = Path('/content/rars-v10-env')
if ENV_ROOT.exists():
    shutil.rmtree(ENV_ROOT)
venv_result = subprocess.run([
    sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages',
    str(ENV_ROOT),
], text=True, capture_output=True)
if venv_result.returncode != 0:
    print(venv_result.stdout)
    print(venv_result.stderr, file=sys.stderr)
    venv_result.check_returncode()
EXPERIMENT_PYTHON = str(ENV_ROOT / 'bin/python')
subprocess.run([
    sys.executable, '-m', 'pip', '--python', EXPERIMENT_PYTHON,
    'install', '-q', 'numpy==1.26.4', 'faiss-gpu-cu12==1.12.0',
    'pytest>=8,<9',
], check=True)
EXPERIMENT_ENV = os.environ.copy()
EXPERIMENT_ENV['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
versions = subprocess.check_output([
    EXPERIMENT_PYTHON, '-c',
    'import numpy, torch; print(numpy.__version__); print(torch.__version__); print(torch.version.cuda)',
], text=True, env=EXPERIMENT_ENV).splitlines()
assert versions == ['1.26.4', '2.11.0+cu128', '12.8'], versions

TRAIN_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v2_2')
V3_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v3')
V10_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v10')
PARENT_WORK = Path('/content') / f'rars-v2.2-{{TRAINING_COMMIT[:12]}}'
V3_WORK = Path('/content') / f'rars-v3-{{V3_IMPLEMENTATION_COMMIT[:12]}}'
for work in (PARENT_WORK, V3_WORK):
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
PARENT_BUNDLES = PARENT_WORK / 'bundles'
PARENT_CANDIDATE_CACHE = PARENT_WORK / 'candidate-cache'
V3_BUNDLES = V3_WORK / 'bundles'

def clone_exact(destination, commit):
    if destination.exists():
        shutil.rmtree(destination)
    subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True).strip()
    assert actual == commit and not dirty, (actual, commit, dirty)

clone_exact(TRAIN_REPO, TRAINING_COMMIT)
clone_exact(V3_REPO, V3_IMPLEMENTATION_COMMIT)
resolved = subprocess.check_output([
    'git', 'ls-remote', REPO_URL, f'refs/heads/{{SOURCE_BRANCH}}'
], text=True).split()[0]
clone_exact(V10_REPO, resolved)
V10_IMPLEMENTATION_COMMIT = resolved

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

SOURCE_HASHES = {json.dumps(source_hashes, indent=4)}
for relative, expected in SOURCE_HASHES.items():
    actual = sha256_file(V10_REPO / relative)
    assert actual == expected, (relative, actual, expected)
print('Exact V10 run commit:', V10_IMPLEMENTATION_COMMIT)
print('All V10 source hashes verified; isolated NumPy/Torch/CUDA:', versions)
"""
        ),
        code(
            """V3_PROTOCOL = V3_REPO / 'protocols/rars_v3_oracle_first_feasibility_v1.json'
V10_PROTOCOL = V10_REPO / 'protocols/rars_v10_pca_anchored_harm_constrained_v1.json'
protocol = json.loads(V10_PROTOCOL.read_text())
assert protocol['status'] == 'FROZEN_BEFORE_FIRST_V10_DEVELOPMENT_RUN'
assert protocol['data_policy']['cross_validation']['configuration_count'] == 1
assert protocol['development_gate']['v9_reuse_authorized'] is False
assert protocol['method']['single_sidecar_only'] is True
assert protocol['optimization']['accepted_objective_must_be_monotone'] is True
assert protocol['avq_scalar_headroom_diagnostic']['diagnostic_only'] is True
assert protocol['avq_scalar_headroom_diagnostic']['codebook_training_performed'] is False
subprocess.run([
    EXPERIMENT_PYTHON, '-m', 'pytest', '-q',
    'tests/test_rars_v10_stable_core.py',
    'tests/test_rars_v10_protocol_contract.py',
    'tests/test_train_rars_v10_stable_sidecar_contract.py',
    'tests/test_rars_v8_cutoff_sidecar_core.py',
    'tests/test_verify_rars_v6_1m_headroom_packet.py',
], cwd=V10_REPO, check=True, env=EXPERIMENT_ENV)
print('V10 numerical and isolation contracts passed.')
"""
        ),
        code(
            """DRIVE = Path('/content/drive/MyDrive/rag-pq-checkpoints')
CACHE = DRIVE / 'msmarco_basis_gate0_cache'
CLEAN = DRIVE / 'rars_clean_split_v1'
PCA = DRIVE / 'rars_pca_comparator_v1'
INDEX = DRIVE / 'msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index'
V6_PACKET = DRIVE / 'rars-v6-1m-headroom' / V6_IMPLEMENTATION_COMMIT[:12]
V10_ROOT = DRIVE / 'rars-v10-stable-sidecar' / V10_IMPLEMENTATION_COMMIT[:12]
DEVELOPMENT = V10_ROOT / 'development-once'
required = [
    CACHE / 'embeddings.fp16.memmap',
    CACHE / 'doc_ids.int64.memmap',
    CACHE / 'query_vectors.fp32.npy',
    CACHE / 'qrels_subset.json',
    INDEX,
    PCA / 'bases/pca_unweighted_rank16.float32.npy',
    PCA / 'sidecars/scales_pca_rank16.float32.npy',
    PCA / 'sidecars/codes_pca_rank16.int8.memmap',
    CLEAN / 'selected_config.json',
    CLEAN / 'bases/score_error_weighted_rank16.npy',
    CLEAN / 'sidecars/scales_score_error_weighted_rank16.float32.npy',
    CLEAN / 'sidecars/codes_score_error_weighted_rank16.int8.memmap',
    V6_PACKET / 'headroom_result.json',
    V6_PACKET / 'headroom_complete.json',
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, {'missing_artifacts': missing}
assert shutil.disk_usage('/content').free >= 8_000_000_000, 'Need 8 GB local disk'
assert not DEVELOPMENT.exists() or not any(DEVELOPMENT.iterdir()), (
    'V10 development-once is non-empty. Do not overwrite or rerun.'
)
v6_summary = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V10_REPO / 'scripts/verify_rars_v6_1m_headroom_packet.py'),
    '--packet-root', str(V6_PACKET),
], text=True, cwd=V10_REPO, env=EXPERIMENT_ENV))
assert v6_summary['status'] == 'RARS_V6_1M_HEADROOM_PACKET_VERIFIED'
assert v6_summary['formal_decision'] == 'GO_TO_V6_LOSS_IMPLEMENTATION'
print('Artifacts verified; V10 output is empty; no V9 path was opened.')
"""
        ),
        markdown(
            """## Rebuild the historical development bundle

The next three cells reproduce the exact V2.2 parent bundle and qrels-free V3
role split used by V8. Only the already opened `oracle_design` labels are
materialized. `oracle_audit` remains unlabeled and `future_method_holdout`
remains identity-only. This is development rematerialization, not a new
confirmation outcome access.
"""
        ),
        code(
            """builder = [
    EXPERIMENT_PYTHON, str(TRAIN_REPO / 'scripts/build_msmarco_rars_v2_boundary_bundles.py'),
    '--inner-only',
    '--embeddings', str(CACHE / 'embeddings.fp16.memmap'),
    '--doc-ids', str(CACHE / 'doc_ids.int64.memmap'),
    '--query-vectors', str(CACHE / 'query_vectors.fp32.npy'),
    '--index', str(INDEX),
    '--qrels', str(CACHE / 'qrels_subset.json'),
    '--train-split', str(TRAIN_REPO / 'splits/msmarco_rars_train_split.json'),
    '--validation-split', str(TRAIN_REPO / 'splits/msmarco_rars_validation_split.json'),
    '--cache-root', str(PARENT_CANDIDATE_CACHE),
    '--pca-config', str(TRAIN_REPO / 'results/rars_pca_comparator/selected_pca_config.json'),
    '--pca-basis', str(PCA / 'bases/pca_unweighted_rank16.float32.npy'),
    '--pca-scales', str(PCA / 'sidecars/scales_pca_rank16.float32.npy'),
    '--pca-codes', str(PCA / 'sidecars/codes_pca_rank16.int8.memmap'),
    '--rars-config', str(CLEAN / 'selected_config.json'),
    '--rars-basis', str(CLEAN / 'bases/score_error_weighted_rank16.npy'),
    '--rars-scales', str(CLEAN / 'sidecars/scales_score_error_weighted_rank16.float32.npy'),
    '--rars-codes', str(CLEAN / 'sidecars/codes_score_error_weighted_rank16.int8.memmap'),
    '--output-root', str(PARENT_BUNDLES),
    '--residual-batch-size', '20000',
]
subprocess.run(builder, check=True, cwd=TRAIN_REPO, env=EXPERIMENT_ENV)
subprocess.run([
    EXPERIMENT_PYTHON, str(TRAIN_REPO / 'scripts/freeze_rars_v2_2_inner_bundles.py'),
    '--bundle-root', str(PARENT_BUNDLES),
    '--query-vectors', str(CACHE / 'query_vectors.fp32.npy'),
    '--train-split', str(TRAIN_REPO / 'splits/msmarco_rars_train_split.json'),
    '--outer-validation-split', str(TRAIN_REPO / 'splits/msmarco_rars_validation_split.json'),
    '--clean-test-split', str(TRAIN_REPO / 'splits/msmarco_rars_test_split.json'),
    '--source-commit', TRAINING_COMMIT,
], check=True, cwd=TRAIN_REPO, env=EXPERIMENT_ENV)
print('Exact V2.2 parent development bundle rematerialized.')
"""
        ),
        code(
            """subprocess.run([
    EXPERIMENT_PYTHON, str(V3_REPO / 'scripts/build_msmarco_rars_v3_oracle_bundles.py'),
    '--parent-inner-train-bundle', str(PARENT_BUNDLES / 'inner_train'),
    '--doc-ids', str(CACHE / 'doc_ids.int64.memmap'),
    '--output-root', str(V3_BUNDLES),
    '--protocol', str(V3_PROTOCOL),
    '--source-commit', V3_IMPLEMENTATION_COMMIT,
    '--n-docs', '1000000',
], check=True, cwd=V3_REPO, env=EXPERIMENT_ENV)
summary = json.loads((V3_BUNDLES / 'v3_oracle_bundle_freeze_summary.json').read_text())
assert summary['status'] == 'V3_QRELS_FREE_CANDIDATE_BUNDLES_FROZEN'
assert summary['qrels_opened_or_parsed'] is False
ROLE_LABEL_FILES = {
    'candidate_relevance.uint8.npy', 'relevant_counts.int32.npy',
    'v3_role_labels_started.json', 'v3_role_labels_manifest.json',
}
for role in ('oracle_design', 'oracle_audit'):
    assert not ROLE_LABEL_FILES.intersection(
        path.name for path in (V3_BUNDLES / role).iterdir()
    )
future_files = {path.name for path in (V3_BUNDLES / 'future_method_holdout').iterdir()}
assert future_files == {'query_manifest.json', 'v3_identity_manifest.json'}
print('V3 candidates frozen; future role remains identity-only.')
"""
        ),
        code(
            """subprocess.run([
    EXPERIMENT_PYTHON, str(V3_REPO / 'scripts/materialize_rars_v3_role_labels.py'),
    '--bundle-root', str(V3_BUNDLES),
    '--parent-inner-train-bundle', str(PARENT_BUNDLES / 'inner_train'),
    '--role', 'oracle_design',
    '--source-commit', V3_IMPLEMENTATION_COMMIT,
    '--protocol', str(V3_PROTOCOL),
], check=True, cwd=V3_REPO, env=EXPERIMENT_ENV)
labels = json.loads(
    (V3_BUNDLES / 'oracle_design/v3_role_labels_manifest.json').read_text()
)
assert labels['status'] == 'ROLE_LABELS_MATERIALIZED_FROM_FROZEN_PARENT'
assert labels['role_id'] == 'oracle_design'
assert labels['label_source']['qrels_opened_or_parsed'] is False
assert not ROLE_LABEL_FILES.intersection(
    path.name for path in (V3_BUNDLES / 'oracle_audit').iterdir()
)
assert {path.name for path in (V3_BUNDLES / 'future_method_holdout').iterdir()} == future_files
print('Only historical oracle_design labels materialized.')
"""
        ),
        markdown(
            """## Run the single V10 development configuration

This cell first performs finite-difference gradient audits, then fits five OOF
bases plus one final development basis. Every accepted QR-retracted step must
reduce the registered objective and remain within 20 degrees of PCA. Do not
interrupt, modify the output directory, or rerun after metrics appear.
"""
        ),
        code(
            """DEVELOPMENT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    EXPERIMENT_PYTHON, str(V10_REPO / 'scripts/train_rars_v10_stable_sidecar.py'),
    '--design-role-dir', str(V3_BUNDLES / 'oracle_design'),
    '--v6-packet-root', str(V6_PACKET),
    '--output-dir', str(DEVELOPMENT),
    '--protocol', str(V10_PROTOCOL),
    '--source-commit', V10_IMPLEMENTATION_COMMIT,
], check=True, cwd=V10_REPO, env=EXPERIMENT_ENV)
print('Single V10 development run completed.')
"""
        ),
        code(
            """complete = json.loads((DEVELOPMENT / 'development_complete.json').read_text())
result = json.loads((DEVELOPMENT / 'development_result.json').read_text())
freeze = json.loads((DEVELOPMENT / 'method_freeze.json').read_text())
audit = json.loads((DEVELOPMENT / 'optimizer_audit.json').read_text())
allowed = {
    'GO_TO_FRESH_EXTERNAL_V10_PROTOCOL',
    'STOP_V10_NO_STABLE_PCA_ADVANTAGE',
}
assert complete['status'] == result['status'] == 'RARS_V10_DEVELOPMENT_COMPLETE'
assert freeze['status'] == 'RARS_V10_METHOD_CLOSED_AFTER_SINGLE_DEVELOPMENT_RUN'
assert complete['formal_decision'] == result['formal_decision'] == freeze['formal_decision']
assert result['formal_decision'] in allowed
assert complete['configuration_count'] == 1
assert complete['v9_files_opened'] is False
assert complete['future_method_holdout_opened'] is False
assert result['fresh_external_access_authorized'] is False
assert freeze['v9_reuse_authorized'] is False
assert audit['all_gradient_audits_pass'] is True
assert audit['all_accepted_losses_monotone'] is True
for filename, record in complete['outputs'].items():
    path = DEVELOPMENT / filename
    assert path.stat().st_size == record['bytes'], filename
    assert sha256_file(path) == record['sha256'], filename
report = {
    'formal_decision': result['formal_decision'],
    'metrics': result['metrics'],
    'comparisons': result['comparisons'],
    'fold_gains_over_pca': result['fold_gains_over_pca'],
    'candidate_gap_recovery_fraction': result['candidate_gap_recovery_fraction'],
    'avq_scalar_headroom_diagnostic': result['avq_scalar_headroom_diagnostic'],
    'optimizer': result['optimizer'],
    'failed_gates': result['decision']['failed_gates'],
    'development_result_sha256': sha256_file(DEVELOPMENT / 'development_result.json'),
    'method_freeze_sha256': sha256_file(DEVELOPMENT / 'method_freeze.json'),
}
print(json.dumps(report, indent=2, allow_nan=False))
"""
        ),
        markdown(
            """## Return the complete packet

Download the ZIP and the executed notebook. Return both for independent hash,
gradient, metric, and gate recomputation. Do not tune V10 from this output. A
GO requires a new external dataset/model protocol; a STOP closes this method.
"""
        ),
        code(
            """archive_base = Path('/content') / f'rars-v10-development-{V10_IMPLEMENTATION_COMMIT[:12]}'
archive_path = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=DEVELOPMENT))
print('Download:', archive_path, archive_path.stat().st_size, 'bytes')
from google.colab import files
files.download(str(archive_path))
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
