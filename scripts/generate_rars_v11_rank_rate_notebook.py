#!/usr/bin/env python3
"""Generate the clean source-hash-pinned RARS-v11 diagnostic notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/MSMARCO_RARS_v11_Rank_Rate_Diagnostic.ipynb"
PINNED_SOURCES = (
    "protocols/rars_v11_rank_rate_diagnostic_v1.json",
    "scripts/rars_v11_rank_rate_core.py",
    "scripts/evaluate_rars_v11_rank_rate.py",
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
            """# MS MARCO RARS-v11 Rank–Rate Diagnostic

## TL;DR

This fixed architecture screen tests the bottleneck identified by V10. It does
not train a cutoff-aware method. Under the same frozen M32 IVF-PQ candidates,
`alpha=0.75`, Top-B `40`, and `16 B/document`, it compares packed rank-32 int4
and rank-32/rank-64 residual product codes against the rank-16 int8 PCA
sidecar. FP32 rank-32/rank-64 results are non-deployable capacity ceilings.

Run once. Do not edit ranks, product partitions, seeds, gates, output files, or
rerun after metrics appear.
"""
        ),
        markdown(
            """## Context & Methods

The primary decision is hierarchical:

1. If rank-64 FP32 does not establish at least `+0.005` Recall@10 over PCA
   rank-16 int8, stop global linear rank expansion.
2. If the rank ceiling passes but rank-64 RPQ does not improve PCA by at least
   `+0.003` while retaining half the ceiling, stop this 16-byte encoder.
3. Passing both only authorizes writing a separate CA-RPQ cutoff-development
   protocol on fresh development data.

The notebook uses only the historically opened `oracle_design` role. It never
reads a V9/V10 result packet, `future_method_holdout`, or `oracle_audit`.
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

ENV_ROOT = Path('/content/rars-v11-env')
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
V11_REPO = Path('/content/Embedding_Compression_for_RAG_Retrieval_rars_v11')
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
clone_exact(V11_REPO, resolved)
V11_IMPLEMENTATION_COMMIT = resolved

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

SOURCE_HASHES = {json.dumps(source_hashes, indent=4)}
for relative, expected in SOURCE_HASHES.items():
    actual = sha256_file(V11_REPO / relative)
    assert actual == expected, (relative, actual, expected)
print('Exact V11 run commit:', V11_IMPLEMENTATION_COMMIT)
print('All V11 source hashes verified; isolated NumPy/Torch/CUDA:', versions)
"""
        ),
        code(
            """V3_PROTOCOL = V3_REPO / 'protocols/rars_v3_oracle_first_feasibility_v1.json'
V11_PROTOCOL = V11_REPO / 'protocols/rars_v11_rank_rate_diagnostic_v1.json'
protocol = json.loads(V11_PROTOCOL.read_text())
assert protocol['status'] == 'FROZEN_BEFORE_FIRST_V11_DIAGNOSTIC_RUN'
assert protocol['data_policy']['configuration_policy']['fixed_sidecar_screen_count'] == 7
assert protocol['data_policy']['configuration_policy']['cutoff_training_performed'] is False
assert protocol['diagnostic_gate']['old_holdout_reuse_authorized'] is False
assert protocol['diagnostic_gate']['fresh_confirmation_access_authorized'] is False
assert protocol['rpq_training']['subquantizers'] == 16
assert protocol['rpq_training']['bits_per_subquantizer'] == 8
subprocess.run([
    EXPERIMENT_PYTHON, '-m', 'pytest', '-q',
    'tests/test_rars_v11_rank_rate_core.py',
    'tests/test_rars_v11_protocol_contract.py',
    'tests/test_evaluate_rars_v11_rank_rate_contract.py',
    'tests/test_rars_v8_cutoff_sidecar_core.py',
    'tests/test_verify_rars_v6_1m_headroom_packet.py',
], cwd=V11_REPO, check=True, env=EXPERIMENT_ENV)
print('V11 numerical, payload, and isolation contracts passed.')
"""
        ),
        markdown(
            """## Data

Verify the frozen 1M corpus/index artifacts, then rematerialize the historical
V2.2/V3 development bundle. Only `oracle_design` labels are opened. The future
role remains identity-only and `oracle_audit` remains unlabeled.
"""
        ),
        code(
            """DRIVE = Path('/content/drive/MyDrive/rag-pq-checkpoints')
CACHE = DRIVE / 'msmarco_basis_gate0_cache'
CLEAN = DRIVE / 'rars_clean_split_v1'
PCA = DRIVE / 'rars_pca_comparator_v1'
INDEX = DRIVE / 'msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index'
V6_PACKET = DRIVE / 'rars-v6-1m-headroom' / V6_IMPLEMENTATION_COMMIT[:12]
V11_ROOT = DRIVE / 'rars-v11-rank-rate' / V11_IMPLEMENTATION_COMMIT[:12]
DIAGNOSTIC = V11_ROOT / 'diagnostic-once'
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
assert not DIAGNOSTIC.exists() or not any(DIAGNOSTIC.iterdir()), (
    'V11 diagnostic-once is non-empty. Do not overwrite or rerun.'
)
v6_summary = json.loads(subprocess.check_output([
    EXPERIMENT_PYTHON, str(V11_REPO / 'scripts/verify_rars_v6_1m_headroom_packet.py'),
    '--packet-root', str(V6_PACKET),
], text=True, cwd=V11_REPO, env=EXPERIMENT_ENV))
assert v6_summary['status'] == 'RARS_V6_1M_HEADROOM_PACKET_VERIFIED'
assert v6_summary['formal_decision'] == 'GO_TO_V6_LOSS_IMPLEMENTATION'
print('Artifacts verified; V11 output is empty; no V9/V10 packet path was opened.')
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
    assert not ROLE_LABEL_FILES.intersection(path.name for path in (V3_BUNDLES / role).iterdir())
future_files = {path.name for path in (V3_BUNDLES / 'future_method_holdout').iterdir()}
assert future_files == {'query_manifest.json', 'v3_identity_manifest.json'}
subprocess.run([
    EXPERIMENT_PYTHON, str(V3_REPO / 'scripts/materialize_rars_v3_role_labels.py'),
    '--bundle-root', str(V3_BUNDLES),
    '--parent-inner-train-bundle', str(PARENT_BUNDLES / 'inner_train'),
    '--role', 'oracle_design',
    '--source-commit', V3_IMPLEMENTATION_COMMIT,
    '--protocol', str(V3_PROTOCOL),
], check=True, cwd=V3_REPO, env=EXPERIMENT_ENV)
labels = json.loads((V3_BUNDLES / 'oracle_design/v3_role_labels_manifest.json').read_text())
assert labels['status'] == 'ROLE_LABELS_MATERIALIZED_FROM_FROZEN_PARENT'
assert labels['role_id'] == 'oracle_design'
assert not ROLE_LABEL_FILES.intersection(path.name for path in (V3_BUNDLES / 'oracle_audit').iterdir())
assert {path.name for path in (V3_BUNDLES / 'future_method_holdout').iterdir()} == future_files
print('Only historical oracle_design labels materialized; protected roles remain unopened.')
"""
        ),
        markdown(
            """## Results

Run the complete fixed screen once. Product codebooks are trained only on
fixed PCA residual coefficients; no relevance/cutoff loss or basis learning is
performed. The CPU RPQ training may take several minutes even in a GPU runtime.
Do not interrupt or reuse a partially written output directory.
"""
        ),
        code(
            """DIAGNOSTIC.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    EXPERIMENT_PYTHON, str(V11_REPO / 'scripts/evaluate_rars_v11_rank_rate.py'),
    '--design-role-dir', str(V3_BUNDLES / 'oracle_design'),
    '--v6-packet-root', str(V6_PACKET),
    '--output-dir', str(DIAGNOSTIC),
    '--protocol', str(V11_PROTOCOL),
    '--source-commit', V11_IMPLEMENTATION_COMMIT,
], check=True, cwd=V11_REPO, env=EXPERIMENT_ENV)
print('Single V11 fixed rank-rate diagnostic completed.')
"""
        ),
        code(
            """complete = json.loads((DIAGNOSTIC / 'diagnostic_complete.json').read_text())
result = json.loads((DIAGNOSTIC / 'rank_rate_result.json').read_text())
freeze = json.loads((DIAGNOSTIC / 'diagnostic_freeze.json').read_text())
allowed = {
    'STOP_LINEAR_RANK_EXPANSION_NO_HEADROOM',
    'STOP_RPQ_16B_CANNOT_RETAIN_HEADROOM',
    'GO_TO_SEPARATE_CA_RPQ_CUTOFF_PROTOCOL',
}
assert complete['status'] == result['status'] == 'RARS_V11_RANK_RATE_DIAGNOSTIC_COMPLETE'
assert freeze['status'] == 'RARS_V11_RANK_RATE_DIAGNOSTIC_CLOSED'
assert complete['formal_decision'] == result['formal_decision'] == freeze['formal_decision']
assert result['formal_decision'] in allowed
assert result['v9_packet_opened'] is False
assert result['v10_packet_opened'] is False
assert result['future_method_holdout_opened'] is False
assert result['cutoff_training_performed'] is False
assert result['fresh_confirmation_access_authorized'] is False
for filename, record in complete['outputs'].items():
    path = DIAGNOSTIC / filename
    assert path.stat().st_size == record['bytes'], filename
    assert sha256_file(path) == record['sha256'], filename
report = {
    'formal_decision': result['formal_decision'],
    'metrics': result['metrics'],
    'comparisons': result['comparisons'],
    'rank64_rpq_fold_gains_over_pca': result['rank64_rpq_fold_gains_over_pca'],
    'rank64_headroom_retention_fraction': result['rank64_headroom_retention_fraction'],
    'rank64_rpq_candidate_gap_recovery_fraction': result['rank64_rpq_candidate_gap_recovery_fraction'],
    'rpq_diagnostics': result['rpq_diagnostics'],
    'failed_gates': result['decision']['failed_gates'],
    'rank_rate_result_sha256': sha256_file(DIAGNOSTIC / 'rank_rate_result.json'),
}
print(json.dumps(report, indent=2, allow_nan=False))
"""
        ),
        markdown(
            """## Takeaways and handoff

The printed formal decision must be accepted unchanged. A GO is not a CA-RPQ
success result; it permits only a new, separately frozen cutoff-development
protocol. A STOP closes the corresponding architecture path.

Download the ZIP and this executed notebook, return both for independent hash,
metric, payload, inference, and gate recomputation, and do not rerun.
"""
        ),
        code(
            """archive_base = Path('/content') / f'rars-v11-rank-rate-{V11_IMPLEMENTATION_COMMIT[:12]}'
archive_path = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=DIAGNOSTIC))
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
