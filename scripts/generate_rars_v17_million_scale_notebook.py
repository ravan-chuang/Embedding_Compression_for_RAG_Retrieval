#!/usr/bin/env python3
"""Generate the source-pinned Colab runner for the RARS-v17 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


PLACEHOLDER = "PASTE_FULL_40_CHARACTER_COMMIT_HERE"


def markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(implementation_commit: str = PLACEHOLDER) -> dict[str, Any]:
    cells = [
        markdown(
            """# RARS-v17 Million-Scale Setting-Transfer Diagnostic

V16 stopped before metrics because FiQA and SciFact were below the required
million-document scale. V17 uses two already materialized frozen settings:

- MS MARCO deterministic 1M corpus;
- full BEIR Natural Questions corpus (about 2.68M documents).

This is an outcome-informed mechanism diagnostic. The NQ official test role was
opened by the earlier one-shot confirmation and is reused only as development
evidence. The notebook must not be described as a new independent confirmation
or a pure domain-shift causal test.
"""
        ),
        code(
            """# Cell 1 — immutable execution configuration
import re
from pathlib import Path

V17_IMPLEMENTATION_COMMIT = "__V17_IMPLEMENTATION_COMMIT__"
assert re.fullmatch(r"[0-9a-f]{40}", V17_IMPLEMENTATION_COMMIT), (
    "Paste the full pushed V17 implementation commit before running."
)

MSMARCO_PARENT_COMMIT = "bb9b106e69b9a453756fd800665f701614ce67b3"
REPO_URL = "https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval.git"
DRIVE_ROOT = Path("/content/drive/MyDrive/rag-pq-checkpoints")
RUN_ROOT = DRIVE_ROOT / "rars-v17-million-scale" / V17_IMPLEMENTATION_COMMIT[:12]
BUNDLE_ROOT = RUN_ROOT / "bundles"
NQ_PREP_ROOT = RUN_ROOT / "prepared-nq"
OUTPUT = RUN_ROOT / "diagnostic"
NQ_ARTIFACT_ROOT = Path("/content/drive/MyDrive/rars-beir-nq-confirmation-v2")

V17_REPO = Path("/content") / (
    "Embedding_Compression_for_RAG_Retrieval_v17_"
    + V17_IMPLEMENTATION_COMMIT[:12]
)
PARENT_REPO = Path("/content/Embedding_Compression_for_RAG_Retrieval_rars_v2_2")
PARENT_WORK = Path("/content") / f"rars-v2.2-{MSMARCO_PARENT_COMMIT[:12]}"
PARENT_BUNDLES = PARENT_WORK / "bundles"
PARENT_CANDIDATE_CACHE = PARENT_WORK / "candidate-cache"
ENV_ROOT = Path("/content/rars-v17-env")

CACHE = DRIVE_ROOT / "msmarco_basis_gate0_cache"
CLEAN = DRIVE_ROOT / "rars_clean_split_v1"
PCA = DRIVE_ROOT / "rars_pca_comparator_v1"
MSMARCO_INDEX = (
    DRIVE_ROOT
    / "msmarco_1m_pq_residual_gate3/frozen_ivfpq_m32_nlist512.index"
)
print("RUN_ROOT:", RUN_ROOT)
""".replace("__V17_IMPLEMENTATION_COMMIT__", implementation_commit)
        ),
        code(
            """# Cell 2 — mount Drive, isolate packages, and checkout exact sources
from google.colab import drive
drive.mount("/content/drive")

import json, os, shutil, subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "virtualenv>=20.26,<21"],
    check=True,
)
if ENV_ROOT.exists():
    shutil.rmtree(ENV_ROOT)
subprocess.run(
    [
        sys.executable, "-m", "virtualenv", "--system-site-packages",
        str(ENV_ROOT),
    ],
    check=True,
)
EXPERIMENT_PYTHON = str(ENV_ROOT / "bin/python")
subprocess.run(
    [
        EXPERIMENT_PYTHON, "-m", "pip", "install", "-q",
        "numpy==1.26.4", "faiss-gpu-cu12==1.12.0", "pytest>=8,<9",
    ],
    check=True,
)

def clone_exact(destination, commit):
    if destination.exists():
        shutil.rmtree(destination)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout",
         REPO_URL, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "origin", commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", commit],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain"], text=True
    ).strip()
    assert actual == commit, (actual, commit)
    assert not dirty, dirty

clone_exact(V17_REPO, V17_IMPLEMENTATION_COMMIT)
clone_exact(PARENT_REPO, MSMARCO_PARENT_COMMIT)
subprocess.run(
    [
        EXPERIMENT_PYTHON, "-c",
        "import numpy, faiss; "
        "assert numpy.__version__ == '1.26.4'; "
        "assert faiss.__version__ == '1.12.0'; "
        "print('numpy', numpy.__version__, 'faiss', faiss.__version__)",
    ],
    check=True,
)
"""
        ),
        code(
            """# Cell 3 — verify required million-scale source artifacts
required = [
    CACHE / "embeddings.fp16.memmap",
    CACHE / "doc_ids.int64.memmap",
    CACHE / "query_vectors.fp32.npy",
    CACHE / "qrels_subset.json",
    MSMARCO_INDEX,
    PCA / "bases/pca_unweighted_rank16.float32.npy",
    PCA / "sidecars/scales_pca_rank16.float32.npy",
    PCA / "sidecars/codes_pca_rank16.int8.memmap",
    CLEAN / "selected_config.json",
    CLEAN / "bases/score_error_weighted_rank16.npy",
    CLEAN / "sidecars/scales_score_error_weighted_rank16.float32.npy",
    CLEAN / "sidecars/codes_score_error_weighted_rank16.int8.memmap",
    NQ_ARTIFACT_ROOT / "stage1/corpus/corpus_artifacts_manifest.json",
    NQ_ARTIFACT_ROOT / "stage1/index/index_manifest.json",
    NQ_ARTIFACT_ROOT / "stage3/audit/eligible_test_query_audit.json",
    NQ_ARTIFACT_ROOT / "stage3/evaluation/evaluation_complete.json",
    NQ_ARTIFACT_ROOT / "stage3/evaluation/metrics_summary.json",
    NQ_ARTIFACT_ROOT / "stage3/evaluation/per_query_metrics.csv",
    NQ_ARTIFACT_ROOT / "stage3/evaluation/test_query_vectors.float32.npy",
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, {"missing_million_scale_artifacts": missing}
assert shutil.disk_usage("/content").free >= 4_000_000_000, (
    "Need at least 4 GB free local scratch space."
)
print("All MS MARCO 1M and full NQ artifacts are present.")
"""
        ),
        code(
            """# Cell 4 — run V17 source contracts before metric-bearing work
v17_tests = [
    "tests/test_rars_v17_million_scale_core.py",
    "tests/test_prepare_rars_v17_nq_roles.py",
    "tests/test_build_rars_v17_setting_bundle_contract.py",
    "tests/test_freeze_rars_v17_setting_manifest_contract.py",
    "tests/test_evaluate_rars_v17_million_scale_contract.py",
    "tests/test_rars_v17_million_scale_protocol_contract.py",
    "tests/test_rars_v17_million_scale_notebook_contract.py",
]
subprocess.run(
    [EXPERIMENT_PYTHON, "-m", "pytest", "-q", *v17_tests],
    cwd=V17_REPO,
    check=True,
)
"""
        ),
        code(
            """# Cell 5 — rematerialize only opened MS MARCO development bundles
if PARENT_WORK.exists():
    shutil.rmtree(PARENT_WORK)
PARENT_WORK.mkdir(parents=True)
builder = [
    EXPERIMENT_PYTHON,
    str(PARENT_REPO / "scripts/build_msmarco_rars_v2_boundary_bundles.py"),
    "--inner-only",
    "--embeddings", str(CACHE / "embeddings.fp16.memmap"),
    "--doc-ids", str(CACHE / "doc_ids.int64.memmap"),
    "--query-vectors", str(CACHE / "query_vectors.fp32.npy"),
    "--index", str(MSMARCO_INDEX),
    "--qrels", str(CACHE / "qrels_subset.json"),
    "--train-split", str(PARENT_REPO / "splits/msmarco_rars_train_split.json"),
    "--validation-split",
    str(PARENT_REPO / "splits/msmarco_rars_validation_split.json"),
    "--cache-root", str(PARENT_CANDIDATE_CACHE),
    "--pca-config",
    str(PARENT_REPO / "results/rars_pca_comparator/selected_pca_config.json"),
    "--pca-basis", str(PCA / "bases/pca_unweighted_rank16.float32.npy"),
    "--pca-scales",
    str(PCA / "sidecars/scales_pca_rank16.float32.npy"),
    "--pca-codes",
    str(PCA / "sidecars/codes_pca_rank16.int8.memmap"),
    "--rars-config", str(CLEAN / "selected_config.json"),
    "--rars-basis", str(CLEAN / "bases/score_error_weighted_rank16.npy"),
    "--rars-scales",
    str(CLEAN / "sidecars/scales_score_error_weighted_rank16.float32.npy"),
    "--rars-codes",
    str(CLEAN / "sidecars/codes_score_error_weighted_rank16.int8.memmap"),
    "--output-root", str(PARENT_BUNDLES),
    "--residual-batch-size", "20000",
]
subprocess.run(builder, cwd=PARENT_REPO, check=True)
subprocess.run(
    [
        EXPERIMENT_PYTHON,
        str(PARENT_REPO / "scripts/freeze_rars_v2_2_inner_bundles.py"),
        "--bundle-root", str(PARENT_BUNDLES),
        "--query-vectors", str(CACHE / "query_vectors.fp32.npy"),
        "--train-split",
        str(PARENT_REPO / "splits/msmarco_rars_train_split.json"),
        "--outer-validation-split",
        str(PARENT_REPO / "splits/msmarco_rars_validation_split.json"),
        "--clean-test-split",
        str(PARENT_REPO / "splits/msmarco_rars_test_split.json"),
        "--source-commit", MSMARCO_PARENT_COMMIT,
    ],
    cwd=PARENT_REPO,
    check=True,
)
"""
        ),
        code(
            """# Cell 6 — adapt the MS MARCO 1M roles to the V17 contract
PROTOCOL = (
    V17_REPO / "protocols/rars_v17_million_scale_setting_transfer_v1.json"
)
MSMARCO_SETTING = "msmarco_1m_bge_opened_development"
for role, parent_role in (
    ("fit", "inner_train"),
    ("evaluation", "inner_validation"),
):
    output_dir = BUNDLE_ROOT / MSMARCO_SETTING / role
    complete = output_dir / "bundle_complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text())
        assert payload["status"] == "RARS_V17_DOMAIN_BUNDLE_COMPLETE"
        print("Reusing:", output_dir)
        continue
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            EXPERIMENT_PYTHON,
            str(V17_REPO / "scripts/adapt_rars_v17_msmarco_bundle.py"),
            "--parent-bundle", str(PARENT_BUNDLES / parent_role),
            "--role", role,
            "--output-dir", str(output_dir),
            "--protocol", str(PROTOCOL),
            "--source-commit", V17_IMPLEMENTATION_COMMIT,
        ],
        cwd=V17_REPO,
        check=True,
    )
"""
        ),
        code(
            """# Cell 7 — freeze deterministic opened-NQ fit/evaluation roles
NQ_PREP_COMPLETE = NQ_PREP_ROOT / "preparation_complete.json"
if not NQ_PREP_COMPLETE.exists():
    NQ_PREP_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            EXPERIMENT_PYTHON,
            str(V17_REPO / "scripts/prepare_rars_v17_nq_roles.py"),
            "--artifact-root", str(NQ_ARTIFACT_ROOT),
            "--output-root", str(NQ_PREP_ROOT),
            "--protocol", str(PROTOCOL),
            "--source-commit", V17_IMPLEMENTATION_COMMIT,
        ],
        cwd=V17_REPO,
        check=True,
    )
nq_preparation = json.loads(NQ_PREP_COMPLETE.read_text())
assert nq_preparation["status"] == "RARS_V17_NQ_ROLES_PREPARED"
assert nq_preparation["document_count"] >= 2_000_000
assert nq_preparation["prior_confirmation_outcomes_known"] is True
print(json.dumps(nq_preparation, indent=2)[:6000])
"""
        ),
        code(
            """# Cell 8 — build full-NQ metric-free candidate/residual bundles
NQ_SETTING = "beir_nq_2_68m_bge_opened_test_diagnostic"
NQ_SETTING_DIR = NQ_PREP_ROOT / NQ_SETTING
nq_prepared = json.loads((NQ_SETTING_DIR / "prepared_domain.json").read_text())
for role in ("fit", "evaluation"):
    output_dir = BUNDLE_ROOT / NQ_SETTING / role
    complete = output_dir / "bundle_complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text())
        assert payload["status"] == "RARS_V17_DOMAIN_BUNDLE_COMPLETE"
        print("Reusing:", output_dir)
        continue
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            EXPERIMENT_PYTHON,
            str(V17_REPO / "scripts/build_rars_v17_setting_bundle.py"),
            "--domain-id", NQ_SETTING,
            "--encoder-id", nq_prepared["encoder"]["id"],
            "--encoder-revision", nq_prepared["encoder"]["revision"],
            "--evidence-role", role,
            "--query-ids", str(NQ_SETTING_DIR / role / "query_ids.utf8.txt"),
            "--query-vectors",
            str(NQ_SETTING_DIR / role / "query_vectors.float32.npy"),
            "--qrels-rows", str(NQ_SETTING_DIR / role / "qrels_rows.json"),
            "--embeddings", nq_prepared["embeddings"]["path"],
            "--embeddings-dtype", "float16",
            "--index", nq_prepared["index"]["path"],
            "--output-dir", str(output_dir),
            "--protocol", str(PROTOCOL),
            "--source-commit", V17_IMPLEMENTATION_COMMIT,
            "--document-count", str(nq_prepared["document_count"]),
            "--dimension", "384",
            "--nprobe", "32",
            "--candidate-pool", "100",
            "--fold-count", "5",
        ],
        cwd=V17_REPO,
        check=True,
    )
"""
        ),
        code(
            """# Cell 9 — freeze all four role bundles before fitting or metrics
SETTING_MANIFEST = BUNDLE_ROOT / "setting_manifest.json"
if not SETTING_MANIFEST.exists():
    subprocess.run(
        [
            EXPERIMENT_PYTHON,
            str(V17_REPO / "scripts/freeze_rars_v17_setting_manifest.py"),
            "--msmarco-fit", str(BUNDLE_ROOT / MSMARCO_SETTING / "fit"),
            "--msmarco-evaluation",
            str(BUNDLE_ROOT / MSMARCO_SETTING / "evaluation"),
            "--nq-fit", str(BUNDLE_ROOT / NQ_SETTING / "fit"),
            "--nq-evaluation", str(BUNDLE_ROOT / NQ_SETTING / "evaluation"),
            "--output", str(SETTING_MANIFEST),
            "--source-commit", V17_IMPLEMENTATION_COMMIT,
        ],
        cwd=V17_REPO,
        check=True,
    )
setting_manifest = json.loads(SETTING_MANIFEST.read_text())
assert setting_manifest["status"] == "V17_DOMAIN_BUNDLES_FROZEN"
assert setting_manifest["minimum_document_count_verified"] is True
print(json.dumps(setting_manifest, indent=2)[:6000])
"""
        ),
        code(
            """# Cell 10 — execute the one-shot million-scale decomposition
COMPLETE = OUTPUT / "diagnostic_complete.json"
if COMPLETE.exists():
    payload = json.loads(COMPLETE.read_text())
    assert payload["status"] == "RARS_V17_MECHANISM_DIAGNOSTIC_COMPLETE"
    print("Reusing complete diagnostic:", COMPLETE)
else:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            EXPERIMENT_PYTHON,
            str(V17_REPO / "scripts/evaluate_rars_v17_million_scale.py"),
            "--domain-manifest", str(SETTING_MANIFEST),
            "--output-dir", str(OUTPUT),
            "--protocol", str(PROTOCOL),
            "--source-commit", V17_IMPLEMENTATION_COMMIT,
        ],
        cwd=V17_REPO,
        check=True,
    )
"""
        ),
        code(
            """# Cell 11 — inspect the frozen decision without retuning
RESULT = json.loads((OUTPUT / "diagnostic_result.json").read_text())
assert RESULT["status"] == "RARS_V17_MECHANISM_DIAGNOSTIC_COMPLETE"
assert RESULT["confirmatory_claim_allowed"] is False
assert RESULT["nq_prior_confirmation_outcomes_known"] is True
print("Decision:", RESULT["diagnostic_decision"]["decision"])
print("\\nEqual-setting descriptive summary:")
print(json.dumps(RESULT["equal_domain_factor_summary"], indent=2))
for setting_id, row in RESULT["domains"].items():
    print("\\nSETTING:", setting_id)
    print("Documents:", row["document_count"])
    print("Index:", row["index_contract"])
    print("Base:", row["mean_metrics"]["base"]["recall"])
    print("Exact ceiling:", row["mean_metrics"]["same_candidate_exact"]["recall"])
    for name, contrast in row["comparisons"].items():
        print(
            f"{name:24s} delta={contrast['mean_difference']:+.6f} "
            f"CI=[{contrast['lower']:+.6f}, {contrast['upper']:+.6f}] "
            f"p={contrast['randomization_p_value']:.6g}"
        )
"""
        ),
        code(
            """# Cell 12 — package the complete packet for local audit
archive_base = (
    Path("/content")
    / f"rars-v17-million-scale-{V17_IMPLEMENTATION_COMMIT[:12]}"
)
archive = shutil.make_archive(str(archive_base), "zip", root_dir=RUN_ROOT)
print("Archive:", archive)

from google.colab import files
files.download(archive)
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "RARS_V17_Million_Scale_Setting_Transfer.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", default=PLACEHOLDER)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.implementation_commit != PLACEHOLDER and not re.fullmatch(
        r"[0-9a-f]{40}", args.implementation_commit
    ):
        raise ValueError("--implementation-commit must be exact lowercase 40-hex")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            build_notebook(args.implementation_commit),
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
