#!/usr/bin/env python3
"""Generate the Colab runner for the RARS-v16 mechanism diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


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


def build_notebook(
    implementation_commit: str = "PASTE_FULL_40_CHARACTER_COMMIT_HERE",
) -> dict[str, Any]:
    cells = [
        markdown(
            """# RARS-v16 Same-Encoder Mechanism Decomposition

This outcome-informed development diagnostic uses the same pinned BGE encoder
on FiQA and SciFact. It separates frozen-candidate headroom, rank capacity,
int8 coding loss, cutoff-aware objective value, fit-domain interaction, and
pooled-fit repair.

It is **not** an unseen confirmation and cannot support universal
generalization claims. Run top-to-bottom on a Colab GPU runtime. The only
manual edit is the full implementation commit in Cell 1 after that commit has
been pushed.
"""
        ),
        code(
            """# Cell 1 — immutable execution configuration
import re
from pathlib import Path

V16_IMPLEMENTATION_COMMIT = "__V16_IMPLEMENTATION_COMMIT__"
assert re.fullmatch(r"[0-9a-f]{40}", V16_IMPLEMENTATION_COMMIT), (
    "Paste the full pushed V16 implementation commit before running."
)

REPO_URL = "https://github.com/ravan-chuang/Embedding_Compression_for_RAG_Retrieval.git"
DRIVE_ROOT = Path("/content/drive/MyDrive/rag-pq-checkpoints")
RUN_ROOT = DRIVE_ROOT / "rars-v16-mechanism" / V16_IMPLEMENTATION_COMMIT[:12]
PREP_ROOT = RUN_ROOT / "prepared-domains"
BUNDLE_ROOT = RUN_ROOT / "bundles"
OUTPUT = RUN_ROOT / "diagnostic"
CACHE_ROOT = DRIVE_ROOT / "beir_data"
CLONE = Path("/content") / f"Embedding_Compression_for_RAG_Retrieval_v16_{V16_IMPLEMENTATION_COMMIT[:12]}"
ENV_ROOT = Path("/content/rars-v16-env")
print("RUN_ROOT:", RUN_ROOT)
""".replace("__V16_IMPLEMENTATION_COMMIT__", implementation_commit)
        ),
        code(
            """# Cell 2 — mount Drive, create isolated environment, and checkout exact source
from google.colab import drive
drive.mount("/content/drive")

import json, os, shutil, subprocess, sys
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "virtualenv>=20.26,<21"],
    check=True,
)
if ENV_ROOT.exists():
    shutil.rmtree(ENV_ROOT)
subprocess.run(
    [sys.executable, "-m", "virtualenv", "--system-site-packages", str(ENV_ROOT)],
    check=True,
)
EXPERIMENT_PYTHON = str(ENV_ROOT / "bin/python")
subprocess.run(
    [
        EXPERIMENT_PYTHON, "-m", "pip", "install", "-q",
        "numpy==1.26.4", "faiss-cpu==1.12.0",
        "sentence-transformers==3.4.1", "transformers==4.48.3",
        "huggingface-hub==0.28.1", "pytest>=8,<9",
    ],
    check=True,
)

if CLONE.exists():
    shutil.rmtree(CLONE)
subprocess.run(["git", "clone", "--filter=blob:none", REPO_URL, str(CLONE)], check=True)
subprocess.run(
    ["git", "-C", str(CLONE), "fetch", "origin", V16_IMPLEMENTATION_COMMIT],
    check=True,
)
subprocess.run(
    ["git", "-C", str(CLONE), "checkout", "--detach", V16_IMPLEMENTATION_COMMIT],
    check=True,
)
actual = subprocess.check_output(
    ["git", "-C", str(CLONE), "rev-parse", "HEAD"], text=True
).strip()
assert actual == V16_IMPLEMENTATION_COMMIT
assert not subprocess.check_output(
    ["git", "-C", str(CLONE), "status", "--porcelain"], text=True
).strip()
subprocess.run(
    [
        EXPERIMENT_PYTHON, "-c",
        "import importlib.metadata, numpy, faiss, torch; "
        "assert numpy.__version__ == '1.26.4'; "
        "assert importlib.metadata.version('sentence-transformers') == '3.4.1'; "
        "assert importlib.metadata.version('transformers') == '4.48.3'; "
        "assert importlib.metadata.version('huggingface-hub') == '0.28.1'; "
        "print('numpy', numpy.__version__, 'faiss', faiss.__version__, "
        "'torch', torch.__version__, 'cuda', torch.cuda.is_available())",
    ],
    check=True,
)
"""
        ),
        code(
            """# Cell 3 — run source contracts before spending GPU time
subprocess.run(
    [
        EXPERIMENT_PYTHON, "-m", "pytest", "-q",
        "tests/test_rars_v16_causal_generalization_core.py",
        "tests/test_prepare_rars_v16_beir_domains_contract.py",
        "tests/test_build_rars_v16_domain_bundle_contract.py",
        "tests/test_freeze_rars_v16_domain_manifest_contract.py",
        "tests/test_evaluate_rars_v16_causal_generalization_contract.py",
        "tests/test_rars_v16_causal_generalization_protocol_contract.py",
    ],
    cwd=CLONE,
    check=True,
)
"""
        ),
        code(
            """# Cell 4 — prepare the two same-encoder BEIR domains (long GPU cell)
PREP_COMPLETE = PREP_ROOT / "preparation_complete.json"
if PREP_COMPLETE.exists():
    existing = json.loads(PREP_COMPLETE.read_text())
    assert existing["status"] == "RARS_V16_SAME_ENCODER_DOMAIN_INPUTS_PREPARED"
    print("Reusing complete prepared inputs:", PREP_COMPLETE)
else:
    PREP_ROOT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        EXPERIMENT_PYTHON,
        str(CLONE / "scripts/prepare_rars_v16_beir_domains.py"),
        "--cache-root", str(CACHE_ROOT),
        "--output-root", str(PREP_ROOT),
        "--seed", "20261600",
        "--nlist", "128",
        "--nprobe", "16",
        "--subquantizers", "32",
        "--nbits", "8",
    ]
    process = subprocess.Popen(
        command,
        cwd=CLONE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_tail = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_tail.append(line)
        if len(output_tail) > 400:
            output_tail.pop(0)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(
            "V16 preparation failed with return code "
            f"{return_code}. Last combined output:\\n"
            + "".join(output_tail)
        )
print(json.dumps(json.loads(PREP_COMPLETE.read_text()), indent=2)[:5000])
"""
        ),
        code(
            """# Cell 5 — build the four metric-free candidate/residual bundles
PROTOCOL = CLONE / "protocols/rars_v16_causal_generalization_diagnostic_v1.json"
DOMAIN_IDS = ("fiqa_bge_same_encoder", "scifact_bge_same_encoder")
for domain_id in DOMAIN_IDS:
    prepared_dir = PREP_ROOT / domain_id
    prepared = json.loads((prepared_dir / "prepared_domain.json").read_text())
    for role in ("fit", "evaluation"):
        output_dir = BUNDLE_ROOT / domain_id / role
        complete = output_dir / "bundle_complete.json"
        if complete.exists():
            assert json.loads(complete.read_text())["status"] == "RARS_V16_DOMAIN_BUNDLE_COMPLETE"
            print("Reusing:", output_dir)
            continue
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                EXPERIMENT_PYTHON,
                str(CLONE / "scripts/build_rars_v16_domain_bundle.py"),
                "--domain-id", domain_id,
                "--encoder-id", prepared["encoder"]["id"],
                "--encoder-revision", prepared["encoder"]["revision"],
                "--evidence-role", role,
                "--query-ids", str(prepared_dir / role / "query_ids.utf8.txt"),
                "--query-vectors", str(prepared_dir / role / "query_vectors.float32.npy"),
                "--qrels-rows", str(prepared_dir / role / "qrels_rows.json"),
                "--embeddings", str(prepared_dir / "embeddings.float16.memmap"),
                "--embeddings-dtype", "float16",
                "--index", str(prepared_dir / "frozen_ivfpq.index"),
                "--output-dir", str(output_dir),
                "--protocol", str(PROTOCOL),
                "--source-commit", V16_IMPLEMENTATION_COMMIT,
                "--document-count", str(prepared["document_count"]),
                "--dimension", "384",
                "--nprobe", "16",
                "--candidate-pool", "100",
                "--fold-count", "5",
            ],
            cwd=CLONE,
            check=True,
        )
"""
        ),
        code(
            """# Cell 6 — freeze the domain manifest before fitting or metrics
DOMAIN_MANIFEST = BUNDLE_ROOT / "domain_manifest.json"
if not DOMAIN_MANIFEST.exists():
    subprocess.run(
        [
            EXPERIMENT_PYTHON,
            str(CLONE / "scripts/freeze_rars_v16_domain_manifest.py"),
            "--fiqa-fit", str(BUNDLE_ROOT / "fiqa_bge_same_encoder/fit"),
            "--fiqa-evaluation", str(BUNDLE_ROOT / "fiqa_bge_same_encoder/evaluation"),
            "--scifact-fit", str(BUNDLE_ROOT / "scifact_bge_same_encoder/fit"),
            "--scifact-evaluation", str(BUNDLE_ROOT / "scifact_bge_same_encoder/evaluation"),
            "--output", str(DOMAIN_MANIFEST),
            "--source-commit", V16_IMPLEMENTATION_COMMIT,
        ],
        cwd=CLONE,
        check=True,
    )
manifest = json.loads(DOMAIN_MANIFEST.read_text())
assert manifest["status"] == "V16_DOMAIN_BUNDLES_FROZEN"
assert manifest["source_domain_id"] == "fiqa_bge_same_encoder"
print(json.dumps(manifest, indent=2)[:5000])
"""
        ),
        code(
            """# Cell 7 — execute the one-shot mechanism decomposition (long CPU cell)
COMPLETE = OUTPUT / "diagnostic_complete.json"
if COMPLETE.exists():
    assert json.loads(COMPLETE.read_text())["status"] == "RARS_V16_MECHANISM_DIAGNOSTIC_COMPLETE"
    print("Reusing complete diagnostic:", COMPLETE)
else:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            EXPERIMENT_PYTHON,
            str(CLONE / "scripts/evaluate_rars_v16_causal_generalization.py"),
            "--domain-manifest", str(DOMAIN_MANIFEST),
            "--output-dir", str(OUTPUT),
            "--protocol", str(PROTOCOL),
            "--source-commit", V16_IMPLEMENTATION_COMMIT,
        ],
        cwd=CLONE,
        check=True,
    )
"""
        ),
        code(
            """# Cell 8 — inspect the frozen decision without retuning
RESULT = json.loads((OUTPUT / "diagnostic_result.json").read_text())
assert RESULT["status"] == "RARS_V16_MECHANISM_DIAGNOSTIC_COMPLETE"
print("Decision:", RESULT["diagnostic_decision"]["decision"])
print("\\nEqual-domain factor summary:")
print(json.dumps(RESULT["equal_domain_factor_summary"], indent=2))
for domain_id, row in RESULT["domains"].items():
    print("\\nDOMAIN:", domain_id)
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
            """# Cell 9 — package the complete packet for local audit
import shutil
archive_base = Path("/content") / f"rars-v16-mechanism-{V16_IMPLEMENTATION_COMMIT[:12]}"
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
            "colab": {"name": "RARS_V16_Same_Encoder_Mechanism_Diagnostic.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("--commit must be an exact lowercase 40-hex commit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            build_notebook(
                args.commit or "PASTE_FULL_40_CHARACTER_COMMIT_HERE"
            ),
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
