from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_rars_v16_beir_domains.py"
SPEC = importlib.util.spec_from_file_location("prepare_rars_v16", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SOURCE = SCRIPT.read_text(encoding="utf-8")


def test_v16_preparer_pins_one_encoder_and_two_domains() -> None:
    assert MODULE.MODEL_ID == "BAAI/bge-small-en-v1.5"
    assert MODULE.MODEL_REVISION == (
        "88885630388d6249d876a3ab145b78b34665b79a"
    )
    assert set(MODULE.DATASETS) == {
        "fiqa_bge_same_encoder",
        "scifact_bge_same_encoder",
    }
    assert "normalize_embeddings=True" in SOURCE
    assert "snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)" in SOURCE
    assert "RARS_V16_PINNED_BGE_SNAPSHOT_VERIFIED" in SOURCE
    assert 'config.get("model_type") != "bert"' in SOURCE
    assert "Pinned BGE encoder preflight" in SOURCE


def test_v16_query_role_is_deterministic_and_domain_separated() -> None:
    query_ids = [str(index) for index in range(300)]
    fiqa = MODULE.deterministic_roles(query_ids, "fiqa_bge_same_encoder")
    scifact = MODULE.deterministic_roles(
        query_ids, "scifact_bge_same_encoder"
    )
    assert fiqa == MODULE.deterministic_roles(
        query_ids, "fiqa_bge_same_encoder"
    )
    assert len(fiqa["fit"]) == 180
    assert len(fiqa["evaluation"]) == 120
    assert set(fiqa["fit"]).isdisjoint(fiqa["evaluation"])
    assert set(fiqa["fit"]) | set(fiqa["evaluation"]) == set(query_ids)
    assert fiqa != scifact


def test_v16_role_minima_are_feasible_for_scifact() -> None:
    assert MODULE.MINIMUM_ROLE_QUERIES == {
        "fit": 150,
        "evaluation": 100,
    }
    assert sum(MODULE.MINIMUM_ROLE_QUERIES.values()) <= 300


def test_v16_preparer_has_no_metric_or_sidecar_training_path() -> None:
    assert '"metrics_computed": False' in SOURCE
    assert '"sidecar_basis_fitted": False' in SOURCE
    assert "Recall@" not in SOURCE
    assert "fit_cutoff_aware_basis" not in SOURCE
    assert "RARS_V16_SAME_ENCODER_DOMAIN_INPUTS_PREPARED" in SOURCE


def test_v16_pinned_model_snapshot_requires_sentence_transformer_files() -> None:
    required = (
        "modules.json",
        "config.json",
        "1_Pooling/config.json",
        "pytorch_model.bin",
        "tokenizer.json",
    )
    assert all(name in SOURCE for name in required)
