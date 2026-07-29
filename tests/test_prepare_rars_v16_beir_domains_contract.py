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
        "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"
    )
    assert set(MODULE.DATASETS) == {
        "fiqa_bge_same_encoder",
        "scifact_bge_same_encoder",
    }
    assert "normalize_embeddings=True" in SOURCE


def test_v16_query_role_is_deterministic_and_domain_separated() -> None:
    assert MODULE.deterministic_role("q1", "fiqa_bge_same_encoder") == (
        MODULE.deterministic_role("q1", "fiqa_bge_same_encoder")
    )
    pairs = {
        (
            MODULE.deterministic_role(str(index), "fiqa_bge_same_encoder"),
            MODULE.deterministic_role(str(index), "scifact_bge_same_encoder"),
        )
        for index in range(100)
    }
    assert any(left != right for left, right in pairs)


def test_v16_preparer_has_no_metric_or_sidecar_training_path() -> None:
    assert '"metrics_computed": False' in SOURCE
    assert '"sidecar_basis_fitted": False' in SOURCE
    assert "Recall@" not in SOURCE
    assert "fit_cutoff_aware_basis" not in SOURCE
    assert "RARS_V16_SAME_ENCODER_DOMAIN_INPUTS_PREPARED" in SOURCE
