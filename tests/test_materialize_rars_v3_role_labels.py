from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
SCRIPT = ROOT / "scripts/materialize_rars_v3_role_labels.py"
SPEC = importlib.util.spec_from_file_location("materialize_rars_v3_labels", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_materializer_rejects_future_role_before_any_input_access() -> None:
    with pytest.raises(ValueError, match="Only oracle_design and oracle_audit"):
        MODULE.materialize(
            argparse.Namespace(
                role="future_method_holdout",
                bundle_root=Path("never-opened"),
                parent_inner_train_bundle=Path("never-opened"),
                protocol=Path("never-opened"),
                source_commit="a" * 40,
                design_freeze=None,
            )
        )


def test_registered_output_validation_rejects_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe design-freeze"):
        MODULE._verify_registered_outputs(
            tmp_path,
            {"../outside": {"bytes": 0, "sha256": "0" * 64}},
        )


def test_materializer_has_no_qrels_reader() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "load_qrels" not in source
    assert "future_method_holdout" not in source.split("choices=", 1)[-1]
