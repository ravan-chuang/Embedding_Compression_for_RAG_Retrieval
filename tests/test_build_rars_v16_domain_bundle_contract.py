from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_rars_v16_domain_bundle.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v16_bundle_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v16_bundle_builder_declares_outcome_free_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "RARS_V16_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS" in source
    assert '"metrics_computed": False' in source
    assert '"basis_fitted": False' in source
    assert '"closed_test_opened": False' in source
    assert "ivf.make_direct_map()" in source
    assert "index.reconstruct_batch" in source
    assert "--qrels-rows" in source
    assert "--source-commit" in source


def test_v16_fold_assignment_is_domain_separated_and_deterministic() -> None:
    module = _load_module()
    first = module._deterministic_fold("123", "fiqa_bge", 5)
    assert first == module._deterministic_fold("123", "fiqa_bge", 5)
    assert 0 <= first < 5
    pairs = {
        (
            module._deterministic_fold(str(index), "fiqa_bge", 5),
            module._deterministic_fold(str(index), "scifact_bge", 5),
        )
        for index in range(100)
    }
    assert any(left != right for left, right in pairs)


def test_v16_qrels_rows_must_exactly_cover_queries(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "qrels.json"
    path.write_text(json.dumps({"q1": [1, 3], "q2": [2]}), encoding="utf-8")
    observed = module._load_qrels_rows(path, ["q1", "q2"], 4)
    assert observed == {"q1": [1, 3], "q2": [2]}

    path.write_text(json.dumps({"q1": [1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="cover exactly"):
        module._load_qrels_rows(path, ["q1", "q2"], 4)


def test_v16_matrix_loader_checks_raw_bytes_and_npy_dtype(tmp_path: Path) -> None:
    module = _load_module()
    array = np.arange(12, dtype=np.float16).reshape(3, 4)
    npy = tmp_path / "values.npy"
    np.save(npy, array)
    observed = module._load_matrix(
        npy, dtype=np.dtype(np.float16), shape=(3, 4)
    )
    assert np.array_equal(observed, array)

    raw = tmp_path / "values.memmap"
    raw.write_bytes(array.tobytes())
    observed_raw = module._load_matrix(
        raw, dtype=np.dtype(np.float16), shape=(3, 4)
    )
    assert np.array_equal(observed_raw, array)

    with pytest.raises(ValueError, match="expected"):
        module._load_matrix(raw, dtype=np.dtype(np.float32), shape=(3, 4))
