from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT / "notebooks/RARS_V16_Same_Encoder_Mechanism_Diagnostic.ipynb"
)
PINNED_IMPLEMENTATION_COMMIT = (
    "deef9f33d5dcf29bb2c6c5852ff914da924637ea"
)


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_v16_notebook_is_commit_pinned_and_uses_isolated_virtualenv() -> None:
    source = _source()
    assert PINNED_IMPLEMENTATION_COMMIT in source
    assert "PASTE_FULL_40_CHARACTER_COMMIT_HERE" not in source
    assert 're.fullmatch(r"[0-9a-f]{40}"' in source
    assert '"--system-site-packages"' in source
    assert '"numpy==1.26.4"' in source
    assert "actual == V16_IMPLEMENTATION_COMMIT" in source


def test_v16_notebook_runs_prepare_bundle_freeze_evaluate_in_order() -> None:
    source = _source()
    positions = [
        source.index("prepare_rars_v16_beir_domains.py"),
        source.index("build_rars_v16_domain_bundle.py"),
        source.index("freeze_rars_v16_domain_manifest.py"),
        source.index("evaluate_rars_v16_causal_generalization.py"),
    ]
    assert positions == sorted(positions)
    assert "diagnostic_result.json" in source
    assert "files.download(archive)" in source
