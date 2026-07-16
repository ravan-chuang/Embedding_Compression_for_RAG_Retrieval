from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/MSMARCO_RARS_v2_2_FP32_Development.ipynb"
)
PINNED_IMPLEMENTATION = "b5607e6e13e63015af7b93ff42247419e3a81079"


def test_v2_2_notebook_is_clean_ordered_and_pinned() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert len({cell["id"] for cell in cells}) == len(cells)
    assert all(cell.get("execution_count") is None for cell in cells if cell["cell_type"] == "code")
    assert all(not cell.get("outputs") for cell in cells if cell["cell_type"] == "code")
    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    assert markdown.index("## tl;dr") < markdown.index("## Context & Methods")
    assert markdown.index("## Context & Methods") < markdown.index("## Data")
    assert markdown.index("## Data") < markdown.index("## Results")
    assert markdown.index("## Results") < markdown.index("## Takeaways")
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert PINNED_IMPLEMENTATION in source


def test_v2_2_notebook_never_builds_or_trains_on_outer() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "'--inner-only'" in code
    assert "--validation-bundle-dir" not in code
    assert "--cached-full-residuals" not in code
    assert "train_boundary_loss_sidecar_v2_2.py" in code
    assert "query_gate" not in code.split("trainer = [", 1)[1].split("]", 1)[0]
