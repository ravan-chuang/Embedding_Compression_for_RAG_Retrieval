from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/build_rars_v8_frozen_sidecars.py").read_text(
    encoding="utf-8"
)


def test_builder_has_no_label_or_query_inputs() -> None:
    arguments = SOURCE.split("def parse_args()", 1)[1]
    assert "--qrels" not in arguments
    assert "--query" not in arguments
    assert "--future" not in arguments
    assert '"qrels_argument_accepted": False' in SOURCE
    assert '"query_argument_accepted": False' in SOURCE
    assert '"future_method_holdout_opened": False' in SOURCE


def test_builder_requires_a_closed_go_development_packet() -> None:
    assert "GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL" in SOURCE
    assert "GO_TO_GENERIC_SIDECAR_CONFIRMATION_PROTOCOL" in SOURCE
    assert "verify_development_packet(" in SOURCE
    assert "does not authorize artifact build" in SOURCE
    assert "verify_record(result_path" in SOURCE
    assert "verify_record(freeze_path" in SOURCE


def test_builder_keeps_index_byte_identical_and_uses_ivf_direct_map() -> None:
    assert "index_before = file_record(args.index)" in SOURCE
    assert "index_after = file_record(args.index)" in SOURCE
    assert "if index_after != index_before" in SOURCE
    assert "ivf.make_direct_map()" in SOURCE
    assert "index=ivf" in SOURCE
    assert "index=cpu_index" not in SOURCE


def test_builder_calibrates_then_encodes_full_corpus_without_residual_artifact() -> None:
    assert SOURCE.count("for start in range(0, len(embeddings), batch_size)") >= 2
    assert "maximum / 127.0" in SOURCE
    assert "np.lib.format.open_memmap" in SOURCE
    assert "codes.int8.npy" in SOURCE
    assert "residuals.float32.npy" not in SOURCE
    assert '"external_document_id_bytes_duplicated": 0' in SOURCE
