from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_rars_v8_cutoff_sidecar_packet import verify_packet


PACKET = (
    Path(__file__).resolve().parents[1] / "results/rars_v8_cutoff_sidecar"
)


def test_v8_closure_recomputes_primary_result_and_gate() -> None:
    summary = verify_packet(PACKET)
    assert summary["status"] == "RARS_V8_CUTOFF_SIDECAR_CLOSURE_VERIFIED"
    assert summary["formal_decision"] == "GO_TO_RARS_ALGORITHM_CONFIRMATION_PROTOCOL"
    assert summary["rars_gain_over_base"] == pytest.approx(0.02290131483889611)
    assert summary["rars_gain_over_pca"] == pytest.approx(0.010186389250108365)
    assert summary["full_sidecar_codes_external"] is True
    assert summary["diagnostics"]["final_recorded_loss_decreased"] is False


def test_v8_closure_rejects_oof_array_tampering(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    shutil.copytree(PACKET, packet)
    path = packet / "development/oof_rars_recall_at_10.float64.npy"
    values = np.load(path, allow_pickle=False)
    values[0] = 1.0 - values[0]
    np.save(path, values, allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_packet(packet)
