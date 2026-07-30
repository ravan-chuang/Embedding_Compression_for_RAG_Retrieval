from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_rars_v7_query_adapter.py").read_text(encoding="utf-8")


def test_v7_trainer_requires_verified_v6_packet_and_epoch_zero_parity() -> None:
    assert "verify_v6_packet(args.v6_packet_root)" in SOURCE
    assert "Epoch-zero reconstructed-PQ retrieval does not reproduce v6 Recall" in SOURCE
    assert "Epoch-zero same-route FP32 retrieval does not reproduce v6 Recall" in SOURCE


def test_v7_trainer_uses_original_routing_and_fixed_pq_reconstruction() -> None:
    assert "probed_candidate_rows" in SOURCE
    assert "np.array_equal(probed_lists, registered_probes)" in SOURCE
    direct_map = SOURCE.index("ivf.make_direct_map()")
    first_reconstruction = SOURCE.index(
        "exact_union_scores, pq_union_scores = v6_eval.score_flip_candidate_union"
    )
    assert direct_map < first_reconstruction
    assert "index.reconstruct_batch(rows)" in SOURCE
    assert "index_record_after != index_record_before" in SOURCE
    assert '"in_memory_direct_map_built": True' in SOURCE
    assert "document_reencoding_performed\": False" in SOURCE


def test_v7_trainer_never_opens_audit_or_future_paths() -> None:
    assert "--audit-role-dir" not in SOURCE
    assert "--future" not in SOURCE
    assert "future_method_holdout_opened\": False" in SOURCE
    assert "rars_used\": False" in SOURCE
