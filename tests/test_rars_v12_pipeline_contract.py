from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = (ROOT / "scripts/freeze_rars_v12_fresh_queries.py").read_text()
BUNDLER = (ROOT / "scripts/build_rars_v12_fresh_bundle.py").read_text()
TRAINER = (ROOT / "scripts/train_rars_v12_ca_rpq.py").read_text()
VERIFIER = (ROOT / "scripts/verify_rars_v12_ca_rpq_packet.py").read_text()


def test_query_freezer_excludes_prior_qids_before_encoding_or_candidates() -> None:
    assert "deterministic_query_priority" in FREEZER
    assert "eligible_ids = set(covered).difference(prior_qids)" in FREEZER
    assert "expected_prior_count = 6980" in FREEZER
    assert '"candidate_retrieval_performed": False' in FREEZER
    assert "normalize_embeddings" in FREEZER
    assert "model_revision" in FREEZER
    assert "validate_runtime(protocol)" in FREEZER
    assert 'environment["faiss_version"]' in FREEZER


def test_bundle_build_occurs_after_freeze_and_computes_no_metrics() -> None:
    assert "RARS_V12_FRESH_QUERY_FREEZE_COMPLETE" in BUNDLER
    assert "RARS_V12_FRESH_QUERIES_FROZEN_BEFORE_CANDIDATES" in BUNDLER
    assert "reconstruct_batch" in BUNDLER
    assert '"metrics_computed": False' in BUNDLER
    assert '"old_rars_holdout_opened": False' in BUNDLER
    assert 'freeze.get("source_commit") != source_commit' in BUNDLER
    assert 'f"fresh-query source blob {relative}"' in BUNDLER
    assert "prior.intersection(query_ids)" in BUNDLER
    assert "ivf.make_direct_map()" in BUNDLER


def test_trainer_is_oof_all_seed_and_full_corpus_materialized() -> None:
    assert "for fold in range" in TRAINER
    assert "train_queries = np.flatnonzero(folds != fold)" in TRAINER
    assert "heldout_queries = np.flatnonzero(folds == fold)" in TRAINER
    assert "for seed_index, seed in enumerate(seeds)" in TRAINER
    assert "fit_anchored_cutoff_codebooks" in TRAINER
    assert "full_corpus_ca_rpq_codes.uint8.memmap" in TRAINER
    assert "index.reconstruct_batch" in TRAINER
    assert '"v11_packet_opened": False' in TRAINER
    assert "rars-v11" in TRAINER
    assert 'input_records.pop("registered_embeddings")' in TRAINER
    assert 'input_records.pop("registered_index")' in TRAINER
    assert "code_histograms" in TRAINER


def test_every_rpq_scorer_call_uses_the_exact_positional_interface() -> None:
    tree = ast.parse(TRAINER)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "score_product_sidecar_candidates"
    ]
    assert len(calls) == 2
    assert all(len(call.args) == 7 for call in calls)
    assert all(
        {keyword.arg for keyword in call.keywords} == {"alpha", "top_b"}
        for call in calls
    )


def test_verifier_recomputes_every_previously_missing_guardrail() -> None:
    assert "query_ids.utf8.txt" in VERIFIER
    assert "fold_ids.int64.npy" in VERIFIER
    assert 'for name in ("recall", "mrr", "ndcg")' in VERIFIER
    assert "paired_inference" in VERIFIER
    assert "fold_gains" in VERIFIER
    assert "seed_gains" in VERIFIER
    assert "ca_rpq_decision" in VERIFIER
    assert "full_corpus_ca_rpq_codes.uint8.memmap" in VERIFIER
    assert "V12 full-corpus codes are degenerate" in VERIFIER
    assert "objective audit flag does not recompute" in VERIFIER
