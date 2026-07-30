from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "protocols/rars_v17_million_scale_setting_transfer_v1.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_v17_replaces_small_v16_corpora_with_million_scale_settings() -> None:
    payload = _protocol()
    assert payload["protocol_id"] == (
        "rars_v17_million_scale_setting_transfer_v1"
    )
    assert payload["status"] == (
        "FROZEN_BEFORE_FIRST_V17_MILLION_SCALE_DIAGNOSTIC_RUN"
    )
    assert payload["supersedes"]["status"] == (
        "ABORTED_PRE_METRIC_SMALL_CORPUS_DESIGN"
    )
    policy = payload["data_policy"]
    assert policy["minimum_document_count_per_setting"] == 1_000_000
    assert policy["allowed_development_settings"] == [
        "msmarco_1m_bge_opened_development",
        "beir_nq_2_68m_bge_opened_test_diagnostic",
    ]
    assert "fiqa" not in json.dumps(policy).casefold()
    assert "scifact" not in json.dumps(policy).casefold()


def test_v17_discloses_nq_reuse_and_noncausal_setting_contrast() -> None:
    payload = _protocol()
    boundary = payload["evidence_boundary"]
    assert "independent confirmation" in boundary["forbidden_claims"]
    assert "pure domain-shift causality" in boundary["forbidden_claims"]
    assert payload["data_policy"]["nq_roles"]["prior_confirmation_outcomes_known"]
    assert not payload["data_policy"]["nq_roles"][
        "independent_confirmation_claim_allowed"
    ]
    assert payload["index_policy"]["different_nlist_and_nprobe_are_expected"]
    assert payload["index_policy"]["cross_setting_not_pure_domain_contrast"]


def test_v17_keeps_matched_sidecar_budget_and_query_level_inference() -> None:
    payload = _protocol()
    config = payload["diagnostic_configuration"]
    assert config["rars_rank"] == 16
    assert config["coefficient_dtype"] == "int8"
    assert config["payload_bytes_per_document_at_rank16"] == 16
    assert config["top_b"] == 40
    assert config["final_k"] == 10
    assert payload["inference"]["resampling_unit"] == "query"
    assert payload["inference"]["equal_setting_summary_is_descriptive_only"]
