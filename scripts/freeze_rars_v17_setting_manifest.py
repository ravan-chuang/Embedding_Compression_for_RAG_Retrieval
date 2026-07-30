#!/usr/bin/env python3
"""Freeze the four million-scale V17 setting roles into one manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROTOCOL_ID = "rars_v17_million_scale_setting_transfer_v1"
CANONICAL_PROTOCOL = Path(
    "protocols/rars_v17_million_scale_setting_transfer_v1.json"
)
CANONICAL_SOURCES = (
    CANONICAL_PROTOCOL,
    Path("scripts/adapt_rars_v17_msmarco_bundle.py"),
    Path("scripts/build_rars_v17_setting_bundle.py"),
    Path("scripts/evaluate_rars_v17_million_scale.py"),
    Path("scripts/freeze_rars_v17_setting_manifest.py"),
    Path("scripts/prepare_rars_v17_nq_roles.py"),
    Path("scripts/rars_v17_million_scale_core.py"),
)
MSMARCO_SETTING = "msmarco_1m_bge_opened_development"
NQ_SETTING = "beir_nq_2_68m_bge_opened_test_diagnostic"
FAISS_METRIC_TYPES = {"inner_product": 0, "l2": 1}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("--source-commit must be exact lowercase 40-hex")


def _validate_source(
    repo_root: Path, source_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_commit(source_commit)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if head != source_commit or dirty:
        raise ValueError("V17 manifest freeze requires a clean exact checkout")
    protocol_path = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    protocol = _read(protocol_path)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status")
        != "FROZEN_BEFORE_FIRST_V17_MILLION_SCALE_DIAGNOSTIC_RUN"
    ):
        raise ValueError("Unexpected V17 protocol")
    source_blobs = {
        relative.as_posix(): file_record(
            (repo_root / relative).resolve(strict=True)
        )
        for relative in CANONICAL_SOURCES
    }
    return protocol, source_blobs


def _load_bundle(
    path: Path,
    *,
    setting_id: str,
    role: str,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    bundle = path.resolve(strict=True)
    manifest_path = bundle / "bundle_manifest.json"
    manifest = _read(manifest_path)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status")
        != "RARS_V17_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS"
        or manifest.get("setting_id", manifest.get("domain_id")) != setting_id
        or manifest.get("evidence_role") != role
    ):
        raise ValueError(f"Unexpected {setting_id} {role} bundle identity")
    qids_path = bundle / "query_ids.utf8.txt"
    qids = set(qids_path.read_text(encoding="utf-8").splitlines())
    if len(qids) != int(manifest["query_count"]):
        raise ValueError(f"{setting_id} {role} query count changed")
    minimum_key = (
        "minimum_fit_queries_per_domain"
        if role == "fit"
        else "minimum_evaluation_queries_per_domain"
    )
    minimum_queries = int(protocol["data_policy"][minimum_key])
    if len(qids) < minimum_queries:
        raise ValueError(
            f"{setting_id} {role} has {len(qids)} queries; "
            f"requires {minimum_queries}"
        )
    setting_contract = protocol["index_policy"][setting_id]
    common = protocol["index_policy"]["common"]
    document_count = int(manifest.get("document_count", -1))
    minimum_documents = max(
        int(protocol["data_policy"]["minimum_document_count_per_setting"]),
        int(setting_contract["minimum_documents"]),
    )
    if document_count < minimum_documents:
        raise ValueError(
            f"{setting_id} has {document_count} documents; "
            f"requires {minimum_documents}"
        )
    index = manifest.get("index_contract", {})
    expected_metric_type = FAISS_METRIC_TYPES[str(common["metric"])]
    expected_index = {
        "dimension": int(common["dimension"]),
        "document_count": document_count,
        "nlist": int(setting_contract["nlist"]),
        "nprobe": int(setting_contract["nprobe"]),
        "subquantizers": int(common["subquantizers"]),
        "bits_per_subquantizer": int(common["bits_per_subquantizer"]),
        "metric_type": expected_metric_type,
    }
    observed_index = {
        key: int(index.get(key, -1)) for key in expected_index
    }
    if observed_index != expected_index:
        raise ValueError(
            f"{setting_id} {role} index contract changed: "
            f"{observed_index} != {expected_index}"
        )
    opened_nq = manifest.get(
        "opened_nq_test_evidence", manifest.get("closed_test_opened")
    )
    if setting_id == NQ_SETTING and opened_nq is not True:
        raise ValueError("NQ bundle must disclose its previously opened test evidence")
    if setting_id == MSMARCO_SETTING and opened_nq is True:
        raise ValueError("MS MARCO development bundle cannot claim NQ test evidence")
    return manifest, file_record(manifest_path), qids


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    protocol, source_blobs = _validate_source(repo_root, args.source_commit)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing V17 domain manifest")
    role_paths = {
        MSMARCO_SETTING: {
            "fit": args.msmarco_fit,
            "evaluation": args.msmarco_evaluation,
        },
        NQ_SETTING: {
            "fit": args.nq_fit,
            "evaluation": args.nq_evaluation,
        },
    }
    settings: list[dict[str, Any]] = []
    encoder_families: set[tuple[str, int]] = set()
    index_recipes: dict[str, list[int]] = {}
    for setting_id, paths in role_paths.items():
        records: dict[str, Any] = {}
        qids_by_role: dict[str, set[str]] = {}
        role_recipes: set[tuple[int, int, int, int, int, int]] = set()
        for role, path in paths.items():
            manifest, record, qids = _load_bundle(
                path,
                setting_id=setting_id,
                role=role,
                protocol=protocol,
            )
            records[role] = record
            qids_by_role[role] = qids
            encoder = manifest.get("encoder", {})
            encoder_families.add(
                (
                    str(manifest.get("encoder_id", encoder.get("id", ""))),
                    int(manifest.get("dimension", encoder.get("dimension", 0))),
                )
            )
            contract = manifest["index_contract"]
            role_recipes.add(
                tuple(
                    int(contract[key])
                    for key in (
                        "dimension",
                        "nlist",
                        "nprobe",
                        "subquantizers",
                        "bits_per_subquantizer",
                        "metric_type",
                    )
                )
            )
        if qids_by_role["fit"] & qids_by_role["evaluation"]:
            raise ValueError(f"{setting_id} fit/evaluation query IDs overlap")
        if len(role_recipes) != 1:
            raise ValueError(f"{setting_id} fit/evaluation index recipe changed")
        index_recipes[setting_id] = list(next(iter(role_recipes)))
        settings.append(
            {
                "setting_id": setting_id,
                "domain_id": setting_id,
                "roles": {
                    role: str(path.resolve(strict=True))
                    for role, path in paths.items()
                },
                "role_manifests": records,
            }
        )
    if len(encoder_families) != 1:
        raise ValueError("V17 settings do not share one encoder family/dimension")
    source_setting = protocol["data_policy"]["source_setting_id"]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "V17_DOMAIN_BUNDLES_FROZEN",
        "source_commit": args.source_commit,
        "source_setting_id": source_setting,
        "source_domain_id": source_setting,
        "settings": settings,
        "domains": settings,
        "encoder_family_and_dimension": list(next(iter(encoder_families))),
        "exact_encoder_revision_match_claimed": False,
        "index_recipes_by_setting": index_recipes,
        "different_nlist_and_nprobe_expected": True,
        "minimum_document_count_verified": True,
        "source_blobs": source_blobs,
        "nq_prior_opened_test_artifact_reused": True,
        "nq_prior_confirmation_outcomes_known": True,
        "closed_confirmation_role_opened": True,
        "new_unseen_confirmation_role_opened": False,
        "independent_confirmation_claim_allowed": False,
        "evaluation_used_for_selection": False,
        "metrics_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco-fit", type=Path, required=True)
    parser.add_argument("--msmarco-evaluation", type=Path, required=True)
    parser.add_argument("--nq-fit", type=Path, required=True)
    parser.add_argument("--nq-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
