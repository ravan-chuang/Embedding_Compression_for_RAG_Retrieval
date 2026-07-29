#!/usr/bin/env python3
"""Freeze the four prepared V16 bundle roles into one execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROTOCOL_ID = "rars_v16_causal_generalization_diagnostic_v1"
CANONICAL_PROTOCOL = Path(
    "protocols/rars_v16_causal_generalization_diagnostic_v1.json"
)


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


def _validate_source(repo_root: Path, source_commit: str) -> dict[str, Any]:
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
        raise ValueError("V16 manifest freeze requires a clean exact checkout")
    protocol_path = (repo_root / CANONICAL_PROTOCOL).resolve(strict=True)
    protocol = _read(protocol_path)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status")
        != "FROZEN_BEFORE_FIRST_V16_MECHANISM_DIAGNOSTIC_RUN"
    ):
        raise ValueError("Unexpected V16 protocol")
    return protocol


def _load_bundle(
    path: Path, *, domain_id: str, role: str
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    bundle = path.resolve(strict=True)
    manifest_path = bundle / "bundle_manifest.json"
    manifest = _read(manifest_path)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status")
        != "RARS_V16_DOMAIN_BUNDLE_FROZEN_BEFORE_METRICS"
        or manifest.get("domain_id") != domain_id
        or manifest.get("evidence_role") != role
    ):
        raise ValueError(f"Unexpected {domain_id} {role} bundle identity")
    qids_path = bundle / "query_ids.utf8.txt"
    qids = set(qids_path.read_text(encoding="utf-8").splitlines())
    if len(qids) != int(manifest["query_count"]):
        raise ValueError(f"{domain_id} {role} query count changed")
    return manifest, file_record(manifest_path), qids


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    protocol = _validate_source(repo_root, args.source_commit)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing V16 domain manifest")
    role_paths = {
        "fiqa_bge_same_encoder": {
            "fit": args.fiqa_fit,
            "evaluation": args.fiqa_evaluation,
        },
        "scifact_bge_same_encoder": {
            "fit": args.scifact_fit,
            "evaluation": args.scifact_evaluation,
        },
    }
    domains: list[dict[str, Any]] = []
    encoder_keys: set[tuple[str, str, int]] = set()
    index_recipes: set[tuple[int, int, int, int, int]] = set()
    for domain_id, paths in role_paths.items():
        records: dict[str, Any] = {}
        qids_by_role: dict[str, set[str]] = {}
        for role, path in paths.items():
            manifest, record, qids = _load_bundle(
                path, domain_id=domain_id, role=role
            )
            records[role] = record
            qids_by_role[role] = qids
            encoder = manifest.get("encoder", {})
            encoder_keys.add(
                (
                    str(manifest.get("encoder_id", encoder.get("id", ""))),
                    str(
                        manifest.get(
                            "encoder_revision", encoder.get("revision", "")
                        )
                    ),
                    int(manifest.get("dimension", encoder.get("dimension", 0))),
                )
            )
            contract = manifest["index_contract"]
            index_recipes.add(
                tuple(
                    int(contract[key])
                    for key in (
                        "nlist",
                        "nprobe",
                        "subquantizers",
                        "bits_per_subquantizer",
                        "metric_type",
                    )
                )
            )
        if qids_by_role["fit"] & qids_by_role["evaluation"]:
            raise ValueError(f"{domain_id} fit/evaluation query IDs overlap")
        domains.append(
            {
                "domain_id": domain_id,
                "roles": {
                    role: str(path.resolve(strict=True))
                    for role, path in paths.items()
                },
                "role_manifests": records,
            }
        )
    if len(encoder_keys) != 1 or len(index_recipes) != 1:
        raise ValueError("V16 domains do not share one encoder and index recipe")
    source_domain = protocol["data_policy"]["source_domain_id"]
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "V16_DOMAIN_BUNDLES_FROZEN",
        "source_commit": args.source_commit,
        "source_domain_id": source_domain,
        "domains": domains,
        "encoder_key": list(next(iter(encoder_keys))),
        "index_recipe": list(next(iter(index_recipes))),
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
    parser.add_argument("--fiqa-fit", type=Path, required=True)
    parser.add_argument("--fiqa-evaluation", type=Path, required=True)
    parser.add_argument("--scifact-fit", type=Path, required=True)
    parser.add_argument("--scifact-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
