#!/usr/bin/env python3
"""Materialize explicit P/N/U labels for the RARS-v4 Phase-0 gate.

Missing qrel rows are always UNJUDGED.  The script never reads the inherited
binary ``candidate_relevance`` arrays because those collapsed missing rows and
non-relevance into the same value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


PROTOCOL_ID = "rars_v4_tristate_action_feasibility_v1"
V3_PROTOCOL_ID = "rars_v3_oracle_first_feasibility_v1"
DESIGN_ROLE_ID = "v4_design_observed"
AUDIT_ROLE_ID = "v4_diagnostic_audit"
ROLE_TO_V3 = {
    DESIGN_ROLE_ID: "oracle_design",
    AUDIT_ROLE_ID: "oracle_audit",
}
CANONICAL_PROTOCOL = Path("protocols/rars_v4_tristate_action_feasibility_v1.json")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not Path(path).is_file():
        raise ValueError(f"Missing registered {label}: {path}")
    if Path(path).stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Registered {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Registered {label} hash changed")


def _validate_exact_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source_commit must be an exact lowercase 40-hex commit")


def _validate_clean_git_head(repo_root: Path, source_commit: str) -> None:
    _validate_exact_commit(source_commit)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if head != source_commit:
        raise ValueError(f"Expected exact Git HEAD {source_commit}, found {head}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("RARS-v4 label materialization requires a clean worktree")


def _validate_protocol(
    repo_root: Path, protocol_path: Path, source_commit: str
) -> dict[str, Any]:
    canonical = (repo_root / CANONICAL_PROTOCOL).resolve()
    if protocol_path.resolve() != canonical:
        raise ValueError("Only the canonical RARS-v4 protocol path is allowed")
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected RARS-v4 protocol ID")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_TRISTATE_LABEL_AUDIT":
        raise ValueError("RARS-v4 protocol is not frozen for Phase-0")
    if protocol.get("method_revision_allowed") is not False or protocol.get(
        "outcome_informed_revision_allowed"
    ) is not False:
        raise ValueError("RARS-v4 protocol permits outcome-informed revision")
    _validate_clean_git_head(repo_root, source_commit)
    return protocol


def _validate_candidate_bundle(
    bundle_dir: Path,
    *,
    role_id: str,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    expected_v3_role = ROLE_TO_V3[role_id]
    manifest_path = bundle_dir / "v3_candidate_manifest.json"
    query_manifest_path = bundle_dir / "query_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("protocol_id") != V3_PROTOCOL_ID:
        raise ValueError("Candidate bundle is not from the frozen v3 protocol")
    if manifest.get("source_commit") != protocol["parent_lineage"][
        "v3_implementation_commit"
    ]:
        raise ValueError("Candidate bundle v3 source commit changed")
    if manifest.get("role_id") != expected_v3_role:
        raise ValueError("Candidate bundle role does not match the requested v4 role")
    if manifest.get("evidence_status") != "DEVELOPMENT_ONLY":
        raise ValueError("Candidate bundle is not development-only")
    query_record = manifest.get("query_manifest")
    if not isinstance(query_record, dict):
        raise ValueError("Candidate manifest lacks its query manifest")
    _verify_record(query_manifest_path, query_record, "candidate query manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Candidate manifest lacks registered files")
    document_record = files.get("candidate_doc_ids.int64.npy")
    if not isinstance(document_record, dict):
        raise ValueError("Candidate manifest lacks candidate document IDs")
    document_path = bundle_dir / "candidate_doc_ids.int64.npy"
    _verify_record(document_path, document_record, "candidate document IDs")
    query_manifest = read_json(query_manifest_path)
    qids = [str(value) for value in query_manifest.get("query_ids", [])]
    if len(qids) != int(manifest.get("query_count", -1)) or len(qids) != len(set(qids)):
        raise ValueError("Candidate query identity is invalid")
    expected_count = int(protocol["data_policy"]["roles"][role_id]["query_count"])
    if len(qids) != expected_count:
        raise ValueError("Candidate role query count changed")
    document_ids = np.load(document_path, mmap_mode="r")
    expected_shape = (len(qids), int(manifest.get("candidate_count", -1)))
    if document_ids.dtype != np.dtype(np.int64) or document_ids.shape != expected_shape:
        raise ValueError("Candidate document-ID matrix changed")
    for query_index, row in enumerate(document_ids):
        if len(row) != len(set(np.asarray(row).tolist())):
            raise ValueError(f"Duplicate candidate document IDs for query {query_index}")
    return manifest, qids, document_ids


def _validate_design_freeze(
    design_freeze_path: Path,
    *,
    protocol: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    freeze = read_json(design_freeze_path)
    if freeze.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Audit label release received the wrong design protocol")
    if freeze.get("source_commit") != source_commit:
        raise ValueError("Audit label release design source commit changed")
    if freeze.get("status") != "V4_DESIGN_GO_TO_DIAGNOSTIC_AUDIT":
        raise ValueError("Audit labels require a durable v4 design GO freeze")
    if freeze.get("formal_decision") != "DESIGN_GO_TO_DIAGNOSTIC_AUDIT":
        raise ValueError("Design freeze does not authorize diagnostic audit labels")
    if freeze.get("all_required_checks_passed") is not True:
        raise ValueError("Design freeze did not pass every Phase-0 design check")
    return freeze


def _parse_role_judgments(
    payload: Any, qids: list[str]
) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Judgment source must be a JSON object keyed by query ID")
    parsed: dict[str, dict[int, float]] = {}
    mapping_entries = 0
    list_entries = 0
    positive_rows = 0
    explicit_negative_rows = 0
    missing_qids: list[str] = []
    for qid in qids:
        if qid not in payload:
            missing_qids.append(qid)
            continue
        value = payload[qid]
        if isinstance(value, dict):
            mapping_entries += 1
            judgments: dict[int, float] = {}
            for document_id, grade in value.items():
                try:
                    normalized_grade = float(grade)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Non-numeric judgment grade for query {qid}") from error
                if not np.isfinite(normalized_grade):
                    raise ValueError(f"Non-finite judgment grade for query {qid}")
                normalized_document_id = int(document_id)
                if normalized_document_id in judgments:
                    raise ValueError(f"Duplicate judgment document for query {qid}")
                judgments[normalized_document_id] = normalized_grade
                if normalized_grade > 0:
                    positive_rows += 1
                else:
                    explicit_negative_rows += 1
            parsed[qid] = judgments
        elif isinstance(value, list):
            list_entries += 1
            judgments = {int(document_id): 1.0 for document_id in value}
            if len(judgments) != len(value):
                raise ValueError(f"Duplicate positive-only judgment for query {qid}")
            positive_rows += len(judgments)
            parsed[qid] = judgments
        else:
            raise ValueError(f"Unsupported judgment entry for query {qid}")
    if missing_qids:
        raise ValueError(
            f"Judgment source lacks {len(missing_qids)} role queries; first={missing_qids[0]}"
        )
    all_role_entries_graded_mapping = mapping_entries == len(qids) and list_entries == 0
    semantics_preserved = (
        all_role_entries_graded_mapping and explicit_negative_rows > 0
    )
    return parsed, {
        "role_query_count": len(qids),
        "graded_mapping_query_count": mapping_entries,
        "positive_only_list_query_count": list_entries,
        "positive_source_rows": positive_rows,
        "explicit_negative_source_rows": explicit_negative_rows,
        "all_role_entries_graded_mapping": all_role_entries_graded_mapping,
        "explicit_negative_semantics_preserved": semantics_preserved,
        "interpretation": (
            "PRESERVED_GRADED_TRISTATE"
            if semantics_preserved
            else "POSITIVE_ONLY_OR_NO_EXPLICIT_NEGATIVE_ROWS"
        ),
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.role not in ROLE_TO_V3:
        raise ValueError("Only registered v4 design/audit roles are allowed")
    repo_root = Path(__file__).resolve().parents[1]
    protocol = _validate_protocol(repo_root, args.protocol, args.source_commit)
    candidate_manifest, qids, document_ids = _validate_candidate_bundle(
        args.candidate_bundle, role_id=args.role, protocol=protocol
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to reuse a non-empty tri-state label directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.role == AUDIT_ROLE_ID:
        if args.design_freeze is None:
            raise ValueError("Diagnostic-audit labels require --design-freeze")
        _validate_design_freeze(
            args.design_freeze,
            protocol=protocol,
            source_commit=args.source_commit,
        )
        design_freeze_record = file_record(args.design_freeze)
    else:
        if args.design_freeze is not None:
            raise ValueError("Design labels must not receive a design freeze")
        design_freeze_record = None

    started_path = args.output_dir / "v4_tristate_labels_started.json"
    materializer_sha256 = sha256_file(Path(__file__).resolve())
    atomic_json(
        started_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "TRISTATE_LABEL_MATERIALIZATION_STARTED",
            "role_id": args.role,
            "source_commit": args.source_commit,
            "materializer_sha256": materializer_sha256,
            "candidate_manifest": file_record(
                args.candidate_bundle / "v3_candidate_manifest.json"
            ),
            "judgment_source": file_record(args.judgments),
            "design_freeze": design_freeze_record,
            "binary_candidate_relevance_read": False,
            "future_method_holdout_accessed": False,
        },
    )
    payload = read_json(args.judgments)
    judgments, source_schema = _parse_role_judgments(payload, qids)
    states = np.zeros(document_ids.shape, dtype=np.int8)
    relevant_counts = np.empty(len(qids), dtype=np.int32)
    candidate_positive_rows = 0
    candidate_explicit_negative_rows = 0
    for query_index, qid in enumerate(qids):
        row_judgments = judgments[qid]
        positives = {docid for docid, grade in row_judgments.items() if grade > 0}
        explicit_negatives = {
            docid for docid, grade in row_judgments.items() if grade <= 0
        }
        if not positives:
            raise ValueError(f"No judged positive for Phase-0 query {qid}")
        relevant_counts[query_index] = len(positives)
        candidate_row = np.asarray(document_ids[query_index], dtype=np.int64)
        positive_mask = np.isin(candidate_row, list(positives))
        negative_mask = np.isin(candidate_row, list(explicit_negatives))
        if np.any(positive_mask & negative_mask):
            raise AssertionError("A candidate cannot be both positive and explicit negative")
        states[query_index, positive_mask] = np.int8(1)
        states[query_index, negative_mask] = np.int8(-1)
        candidate_positive_rows += int(np.sum(positive_mask))
        candidate_explicit_negative_rows += int(np.sum(negative_mask))

    states_path = args.output_dir / "candidate_judgment_state.int8.npy"
    counts_path = args.output_dir / "relevant_counts.int32.npy"
    atomic_save(states_path, states)
    atomic_save(counts_path, relevant_counts)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "TRISTATE_ROLE_LABELS_MATERIALIZED",
        "role_id": args.role,
        "source_v3_role": ROLE_TO_V3[args.role],
        "source_commit": args.source_commit,
        "materializer_sha256": materializer_sha256,
        "query_count": len(qids),
        "candidate_count": int(document_ids.shape[1]),
        "candidate_manifest": file_record(
            args.candidate_bundle / "v3_candidate_manifest.json"
        ),
        "candidate_manifest_protocol_id": candidate_manifest["protocol_id"],
        "judgment_source": file_record(args.judgments),
        "started": file_record(started_path),
        "design_freeze": design_freeze_record,
        "materialized_after_design_go_freeze": args.role == AUDIT_ROLE_ID,
        "source_schema": source_schema,
        "candidate_intersections": {
            "positive_rows": candidate_positive_rows,
            "explicit_negative_rows": candidate_explicit_negative_rows,
            "unjudged_rows": int(
                states.size - candidate_positive_rows - candidate_explicit_negative_rows
            ),
        },
        "state_contract": {"POSITIVE": 1, "EXPLICIT_NEGATIVE": -1, "UNJUDGED": 0},
        "files": {
            states_path.name: file_record(states_path),
            counts_path.name: file_record(counts_path),
        },
        "binary_candidate_relevance_read": False,
        "missing_rows_interpreted_as_explicit_negative": False,
        "future_method_holdout_accessed": False,
    }
    manifest_path = args.output_dir / "v4_tristate_labels_manifest.json"
    atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": file_record(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bundle", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=tuple(ROLE_TO_V3))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / CANONICAL_PROTOCOL,
    )
    parser.add_argument("--design-freeze", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(materialize(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
