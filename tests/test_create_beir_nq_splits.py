from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "create_beir_nq_train_validation_splits.py"
SPEC = importlib.util.spec_from_file_location("nq_splits", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def query_ids_for_both_partitions() -> list[str]:
    validation = next(
        str(value) for value in range(1000) if MODULE.split_bucket(str(value)) == 0
    )
    fit = next(
        str(value) for value in range(1000) if MODULE.split_bucket(str(value)) != 0
    )
    return [validation, fit]


def test_hash_partition_is_deterministic_and_disjoint() -> None:
    qids = [str(value) for value in range(100)]
    first = MODULE.partition_train_qids(qids)
    second = MODULE.partition_train_qids(qids)

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(qids)


def test_create_splits_reads_train_membership_but_not_relevance(
    tmp_path: Path,
) -> None:
    qids = query_ids_for_both_partitions()
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "".join(
            json.dumps({"_id": qid, "text": f"question {qid}"}) + "\n"
            for qid in qids
        ),
        encoding="utf-8",
    )
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    train_qrels = qrels_dir / "train.tsv"
    train_qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"{qid}\tdoc-{qid}\t999\n" for qid in qids),
        encoding="utf-8",
    )

    result = MODULE.create_splits(queries, train_qrels, tmp_path / "out")

    assert result["official_train_query_count"] == 2
    assert result["fit_query_count"] == 1
    assert result["validation_query_count"] == 1
    assert result["train_membership_source"]["relevance_values_parsed"] is False
    assert result["test_qrels_accessed"] is False


def test_test_qrels_path_is_rejected(tmp_path: Path) -> None:
    test_qrels = tmp_path / "test.tsv"
    test_qrels.write_text("q1\td1\t1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="train.tsv"):
        MODULE.load_train_qids(test_qrels)


def test_missing_train_query_text_is_rejected(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"_id": "known", "text": "known question"}) + "\n",
        encoding="utf-8",
    )
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    train_qrels = qrels_dir / "train.tsv"
    train_qrels.write_text("missing\tdoc\t1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from queries.jsonl"):
        MODULE.create_splits(queries, train_qrels, tmp_path / "out")


def test_non_train_query_text_is_not_retained(tmp_path: Path) -> None:
    qids = query_ids_for_both_partitions()
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "".join([
            json.dumps({"_id": qids[0], "text": "train zero"}) + "\n",
            json.dumps({"_id": qids[1], "text": "train one"}) + "\n",
            json.dumps({"_id": "not-train", "text": "must not be retained"}) + "\n",
        ]),
        encoding="utf-8",
    )
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    train_qrels = qrels_dir / "train.tsv"
    train_qrels.write_text(
        "".join(f"{qid}\tdoc\t1\n" for qid in qids),
        encoding="utf-8",
    )

    MODULE.create_splits(queries, train_qrels, tmp_path / "out")

    fit = json.loads((tmp_path / "out" / "train_query_manifest.json").read_text())
    validation = json.loads(
        (tmp_path / "out" / "validation_query_manifest.json").read_text()
    )
    assert "not-train" not in fit["query_ids"] + validation["query_ids"]
    assert "must not be retained" not in fit["query_texts"] + validation["query_texts"]
