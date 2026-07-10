"""Faiss-backed retrieval service primitives."""

from __future__ import annotations

from app.reranker import CrossEncoderReranker
from app.sidecar import RARSSidecar

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RetrievalService:
    """Load a serialized Faiss index and optional retrieval components."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

        self.index: faiss.Index | None = None
        self.model: SentenceTransformer | None = None
        self.documents: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}
        self.query_rotation: np.ndarray | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.sidecar: RARSSidecar | None = None
        self.sidecar_artifact_dir: Path | None = None

    @property
    def is_ready(self) -> bool:
        return (
            self.index is not None
            and self.model is not None
            and bool(self.documents)
        )

    def load(self) -> None:
        config_path = self.artifact_dir / "service_config.json"
        index_path = self.artifact_dir / "index.faiss"
        documents_path = self.artifact_dir / "documents.jsonl"

        required_paths = [
            config_path,
            index_path,
            documents_path,
        ]

        missing = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing retrieval artifacts: " + ", ".join(missing)
            )

        self.config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        self.index = faiss.read_index(str(index_path))

        with documents_path.open("r", encoding="utf-8") as file:
            self.documents = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "Index/document mismatch: "
                f"{self.index.ntotal} vectors vs "
                f"{len(self.documents)} document rows."
            )

        self._load_query_transform()

        self.model = SentenceTransformer(
            self.config["embedding_model"],
            device="cpu",
        )

        default_nprobe = self.config.get("default_nprobe")

        if default_nprobe is not None:
            self.set_nprobe(int(default_nprobe))

        self._load_reranker()
        self._load_sidecar()

    def _load_query_transform(self) -> None:
        """Load the query-side OPQ rotation when configured."""
        transform_config = self.config.get("query_transform", {})

        if not transform_config.get("enabled", False):
            self.query_rotation = None
            return

        filename = transform_config.get("file")

        if not filename:
            raise ValueError(
                "query_transform.enabled is true but no file is configured."
            )

        rotation_path = self.artifact_dir / filename

        if not rotation_path.exists():
            raise FileNotFoundError(
                f"Missing query transform: {rotation_path}"
            )

        if self.index is None:
            raise RuntimeError(
                "Index must load before validating query rotation."
            )

        rotation = np.load(rotation_path).astype(np.float32)
        expected_shape = (self.index.d, self.index.d)

        if rotation.shape != expected_shape:
            raise ValueError(
                "Invalid OPQ rotation shape: "
                f"expected {expected_shape}, got {rotation.shape}."
            )

        self.query_rotation = np.ascontiguousarray(rotation)

    def _load_reranker(self) -> None:
        reranker_config = self.config.get("reranker", {})

        if not reranker_config.get("enabled", False):
            self.reranker = None
            return

        self.reranker = CrossEncoderReranker(
            model_name=reranker_config.get(
                "model_name",
                "BAAI/bge-reranker-base",
            ),
            device=reranker_config.get("device", "cpu"),
            batch_size=int(reranker_config.get("batch_size", 16)),
        )
        self.reranker.load()

    def _load_sidecar(self) -> None:
        """Load optional RARS / residual sidecar artifact.

        Expected service_config block:

        {
          "sidecar": {
            "enabled": true,
            "artifact_dir": "../msmarco_rars_sidecar_m32_rank16",
            "config_file": "sidecar_config.json"
          }
        }

        Relative artifact paths are resolved relative to the retrieval artifact
        directory.
        """

        sidecar_config = self.config.get("sidecar", {})

        if not sidecar_config.get("enabled", False):
            self.sidecar = None
            self.sidecar_artifact_dir = None
            return

        artifact_dir_value = sidecar_config.get("artifact_dir")

        if not artifact_dir_value:
            raise ValueError(
                "sidecar.enabled is true but no artifact_dir is configured."
            )

        sidecar_artifact_dir = Path(artifact_dir_value)

        if not sidecar_artifact_dir.is_absolute():
            sidecar_artifact_dir = self.artifact_dir / sidecar_artifact_dir

        config_name = sidecar_config.get(
            "config_file",
            "sidecar_config.json",
        )

        self.sidecar = RARSSidecar(
            sidecar_artifact_dir,
            config_name=config_name,
        )
        self.sidecar_artifact_dir = sidecar_artifact_dir

        if self.index is not None and self.sidecar.dim != self.index.d:
            raise ValueError(
                "Sidecar/index dimension mismatch: "
                f"sidecar dim={self.sidecar.dim}, index dim={self.index.d}."
            )

        if self.sidecar.num_docs != len(self.documents):
            raise ValueError(
                "Sidecar/document count mismatch: "
                f"sidecar docs={self.sidecar.num_docs}, "
                f"documents={len(self.documents)}."
            )

    def set_nprobe(self, nprobe: int) -> None:
        """Set IVF probing depth when the loaded index supports it."""
        if self.index is None:
            raise RuntimeError("Retriever is not loaded.")

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = int(nprobe)

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode normalized queries and apply the matching OPQ transform."""
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")

        vectors = self.model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        if self.query_rotation is not None:
            vectors = vectors @ self.query_rotation

        return np.ascontiguousarray(vectors.astype(np.float32))

    def _format_results(
        self,
        query: str,
        scores: np.ndarray,
        indices: np.ndarray,
        top_k: int,
        nprobe: int | None,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []

        for rank, (score, idx) in enumerate(
            zip(scores, indices),
            start=1,
        ):
            if idx < 0:
                continue

            document = self.documents[int(idx)]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "doc_id": document["doc_id"],
                    "title": document.get("title", ""),
                    "text": document.get("text", ""),
                }
            )

        return {
            "query": query,
            "top_k": top_k,
            "nprobe": (
                nprobe
                if nprobe is not None
                else self.config.get("default_nprobe")
            ),
            "index_type": self.config.get(
                "index_type",
                type(self.index).__name__,
            ),
            "query_transform_enabled": self.query_rotation is not None,
            "results": results,
        }

    def _format_sidecar_results(
        self,
        query: str,
        sidecar_result: dict[str, Any],
        top_k: int,
        nprobe: int | None,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []

        rows = sidecar_result["candidate_rows"]
        ann_scores = sidecar_result["ann_scores"]
        corrected_scores = sidecar_result["corrected_scores"]
        corrections = sidecar_result["corrections"]

        for rank, (idx, ann_score, corrected_score, correction) in enumerate(
            zip(rows, ann_scores, corrected_scores, corrections),
            start=1,
        ):
            if idx < 0:
                continue

            document = self.documents[int(idx)]

            results.append(
                {
                    "rank": rank,
                    "score": float(corrected_score),
                    "ann_score": float(ann_score),
                    "sidecar_correction": float(correction),
                    "corrected_score": float(corrected_score),
                    "doc_id": document["doc_id"],
                    "title": document.get("title", ""),
                    "text": document.get("text", ""),
                }
            )

        return {
            "query": query,
            "top_k": top_k,
            "nprobe": (
                nprobe
                if nprobe is not None
                else self.config.get("default_nprobe")
            ),
            "index_type": self.config.get(
                "index_type",
                type(self.index).__name__,
            ),
            "query_transform_enabled": self.query_rotation is not None,
            "results": results,
            "sidecar_enabled": True,
            "sidecar_top_b": sidecar_result["top_b"],
            "sidecar_actual_top_b": sidecar_result["actual_top_b"],
            "sidecar_alpha": sidecar_result["alpha"],
        }

    def search(
        self,
        query: str,
        top_k: int,
        nprobe: int | None = None,
        rerank: bool = False,
        candidate_k: int | None = None,
        sidecar: bool = False,
        sidecar_top_b: int | None = None,
    ) -> dict[str, Any]:
        batch = self.search_many(
            queries=[query],
            top_k=top_k,
            nprobe=nprobe,
            rerank=rerank,
            candidate_k=candidate_k,
            sidecar=sidecar,
            sidecar_top_b=sidecar_top_b,
        )

        item = batch["items"][0]
        item["latency_ms"] = batch["latency_ms_total"]

        return item

    def search_many(
        self,
        queries: list[str],
        top_k: int,
        nprobe: int | None = None,
        rerank: bool = False,
        candidate_k: int | None = None,
        sidecar: bool = False,
        sidecar_top_b: int | None = None,
    ) -> dict[str, Any]:
        """Embed queries, search ANN candidates, and optionally rerank in batch."""

        if not self.is_ready or self.index is None:
            raise RuntimeError("Retriever is not ready.")

        if not queries:
            raise ValueError("At least one query is required.")

        if rerank and self.reranker is None:
            raise RuntimeError(
                "Reranking was requested but no reranker is configured."
            )

        if sidecar and self.sidecar is None:
            raise RuntimeError(
                "RARS sidecar correction was requested but no sidecar "
                "artifact is configured or loaded."
            )

        requested_sidecar_top_b = sidecar_top_b

        if sidecar:
            assert self.sidecar is not None

            if requested_sidecar_top_b is None:
                requested_sidecar_top_b = self.sidecar.config.default_top_b

            if requested_sidecar_top_b > self.sidecar.config.max_top_b:
                raise ValueError(
                    "sidecar_top_b exceeds configured sidecar max_top_b: "
                    f"{requested_sidecar_top_b} > "
                    f"{self.sidecar.config.max_top_b}"
                )

        if candidate_k is None:
            candidate_k = max(
                top_k,
                50 if rerank else top_k,
                100 if sidecar else top_k,
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        if sidecar and requested_sidecar_top_b is not None:
            if candidate_k < requested_sidecar_top_b:
                raise ValueError(
                    "candidate_k must be greater than or equal to "
                    "sidecar_top_b when sidecar correction is enabled."
                )

        if nprobe is not None:
            self.set_nprobe(nprobe)

        total_start = time.perf_counter()

        embedding_start = time.perf_counter()
        vectors = self._encode_queries(queries)
        embedding_latency_ms = (
            time.perf_counter() - embedding_start
        ) * 1000.0

        retrieval_start = time.perf_counter()
        scores, indices = self.index.search(vectors, candidate_k)
        ann_search_latency_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000.0

        sidecar_latency_ms = 0.0

        if sidecar:
            assert self.sidecar is not None
            assert requested_sidecar_top_b is not None

            sidecar_start = time.perf_counter()

            sidecar_results = self.sidecar.rerank_batch(
                query_embeddings=vectors,
                candidate_rows=indices,
                ann_scores=scores,
                top_k=candidate_k,
                top_b=requested_sidecar_top_b,
            )

            sidecar_latency_ms = (
                time.perf_counter() - sidecar_start
            ) * 1000.0

            items = [
                self._format_sidecar_results(
                    query=query,
                    sidecar_result=sidecar_result,
                    top_k=candidate_k,
                    nprobe=nprobe,
                )
                for query, sidecar_result in zip(queries, sidecar_results)
            ]
        else:
            items = [
                self._format_results(
                    query=query,
                    scores=query_scores,
                    indices=query_indices,
                    top_k=candidate_k,
                    nprobe=nprobe,
                )
                for query, query_scores, query_indices in zip(
                    queries,
                    scores,
                    indices,
                )
            ]

            for item in items:
                item["sidecar_enabled"] = False
                item["sidecar_top_b"] = None
                item["sidecar_actual_top_b"] = 0
                item["sidecar_alpha"] = None

        rerank_latency_ms = 0.0

        if rerank:
            rerank_start = time.perf_counter()

            ranked_groups = self.reranker.rerank_many(
                [
                    (item["query"], item["results"])
                    for item in items
                ]
            )

            rerank_latency_ms = (
                time.perf_counter() - rerank_start
            ) * 1000.0

            for item, ranked in zip(items, ranked_groups):
                item["results"] = ranked[:top_k]
                item["rerank_enabled"] = True
                item["candidate_k"] = candidate_k
        else:
            for item in items:
                item["results"] = item["results"][:top_k]
                item["rerank_enabled"] = False
                item["candidate_k"] = candidate_k

        for item in items:
            item["top_k"] = top_k

        latency_ms_total = (time.perf_counter() - total_start) * 1000.0

        return {
            "count": len(queries),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "nprobe": (
                nprobe
                if nprobe is not None
                else self.config.get("default_nprobe")
            ),
            "index_type": self.config.get(
                "index_type",
                type(self.index).__name__,
            ),
            "query_transform_enabled": self.query_rotation is not None,
            "rerank_enabled": rerank,
            "sidecar_enabled": sidecar,
            "sidecar_top_b": requested_sidecar_top_b if sidecar else None,
            "latency_ms_total": round(latency_ms_total, 3),
            "embedding_latency_ms": round(embedding_latency_ms, 3),
            "ann_search_latency_ms": round(ann_search_latency_ms, 3),
            "sidecar_latency_ms": round(sidecar_latency_ms, 3),
            "rerank_latency_ms": round(rerank_latency_ms, 3),
            "latency_ms_per_query": round(
                latency_ms_total / len(queries),
                3,
            ),
            "items": items,
        }