"""RARS / PQ-residual sidecar serving utilities.

This module implements a small deployable reranking layer for a frozen IVF-PQ
retrieval path. The sidecar stores a low-rank residual correction:

    corrected_score(q, x)
    = ann_score(q, x) + alpha * q^T B a_x

where:
- q is the normalized query embedding.
- B is a shared residual basis with shape [dim, rank].
- a_x is the dequantized int8 coefficient vector for a document.
- only the first top_b ANN candidates are corrected and reranked.

The class is intentionally independent from FastAPI so it can be unit-tested
offline and later integrated into app.retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SidecarConfig:
    """Configuration for a deployable residual sidecar artifact."""

    dim: int
    rank: int
    alpha: float
    default_top_b: int
    max_top_b: int
    code_dtype: str
    doc_id_dtype: str
    basis_file: str
    scales_file: str
    codes_file: str
    doc_ids_file: str

    @classmethod
    def from_json(cls, path: str | Path) -> "SidecarConfig":
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        required = {
            "dim",
            "rank",
            "alpha",
            "default_top_b",
            "max_top_b",
            "code_dtype",
            "doc_id_dtype",
            "basis_file",
            "scales_file",
            "codes_file",
            "doc_ids_file",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"sidecar config missing fields: {missing}")

        return cls(
            dim=int(data["dim"]),
            rank=int(data["rank"]),
            alpha=float(data["alpha"]),
            default_top_b=int(data["default_top_b"]),
            max_top_b=int(data["max_top_b"]),
            code_dtype=str(data["code_dtype"]),
            doc_id_dtype=str(data["doc_id_dtype"]),
            basis_file=str(data["basis_file"]),
            scales_file=str(data["scales_file"]),
            codes_file=str(data["codes_file"]),
            doc_ids_file=str(data["doc_ids_file"]),
        )


class RARSSidecar:
    """Low-rank int8 residual sidecar reranker.

    The sidecar is designed for candidate reranking after a frozen IVF-PQ search.
    It assumes ANN candidates are represented by corpus-internal row ids, not
    external document ids. If the retrieval service only returns external doc ids,
    call ``rows_from_doc_ids`` first.

    Artifact layout:

        sidecar_config.json
        basis.npy
        scales.npy
        codes.int8.npy
        doc_ids.npy

    ``codes`` must have shape [num_docs, rank].
    ``basis`` must have shape [dim, rank].
    ``scales`` must have shape [rank].
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        config_name: str = "sidecar_config.json",
        mmap_codes: bool = True,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config_path = self.artifact_dir / config_name
        self.config = SidecarConfig.from_json(self.config_path)

        self.basis = np.load(self.artifact_dir / self.config.basis_file).astype(
            np.float32
        )
        self.scales = np.load(self.artifact_dir / self.config.scales_file).astype(
            np.float32
        )

        code_mmap_mode = "r" if mmap_codes else None
        self.codes = np.load(
            self.artifact_dir / self.config.codes_file,
            mmap_mode=code_mmap_mode,
        )
        self.doc_ids = np.load(self.artifact_dir / self.config.doc_ids_file)

        self._validate_artifact()
        self._doc_id_to_row: dict[int, int] | None = None

    def _validate_artifact(self) -> None:
        cfg = self.config

        if self.basis.shape != (cfg.dim, cfg.rank):
            raise ValueError(
                f"basis shape mismatch: expected {(cfg.dim, cfg.rank)}, "
                f"got {self.basis.shape}"
            )

        if self.scales.shape != (cfg.rank,):
            raise ValueError(
                f"scales shape mismatch: expected {(cfg.rank,)}, "
                f"got {self.scales.shape}"
            )

        if self.codes.ndim != 2 or self.codes.shape[1] != cfg.rank:
            raise ValueError(
                f"codes shape mismatch: expected [num_docs, {cfg.rank}], "
                f"got {self.codes.shape}"
            )

        if self.doc_ids.ndim != 1:
            raise ValueError(f"doc_ids must be 1D, got {self.doc_ids.shape}")

        if self.doc_ids.shape[0] != self.codes.shape[0]:
            raise ValueError(
                f"doc_ids/codes length mismatch: "
                f"{self.doc_ids.shape[0]} vs {self.codes.shape[0]}"
            )

        if str(self.codes.dtype) != cfg.code_dtype:
            raise ValueError(
                f"codes dtype mismatch: expected {cfg.code_dtype}, "
                f"got {self.codes.dtype}"
            )

    @property
    def num_docs(self) -> int:
        return int(self.codes.shape[0])

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def rank(self) -> int:
        return self.config.rank

    def _ensure_doc_id_map(self) -> dict[int, int]:
        if self._doc_id_to_row is None:
            self._doc_id_to_row = {
                int(doc_id): int(row) for row, doc_id in enumerate(self.doc_ids)
            }
        return self._doc_id_to_row

    def rows_from_doc_ids(self, doc_ids: Iterable[int]) -> np.ndarray:
        """Map external document ids to corpus-internal row ids."""

        mapping = self._ensure_doc_id_map()
        rows: list[int] = []

        missing: list[int] = []
        for doc_id in doc_ids:
            key = int(doc_id)
            row = mapping.get(key)
            if row is None:
                missing.append(key)
            else:
                rows.append(row)

        if missing:
            preview = missing[:5]
            raise KeyError(
                f"{len(missing)} document ids are missing from sidecar doc_ids; "
                f"first missing ids: {preview}"
            )

        return np.asarray(rows, dtype=np.int64)

    def compute_corrections(
        self,
        query_embedding: np.ndarray,
        candidate_rows: np.ndarray,
    ) -> np.ndarray:
        """Compute residual score corrections for candidate rows.

        Parameters
        ----------
        query_embedding:
            Query embedding with shape [dim].
        candidate_rows:
            Corpus-internal row ids with shape [n].

        Returns
        -------
        np.ndarray
            Correction scores with shape [n], before multiplying by alpha.
        """

        q = np.asarray(query_embedding, dtype=np.float32)
        rows = np.asarray(candidate_rows, dtype=np.int64)

        if q.shape != (self.config.dim,):
            raise ValueError(
                f"query_embedding shape mismatch: expected {(self.config.dim,)}, "
                f"got {q.shape}"
            )

        if rows.ndim != 1:
            raise ValueError(f"candidate_rows must be 1D, got {rows.shape}")

        if len(rows) == 0:
            return np.empty((0,), dtype=np.float32)

        if rows.min() < 0 or rows.max() >= self.num_docs:
            raise IndexError(
                f"candidate row out of range: min={rows.min()}, "
                f"max={rows.max()}, num_docs={self.num_docs}"
            )

        # Project query to the sidecar basis, then dot with dequantized codes:
        #
        #   q^T B a_x
        # = (q^T B) · a_x
        #
        # codes are int8; scales are per sidecar dimension.
        q_basis = q @ self.basis  # [rank]
        coeffs = np.asarray(self.codes[rows], dtype=np.float32) * self.scales
        corrections = coeffs @ q_basis

        return corrections.astype(np.float32, copy=False)

    def rerank(
        self,
        query_embedding: np.ndarray,
        candidate_rows: np.ndarray,
        ann_scores: np.ndarray,
        *,
        top_k: int = 10,
        top_b: int | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        """Apply sidecar correction to top-B candidates and rerank.

        Only the first ``top_b`` candidates are corrected. Candidates beyond
        top-B keep their original ANN scores. The final ranking is produced by
        sorting the full candidate set by corrected score.

        Returns a dictionary so the result is easy to integrate into FastAPI
        responses and benchmark scripts.
        """

        rows = np.asarray(candidate_rows, dtype=np.int64)
        scores = np.asarray(ann_scores, dtype=np.float32)

        if rows.ndim != 1:
            raise ValueError(f"candidate_rows must be 1D, got {rows.shape}")

        if scores.ndim != 1:
            raise ValueError(f"ann_scores must be 1D, got {scores.shape}")

        if rows.shape[0] != scores.shape[0]:
            raise ValueError(
                f"candidate_rows and ann_scores length mismatch: "
                f"{rows.shape[0]} vs {scores.shape[0]}"
            )

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if top_b is None:
            top_b = self.config.default_top_b

        top_b = int(top_b)
        if top_b < 0:
            raise ValueError(f"top_b must be non-negative, got {top_b}")

        if top_b > self.config.max_top_b:
            raise ValueError(
                f"top_b={top_b} exceeds artifact max_top_b={self.config.max_top_b}"
            )

        if alpha is None:
            alpha = self.config.alpha

        alpha = float(alpha)

        n = len(rows)
        actual_top_b = min(top_b, n)
        corrected_scores = scores.copy()
        corrections = np.zeros_like(scores, dtype=np.float32)

        if actual_top_b > 0 and alpha != 0.0:
            raw_corr = self.compute_corrections(
                query_embedding=query_embedding,
                candidate_rows=rows[:actual_top_b],
            )
            corrections[:actual_top_b] = raw_corr
            corrected_scores[:actual_top_b] = (
                corrected_scores[:actual_top_b] + alpha * raw_corr
            )

        order = np.argsort(-corrected_scores, kind="stable")
        top_order = order[: min(top_k, n)]

        return {
            "candidate_rows": rows[top_order].astype(np.int64),
            "doc_ids": self.doc_ids[rows[top_order]].astype(self.doc_ids.dtype),
            "ann_scores": scores[top_order].astype(np.float32),
            "corrected_scores": corrected_scores[top_order].astype(np.float32),
            "corrections": corrections[top_order].astype(np.float32),
            "rerank_order": top_order.astype(np.int64),
            "top_k": int(top_k),
            "top_b": int(top_b),
            "actual_top_b": int(actual_top_b),
            "alpha": float(alpha),
            "sidecar_enabled": True,
        }

    def rerank_batch(
        self,
        query_embeddings: np.ndarray,
        candidate_rows: np.ndarray,
        ann_scores: np.ndarray,
        *,
        top_k: int = 10,
        top_b: int | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Apply sidecar reranking to a batch of queries.

        Parameters
        ----------
        query_embeddings:
            Array with shape [batch, dim].
        candidate_rows:
            Array with shape [batch, candidate_count].
        ann_scores:
            Array with shape [batch, candidate_count].
        """

        Q = np.asarray(query_embeddings, dtype=np.float32)
        rows = np.asarray(candidate_rows, dtype=np.int64)
        scores = np.asarray(ann_scores, dtype=np.float32)

        if Q.ndim != 2 or Q.shape[1] != self.config.dim:
            raise ValueError(
                f"query_embeddings shape mismatch: expected [batch, {self.config.dim}], "
                f"got {Q.shape}"
            )

        if rows.ndim != 2:
            raise ValueError(f"candidate_rows must be 2D, got {rows.shape}")

        if scores.shape != rows.shape:
            raise ValueError(
                f"ann_scores shape mismatch: expected {rows.shape}, got {scores.shape}"
            )

        if Q.shape[0] != rows.shape[0]:
            raise ValueError(
                f"batch mismatch: Q={Q.shape[0]}, candidates={rows.shape[0]}"
            )

        return [
            self.rerank(
                query_embedding=Q[i],
                candidate_rows=rows[i],
                ann_scores=scores[i],
                top_k=top_k,
                top_b=top_b,
                alpha=alpha,
            )
            for i in range(Q.shape[0])
        ]