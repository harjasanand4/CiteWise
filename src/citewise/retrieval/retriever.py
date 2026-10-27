"""Hybrid retrieval: keyword + vector search, fused with Reciprocal Rank Fusion, then reranked.

Why both: keyword search nails exact terms ("FMLA", "Brex", "SF-182") that embeddings blur,
and embeddings catch paraphrases ("how much can I expense" vs "User Limit of $5,000") that
keywords miss. RRF combines the two rankings without having to calibrate their scores,
which live on completely different scales.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..embeddings import Embedder
from ..models import Hit, Principal
from ..store.base import AccessFilter, Store

Mode = Literal["lexical", "dense", "hybrid"]


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """fastembed cross-encoder. Returns raw logits; Hit.rerank_score stores sigmoid(logit)."""

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from ..embeddings import model_cache_dir

        self.name = f"fastembed:{model}"
        self._model = TextCrossEncoder(model_name=model, cache_dir=model_cache_dir())

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, passages, batch_size=32)]


def get_reranker(spec: str) -> Reranker | None:
    spec = spec.strip()
    if spec in ("", "none"):
        return None
    if spec.startswith("fastembed:"):
        return CrossEncoderReranker(spec.split(":", 1)[1])
    raise ValueError(f"Unknown reranker spec: {spec!r}")


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank of d)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder | None,
        reranker: Reranker | None,
        mode: Mode = "hybrid",
        candidates: int = 30,
        rrf_k: int = 60,
    ) -> None:
        if mode in ("dense", "hybrid") and embedder is None:
            mode = "lexical"
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.mode: Mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k

    def describe(self) -> str:
        parts: list[str] = [self.mode]
        if self.mode != "lexical" and self.embedder:
            parts.append(self.embedder.name)
        if self.reranker:
            parts.append(f"rerank:{self.reranker.name}")
        return " + ".join(parts)

    def retrieve(
        self,
        query: str,
        workspace: str,
        principal: Principal,
        k: int = 6,
        mode: Mode | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        mode = mode or self.mode
        if mode != "lexical" and self.embedder is None:
            mode = "lexical"
        use_rerank = self.reranker is not None if rerank is None else (rerank and self.reranker is not None)
        timings: dict[str, float] = {}
        f = AccessFilter(workspace=workspace, groups=principal.groups)

        if workspace not in principal.workspaces:  # belt and braces; the API checks this too
            return RetrievalResult([], timings)

        lexical: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []
        t = time.perf_counter()
        if mode in ("lexical", "hybrid"):
            lexical = self.store.lexical_search(query, f, self.candidates)
        timings["lexical"] = _ms(t)
        if mode in ("dense", "hybrid") and self.embedder is not None:
            t = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            timings["embed_query"] = _ms(t)
            t = time.perf_counter()
            dense = self.store.dense_search(qvec, f, self.candidates)
            timings["dense"] = _ms(t)

        lex_rank = {cid: i for i, (cid, _) in enumerate(lexical, start=1)}
        dense_rank = {cid: i for i, (cid, _) in enumerate(dense, start=1)}
        dense_score = dict(dense)
        if mode == "hybrid":
            fused = rrf([[c for c, _ in lexical], [c for c, _ in dense]], k=self.rrf_k)
            order = sorted(fused, key=lambda c: -fused[c])
            score_of = fused
        elif mode == "dense":
            order = [c for c, _ in dense]
            score_of = dense_score
        else:
            order = [c for c, _ in lexical]
            score_of = dict(lexical)

        order = order[: self.candidates]
        chunks = self.store.get_chunks(order)
        hits = [
            Hit(
                chunk=chunks[c],
                score=score_of[c],
                lexical_rank=lex_rank.get(c),
                dense_rank=dense_rank.get(c),
                dense_score=dense_score.get(c),
            )
            for c in order
            if c in chunks
        ]
        # Defense in depth: the store already filtered, but never hand back something unreadable.
        hits = [h for h in hits if principal.can_read(h.chunk.workspace, h.chunk.acl)]

        if use_rerank and hits and self.reranker is not None:
            t = time.perf_counter()
            logits = self.reranker.score(query, [h.chunk.embed_text for h in hits])
            for h, logit in zip(hits, logits, strict=True):
                h.rerank_score = sigmoid(logit)
                h.score = logit
            hits.sort(key=lambda h: -h.score)
            timings["rerank"] = _ms(t)

        return RetrievalResult(hits[:k], timings)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
