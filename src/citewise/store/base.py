"""Storage interface. SQLite (zero setup) and Postgres + pgvector implement the same calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..models import Answer, ChunkRecord, DocumentRecord


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model than the one configured."""


@dataclass(frozen=True)
class AccessFilter:
    """Applied inside the database query, before ranking. A chunk the principal can't read
    is never scored, so it can't leak into the prompt even through a ranking bug."""

    workspace: str
    groups: tuple[str, ...]


class Store(Protocol):
    def ensure_schema(self, embedding_name: str | None, dim: int | None) -> None: ...

    def reset(self) -> None: ...

    def index_info(self) -> dict[str, Any]: ...

    def document_hashes(self, workspace: str) -> dict[str, str]: ...

    def upsert_document(
        self, doc: DocumentRecord, chunks: list[ChunkRecord], vectors: np.ndarray | None
    ) -> None: ...

    def delete_documents(self, workspace: str, paths: list[str]) -> None: ...

    def lexical_search(self, query: str, f: AccessFilter, k: int) -> list[tuple[str, float]]: ...

    def dense_search(self, vector: np.ndarray, f: AccessFilter, k: int) -> list[tuple[str, float]]: ...

    def get_chunks(self, ids: list[str]) -> dict[str, ChunkRecord]: ...

    def log_answer(self, answer: Answer, principal_id: str) -> None: ...

    def add_feedback(self, answer_id: str, principal_id: str, rating: int, comment: str | None) -> bool: ...

    def list_feedback(self) -> list[dict[str, Any]]: ...

    def ping(self) -> bool: ...


STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from how i if in into is it its me my of on or our
    should so than that the their them then there these they this to us was we what when where which
    who whom why will with would you your yours have has had am been being did get got any some""".split()
)
TERM = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def query_terms(query: str, limit: int = 32) -> list[str]:
    """Keyword-search terms: lowercase words without stopwords, deduplicated, order kept."""
    seen: dict[str, None] = {}
    for t in TERM.findall(query.lower()):
        t = t.split("'")[0]
        keep = t.isdigit() or (len(t) > 1 and t not in STOPWORDS)
        if keep:
            seen.setdefault(t, None)
    return list(seen)[:limit]


def open_store(database_url: str) -> Store:
    if database_url.startswith("sqlite:///"):
        from .sqlite import SQLiteStore

        return SQLiteStore(database_url.removeprefix("sqlite:///"))
    if database_url.startswith(("postgresql://", "postgres://", "postgresql+psycopg://")):
        from .postgres import PostgresStore

        return PostgresStore(database_url.replace("postgresql+psycopg://", "postgresql://", 1))
    raise ValueError(f"Unsupported database URL: {database_url}")
