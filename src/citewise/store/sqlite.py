"""SQLite store: FTS5 (BM25) for keyword search, NumPy for vector search.

Zero setup, and fast enough for a few hundred thousand chunks: vectors are loaded once per
workspace into one float32 matrix and searched with a single matrix-vector product. The
cache is invalidated by an index version that every write bumps.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..models import Answer, ChunkRecord, DocumentRecord
from .base import AccessFilter, IndexMismatchError, query_terms

SCHEMA = """
create table if not exists meta (key text primary key, value text not null);
create table if not exists documents (
  workspace text not null, path text not null, title text, url text, content_hash text not null,
  acl text not null default '[]', ingested_at text, primary key (workspace, path));
create table if not exists chunks (
  rid integer primary key autoincrement,
  id text unique not null, workspace text not null, path text not null, ordinal integer not null,
  title text, headings text, text text not null, url text,
  acl text not null default '[]', flags text not null default '[]', keywords text not null default '',
  word_count integer, embedding blob);
create index if not exists chunks_doc on chunks (workspace, path);
create virtual table if not exists chunks_fts using fts5(title, headings, keywords, body, tokenize = 'porter unicode61');
create table if not exists answers (
  id text primary key, created_at text, principal text, workspace text, question text, answer text,
  abstained integer, abstain_reason text, sources text, mode text, timings text);
create table if not exists feedback (
  id integer primary key autoincrement, answer_id text not null references answers(id),
  principal text, rating integer not null, comment text, created_at text);
"""

# Column weights for bm25(): a hit in the title counts more than one in the body.
BM25_WEIGHTS = "3.0, 2.0, 1.5, 1.0"
ACL_CLAUSE = "(c.acl = '[]' or exists (select 1 from json_each(c.acl) j where j.value in (select value from json_each(?))))"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._vectors: dict[str, tuple[int, list[str], list[list[str]], np.ndarray]] = {}
        with self._tx() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("pragma journal_mode = wal")
            conn.execute("pragma foreign_keys = on")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _meta(self, conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("select value from meta where key = ?", (key,)).fetchone()
        return row["value"] if row else None

    @staticmethod
    def _bump(conn: sqlite3.Connection) -> None:
        conn.execute(
            "insert into meta (key, value) values ('index_version', '1') "
            "on conflict(key) do update set value = cast(value as integer) + 1"
        )

    # ---- schema / admin -----------------------------------------------------------------

    def ensure_schema(self, embedding_name: str | None, dim: int | None) -> None:
        with self._tx() as conn:
            existing = self._meta(conn, "embedding_model")
            has_chunks = conn.execute("select 1 from chunks limit 1").fetchone() is not None
            if has_chunks and existing != (embedding_name or "none"):
                raise IndexMismatchError(
                    f"Index was built with embeddings {existing!r} but {embedding_name or 'none'!r} is configured. "
                    "Run `citewise ingest --rebuild`."
                )
            conn.execute(
                "insert or replace into meta values ('embedding_model', ?)", (embedding_name or "none",)
            )
            conn.execute("insert or replace into meta values ('embedding_dim', ?)", (str(dim or 0),))

    def reset(self) -> None:
        with self._tx() as conn:
            conn.execute("delete from chunks_fts")
            conn.execute("delete from chunks")
            conn.execute("delete from documents")
            conn.execute("delete from meta where key in ('embedding_model', 'embedding_dim')")
            self._bump(conn)
        self._vectors.clear()

    def index_info(self) -> dict[str, Any]:
        with self._tx() as conn:
            rows = conn.execute(
                "select workspace, count(distinct path) docs, count(*) chunks from chunks group by workspace"
            ).fetchall()
            return {
                "backend": "sqlite",
                "embedding_model": self._meta(conn, "embedding_model"),
                "embedding_dim": int(self._meta(conn, "embedding_dim") or 0),
                "workspaces": {r["workspace"]: {"documents": r["docs"], "chunks": r["chunks"]} for r in rows},
            }

    def ping(self) -> bool:
        with self._tx() as conn:
            conn.execute("select 1").fetchone()
        return True

    # ---- documents ----------------------------------------------------------------------

    def document_hashes(self, workspace: str) -> dict[str, str]:
        with self._tx() as conn:
            rows = conn.execute("select path, content_hash from documents where workspace = ?", (workspace,))
            return {r["path"]: r["content_hash"] for r in rows}

    def _delete_doc(self, conn: sqlite3.Connection, workspace: str, path: str) -> None:
        rids = [
            r["rid"]
            for r in conn.execute(
                "select rid from chunks where workspace = ? and path = ?", (workspace, path)
            )
        ]
        conn.executemany("delete from chunks_fts where rowid = ?", [(r,) for r in rids])
        conn.execute("delete from chunks where workspace = ? and path = ?", (workspace, path))
        conn.execute("delete from documents where workspace = ? and path = ?", (workspace, path))

    def upsert_document(
        self, doc: DocumentRecord, chunks: list[ChunkRecord], vectors: np.ndarray | None
    ) -> None:
        with self._lock, self._tx() as conn:
            self._delete_doc(conn, doc.workspace, doc.path)
            conn.execute(
                "insert into documents values (?, ?, ?, ?, ?, ?, ?)",
                (doc.workspace, doc.path, doc.title, doc.url, doc.content_hash, json.dumps(doc.acl), _now()),
            )
            for i, c in enumerate(chunks):
                blob = vectors[i].astype(np.float32).tobytes() if vectors is not None else None
                cur = conn.execute(
                    "insert into chunks (id, workspace, path, ordinal, title, headings, text, url, acl, flags, "
                    "keywords, word_count, embedding) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        c.id,
                        c.workspace,
                        c.path,
                        c.ordinal,
                        c.title,
                        json.dumps(c.headings),
                        c.text,
                        c.url,
                        json.dumps(c.acl),
                        json.dumps(c.flags),
                        " ".join(c.keywords),
                        c.word_count,
                        blob,
                    ),
                )
                conn.execute(
                    "insert into chunks_fts (rowid, title, headings, keywords, body) values (?, ?, ?, ?, ?)",
                    (cur.lastrowid, c.title, " ".join(c.headings), " ".join(c.keywords), c.text),
                )
            self._bump(conn)

    def delete_documents(self, workspace: str, paths: list[str]) -> None:
        with self._lock, self._tx() as conn:
            for p in paths:
                self._delete_doc(conn, workspace, p)
            self._bump(conn)

    # ---- search -------------------------------------------------------------------------

    def lexical_search(self, query: str, f: AccessFilter, k: int) -> list[tuple[str, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = (
            f"select c.id, -bm25(chunks_fts, {BM25_WEIGHTS}) as score from chunks_fts "
            "join chunks c on c.rid = chunks_fts.rowid "
            f"where chunks_fts match ? and c.workspace = ? and {ACL_CLAUSE} "
            f"order by bm25(chunks_fts, {BM25_WEIGHTS}) limit ?"
        )
        with self._tx() as conn:
            rows = conn.execute(sql, (match, f.workspace, json.dumps(list(f.groups)), k)).fetchall()
        return [(r["id"], float(r["score"])) for r in rows]

    def _matrix(self, workspace: str) -> tuple[list[str], list[list[str]], np.ndarray]:
        with self._tx() as conn:
            version = int(self._meta(conn, "index_version") or 0)
            cached = self._vectors.get(workspace)
            if cached and cached[0] == version:
                return cached[1], cached[2], cached[3]
            dim = int(self._meta(conn, "embedding_dim") or 0)
            rows = conn.execute(
                "select id, acl, embedding from chunks where workspace = ? and embedding is not null order by rid",
                (workspace,),
            ).fetchall()
        ids = [r["id"] for r in rows]
        acls = [json.loads(r["acl"]) for r in rows]
        matrix = (
            np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), dim)
            if rows and dim
            else np.zeros((0, max(dim, 1)), dtype=np.float32)
        )
        self._vectors[workspace] = (version, ids, acls, matrix)
        return ids, acls, matrix

    def dense_search(self, vector: np.ndarray, f: AccessFilter, k: int) -> list[tuple[str, float]]:
        ids, acls, matrix = self._matrix(f.workspace)
        if not ids:
            return []
        groups = set(f.groups)
        allowed = np.fromiter((not a or bool(groups & set(a)) for a in acls), dtype=bool, count=len(acls))
        scores = matrix @ vector.astype(np.float32)
        scores = np.where(allowed, scores, -np.inf)
        n = min(k, int(allowed.sum()))
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(ids[i], float(scores[i])) for i in top]

    def get_chunks(self, ids: list[str]) -> dict[str, ChunkRecord]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._tx() as conn:
            rows = conn.execute(f"select * from chunks where id in ({marks})", ids).fetchall()
        return {r["id"]: _row_to_chunk(r) for r in rows}

    # ---- answers & feedback -------------------------------------------------------------

    def log_answer(self, answer: Answer, principal_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "insert or replace into answers values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    answer.id,
                    _now(),
                    principal_id,
                    answer.workspace,
                    answer.question,
                    answer.text,
                    int(answer.abstained),
                    answer.abstain_reason,
                    json.dumps([s.chunk_id for s in answer.sources]),
                    answer.mode,
                    json.dumps(answer.timings_ms),
                ),
            )

    def add_feedback(self, answer_id: str, principal_id: str, rating: int, comment: str | None) -> bool:
        with self._tx() as conn:
            owner = conn.execute("select principal from answers where id = ?", (answer_id,)).fetchone()
            if not owner or owner["principal"] != principal_id:
                return False
            conn.execute(
                "insert into feedback (answer_id, principal, rating, comment, created_at) values (?, ?, ?, ?, ?)",
                (answer_id, principal_id, rating, comment, _now()),
            )
        return True

    def list_feedback(self) -> list[dict[str, Any]]:
        with self._tx() as conn:
            rows = conn.execute(
                "select f.rating, f.comment, f.created_at, a.workspace, a.principal, a.question, a.answer, "
                "a.abstained, a.sources from feedback f join answers a on a.id = f.answer_id order by f.id"
            ).fetchall()
        return [dict(r) | {"sources": json.loads(r["sources"])} for r in rows]


def _row_to_chunk(r: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        id=r["id"],
        workspace=r["workspace"],
        path=r["path"],
        ordinal=r["ordinal"],
        title=r["title"],
        headings=json.loads(r["headings"]),
        text=r["text"],
        url=r["url"],
        acl=json.loads(r["acl"]),
        flags=json.loads(r["flags"]),
        word_count=r["word_count"] or 0,
        keywords=r["keywords"].split() if r["keywords"] else [],
    )
