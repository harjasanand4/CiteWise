"""Postgres store: pgvector (HNSW, cosine) for vectors, tsvector + GIN for keyword search.

Access filtering happens in the WHERE clause. With an HNSW index that matters: the index
returns its nearest neighbours first and the filter runs after, so a selective filter can
leave you with fewer than k rows. pgvector 0.8+ fixes this with iterative index scans,
which are switched on per query when the server supports them.
"""

from __future__ import annotations

import json
import threading
from typing import Any, cast

import numpy as np

from ..models import Answer, ChunkRecord, DocumentRecord
from .base import AccessFilter, IndexMismatchError, query_terms

try:  # optional dependency: pip install "citewise[postgres]"
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError('Postgres support needs: pip install "citewise[postgres]"') from exc

BASE_SCHEMA = """
create extension if not exists vector;
create table if not exists meta (key text primary key, value text not null);
create table if not exists documents (
  workspace text not null, path text not null, title text, url text, content_hash text not null,
  acl text[] not null default '{}', ingested_at timestamptz default now(), primary key (workspace, path));
create table if not exists answers (
  id text primary key, created_at timestamptz default now(), principal text, workspace text, question text,
  answer text, abstained boolean, abstain_reason text, sources jsonb, mode text, timings jsonb);
create table if not exists feedback (
  id bigserial primary key, answer_id text not null references answers(id) on delete cascade,
  principal text, rating smallint not null, comment text, created_at timestamptz default now());
"""

CHUNKS_SCHEMA = """
create table if not exists chunks (
  id text primary key, workspace text not null, path text not null, ordinal int not null,
  title text, headings text[] not null default '{{}}', section text not null default '', text text not null, url text,
  acl text[] not null default '{{}}', flags text[] not null default '{{}}', keywords text not null default '',
  word_count int,
  tsv tsvector generated always as (
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', section), 'B') ||
    setweight(to_tsvector('english', keywords), 'B') ||
    setweight(to_tsvector('english', text), 'C')) stored,
  embedding {vector_type},
  foreign key (workspace, path) references documents (workspace, path) on delete cascade);
create index if not exists chunks_doc_idx on chunks (workspace, path);
create index if not exists chunks_tsv_idx on chunks using gin (tsv);
create index if not exists chunks_acl_idx on chunks using gin (acl);
"""

HNSW_INDEX = (
    "create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops)"
)
ACL_CLAUSE = "(cardinality(acl) = 0 or acl && %(groups)s::text[])"

Row = dict[str, Any]


def _one(conn: Any, sql: str, params: Any = None) -> Row | None:
    """Rows come back as dicts (dict_row); these helpers just give them a stable type."""
    return cast("Row | None", conn.execute(sql, params).fetchone())


def _all(conn: Any, sql: str, params: Any = None) -> list[Row]:
    return cast("list[Row]", conn.execute(sql, params).fetchall())


class PostgresStore:
    def __init__(self, url: str) -> None:
        self.url = url
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()
        self._iterative_scan: bool | None = None
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(BASE_SCHEMA)

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            with self._lock:
                if self._pool is None:

                    def configure(conn: psycopg.Connection) -> None:
                        register_vector(conn)
                        conn.commit()

                    self._pool = ConnectionPool(
                        self.url,
                        min_size=1,
                        max_size=8,
                        configure=configure,
                        kwargs={"row_factory": dict_row},
                        open=True,
                    )
        return self._pool

    def _meta(self, conn: Any, key: str) -> str | None:
        row = _one(conn, "select value from meta where key = %s", (key,))
        return row["value"] if row else None

    def _has_chunks_table(self, conn: Any) -> bool:
        row = _one(conn, "select to_regclass('chunks') is not null as e")
        return bool(row and row["e"])

    # ---- schema / admin -----------------------------------------------------------------

    def ensure_schema(self, embedding_name: str | None, dim: int | None) -> None:
        with self.pool.connection() as conn:
            exists = self._has_chunks_table(conn)
            if exists:
                existing = self._meta(conn, "embedding_model")
                has_rows = bool((_one(conn, "select exists (select 1 from chunks) as e") or {}).get("e"))
                if existing != (embedding_name or "none"):
                    if has_rows:
                        raise IndexMismatchError(
                            f"Index was built with embeddings {existing!r} but {embedding_name or 'none'!r} is "
                            "configured. Run `citewise ingest --rebuild`."
                        )
                    conn.execute("drop table chunks")
                    exists = False
            if not exists:
                conn.execute(CHUNKS_SCHEMA.format(vector_type=f"vector({dim})" if dim else "vector"))
                if dim:
                    conn.execute(HNSW_INDEX)
            conn.execute(
                "insert into meta values ('embedding_model', %s), ('embedding_dim', %s) "
                "on conflict (key) do update set value = excluded.value",
                (embedding_name or "none", str(dim or 0)),
            )

    def reset(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("drop table if exists chunks")
            conn.execute("delete from documents")
            conn.execute("delete from meta where key in ('embedding_model', 'embedding_dim')")

    def index_info(self) -> dict[str, Any]:
        with self.pool.connection() as conn:
            info: dict[str, Any] = {
                "backend": "postgres",
                "embedding_model": self._meta(conn, "embedding_model"),
                "embedding_dim": int(self._meta(conn, "embedding_dim") or 0),
                "workspaces": {},
            }
            if self._has_chunks_table(conn):
                rows = _all(
                    conn,
                    "select workspace, count(distinct path) as docs, count(*) as chunks from chunks group by workspace",
                )
                info["workspaces"] = {
                    r["workspace"]: {"documents": r["docs"], "chunks": r["chunks"]} for r in rows
                }
            return info

    def ping(self) -> bool:
        with self.pool.connection() as conn:
            conn.execute("select 1")
        return True

    # ---- documents ----------------------------------------------------------------------

    def document_hashes(self, workspace: str) -> dict[str, str]:
        with self.pool.connection() as conn:
            rows = _all(conn, "select path, content_hash from documents where workspace = %s", (workspace,))
        return {r["path"]: r["content_hash"] for r in rows}

    def upsert_document(
        self, doc: DocumentRecord, chunks: list[ChunkRecord], vectors: np.ndarray | None
    ) -> None:
        with self.pool.connection() as conn, conn.transaction():
            # Deleting the document cascades to its chunks.
            conn.execute(
                "delete from documents where workspace = %s and path = %s", (doc.workspace, doc.path)
            )
            conn.execute(
                "insert into documents (workspace, path, title, url, content_hash, acl) values (%s, %s, %s, %s, %s, %s)",
                (doc.workspace, doc.path, doc.title, doc.url, doc.content_hash, doc.acl),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into chunks (id, workspace, path, ordinal, title, headings, section, text, url, acl, flags, "
                    "keywords, word_count, embedding) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            c.id,
                            c.workspace,
                            c.path,
                            c.ordinal,
                            c.title,
                            c.headings,
                            " ".join(c.headings),
                            c.text,
                            c.url,
                            c.acl,
                            c.flags,
                            " ".join(c.keywords),
                            c.word_count,
                            vectors[i].astype(np.float32) if vectors is not None else None,
                        )
                        for i, c in enumerate(chunks)
                    ],
                )

    def delete_documents(self, workspace: str, paths: list[str]) -> None:
        if not paths:
            return
        with self.pool.connection() as conn:
            conn.execute("delete from documents where workspace = %s and path = any(%s)", (workspace, paths))

    # ---- search -------------------------------------------------------------------------

    def lexical_search(self, query: str, f: AccessFilter, k: int) -> list[tuple[str, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        # OR the terms: plainto_tsquery would AND them, which misses most natural questions.
        tsquery = " | ".join(terms)
        sql = (
            "select id, ts_rank(tsv, q, 1) as score from chunks, to_tsquery('english', %(q)s) q "
            f"where tsv @@ q and workspace = %(ws)s and {ACL_CLAUSE} order by score desc limit %(k)s"
        )
        with self.pool.connection() as conn:
            rows = _all(conn, sql, {"q": tsquery, "ws": f.workspace, "groups": list(f.groups), "k": k})
        return [(r["id"], float(r["score"])) for r in rows]

    def dense_search(self, vector: np.ndarray, f: AccessFilter, k: int) -> list[tuple[str, float]]:
        sql = (
            "select id, 1 - (embedding <=> %(v)s) as score from chunks "
            f"where workspace = %(ws)s and embedding is not null and {ACL_CLAUSE} "
            "order by embedding <=> %(v)s limit %(k)s"
        )
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("set local hnsw.ef_search = 200")
            if self._iterative_scan is not False:
                try:
                    with conn.transaction():  # savepoint: an unknown setting must not abort the query
                        conn.execute("set local hnsw.iterative_scan = relaxed_order")
                    self._iterative_scan = True
                except psycopg.Error:
                    self._iterative_scan = False  # pgvector < 0.8
            rows = _all(
                conn,
                sql,
                {"v": vector.astype(np.float32), "ws": f.workspace, "groups": list(f.groups), "k": k},
            )
        return [(r["id"], float(r["score"])) for r in rows]

    def get_chunks(self, ids: list[str]) -> dict[str, ChunkRecord]:
        if not ids:
            return {}
        with self.pool.connection() as conn:
            rows = _all(
                conn,
                "select id, workspace, path, ordinal, title, headings, text, url, acl, flags, keywords, word_count "
                "from chunks where id = any(%s)",
                (ids,),
            )
        return {
            r["id"]: ChunkRecord(
                id=r["id"],
                workspace=r["workspace"],
                path=r["path"],
                ordinal=r["ordinal"],
                title=r["title"],
                headings=list(r["headings"]),
                text=r["text"],
                url=r["url"],
                acl=list(r["acl"]),
                flags=list(r["flags"]),
                word_count=r["word_count"] or 0,
                keywords=r["keywords"].split() if r["keywords"] else [],
            )
            for r in rows
        }

    # ---- answers & feedback -------------------------------------------------------------

    def log_answer(self, answer: Answer, principal_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "insert into answers (id, principal, workspace, question, answer, abstained, abstain_reason, "
                "sources, mode, timings) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict (id) do nothing",
                (
                    answer.id,
                    principal_id,
                    answer.workspace,
                    answer.question,
                    answer.text,
                    answer.abstained,
                    answer.abstain_reason,
                    json.dumps([s.chunk_id for s in answer.sources]),
                    answer.mode,
                    json.dumps(answer.timings_ms),
                ),
            )

    def add_feedback(self, answer_id: str, principal_id: str, rating: int, comment: str | None) -> bool:
        with self.pool.connection() as conn:
            owner = _one(conn, "select principal from answers where id = %s", (answer_id,))
            if not owner or owner["principal"] != principal_id:
                return False
            conn.execute(
                "insert into feedback (answer_id, principal, rating, comment) values (%s, %s, %s, %s)",
                (answer_id, principal_id, rating, comment),
            )
        return True

    def list_feedback(self) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = _all(
                conn,
                "select f.rating, f.comment, f.created_at::text as created_at, a.workspace, a.principal, a.question, "
                "a.answer, a.abstained, a.sources from feedback f join answers a on a.id = f.answer_id order by f.id",
            )
        return [dict(r) for r in rows]
