"""Incremental ingestion: parse -> chunk -> embed -> store, skipping unchanged documents.

A document's hash covers its bytes, its access rule and the chunking settings, so editing
access.yaml or changing chunk sizes re-indexes exactly the documents affected.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AccessConfig, Workspace
from ..embeddings import Embedder
from ..generation.guardrails import looks_like_injection
from ..models import ChunkRecord, DocumentRecord
from ..store.base import Store
from .chunker import chunk_document
from .markdown import parse_markdown, slugify

PIPELINE_VERSION = "5"  # bump when parsing/chunking logic changes, to force a re-index
EMBED_BATCH = 256


@dataclass
class IngestReport:
    workspace: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    chunks: int = 0
    flagged: list[str] = field(default_factory=list)
    seconds: float = 0.0


def document_hash(raw: bytes, acl: list[str], max_words: int, min_words: int) -> str:
    h = hashlib.sha256(raw)
    h.update(f"|acl={','.join(acl)}|chunk={max_words},{min_words}|v={PIPELINE_VERSION}".encode())
    return h.hexdigest()


def build_chunks(
    ws: Workspace, rel: str, raw: str, acl: list[str], max_words: int, min_words: int
) -> tuple[DocumentRecord | None, list[ChunkRecord], str | None]:
    """Parse one file into a DocumentRecord and its chunks. Returns a skip reason if not indexable."""
    doc = parse_markdown(raw, fallback_title=Path(rel).stem.replace("-", " ").capitalize())
    if doc.skip_reason:
        return None, [], doc.skip_reason
    page_url = ws.page_url(rel, doc.permalink)
    chunks: list[ChunkRecord] = []
    for i, piece in enumerate(chunk_document(doc, max_words=max_words, min_words=min_words)):
        anchor = f"#{slugify(piece.headings[-1])}" if piece.headings else ""
        chunks.append(
            ChunkRecord(
                id=f"{ws.id}:{rel}#{i}",
                workspace=ws.id,
                path=rel,
                ordinal=i,
                title=doc.title,
                headings=piece.headings,
                text=piece.text,
                url=page_url + anchor,
                acl=acl,
                flags=["possible_injection"] if looks_like_injection(piece.text) else [],
                word_count=piece.words,
                keywords=doc.keywords,
            )
        )
    if not chunks:
        return None, [], "empty"
    record = DocumentRecord(
        workspace=ws.id, path=rel, title=doc.title, url=page_url, content_hash="", acl=acl
    )
    return record, chunks, None


def ingest_workspace(
    ws: Workspace,
    root: Path,
    store: Store,
    embedder: Embedder | None,
    access: AccessConfig,
    max_words: int = 220,
    min_words: int = 40,
    progress: Callable[[str], None] | None = None,
) -> IngestReport:
    started = time.perf_counter()
    report = IngestReport(workspace=ws.id)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run `citewise fetch` first.")

    existing = store.document_hashes(ws.id)
    seen: set[str] = set()
    pending: list[tuple[DocumentRecord, list[ChunkRecord]]] = []

    def flush() -> None:
        if not pending:
            return
        texts = [c.embed_text for _, chunks in pending for c in chunks]
        vectors = embedder.embed_documents(texts) if embedder else None
        offset = 0
        for doc, chunks in pending:
            vecs = vectors[offset : offset + len(chunks)] if vectors is not None else None
            store.upsert_document(doc, chunks, vecs)
            offset += len(chunks)
        if progress:
            progress(f"{ws.id}: indexed {report.added + report.updated} documents, {report.chunks} chunks")
        pending.clear()

    files = sorted(p for p in root.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(root).as_posix()
        if not ws.wants(rel):
            continue
        seen.add(rel)
        raw_bytes = path.read_bytes()
        acl = access.acl_for(ws.id, rel)
        digest = document_hash(raw_bytes, acl, max_words, min_words)
        if existing.get(rel) == digest:
            report.unchanged += 1
            continue

        doc, chunks, skip = build_chunks(
            ws, rel, raw_bytes.decode("utf-8", errors="replace"), acl, max_words, min_words
        )
        if skip or doc is None:
            report.skipped[skip or "empty"] = report.skipped.get(skip or "empty", 0) + 1
            if rel in existing:
                store.delete_documents(ws.id, [rel])
            continue

        doc.content_hash = digest
        if rel in existing:
            report.updated += 1
        else:
            report.added += 1
        report.chunks += len(chunks)
        report.flagged.extend(c.id for c in chunks if c.flags)
        pending.append((doc, chunks))
        if sum(len(c) for _, c in pending) >= EMBED_BATCH:
            flush()
    flush()

    gone = sorted(set(existing) - seen)
    if gone:
        store.delete_documents(ws.id, gone)
        report.deleted = len(gone)
    report.seconds = round(time.perf_counter() - started, 2)
    return report
