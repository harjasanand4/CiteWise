"""Core data types shared by ingestion, storage, retrieval and the API."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class DocumentRecord:
    workspace: str
    path: str  # relative to the workspace root, e.g. "people/spending-money.md"
    title: str
    url: str
    content_hash: str
    acl: list[str] = field(default_factory=list)  # empty = every member of the workspace


@dataclass
class ChunkRecord:
    id: str  # "{workspace}:{path}#{ordinal}"
    workspace: str
    path: str
    ordinal: int
    title: str
    headings: list[str]
    text: str
    url: str
    acl: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    word_count: int = 0
    keywords: list[str] = field(default_factory=list)  # page-level search keywords (lexical index only)

    @property
    def breadcrumb(self) -> str:
        return " > ".join([self.title, *self.headings])

    @property
    def embed_text(self) -> str:
        # A contextual header (doc title + section path) is embedded with the chunk, so a
        # paragraph that only says "it's 14 days" still knows it is about expense receipts.
        return f"{self.breadcrumb}\n{self.text}"


@dataclass(frozen=True)
class Principal:
    """Who is asking. Retrieval is filtered by this before anything is ranked."""

    id: str
    name: str
    workspaces: tuple[str, ...]
    groups: tuple[str, ...] = ()

    def can_read(self, workspace: str, acl: list[str]) -> bool:
        if workspace not in self.workspaces:
            return False
        return not acl or bool(set(acl) & set(self.groups))


@dataclass
class Hit:
    chunk: ChunkRecord
    score: float  # the score the final ranking used
    lexical_rank: int | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rerank_score: float | None = None  # sigmoid of the cross-encoder logit, 0..1


# ---- API / answer types -----------------------------------------------------------------


class SourceOut(BaseModel):
    n: int
    chunk_id: str
    title: str
    headings: list[str]
    url: str
    text: str
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None
    flags: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    id: str
    workspace: str
    question: str
    standalone_question: str
    text: str
    abstained: bool
    abstain_reason: str | None = None
    cited: list[int] = Field(default_factory=list)  # source numbers the answer actually cites
    invalid_citations: list[int] = Field(default_factory=list)
    grounded: bool = True  # False if the model answered without citing anything
    sources: list[SourceOut] = Field(default_factory=list)
    mode: str = ""  # "extractive" or the LLM model name
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
