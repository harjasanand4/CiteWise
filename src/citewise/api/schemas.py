from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    workspace: str
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class SearchRequest(BaseModel):
    workspace: str
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=8, ge=1, le=30)
    mode: Literal["lexical", "dense", "hybrid"] | None = None
    rerank: bool | None = None


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=1, max_length=64)
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=1000)


class WorkspaceOut(BaseModel):
    id: str
    name: str
    description: str
    license: str
    examples: list[str]


class PrincipalOut(BaseModel):
    id: str
    name: str
    groups: list[str]
    workspaces: list[WorkspaceOut]


class DemoUser(BaseModel):
    id: str
    name: str
    api_key: str
    groups: list[str]
    workspaces: list[str]
