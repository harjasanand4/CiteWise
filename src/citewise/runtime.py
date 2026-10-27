"""Builds the object graph (store, models, retriever, answerer) from settings. Shared by the CLI and API."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .config import AccessConfig, Settings, Workspace, load_access, load_workspaces
from .embeddings import Embedder, get_embedder
from .generation.answerer import Answerer
from .generation.llm import LLMClient, build_llm
from .retrieval.retriever import Reranker, Retriever, get_reranker
from .store.base import Store, open_store


@dataclass
class Runtime:
    settings: Settings

    @cached_property
    def workspaces(self) -> dict[str, Workspace]:
        return load_workspaces(self.settings.workspaces_file)

    @cached_property
    def access(self) -> AccessConfig:
        return load_access(self.settings.access_file)

    @cached_property
    def store(self) -> Store:
        return open_store(self.settings.database_url)

    @cached_property
    def embedder(self) -> Embedder | None:
        return get_embedder(self.settings.embedding_model)

    @cached_property
    def reranker(self) -> Reranker | None:
        return get_reranker(self.settings.reranker_model)

    @cached_property
    def llm(self) -> LLMClient | None:
        s = self.settings
        key = s.llm_api_key.get_secret_value() if s.llm_api_key else None
        return build_llm(
            s.llm_provider,
            s.llm_base_url,
            s.llm_model,
            key,
            s.llm_timeout_seconds,
            s.llm_max_retries,
            s.llm_temperature,
            s.llm_max_tokens,
        )

    @cached_property
    def retriever(self) -> Retriever:
        s = self.settings
        self.store.ensure_schema(
            self.embedder.name if self.embedder else None, self.embedder.dim if self.embedder else None
        )
        return Retriever(
            self.store,
            self.embedder,
            self.reranker,
            mode=s.retrieval_mode,
            candidates=s.candidates,
            rrf_k=s.rrf_k,
        )

    @cached_property
    def answerer(self) -> Answerer:
        s = self.settings
        return Answerer(
            self.retriever,
            self.llm,
            {w.id: w.name for w in self.workspaces.values()},
            top_k=s.top_k,
            abstain_min_rerank=s.abstain_min_rerank,
            abstain_min_dense=s.abstain_min_dense,
        )
