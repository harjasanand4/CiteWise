"""Settings (from env / .env) and the two YAML files: workspaces and access rules."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import Principal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITEWISE_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/citewise.db"
    workspaces_file: Path = Path("config/workspaces.yaml")
    access_file: Path = Path("config/access.yaml")

    # "fastembed:<model>", "wordllama", "hashing" (tests) or "none" (lexical only)
    embedding_model: str = "fastembed:BAAI/bge-small-en-v1.5"
    # "fastembed:<cross-encoder>" or "none"
    reranker_model: str = "fastembed:Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    candidates: int = 30  # how many each retriever returns before fusion / reranking
    top_k: int = 6  # how many sources go into the prompt
    rrf_k: int = 60

    # Abstain before calling the LLM when the best source is this weak. Chosen with
    # `python -m evals.run --sweep` (see README). 0 disables the gate.
    abstain_min_rerank: float = 0.02  # sigmoid of the cross-encoder logit
    abstain_min_dense: float = 0.0  # cosine similarity, used only when there is no reranker

    chunk_max_words: int = 220
    chunk_min_words: int = 40

    llm_provider: Literal["extractive", "groq", "openai", "ollama", "openai_compatible"] = "extractive"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 4
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700

    # "demo": API keys from access.yaml, and the UI can switch between demo users.
    # "disabled": no auth, every request is a local admin. Never expose either mode publicly
    # without putting real auth (OIDC) in front.
    auth_mode: Literal["demo", "disabled"] = "demo"
    cors_origins: list[str] = ["http://localhost:5173"]
    web_dist: Path = Path("web/dist")
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---- workspaces.yaml --------------------------------------------------------------------


@dataclass
class UrlRule:
    base: str
    trailing_slash: bool = False


@dataclass
class Workspace:
    id: str
    name: str
    description: str
    repo: str
    commit: str
    path: str
    license: str
    url: UrlRule
    include: list[str] = field(default_factory=lambda: ["**/*.md", "**/*.mdx"])
    exclude: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def source_dir(self, data_dir: Path) -> Path:
        return data_dir / "corpora" / self.id / self.path

    def wants(self, rel_path: str) -> bool:
        def match(patterns: list[str]) -> bool:
            # "**/*.md" should also match top-level files, which fnmatch's "**/" does not
            return any(
                fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(rel_path, p.removeprefix("**/"))
                for p in patterns
            )

        return match(self.include) and not match(self.exclude)

    def page_url(self, rel_path: str, permalink: str | None = None) -> str:
        if permalink:
            return (
                self.url.base.rstrip("/")
                + "/"
                + permalink.strip("/")
                + ("/" if self.url.trailing_slash else "")
            )
        stem = rel_path.rsplit(".", 1)[0]
        parts = stem.split("/")
        if parts[-1].lower() in {"index", "readme"}:
            parts = parts[:-1]
        slug = "/".join(parts)
        url = self.url.base.rstrip("/") + "/" + slug
        if self.url.trailing_slash and slug:
            url += "/"
        return url


def load_workspaces(path: Path) -> dict[str, Workspace]:
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, Workspace] = {}
    for wid, w in (raw.get("workspaces") or {}).items():
        out[wid] = Workspace(
            id=wid,
            name=w["name"],
            description=w.get("description", ""),
            repo=w["repo"],
            commit=w["commit"],
            path=w["path"],
            license=w.get("license", ""),
            url=UrlRule(**w["url"]),
            include=w.get("include", ["**/*.md", "**/*.mdx"]),
            exclude=w.get("exclude", []),
            examples=w.get("examples", []),
        )
    return out


# ---- access.yaml ------------------------------------------------------------------------


@dataclass
class AccessRule:
    match: str
    groups: list[str]


@dataclass
class AccessConfig:
    rules: dict[str, list[AccessRule]]
    principals: dict[str, Principal]
    api_keys: dict[str, str]  # api key -> principal id

    def acl_for(self, workspace: str, rel_path: str) -> list[str]:
        """Groups allowed to read a document. First matching rule wins; no match = everyone."""
        for rule in self.rules.get(workspace, []):
            if fnmatch.fnmatch(rel_path, rule.match):
                return sorted(rule.groups)
        return []

    def principal_for_key(self, api_key: str) -> Principal | None:
        pid = self.api_keys.get(api_key)
        return self.principals.get(pid) if pid else None


def load_access(path: Path) -> AccessConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    rules = {
        ws: [AccessRule(match=r["match"], groups=list(r["groups"])) for r in items]
        for ws, items in (raw.get("rules") or {}).items()
    }
    principals: dict[str, Principal] = {}
    keys: dict[str, str] = {}
    for pid, p in (raw.get("principals") or {}).items():
        principals[pid] = Principal(
            id=pid,
            name=p.get("name", pid),
            workspaces=tuple(p.get("workspaces", [])),
            groups=tuple(p.get("groups", [])),
        )
        if p.get("api_key"):
            keys[p["api_key"]] = pid
    return AccessConfig(rules=rules, principals=principals, api_keys=keys)
