"""Test setup: a tiny two-workspace corpus, built into SQLite (always) and Postgres (when
TEST_DATABASE_URL points at a database with pgvector, as it does in CI)."""

from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

import pytest

from citewise.config import Settings
from citewise.ingest.pipeline import ingest_workspace
from citewise.runtime import Runtime

FIXTURES = Path(__file__).parent / "fixtures" / "handbook"

WORKSPACES_YAML = """
workspaces:
  acme:
    name: Acme Handbook
    description: Fixture handbook
    repo: https://example.invalid/acme.git
    commit: "0000000000000000000000000000000000000000"
    path: handbook
    license: test
    url: {base: "https://acme.example/handbook"}
    examples: ["How much can I spend per month?"]
  globex:
    name: Globex Handbook
    description: Second fixture handbook
    repo: https://example.invalid/globex.git
    commit: "0000000000000000000000000000000000000000"
    path: docs
    license: test
    url: {base: "https://globex.example", trailing_slash: true}
"""

ACCESS_YAML = """
rules:
  acme:
    - match: "people/hiring/**"
      groups: [hiring]
principals:
  ana: {name: Ana, api_key: key-ana, workspaces: [acme], groups: []}
  hal: {name: Hal, api_key: key-hal, workspaces: [acme], groups: [hiring]}
  gus: {name: Gus, api_key: key-gus, workspaces: [globex], groups: []}
  root: {name: Root, api_key: key-root, workspaces: [acme, globex], groups: [hiring]}
"""

GLOBEX_DOC = textwrap.dedent("""\
    ---
    title: Globex benefits
    ---

    ## Commuter benefit

    Globex pays for a monthly transit pass for every employee who commutes by train or bus.
    Sign up through the benefits portal before the 20th of the month.
""")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    shutil.copytree(FIXTURES, data / "corpora" / "acme" / "handbook")
    (data / "corpora" / "globex" / "docs").mkdir(parents=True)
    (data / "corpora" / "globex" / "docs" / "benefits.md").write_text(GLOBEX_DOC)
    (tmp_path / "workspaces.yaml").write_text(WORKSPACES_YAML)
    (tmp_path / "access.yaml").write_text(ACCESS_YAML)
    return tmp_path


def _reset_postgres(url: str) -> None:
    import psycopg

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("drop table if exists feedback, answers, chunks, documents, meta cascade")


@pytest.fixture(params=["sqlite", "postgres"])
def database_url(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    if request.param == "sqlite":
        return f"sqlite:///{tmp_path / 'test.db'}"
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    _reset_postgres(url)
    return url


def make_settings(corpus: Path, database_url: str, **overrides: object) -> Settings:
    values: dict[str, object] = dict(
        data_dir=corpus / "data",
        database_url=database_url,
        workspaces_file=corpus / "workspaces.yaml",
        access_file=corpus / "access.yaml",
        embedding_model="hashing:256",
        reranker_model="none",
        llm_provider="extractive",
        abstain_min_rerank=0.0,
        abstain_min_dense=0.0,
        web_dist=corpus / "no-web",
    )
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def build(rt: Runtime) -> Runtime:
    emb = rt.embedder
    rt.store.ensure_schema(emb.name if emb else None, emb.dim if emb else None)
    for ws in rt.workspaces.values():
        ingest_workspace(ws, ws.source_dir(rt.settings.data_dir), rt.store, emb, rt.access)
    return rt


@pytest.fixture
def runtime(corpus: Path, database_url: str) -> Runtime:
    return build(Runtime(make_settings(corpus, database_url)))
