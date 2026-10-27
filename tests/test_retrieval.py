"""Retrieval and ingestion, run against every store backend (SQLite always, Postgres in CI)."""

from pathlib import Path

import pytest

from citewise.ingest.pipeline import ingest_workspace
from citewise.retrieval.retriever import Retriever, rrf
from citewise.runtime import Runtime
from citewise.store.base import IndexMismatchError, query_terms

from .conftest import build, make_settings


def paths(result):
    return [h.chunk.path for h in result.hits]


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert max(fused, key=fused.get) == "b"
    assert fused["d"] < fused["c"]


def test_query_terms_drop_stopwords_and_keep_numbers():
    assert query_terms("What is the limit for 2 people?") == ["limit", "2", "people"]


@pytest.mark.parametrize("mode", ["lexical", "dense", "hybrid"])
def test_finds_the_right_page(runtime: Runtime, mode: str):
    ana = runtime.access.principals["ana"]
    res = runtime.retriever.retrieve("How long do I have to upload receipts?", "acme", ana, k=3, mode=mode)
    assert paths(res)[0] == "people/expenses.mdx"
    assert res.hits[0].chunk.url.startswith("https://acme.example/handbook/people/expenses#")


def test_restricted_pages_never_reach_principals_without_the_group(runtime: Runtime):
    q = "How much is the referral bonus for a hire?"
    ana, hal = runtime.access.principals["ana"], runtime.access.principals["hal"]
    for mode in ("lexical", "dense", "hybrid"):
        assert "people/hiring/process.md" not in paths(
            runtime.retriever.retrieve(q, "acme", ana, k=10, mode=mode)
        )
        assert (
            paths(runtime.retriever.retrieve(q, "acme", hal, k=3, mode=mode))[0] == "people/hiring/process.md"
        )


def test_workspaces_are_isolated(runtime: Runtime):
    root, gus = runtime.access.principals["root"], runtime.access.principals["gus"]
    in_acme = runtime.retriever.retrieve("monthly transit pass commuter benefit", "acme", root, k=10)
    assert all(h.chunk.workspace == "acme" for h in in_acme.hits)
    assert runtime.retriever.retrieve("transit pass", "acme", gus).hits == []  # gus isn't an acme member
    assert paths(runtime.retriever.retrieve("transit pass", "globex", gus, k=1)) == ["benefits.md"]


def test_ingest_report_skips_redirects_and_flags_injection(corpus: Path, database_url: str):
    rt = Runtime(make_settings(corpus, database_url))
    rt.store.ensure_schema(rt.embedder.name, rt.embedder.dim)  # type: ignore[union-attr]
    ws = rt.workspaces["acme"]
    report = ingest_workspace(ws, ws.source_dir(rt.settings.data_dir), rt.store, rt.embedder, rt.access)
    assert report.added == 6
    assert report.skipped == {"redirect": 1}
    assert any("security/agents.md" in cid for cid in report.flagged)


def test_reingest_is_incremental(runtime: Runtime, corpus: Path):
    ws = runtime.workspaces["acme"]
    root = ws.source_dir(runtime.settings.data_dir)

    def again():
        return ingest_workspace(ws, root, runtime.store, runtime.embedder, runtime.access)

    first = again()
    assert first.unchanged == 6 and first.added == first.updated == first.chunks == 0

    page = root / "people" / "time-off.md"
    page.write_text(page.read_text().replace("25 days", "30 days"))
    (root / "engineering" / "incidents.md").unlink()
    second = again()
    assert (second.updated, second.deleted, second.unchanged) == (1, 1, 4)

    ana = runtime.access.principals["ana"]
    hits = runtime.retriever.retrieve("minimum days off a year", "acme", ana, k=1).hits
    assert "30 days" in hits[0].chunk.text
    after_delete = runtime.retriever.retrieve("declare an incident in slack", "acme", ana, k=10)
    assert "engineering/incidents.md" not in paths(after_delete)


def test_changing_an_access_rule_reindexes_the_affected_pages(runtime: Runtime, corpus: Path):
    access_file = corpus / "access.yaml"
    access_file.write_text(access_file.read_text().replace('"people/hiring/**"', '"people/**"'))
    rt = Runtime(runtime.settings)  # reload config
    ws = rt.workspaces["acme"]
    report = ingest_workspace(ws, ws.source_dir(rt.settings.data_dir), rt.store, rt.embedder, rt.access)
    assert report.updated == 2  # expenses.mdx and time-off.md now carry the hiring ACL
    ana = rt.access.principals["ana"]
    assert all(
        not h.chunk.path.startswith("people/")
        for h in rt.retriever.retrieve("receipts", "acme", ana, k=10).hits
    )


def test_switching_embedding_models_requires_a_rebuild(corpus: Path, database_url: str):
    build(Runtime(make_settings(corpus, database_url)))
    other = Runtime(make_settings(corpus, database_url, embedding_model="hashing:128"))
    with pytest.raises(IndexMismatchError):
        other.store.ensure_schema(other.embedder.name, other.embedder.dim)  # type: ignore[union-attr]
    other.store.reset()
    build(other)
    assert other.store.index_info()["embedding_dim"] == 128


def test_lexical_only_mode_works_without_embeddings(corpus: Path, database_url: str):
    rt = build(Runtime(make_settings(corpus, database_url, embedding_model="none")))
    assert rt.retriever.mode == "lexical"
    res = rt.retriever.retrieve("parental leave weeks", "acme", rt.access.principals["ana"], k=1)
    assert paths(res) == ["people/time-off.md"]


class OverlapReranker:
    """Stand-in for the cross-encoder: scores by shared words, deterministic."""

    name = "test-overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(query_terms(query))
        return [float(len(q & set(query_terms(p)))) * 2 - 3 for p in passages]


def test_reranker_reorders_and_sets_scores(runtime: Runtime):
    r = Retriever(runtime.store, runtime.embedder, OverlapReranker(), mode="hybrid")
    res = r.retrieve("declare an incident in slack", "acme", runtime.access.principals["ana"], k=3)
    assert paths(res)[0] == "engineering/incidents.md"
    assert all(h.rerank_score is not None and 0 <= h.rerank_score <= 1 for h in res.hits)
    assert res.hits == sorted(res.hits, key=lambda h: -h.score)
