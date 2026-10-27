"""The eval set is data, and data needs tests: ids unique, labels consistent with the config."""

import json
from pathlib import Path

from citewise.config import load_access, load_workspaces
from evals.metrics import dedupe, hit_at, key_fact_recall, ndcg_at, normalize, reciprocal_rank

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "evals" / "datasets" / "handbook_qa.jsonl"


def rows():
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


def test_ranking_metrics():
    ranked = ["a", "b", "c"]
    assert hit_at(ranked, {"b"}, 1) == 0.0 and hit_at(ranked, {"b"}, 2) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == 1 / 3
    assert ndcg_at(ranked, {"a"}) == 1.0
    assert 0 < ndcg_at(ranked, {"b", "z"}) < 1
    assert dedupe(["a", "a", "b", "a"]) == ["a", "b"]


def test_key_fact_matching_is_format_tolerant():
    assert normalize("You get **$5,000** a month") == "you get $5000 a month"
    assert key_fact_recall("Limit: $5,000 per month.", ["5000"]) == 1.0
    assert key_fact_recall("Tuesday only", ["Tuesday", "Thursday"]) == 0.5
    assert key_fact_recall("anything", []) is None


def test_dataset_is_consistent_with_config():
    workspaces = load_workspaces(ROOT / "config" / "workspaces.yaml")
    access = load_access(ROOT / "config" / "access.yaml")
    data = rows()
    assert len({r["id"] for r in data}) == len(data), "duplicate ids"
    for r in data:
        assert r["type"] in {"answerable", "unanswerable", "restricted"}, r["id"]
        assert r["workspace"] in workspaces, r["id"]
        principal = access.principals[r["as"]]
        assert r["workspace"] in principal.workspaces, r["id"]
        if r["type"] == "unanswerable":
            assert not r["gold_docs"], r["id"]
            continue
        assert r["gold_docs"], r["id"]
        readable = [
            principal.can_read(r["workspace"], access.acl_for(r["workspace"], g)) for g in r["gold_docs"]
        ]
        if r["type"] == "answerable":
            assert any(readable), f"{r['id']}: the asking principal can't read any gold page"
        else:
            assert not any(readable), f"{r['id']}: restricted question, but its page is readable"


def test_dataset_covers_every_slice():
    types = [r["type"] for r in rows()]
    assert types.count("answerable") >= 80
    assert types.count("unanswerable") >= 10
    assert types.count("restricted") >= 5
    assert sum("paraphrase" in r.get("tags", []) for r in rows()) >= 10


class _StubAnswer:
    def __init__(self, question: str) -> None:
        self.question = question

    def model_dump(self) -> dict:
        return {
            "abstained": True,
            "abstain_reason": "model_not_found",
            "text": "not in the handbook",
            "timings_ms": {"total": 5.0},
            "usage": {"total_tokens": 100},
            "invalid_citations": [],
            "sources": [],
            "cited": [],
            "grounded": False,
        }


def _stub_runtime(fail_once_on: set[str], calls: list[str], **settings):
    from types import SimpleNamespace

    from citewise.config import Settings
    from citewise.generation.llm import LLMError

    class StubAnswerer:
        mode = "stub-llm"

        def answer(self, question, workspace, principal):
            calls.append(question)
            if question in fail_once_on:
                fail_once_on.discard(question)
                raise LLMError("LLM API error 429: rate limit reached", 429, retryable=True)
            return _StubAnswer(question)

    return SimpleNamespace(
        settings=Settings(_env_file=None, **settings),
        llm=SimpleNamespace(model="stub-llm"),
        access=SimpleNamespace(principals={"ana": None}),
        answerer=StubAnswerer(),
    )


def test_answers_run_resumes_after_a_rate_limit(tmp_path: Path):
    import pytest

    from evals.run import AnswersIncomplete, eval_answers

    cache = tmp_path / "answers-cache.jsonl"
    data = [
        {
            "id": f"q{i}",
            "type": "unanswerable",
            "workspace": "acme",
            "as": "ana",
            "question": f"question {i}?",
        }
        for i in range(3)
    ]
    calls: list[str] = []
    rt = _stub_runtime({"question 1?"}, calls)

    with pytest.raises(AnswersIncomplete, match="1 finished questions are saved"):
        eval_answers(rt, data, use_judge=False, cache_path=cache)
    assert calls == ["question 0?", "question 1?"]

    cache.write_text(cache.read_text() + '{"key": "cut off by Ctrl+')  # a half-written line is skipped
    result = eval_answers(rt, data, use_judge=False, cache_path=cache)
    assert calls[2:] == ["question 1?", "question 2?"]  # question 0 came from the cache
    assert result["summary"]["abstention_rate"] == 1.0 and len(result["results"]) == 3
    assert result["summary"]["llm_tokens"] == 300

    # A setting that changes answers (here the abstention threshold) invalidates the cache.
    calls.clear()
    eval_answers(_stub_runtime(set(), calls, abstain_min_rerank=0.5), data, use_judge=False, cache_path=cache)
    assert calls == ["question 0?", "question 1?", "question 2?"]
