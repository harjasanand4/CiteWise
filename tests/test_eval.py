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
