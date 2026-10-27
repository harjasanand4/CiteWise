import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from citewise.api.app import create_app
from citewise.runtime import Runtime

from .conftest import build, make_settings


@pytest.fixture
def api(runtime: Runtime) -> TestClient:
    return TestClient(create_app(runtime=runtime))


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_requires_a_valid_api_key(api: TestClient):
    assert api.post("/v1/ask", json={"workspace": "acme", "question": "hi"}).status_code == 401
    assert (
        api.post("/v1/ask", json={"workspace": "acme", "question": "hi"}, headers=auth("nope")).status_code
        == 401
    )


def test_cannot_query_a_workspace_you_are_not_in(api: TestClient):
    r = api.post("/v1/ask", json={"workspace": "acme", "question": "receipts"}, headers=auth("key-gus"))
    assert r.status_code == 403
    assert (
        api.post("/v1/ask", json={"workspace": "nope", "question": "x"}, headers=auth("key-gus")).status_code
        == 404
    )


def test_me_lists_workspaces_and_groups(api: TestClient):
    me = api.get("/v1/me", headers=auth("key-hal")).json()
    assert me["id"] == "hal" and me["groups"] == ["hiring"]
    assert [w["id"] for w in me["workspaces"]] == ["acme"]
    assert me["workspaces"][0]["examples"]


def test_ask_returns_a_cited_answer_and_logs_it(api: TestClient, runtime: Runtime):
    r = api.post(
        "/v1/ask",
        json={"workspace": "acme", "question": "When do receipts need to be uploaded?"},
        headers=auth("key-ana"),
    )
    assert r.status_code == 200 and r.headers["x-request-id"]
    body = r.json()
    assert "14 days" in body["text"] and body["cited"] and not body["abstained"]
    assert body["sources"][0]["url"].startswith("https://acme.example/handbook/people/expenses")
    fb = api.post("/v1/feedback", json={"answer_id": body["id"], "rating": 1}, headers=auth("key-ana"))
    assert fb.json() == {"ok": True}
    assert runtime.store.list_feedback()[0]["question"] == "When do receipts need to be uploaded?"


def test_feedback_only_on_your_own_answers(api: TestClient):
    body = api.post(
        "/v1/ask", json={"workspace": "acme", "question": "parental leave"}, headers=auth("key-ana")
    ).json()
    assert (
        api.post(
            "/v1/feedback", json={"answer_id": body["id"], "rating": -1}, headers=auth("key-hal")
        ).status_code
        == 404
    )


def test_access_control_through_the_api(api: TestClient):
    q = {"workspace": "acme", "question": "How much is the referral bonus?"}
    ana = api.post("/v1/ask", json=q, headers=auth("key-ana")).json()
    hal = api.post("/v1/ask", json=q, headers=auth("key-hal")).json()
    assert all("hiring" not in s["chunk_id"] for s in ana["sources"])
    assert "$2,500" in hal["text"]


def test_stream_sends_server_sent_events(api: TestClient):
    with api.stream(
        "POST",
        "/v1/ask/stream",
        json={"workspace": "acme", "question": "How do I declare an incident?"},
        headers=auth("key-ana"),
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events][0] == "sources" and events[-1]["type"] == "answer"
    assert "declare an incident" in events[-1]["answer"]["text"]


def test_search_shows_per_retriever_ranks(api: TestClient):
    r = api.post(
        "/v1/search",
        json={"workspace": "acme", "query": "parental leave weeks", "k": 3},
        headers=auth("key-ana"),
    )
    hit = r.json()["hits"][0]
    assert hit["title"] == "Time off" and hit["lexical_rank"] and hit["dense_rank"]


def test_validation_errors(api: TestClient):
    assert (
        api.post("/v1/ask", json={"workspace": "acme", "question": ""}, headers=auth("key-ana")).status_code
        == 422
    )
    assert (
        api.post(
            "/v1/search", json={"workspace": "acme", "query": "x", "k": 500}, headers=auth("key-ana")
        ).status_code
        == 422
    )


def test_health_ready_metrics_and_demo_users(api: TestClient):
    assert api.get("/healthz").json() == {"status": "ok"}
    ready = api.get("/readyz").json()
    assert ready["status"] == "ready" and set(ready["index"]["workspaces"]) == {"acme", "globex"}
    api.post("/v1/ask", json={"workspace": "acme", "question": "receipts"}, headers=auth("key-ana"))
    text = api.get("/metrics").text
    assert 'citewise_http_requests_total{method="POST",route="/v1/ask",status="200"}' in text
    assert "citewise_answers_total" in text
    users = api.get("/v1/demo-users").json()
    assert {u["id"] for u in users} == {"ana", "hal", "gus", "root"}


def test_readyz_reports_an_empty_index(corpus: Path, database_url: str):
    api = TestClient(create_app(runtime=Runtime(make_settings(corpus, database_url))))
    assert api.get("/readyz").status_code == 503


def test_disabled_auth_mode_is_a_local_admin(corpus: Path, database_url: str):
    rt = build(Runtime(make_settings(corpus, database_url, auth_mode="disabled")))
    api = TestClient(create_app(runtime=rt))
    me = api.get("/v1/me").json()
    assert me["id"] == "local" and {w["id"] for w in me["workspaces"]} == {"acme", "globex"}
    assert api.get("/v1/demo-users").status_code == 404
