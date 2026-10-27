"""The LLM client against a fake HTTP server (httpx.MockTransport): retries, streaming, errors."""

import json

import httpx
import pytest

from citewise.generation.llm import LLMError, OpenAICompatibleClient


def client(handler, sleeps=None, model="test-model", max_retries=3):
    return OpenAICompatibleClient(
        "https://llm.test/v1",
        model,
        api_key="k",
        max_retries=max_retries,
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def ok(text="Hello [1].", usage=None):
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "choices": [{"message": {"content": text}}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        },
    )


def sse(*events):
    body = "".join(f"data: {json.dumps(e) if not isinstance(e, str) else e}\n\n" for e in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_retries_rate_limits_and_server_errors_then_succeeds():
    calls, sleeps = [], []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "2"})
        if len(calls) == 2:
            return httpx.Response(503, text="overloaded")
        return ok()

    res = client(handler, sleeps).complete([{"role": "user", "content": "hi"}])
    assert res.text == "Hello [1]." and res.attempts == 3
    assert res.usage == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    assert sleeps[0] == 2.0  # Retry-After honoured
    assert 0 <= sleeps[1] <= 2.0  # then exponential backoff with jitter
    assert calls[0].headers["authorization"] == "Bearer k"


def test_does_not_retry_errors_that_retrying_cannot_fix():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    with pytest.raises(LLMError) as e:
        client(handler).complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 1 and e.value.status == 401 and "invalid api key" in str(e.value)


def test_gives_up_after_max_retries():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError):
        client(handler, max_retries=2).complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 3


def test_timeouts_are_retried():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) < 3:
            raise httpx.ReadTimeout("timed out", request=req)
        return ok()

    assert client(handler).complete([{"role": "user", "content": "hi"}]).attempts == 3


def test_streaming_yields_deltas_and_usage():
    def handler(req):
        assert json.loads(req.content)["stream"] is True
        return sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Twelve "}}]},
            {"choices": [{"delta": {"content": "weeks [1]."}}]},
            {"choices": [], "x_groq": {"usage": {"prompt_tokens": 50, "completion_tokens": 4}}},
            "[DONE]",
        )

    stream = client(handler).stream([{"role": "user", "content": "hi"}])
    assert "".join(stream) == "Twelve weeks [1]."
    assert stream.usage == {"prompt_tokens": 50, "completion_tokens": 4}


def test_stream_is_retried_if_it_fails_before_the_first_token():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(503, text="overloaded")
        return sse({"choices": [{"delta": {"content": "ok"}}]}, "[DONE]")

    assert "".join(client(handler).stream([{"role": "user", "content": "hi"}])) == "ok"
    assert len(calls) == 2


def test_reasoning_models_get_low_reasoning_effort():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return ok()

    client(handler, model="openai/gpt-oss-120b").complete([{"role": "user", "content": "hi"}])
    client(handler, model="llama3.2").complete([{"role": "user", "content": "hi"}])
    assert bodies[0]["reasoning_effort"] == "low" and "reasoning_effort" not in bodies[1]
