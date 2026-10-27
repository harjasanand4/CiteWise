"""A small client for any OpenAI-compatible chat API (Groq, OpenAI, Ollama, OpenRouter...).

Written against the HTTP API with httpx instead of a vendor SDK, so one code path covers
every provider and the retry / streaming behaviour is explicit and testable:

- 429 and 5xx responses, timeouts and dropped connections are retried with exponential
  backoff and full jitter, honouring Retry-After when the server sends it.
- Other 4xx errors (bad key, bad request) fail immediately; retrying can't fix them.
- A stream is only retried if it fails before the first token, so the user never sees
  the start of an answer twice.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

PRESETS: dict[str, tuple[str, str]] = {
    # provider: (base_url, default model)
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "ollama": ("http://localhost:11434/v1", "llama3.2"),
}

Message = dict[str, str]


class LLMError(RuntimeError):
    def __init__(
        self, message: str, status: int | None = None, retryable: bool = False, retry_after: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    attempts: int = 1


class LLMClient(Protocol):
    model: str

    def complete(
        self, messages: list[Message], max_tokens: int | None = None, temperature: float | None = None
    ) -> LLMResult: ...

    def stream(
        self, messages: list[Message], max_tokens: int | None = None, temperature: float | None = None
    ) -> TokenStream: ...


class TokenStream:
    """Iterate for text deltas; `usage` is filled in once the stream is exhausted."""

    def __init__(self, gen: Iterator[str], model: str) -> None:
        self._gen = gen
        self.model = model
        self.usage: dict[str, int] = {}

    def __iter__(self) -> Iterator[str]:
        return self._gen


def _parse_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or (payload.get("x_groq") or {}).get("usage") or {}
    return {
        k: int(v)
        for k, v in usage.items()
        if k in ("prompt_tokens", "completion_tokens", "total_tokens") and isinstance(v, int | float)
    }


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        temperature: float = 0.1,
        max_tokens: int = 700,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sleep = sleep
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    # ---- request building ---------------------------------------------------------------

    def _body(
        self, messages: list[Message], max_tokens: int | None, temperature: float | None, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
        }
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"  # reasoning models: keep latency down for Q&A
        return body

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
        return random.uniform(0, min(30.0, 1.0 * 2**attempt))  # full jitter

    @staticmethod
    def _error(resp: httpx.Response) -> LLMError:
        try:
            detail = resp.json().get("error", {}).get("message") or resp.text
        except (json.JSONDecodeError, AttributeError):
            detail = resp.text
        return LLMError(
            f"LLM API error {resp.status_code}: {detail[:300]}",
            resp.status_code,
            retryable=resp.status_code in RETRYABLE_STATUS,
            retry_after=resp.headers.get("retry-after"),
        )

    def _with_retries(self, fn: Callable[[], Any]) -> tuple[Any, int]:
        attempt = 0
        while True:
            try:
                return fn(), attempt + 1
            except LLMError as e:
                if not e.retryable or attempt >= self.max_retries:
                    raise
                retry_after = e.retry_after
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"LLM request failed after {attempt + 1} attempts: {e}", retryable=True
                    ) from e
                retry_after = None
            self._sleep(self._backoff(attempt, retry_after))
            attempt += 1

    # ---- public API ---------------------------------------------------------------------

    def complete(
        self, messages: list[Message], max_tokens: int | None = None, temperature: float | None = None
    ) -> LLMResult:
        def call() -> LLMResult:
            resp = self._http.post(
                "/chat/completions", json=self._body(messages, max_tokens, temperature, False)
            )
            if resp.status_code >= 400:
                raise self._error(resp)
            data = resp.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            return LLMResult(text=text, model=data.get("model", self.model), usage=_parse_usage(data))

        result, attempts = self._with_retries(call)
        result.attempts = attempts
        return result

    def stream(
        self, messages: list[Message], max_tokens: int | None = None, temperature: float | None = None
    ) -> TokenStream:
        body = self._body(messages, max_tokens, temperature, True)
        handle = TokenStream(iter(()), self.model)

        def open_stream() -> tuple[httpx.Response, Iterator[str]]:
            req = self._http.build_request("POST", "/chat/completions", json=body)
            resp = self._http.send(req, stream=True)
            if resp.status_code >= 400:
                resp.read()
                resp.close()
                raise self._error(resp)
            lines = resp.iter_lines()
            # Read until the first content token so that early failures are still retryable.
            buffered: list[str] = []
            try:
                for delta in _sse_deltas(lines, handle):
                    buffered.append(delta)
                    break
            except BaseException:
                resp.close()
                raise
            return resp, _chain(buffered, _sse_deltas(lines, handle))

        (resp, deltas), _ = self._with_retries(open_stream)

        def gen() -> Iterator[str]:
            try:
                yield from deltas
            except (httpx.TransportError, httpx.TimeoutException) as e:
                raise LLMError(f"Stream interrupted: {e}") from e
            finally:
                resp.close()

        handle._gen = gen()
        return handle


def _chain(first: list[str], rest: Iterator[str]) -> Iterator[str]:
    yield from first
    yield from rest


def _sse_deltas(lines: Iterator[str], handle: TokenStream) -> Iterator[str]:
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if payload.get("error"):
            raise LLMError(f"LLM stream error: {payload['error']}")
        usage = _parse_usage(payload)
        if usage:
            handle.usage = usage
        for choice in payload.get("choices") or []:
            delta = (choice.get("delta") or {}).get("content")
            if delta:
                yield delta


def build_llm(
    provider: str,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    timeout: float,
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> OpenAICompatibleClient | None:
    if provider == "extractive":
        return None
    preset_url, preset_model = PRESETS.get(provider, ("", ""))
    url = base_url or preset_url
    if not url:
        raise ValueError("CITEWISE_LLM_BASE_URL is required for provider 'openai_compatible'")
    if provider in ("groq", "openai") and not api_key:
        raise ValueError(f"CITEWISE_LLM_API_KEY is required for provider {provider!r}")
    return OpenAICompatibleClient(
        url, model or preset_model, api_key, timeout, max_retries, temperature, max_tokens
    )
