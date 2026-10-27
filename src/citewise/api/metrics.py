"""Prometheus metrics. Route labels use the route template, never the raw path, to keep cardinality bounded."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)

HTTP_REQUESTS = Counter("citewise_http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_LATENCY = Histogram(
    "citewise_http_request_seconds", "HTTP request latency", ["route"], buckets=LATENCY_BUCKETS
)
STAGE_LATENCY = Histogram(
    "citewise_stage_seconds", "Latency of each answering stage", ["stage"], buckets=LATENCY_BUCKETS
)
ANSWERS = Counter("citewise_answers_total", "Answers by outcome", ["workspace", "outcome"])
LLM_TOKENS = Counter("citewise_llm_tokens_total", "LLM tokens", ["kind"])
LLM_ERRORS = Counter("citewise_llm_errors_total", "LLM calls that failed after retries")
FEEDBACK = Counter("citewise_feedback_total", "Thumbs up / down", ["rating"])


def record_answer(answer: dict) -> None:
    outcome = f"abstained:{answer['abstain_reason']}" if answer["abstained"] else "answered"
    ANSWERS.labels(answer["workspace"], outcome).inc()
    for stage, ms in answer["timings_ms"].items():
        if stage in (
            "lexical",
            "dense",
            "embed_query",
            "rerank",
            "retrieve",
            "generate",
            "condense",
            "total",
        ):
            STAGE_LATENCY.labels(stage).observe(ms / 1000)
    for kind in ("prompt_tokens", "completion_tokens"):
        if answer["usage"].get(kind):
            LLM_TOKENS.labels(kind.removesuffix("_tokens")).inc(answer["usage"][kind])
