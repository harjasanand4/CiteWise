# Operations

## Health

| Endpoint | Meaning |
|---|---|
| `GET /healthz` | The process is up (liveness) |
| `GET /readyz` | The database answers and something is indexed (readiness). 503 otherwise |
| `GET /metrics` | Prometheus metrics |

Every response carries an `X-Request-ID` (pass your own to correlate logs). Set
`CITEWISE_LOG_JSON=true` for one JSON object per request with route, status and latency.

## Metrics

| Metric | Type | Labels |
|---|---|---|
| `citewise_http_requests_total` | counter | route, method, status |
| `citewise_http_request_seconds` | histogram | route |
| `citewise_stage_seconds` | histogram | stage: lexical, embed_query, dense, rerank, retrieve, generate, total |
| `citewise_answers_total` | counter | workspace, outcome: `answered` or `abstained:<reason>` |
| `citewise_llm_tokens_total` | counter | kind: prompt, completion |
| `citewise_llm_errors_total` | counter | |
| `citewise_feedback_total` | counter | rating: up, down |

Run Prometheus with `docker compose --profile observability up -d` and try:

```promql
# p95 end-to-end answer latency
histogram_quantile(0.95, sum by (le) (rate(citewise_stage_seconds_bucket{stage="total"}[5m])))

# Which stage is slow?
histogram_quantile(0.95, sum by (le, stage) (rate(citewise_stage_seconds_bucket[5m])))

# Abstention rate. A sudden jump usually means a bad deploy or a stale index.
sum(rate(citewise_answers_total{outcome=~"abstained.*"}[1h])) / sum(rate(citewise_answers_total[1h]))

# Tokens per minute (cost)
sum by (kind) (rate(citewise_llm_tokens_total[5m])) * 60
```

## Runbook

**`/readyz` returns 503 "Nothing indexed".** Run `citewise fetch && citewise ingest`
(in Docker: `docker compose run --rm api sh -c "citewise fetch && citewise ingest"`).

**503 "Index was built with embeddings X but Y is configured".** The embedding model in
`.env` changed. Rebuild: `citewise ingest --rebuild`.

**502 "The language model is unavailable".** The provider failed after retries. Check
`citewise_llm_errors_total` and the logs. On a free tier this is usually a rate limit:
lower traffic or set `CITEWISE_LLM_MAX_RETRIES` higher. Retrieval keeps working; switch to
`CITEWISE_LLM_PROVIDER=extractive` to keep answering without a model.

**Someone can see a page they shouldn't.** Fix the rule in `config/access.yaml` and run
`citewise ingest`: only the affected pages are re-indexed with the new ACL. The CI eval
fails the build on any access-control leak in the labeled set, so add the case there too.
