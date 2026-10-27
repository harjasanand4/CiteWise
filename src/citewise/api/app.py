"""HTTP API (FastAPI) and the web UI.

POST /v1/ask            cited answer as JSON
POST /v1/ask/stream     the same, streamed as Server-Sent Events
POST /v1/search         retrieval only, with per-retriever ranks
POST /v1/feedback       thumbs up / down on an answer
GET  /v1/me             who you are and which workspaces you can query
GET  /healthz /readyz /metrics
"""

import json
import logging
import time
import uuid
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..config import Settings, get_settings
from ..generation.llm import LLMError
from ..models import Answer, Principal
from ..runtime import Runtime
from ..store.base import IndexMismatchError
from . import metrics
from .schemas import AskRequest, DemoUser, FeedbackRequest, PrincipalOut, SearchRequest, WorkspaceOut

log = logging.getLogger("citewise")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def _configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if settings.log_json else logging.Formatter("%(levelname)s %(message)s")
    )
    log.handlers[:] = [handler]
    log.setLevel(settings.log_level)
    log.propagate = False


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    rt = runtime or Runtime(settings or get_settings())
    settings = rt.settings
    _configure_logging(settings)
    app = FastAPI(title="Citewise", version=__version__, docs_url="/docs", redoc_url=None)
    app.state.rt = rt
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.HTTP_REQUESTS.labels(route, request.method, str(response.status_code)).inc()
        metrics.HTTP_LATENCY.labels(route).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        if route.startswith("/v1"):
            log.info(
                "%s %s %s %.0fms",
                request.method,
                route,
                response.status_code,
                elapsed * 1000,
                extra={
                    "fields": {
                        "request_id": request_id,
                        "route": route,
                        "status": response.status_code,
                        "ms": round(elapsed * 1000, 1),
                    }
                },
            )
        return response

    @app.exception_handler(IndexMismatchError)
    async def index_mismatch(_: Request, exc: IndexMismatchError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        metrics.LLM_ERRORS.inc()
        return JSONResponse(
            status_code=502,
            content={"detail": "The language model is unavailable. Try again shortly.", "error": str(exc)},
        )

    # ---- auth -----------------------------------------------------------------------------

    def principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
        if settings.auth_mode == "disabled":
            groups = sorted({g for rules in rt.access.rules.values() for r in rules for g in r.groups})
            return Principal("local", "Local admin", tuple(rt.workspaces), tuple(groups))
        token = (authorization or "").removeprefix("Bearer ").strip()
        p = rt.access.principal_for_key(token) if token else None
        if p is None:
            raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return p

    User = Annotated[Principal, Depends(principal)]

    def check_workspace(p: Principal, workspace: str) -> None:
        if workspace not in rt.workspaces:
            raise HTTPException(404, f"Unknown workspace {workspace!r}")
        if workspace not in p.workspaces:
            raise HTTPException(403, f"You don't have access to the {workspace!r} workspace")

    def workspace_out(wid: str) -> WorkspaceOut:
        w = rt.workspaces[wid]
        return WorkspaceOut(
            id=w.id, name=w.name, description=w.description, license=w.license, examples=w.examples
        )

    # ---- routes ---------------------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        try:
            rt.store.ping()
            info = rt.store.index_info()
        except Exception as e:  # pragma: no cover - depends on the database being down
            return JSONResponse(status_code=503, content={"status": "unavailable", "detail": str(e)})
        if not info["workspaces"]:
            return JSONResponse(
                status_code=503,
                content={"status": "empty", "detail": "Nothing indexed. Run `citewise ingest`."},
            )
        return JSONResponse({"status": "ready", "index": info})

    @app.get("/metrics")
    def prometheus() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/me", response_model=PrincipalOut)
    def me(p: User) -> PrincipalOut:
        return PrincipalOut(
            id=p.id,
            name=p.name,
            groups=list(p.groups),
            workspaces=[workspace_out(w) for w in p.workspaces if w in rt.workspaces],
        )

    @app.get("/v1/demo-users", response_model=list[DemoUser])
    def demo_users() -> list[DemoUser]:
        """Only in demo mode: lets the UI switch users to show access control working."""
        if settings.auth_mode != "demo":
            raise HTTPException(404)
        keys = {pid: key for key, pid in rt.access.api_keys.items()}
        return [
            DemoUser(
                id=p.id, name=p.name, api_key=keys[p.id], groups=list(p.groups), workspaces=list(p.workspaces)
            )
            for p in rt.access.principals.values()
            if p.id in keys
        ]

    @app.get("/v1/info")
    def info(_: User) -> dict[str, Any]:
        return {
            "version": __version__,
            "retriever": rt.retriever.describe(),
            "answer_mode": rt.answerer.mode,
            "index": rt.store.index_info(),
        }

    @app.post("/v1/search")
    def search(req: SearchRequest, p: User) -> dict[str, Any]:
        check_workspace(p, req.workspace)
        res = rt.retriever.retrieve(req.query, req.workspace, p, k=req.k, mode=req.mode, rerank=req.rerank)
        return {
            "hits": [
                {
                    "chunk_id": h.chunk.id,
                    "title": h.chunk.title,
                    "headings": h.chunk.headings,
                    "url": h.chunk.url,
                    "text": h.chunk.text,
                    "score": h.score,
                    "lexical_rank": h.lexical_rank,
                    "dense_rank": h.dense_rank,
                    "rerank_score": h.rerank_score,
                }
                for h in res.hits
            ],
            "timings_ms": res.timings_ms,
        }

    def _finish(answer: Answer, p: Principal) -> None:
        rt.store.log_answer(answer, p.id)
        metrics.record_answer(answer.model_dump())

    @app.post("/v1/ask", response_model=Answer)
    def ask(req: AskRequest, p: User) -> Answer:
        check_workspace(p, req.workspace)
        answer = rt.answerer.answer(req.question, req.workspace, p, [t.model_dump() for t in req.history])
        _finish(answer, p)
        return answer

    @app.post("/v1/ask/stream")
    def ask_stream(req: AskRequest, p: User) -> StreamingResponse:
        check_workspace(p, req.workspace)
        # Fail fast (with a normal HTTP error) on configuration problems before the stream starts.
        _ = rt.retriever

        def events() -> Iterator[str]:
            try:
                for event in rt.answerer.stream(
                    req.question, req.workspace, p, [t.model_dump() for t in req.history]
                ):
                    if event["type"] == "answer":
                        _finish(Answer.model_validate(event["answer"]), p)
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
            except LLMError as e:
                metrics.LLM_ERRORS.inc()
                log.warning("LLM failed mid-request: %s", e)
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'detail': 'The language model is unavailable. Try again shortly.'})}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/feedback")
    def feedback(req: FeedbackRequest, p: User) -> dict[str, bool]:
        if not rt.store.add_feedback(req.answer_id, p.id, req.rating, req.comment):
            raise HTTPException(404, "Answer not found")
        metrics.FEEDBACK.labels("up" if req.rating > 0 else "down").inc()
        return {"ok": True}

    if settings.web_dist.exists():
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")
    return app
