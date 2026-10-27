"""Command line: fetch the handbooks, build the index, search, ask and serve."""

from __future__ import annotations

import json
import sys
from typing import Annotated, cast

import typer

from .config import get_settings
from .ingest.pipeline import ingest_workspace
from .ingest.sources import fetch_workspace
from .models import Principal
from .retrieval.retriever import Mode
from .runtime import Runtime

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Citewise: cited answers from company handbooks."
)

WorkspaceOpt = Annotated[str | None, typer.Option("--workspace", "-w", help="Workspace id (default: all).")]


def _runtime() -> Runtime:
    return Runtime(get_settings())


def _pick(rt: Runtime, workspace: str | None) -> list[str]:
    if workspace and workspace not in rt.workspaces:
        raise typer.BadParameter(f"Unknown workspace {workspace!r}. Known: {', '.join(rt.workspaces)}")
    return [workspace] if workspace else list(rt.workspaces)


@app.command()
def fetch(workspace: WorkspaceOpt = None) -> None:
    """Download each handbook at its pinned commit (sparse, shallow git checkout)."""
    rt = _runtime()
    for wid in _pick(rt, workspace):
        ws = rt.workspaces[wid]
        path, changed = fetch_workspace(ws, rt.settings.data_dir)
        typer.echo(f"{wid}: {'fetched' if changed else 'already at'} {ws.commit[:10]} -> {path}")


@app.command()
def ingest(
    workspace: WorkspaceOpt = None,
    rebuild: Annotated[bool, typer.Option(help="Drop the index and rebuild from scratch.")] = False,
) -> None:
    """Parse, chunk, embed and index. Unchanged documents are skipped."""
    rt = _runtime()
    if rebuild:
        rt.store.reset()
    emb = rt.embedder
    rt.store.ensure_schema(emb.name if emb else None, emb.dim if emb else None)
    typer.echo(
        f"embeddings: {emb.name if emb else 'none (keyword search only)'} | store: {rt.settings.database_url}"
    )
    for wid in _pick(rt, workspace):
        ws = rt.workspaces[wid]
        report = ingest_workspace(
            ws,
            ws.source_dir(rt.settings.data_dir),
            rt.store,
            emb,
            rt.access,
            max_words=rt.settings.chunk_max_words,
            min_words=rt.settings.chunk_min_words,
            progress=lambda m: typer.echo(f"  {m}", err=True),
        )
        typer.echo(
            f"{wid}: +{report.added} added, ~{report.updated} updated, ={report.unchanged} unchanged, "
            f"-{report.deleted} deleted, {report.chunks} chunks in {report.seconds}s"
            + (f", skipped {report.skipped}" if report.skipped else "")
            + (
                f", {len(report.flagged)} chunks flagged as possible prompt injection"
                if report.flagged
                else ""
            )
        )


@app.command()
def info() -> None:
    """Show what is indexed."""
    typer.echo(json.dumps(_runtime().store.index_info(), indent=2))


def _principal(rt: Runtime, name: str) -> Principal:
    p = rt.access.principals.get(name)
    if not p:
        raise typer.BadParameter(f"Unknown principal {name!r}. Known: {', '.join(rt.access.principals)}")
    return p


@app.command()
def search(
    query: str,
    workspace: Annotated[str, typer.Option("--workspace", "-w")] = "posthog",
    as_: Annotated[str, typer.Option("--as", help="Principal from config/access.yaml")] = "admin",
    k: int = 8,
    mode: str | None = None,
) -> None:
    """Retrieval only: show the ranked chunks and why they ranked."""
    rt = _runtime()
    res = rt.retriever.retrieve(query, workspace, _principal(rt, as_), k=k, mode=cast(Mode | None, mode))
    for i, h in enumerate(res.hits, 1):
        bits = [
            f"lex#{h.lexical_rank}" if h.lexical_rank else "lex -",
            f"vec#{h.dense_rank}" if h.dense_rank else "vec -",
        ]
        if h.rerank_score is not None:
            bits.append(f"rerank {h.rerank_score:.3f}")
        typer.echo(f"{i:>2}. {h.chunk.breadcrumb}  [{', '.join(bits)}]\n    {h.chunk.url}")
    typer.echo(f"\n{rt.retriever.describe()} | {res.timings_ms}", err=True)


@app.command()
def ask(
    question: str,
    workspace: Annotated[str, typer.Option("--workspace", "-w")] = "posthog",
    as_: Annotated[str, typer.Option("--as", help="Principal from config/access.yaml")] = "admin",
    stream: bool = True,
) -> None:
    """Ask a question and print a cited answer."""
    rt = _runtime()
    principal = _principal(rt, as_)
    if stream:
        answer = None
        for event in rt.answerer.stream(question, workspace, principal):
            if event["type"] == "token":
                sys.stdout.write(event["text"])
                sys.stdout.flush()
            elif event["type"] == "answer":
                answer = event["answer"]
        sys.stdout.write("\n")
        if answer and answer["abstained"]:
            typer.echo(answer["text"])
    else:
        answer = rt.answerer.answer(question, workspace, principal).model_dump()
        typer.echo(answer["text"])
    if answer:
        typer.echo("")
        for s in answer["sources"]:
            mark = "*" if s["n"] in answer["cited"] else " "
            typer.echo(f" {mark}[{s['n']}] {' > '.join([s['title'], *s['headings']])}  {s['url']}")
        typer.echo(f"\nmode: {answer['mode']} | timings (ms): {answer['timings_ms']}", err=True)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API and web UI."""
    import uvicorn

    uvicorn.run("citewise.api.app:create_app", factory=True, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
