"""The question-answering flow: condense -> retrieve -> gate -> generate -> validate."""

from __future__ import annotations

import math
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..models import Answer, Hit, Principal, SourceOut
from ..retrieval.retriever import Mode, Retriever
from ..store.base import query_terms
from .guardrails import GateDecision, abstention_gate, is_not_found, validate_citations
from .llm import LLMClient, LLMError, Message
from .prompts import answer_messages, condense_messages

ABSTAIN_TEXT = {
    "no_sources": "I couldn't find anything about that in the {ws}.",
    "low_relevance": "I couldn't find this in the {ws}. The closest pages I found are listed below, but they don't seem to answer it.",
    "model_not_found": "The {ws} pages I found don't answer this. The closest ones are listed below.",
}
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Prepared:
    question: str
    standalone: str
    hits: list[Hit]
    gate: GateDecision
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None,
        workspace_names: dict[str, str],
        top_k: int = 6,
        abstain_min_rerank: float = 0.0,
        abstain_min_dense: float = 0.0,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.workspace_names = workspace_names
        self.top_k = top_k
        self.abstain_min_rerank = abstain_min_rerank
        self.abstain_min_dense = abstain_min_dense

    @property
    def mode(self) -> str:
        return self.llm.model if self.llm else "extractive"

    # ---- steps --------------------------------------------------------------------------

    def condense(self, question: str, history: list[Message]) -> tuple[str, dict[str, int]]:
        if not history:
            return question, {}
        if self.llm is None:
            last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            return f"{last_user} {question}".strip(), {}
        try:
            res = self.llm.complete(condense_messages(history, question), max_tokens=120, temperature=0)
            return (res.text.strip().strip('"') or question), res.usage
        except LLMError:
            return question, {}

    def prepare(
        self,
        question: str,
        workspace: str,
        principal: Principal,
        history: list[Message] | None = None,
        mode: Mode | None = None,
        rerank: bool | None = None,
    ) -> Prepared:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        standalone, usage = self.condense(question, history or [])
        if history:
            timings["condense"] = _ms(t0)
        t = time.perf_counter()
        result = self.retriever.retrieve(
            standalone, workspace, principal, k=self.top_k, mode=mode, rerank=rerank
        )
        timings.update(result.timings_ms)
        timings["retrieve"] = _ms(t)
        gate = abstention_gate(result.hits, self.abstain_min_rerank, self.abstain_min_dense)
        return Prepared(question, standalone, result.hits, gate, timings, usage)

    def _finish(
        self, p: Prepared, workspace: str, raw_text: str, started: float, usage: dict[str, int]
    ) -> Answer:
        ws_name = self.workspace_names.get(workspace, workspace)
        sources = [_source_out(n, h) for n, h in enumerate(p.hits, start=1)]
        base: dict[str, Any] = dict(
            id=uuid.uuid4().hex[:16],
            workspace=workspace,
            question=p.question,
            standalone_question=p.standalone,
            sources=sources,
            mode=self.mode,
            usage=_merge_usage(p.usage, usage),
        )
        if p.gate.abstain or is_not_found(raw_text):
            reason = p.gate.reason if p.gate.abstain else "model_not_found"
            p.timings["total"] = _ms(started)
            return Answer(
                text=ABSTAIN_TEXT[reason or "no_sources"].format(ws=ws_name),
                abstained=True,
                abstain_reason=reason,
                timings_ms=p.timings,
                **base,
            )
        check = validate_citations(raw_text, len(sources))
        p.timings["total"] = _ms(started)
        return Answer(
            text=check.text,
            abstained=False,
            cited=check.cited,
            invalid_citations=check.invalid,
            grounded=bool(check.cited),
            timings_ms=p.timings,
            **base,
        )

    # ---- public -------------------------------------------------------------------------

    def answer(
        self,
        question: str,
        workspace: str,
        principal: Principal,
        history: list[Message] | None = None,
        mode: Mode | None = None,
        rerank: bool | None = None,
    ) -> Answer:
        started = time.perf_counter()
        p = self.prepare(question, workspace, principal, history, mode, rerank)
        if p.gate.abstain:
            return self._finish(p, workspace, "", started, {})
        t = time.perf_counter()
        if self.llm is None:
            text, usage = extractive_answer(p.standalone, p.hits), {}
        else:
            res = self.llm.complete(
                answer_messages(p.standalone, p.hits, self.workspace_names.get(workspace, workspace))
            )
            text, usage = res.text, res.usage
        p.timings["generate"] = _ms(t)
        return self._finish(p, workspace, text, started, usage)

    def stream(
        self, question: str, workspace: str, principal: Principal, history: list[Message] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yields {"type": "sources"}, then {"type": "token"}..., then {"type": "answer"}."""
        started = time.perf_counter()
        p = self.prepare(question, workspace, principal, history)
        yield {
            "type": "sources",
            "standalone_question": p.standalone,
            "sources": [_source_out(n, h).model_dump() for n, h in enumerate(p.hits, start=1)],
        }
        if p.gate.abstain:
            yield {"type": "answer", "answer": self._finish(p, workspace, "", started, {}).model_dump()}
            return
        t = time.perf_counter()
        if self.llm is None:
            text = extractive_answer(p.standalone, p.hits)
            yield {"type": "token", "text": text}
            p.timings["generate"] = _ms(t)
            yield {"type": "answer", "answer": self._finish(p, workspace, text, started, {}).model_dump()}
            return

        stream = self.llm.stream(
            answer_messages(p.standalone, p.hits, self.workspace_names.get(workspace, workspace))
        )
        parts: list[str] = []
        buffer, released = "", False
        for delta in stream:
            parts.append(delta)
            if released:
                yield {"type": "token", "text": delta}
                continue
            # Hold the first characters back until we know the reply isn't the NOT_FOUND sentinel,
            # so the user never sees it flash up before turning into a "couldn't find it" message.
            buffer += delta
            probe = buffer.strip().upper()
            if len(probe) < len("NOT_FOUND") and "NOT_FOUND".startswith(probe):
                continue
            if is_not_found(probe):
                continue
            released = True
            p.timings["first_token"] = _ms(started)
            yield {"type": "token", "text": buffer}
        text = "".join(parts)
        p.timings["generate"] = _ms(t)
        yield {
            "type": "answer",
            "answer": self._finish(p, workspace, text, started, stream.usage).model_dump(),
        }


TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}")


def extractive_answer(
    question: str, hits: list[Hit], max_sentences: int = 3, min_coverage: float = 0.3
) -> str:
    """No-LLM mode: pick the sentences from the top sources that best match the question.

    Query words are weighted by how rare they are among the candidate sentences (so "receipt"
    counts for more than "long"), and sentences from higher-ranked sources get a boost. If the
    best sentence covers under 30% of the question's weight, it isn't an answer, so the
    reply is NOT_FOUND and the caller abstains.
    """
    candidates: list[tuple[int, int, str, set[str]]] = []
    for n, h in enumerate(hits[:3], start=1):
        for i, raw in enumerate(SENTENCE.split(h.chunk.text)):
            sent = raw.strip(" -*>").strip()
            if TABLE_RULE.match(sent) or sent.endswith("?") or len(sent.replace("|", " ").split()) < 6:
                continue  # table rules, questions and fragments are never answers
            candidates.append((n, i, sent, {w[:6] for w in query_terms(sent)}))
    terms = {t[:6] for t in query_terms(question)}
    if not candidates or not terms:
        return "NOT_FOUND"
    df = {t: sum(t in words for *_, words in candidates) for t in terms}
    idf = {t: math.log(1 + len(candidates) / max(df[t], 0.5)) for t in terms}
    total = sum(idf.values())
    scored = []
    for n, i, sent, words in candidates:
        weight = sum(idf[t] for t in terms & words)
        if weight > 0:
            scored.append((weight / math.sqrt(len(words) + 1) * (1.0 + 0.5 / n), n, i, sent, weight / total))
    if not scored:
        return "NOT_FOUND"
    ranked = sorted(scored, key=lambda x: -x[0])
    if max(s[4] for s in ranked[:max_sentences]) < min_coverage:
        return "NOT_FOUND"
    best = [s for s in ranked[:max_sentences] if s[0] >= 0.5 * ranked[0][0]]  # drop weak filler
    best.sort(key=lambda x: (x[1], x[2]))
    return " ".join(f"{s.rstrip()} [{n}]" for _, n, _, s, _ in best)


def _source_out(n: int, h: Hit) -> SourceOut:
    c = h.chunk
    return SourceOut(
        n=n,
        chunk_id=c.id,
        title=c.title,
        headings=c.headings,
        url=c.url,
        text=c.text,
        score=round(h.score, 4),
        lexical_rank=h.lexical_rank,
        dense_rank=h.dense_rank,
        rerank_score=round(h.rerank_score, 4) if h.rerank_score is not None else None,
        flags=c.flags,
    )


def _merge_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
