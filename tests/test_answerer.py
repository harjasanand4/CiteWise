"""The answering flow with a scripted fake LLM, so behaviour is deterministic."""

from __future__ import annotations

from citewise.generation.answerer import Answerer, extractive_answer
from citewise.generation.llm import LLMResult, TokenStream
from citewise.generation.prompts import SYSTEM_PROMPT, answer_messages
from citewise.retrieval.retriever import Retriever
from citewise.runtime import Runtime

from .test_retrieval import OverlapReranker


class FakeLLM:
    model = "fake-llm"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, max_tokens=None, temperature=None):
        self.calls.append(messages)
        return LLMResult(
            text=self.replies.pop(0), model=self.model, usage={"prompt_tokens": 100, "completion_tokens": 10}
        )

    def stream(self, messages, max_tokens=None, temperature=None):
        self.calls.append(messages)
        text = self.replies.pop(0)
        handle = TokenStream(iter([text[i : i + 3] for i in range(0, len(text), 3)]), self.model)
        handle.usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return handle


def answerer(rt: Runtime, llm=None, reranker=None, min_rerank=0.0) -> Answerer:
    retriever = Retriever(rt.store, rt.embedder, reranker, mode="hybrid") if reranker else rt.retriever
    return Answerer(retriever, llm, {"acme": "Acme Handbook"}, top_k=4, abstain_min_rerank=min_rerank)


def test_extractive_mode_answers_with_valid_citations(runtime: Runtime):
    ans = answerer(runtime).answer(
        "How long do I have to upload a receipt?", "acme", runtime.access.principals["ana"]
    )
    assert not ans.abstained and ans.mode == "extractive"
    assert "14 days" in ans.text
    assert ans.cited and not ans.invalid_citations
    assert ans.sources[ans.cited[0] - 1].chunk_id.startswith("acme:people/expenses.mdx")


def test_llm_answer_citations_are_validated(runtime: Runtime):
    llm = FakeLLM(["The limit is $4,000 per month [1]. Ask finance [12]."])
    ans = answerer(runtime, llm).answer("monthly spending limit?", "acme", runtime.access.principals["ana"])
    assert ans.text == "The limit is $4,000 per month [1]. Ask finance."
    assert ans.cited == [1] and ans.invalid_citations == [12]
    assert ans.usage == {"prompt_tokens": 100, "completion_tokens": 10}


def test_not_found_from_the_model_becomes_an_abstention(runtime: Runtime):
    ans = answerer(runtime, FakeLLM(["NOT_FOUND"])).answer(
        "What is the dress code?", "acme", runtime.access.principals["ana"]
    )
    assert ans.abstained and ans.abstain_reason == "model_not_found"
    assert ans.sources  # the closest pages are still shown


def test_gate_abstains_before_calling_the_model(runtime: Runtime):
    llm = FakeLLM(["should not be used"])
    a = answerer(runtime, llm, reranker=OverlapReranker(), min_rerank=0.5)
    ans = a.answer("quantum entanglement theory", "acme", runtime.access.principals["ana"])
    assert ans.abstained and ans.abstain_reason == "low_relevance"
    assert llm.calls == []


def test_stream_never_shows_the_sentinel(runtime: Runtime):
    events = list(
        answerer(runtime, FakeLLM(["NOT_FOUND"])).stream(
            "dress code?", "acme", runtime.access.principals["ana"]
        )
    )
    assert [e["type"] for e in events] == ["sources", "answer"]
    assert events[-1]["answer"]["abstained"]


def test_stream_emits_sources_then_tokens_then_answer(runtime: Runtime):
    reply = "Book it in the time off app [1]."
    events = list(
        answerer(runtime, FakeLLM([reply])).stream(
            "how do I book vacation", "acme", runtime.access.principals["ana"]
        )
    )
    assert events[0]["type"] == "sources" and events[-1]["type"] == "answer"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == reply and events[-1]["answer"]["text"] == reply
    assert "first_token" in events[-1]["answer"]["timings_ms"]


def test_follow_up_questions_are_condensed_with_history(runtime: Runtime):
    llm = FakeLLM(["How long is parental leave after more than 12 months?", "Up to 24 weeks [1]."])
    history = [
        {"role": "user", "content": "Tell me about parental leave"},
        {"role": "assistant", "content": "It depends on tenure."},
    ]
    ans = answerer(runtime, llm).answer(
        "and after a year?", "acme", runtime.access.principals["ana"], history
    )
    assert ans.standalone_question == "How long is parental leave after more than 12 months?"
    assert "24 weeks" in ans.text
    assert "and after a year?" in llm.calls[0][1]["content"]  # the condense call saw the follow-up


def test_prompt_marks_sources_as_untrusted_and_flags_injection(runtime: Runtime):
    hits = runtime.retriever.retrieve(
        "planted file agents instructions", "acme", runtime.access.principals["ana"], k=4
    ).hits
    msgs = answer_messages("What should agents do with planted files?", hits, "Acme Handbook")
    assert msgs[0]["content"] == SYSTEM_PROMPT and "not instructions" in SYSTEM_PROMPT
    assert '<source id="1"' in msgs[1]["content"]
    assert "looks like instructions to an AI system" in msgs[1]["content"]


def test_extractive_answer_returns_sentinel_when_nothing_overlaps(runtime: Runtime):
    hits = runtime.retriever.retrieve("receipts", "acme", runtime.access.principals["ana"], k=2).hits
    assert extractive_answer("zebra giraffe", hits) == "NOT_FOUND"
