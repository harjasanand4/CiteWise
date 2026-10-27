"""Checks that run around the model: prompt-injection flags, the abstention gate and
citation validation. None of them trust the model to police itself."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Hit

# Text inside a document that is trying to talk to the model rather than to employees.
INJECTION_PATTERNS = [
    r"\bignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts?|rules)\b",
    r"\bdisregard (?:all |any |the )?(?:previous|prior|above|earlier|system) (?:instructions|prompts?|rules)\b",
    r"\byou are now (?:a|an) (?:\w+ ){0,3}(?:ai|assistant|model|bot|chatbot)\b",
    r"\byou are now in (?:developer|dan|jailbreak|god|unrestricted) mode\b",
    r"\bai agents?:? ignore\b",
    r"\b(?:reveal|print|show|repeat) (?:your|the) (?:system prompt|instructions|hidden prompt)\b",
    r"\bnew instructions?\s*:",
    r"<\|?(?:im_start|im_end|system|endoftext)\|?>",
    r"\b(?:assistant|ai|chatbot|llm)s?,? (?:must|should) (?:always |now )?(?:say|answer|respond|reply)\b",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.I)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


# ---- abstention -------------------------------------------------------------------------


@dataclass
class GateDecision:
    abstain: bool
    reason: str | None = None
    score: float | None = None
    signal: str | None = None


def abstention_gate(hits: list[Hit], min_rerank: float, min_dense: float) -> GateDecision:
    """Refuse before calling the LLM when retrieval found nothing worth reading.

    The reranker score is the better signal (a cross-encoder reads question and passage
    together). Without a reranker, fall back to dense similarity. Keyword scores are not
    comparable across queries, so lexical-only mode leaves the decision to the model.
    """
    if not hits:
        return GateDecision(True, "no_sources")
    best = hits[0]
    if best.rerank_score is not None and min_rerank > 0:
        top = max(h.rerank_score or 0.0 for h in hits)
        if top < min_rerank:
            return GateDecision(True, "low_relevance", top, "rerank")
        return GateDecision(False, score=top, signal="rerank")
    if best.dense_score is not None and min_dense > 0:
        top = max(h.dense_score or -1.0 for h in hits)
        if top < min_dense:
            return GateDecision(True, "low_relevance", top, "dense")
        return GateDecision(False, score=top, signal="dense")
    return GateDecision(False)


# ---- citations --------------------------------------------------------------------------

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A sentence plus any citation markers that follow its full stop ("... 14 days. [2]").
SENTENCE_WITH_CITES = re.compile(r"[^.!?\n]+(?:[.!?]+|$)(?:\s*\[\d+(?:\s*,\s*\d+)*\])*", re.M)
NOT_FOUND = "NOT_FOUND"


@dataclass
class CitationCheck:
    text: str
    cited: list[int]
    invalid: list[int]


def validate_citations(text: str, n_sources: int) -> CitationCheck:
    """Keep [n] markers that point at a real source; drop the rest and report them."""
    cited: list[int] = []
    invalid: list[int] = []

    def fix(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        good = [n for n in nums if 1 <= n <= n_sources]
        invalid.extend(n for n in nums if n not in good)
        for n in good:
            if n not in cited:
                cited.append(n)
        return "".join(f"[{n}]" for n in good)

    cleaned = CITATION.sub(fix, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return CitationCheck(cleaned.strip(), cited, invalid)


def is_not_found(text: str) -> bool:
    return text.strip().strip(".").strip().upper().startswith(NOT_FOUND)


# ---- cheap faithfulness signal used by the eval -----------------------------------------

WORD = re.compile(r"[a-z0-9$£€%]+")


def support_ratio(claim: str, source: str) -> float:
    """Share of a claim's content words that appear in the cited source (0..1)."""
    claim_words = {
        w for w in WORD.findall(CITATION.sub("", claim.lower())) if len(w) > 3 or any(c.isdigit() for c in w)
    }
    if not claim_words:
        return 1.0
    source_words = set(WORD.findall(source.lower()))
    return len(claim_words & source_words) / len(claim_words)
