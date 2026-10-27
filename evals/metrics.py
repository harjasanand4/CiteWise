"""Retrieval and answer metrics. Relevance is judged at document level: a question is
answered by a page, and any chunk of that page counts as finding it."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from statistics import median


def dedupe(paths: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for p in paths:
        seen.setdefault(p, None)
    return list(seen)


def hit_at(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return float(any(p in gold for p in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    for i, p in enumerate(ranked[:k], start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


def p50(values: Sequence[float]) -> float:
    return float(median(values)) if values else 0.0


def normalize(text: str) -> str:
    """Lowercase, drop thousands separators and markdown emphasis, collapse whitespace."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text)


def key_fact_recall(answer: str, facts: Sequence[str]) -> float | None:
    if not facts:
        return None
    a = normalize(answer)
    return sum(normalize(f) in a for f in facts) / len(facts)


def mean(values: Sequence[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
