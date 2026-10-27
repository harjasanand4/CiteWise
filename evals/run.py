"""Evaluate retrieval and answers against the labeled handbook questions.

    python -m evals.run                          # retrieval: every retriever configuration
    python -m evals.run --answers                # also generate answers (extractive or your LLM)
    python -m evals.run --answers --judge        # plus an LLM-as-judge faithfulness check
    python -m evals.run --sweep                  # pick the abstention threshold
    python -m evals.run --min-hit5 0.85          # exit 1 if the primary config falls below this

Questions come in three types:
  answerable    the page that answers it is in `gold_docs`
  unanswerable  nothing in this workspace answers it (including the other workspace's questions)
  restricted    the answer exists, but in a page the asking principal may not read
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citewise.config import get_settings
from citewise.generation.guardrails import CITATION, SENTENCE_WITH_CITES, support_ratio
from citewise.runtime import Runtime

from .metrics import dedupe, hit_at, key_fact_recall, mean, ndcg_at, p50, percentile, reciprocal_rank

DATASET = Path(__file__).parent / "datasets" / "handbook_qa.jsonl"
REPORTS = Path(__file__).parent / "reports"
ABSTAIN_TYPES = {"unanswerable", "restricted"}


@dataclass
class Config:
    name: str
    mode: str
    rerank: bool


@dataclass
class RetrievalScores:
    config: str
    n: int = 0
    hit1: float = 0.0
    hit3: float = 0.0
    hit5: float = 0.0
    hit10: float = 0.0
    mrr10: float = 0.0
    ndcg10: float = 0.0
    leaks: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    paraphrase_hit5: float = 0.0  # questions written to avoid the page's own wording
    keyword_hit5: float = 0.0  # everything else
    misses: list[dict[str, Any]] = field(default_factory=list)


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def configs_for(rt: Runtime, names: list[str] | None) -> list[Config]:
    available = [Config("lexical", "lexical", False)]
    if rt.embedder is not None:
        available += [Config("dense", "dense", False), Config("hybrid", "hybrid", False)]
        if rt.reranker is not None:
            available.append(Config("hybrid + rerank", "hybrid", True))
    elif rt.reranker is not None:
        available.append(Config("lexical + rerank", "lexical", True))
    if names:
        wanted = [n.strip() for n in names]
        return [c for c in available if c.name.replace(" ", "") in [w.replace(" ", "") for w in wanted]]
    return available


def primary_config(rt: Runtime) -> str:
    mode = rt.retriever.mode
    return f"{mode} + rerank" if rt.reranker is not None else mode


# ---- retrieval ----------------------------------------------------------------------------


def eval_retrieval(rt: Runtime, rows: list[dict[str, Any]], cfg: Config) -> RetrievalScores:
    s = RetrievalScores(config=cfg.name)
    hits1, hits3, hits5, hits10, mrrs, ndcgs, lat = [], [], [], [], [], [], []
    para5, kw5 = [], []
    for row in rows:
        principal = rt.access.principals[row["as"]]
        t = time.perf_counter()
        res = rt.retriever.retrieve(
            row["question"], row["workspace"], principal, k=10, mode=cfg.mode, rerank=cfg.rerank
        )  # type: ignore[arg-type]
        lat.append((time.perf_counter() - t) * 1000)
        # Leakage is checked against access.yaml directly, not the ACL stored in the index,
        # so a stale index (rules changed, not re-ingested) is caught too.
        for h in res.hits:
            if not principal.can_read(h.chunk.workspace, rt.access.acl_for(h.chunk.workspace, h.chunk.path)):
                s.leaks += 1
        if row["type"] != "answerable":
            continue
        ranked = dedupe(h.chunk.path for h in res.hits)
        gold = set(row["gold_docs"])
        hits1.append(hit_at(ranked, gold, 1))
        hits3.append(hit_at(ranked, gold, 3))
        hits5.append(hit_at(ranked, gold, 5))
        (para5 if "paraphrase" in row.get("tags", []) else kw5).append(hits5[-1])
        hits10.append(hit_at(ranked, gold, 10))
        mrrs.append(reciprocal_rank(ranked, gold))
        ndcgs.append(ndcg_at(ranked, gold))
        if not hit_at(ranked, gold, 5):
            s.misses.append(
                {"id": row["id"], "question": row["question"], "gold": row["gold_docs"], "got": ranked[:5]}
            )
    s.n = len(hits5)
    s.hit1, s.hit3, s.hit5, s.hit10 = mean(hits1), mean(hits3), mean(hits5), mean(hits10)
    s.mrr10, s.ndcg10 = mean(mrrs), mean(ndcgs)
    s.paraphrase_hit5, s.keyword_hit5 = mean(para5), mean(kw5)
    s.p50_ms, s.p95_ms = p50(lat), percentile(lat, 0.95)
    return s


# ---- answers ------------------------------------------------------------------------------

JUDGE_PROMPT = """You grade answers from a handbook assistant. You will get a question, the sources the answer cited, and the answer.
Reply with JSON only: {"supported": true or false, "answers_question": true or false, "reason": "<one sentence>"}
"supported" is true only if every factual claim in the answer is stated in the cited sources.
"answers_question" is true if the answer actually addresses what was asked."""


def judge(rt: Runtime, question: str, answer: dict[str, Any]) -> dict[str, Any] | None:
    if rt.llm is None:
        return None
    cited = [s for s in answer["sources"] if s["n"] in answer["cited"]]
    sources = "\n\n".join(f"[{s['n']}] {s['title']}: {s['text']}" for s in cited) or "(none)"
    msg = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nCited sources:\n{sources}\n\nAnswer:\n{answer['text']}",
        },
    ]
    try:
        raw = rt.llm.complete(msg, max_tokens=200, temperature=0).text
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start : end + 1])
    except Exception as e:  # a judge failure shouldn't sink the whole run
        return {"error": str(e)[:200]}


def eval_answers(rt: Runtime, rows: list[dict[str, Any]], use_judge: bool) -> dict[str, Any]:
    results = []
    for i, row in enumerate(rows, 1):
        principal = rt.access.principals[row["as"]]
        ans = rt.answerer.answer(row["question"], row["workspace"], principal).model_dump()
        rec: dict[str, Any] = {
            "id": row["id"],
            "type": row["type"],
            "question": row["question"],
            "abstained": ans["abstained"],
            "abstain_reason": ans["abstain_reason"],
            "text": ans["text"],
            "ms": ans["timings_ms"].get("total", 0.0),
            "invalid_citations": ans["invalid_citations"],
        }
        if row["type"] == "answerable" and not ans["abstained"]:
            by_n = {s["n"]: s for s in ans["sources"]}
            cited_paths = {
                by_n[n]["chunk_id"].split(":", 1)[1].rsplit("#", 1)[0] for n in ans["cited"] if n in by_n
            }
            rec["cited_gold"] = bool(cited_paths & set(row["gold_docs"]))
            rec["key_facts"] = key_fact_recall(ans["text"], row.get("must_include", []))
            rec["grounded"] = ans["grounded"]
            ratios = []
            for sentence in SENTENCE_WITH_CITES.findall(ans["text"]):
                nums = [int(n) for m in CITATION.finditer(sentence) for n in m.group(1).split(",")]
                texts = " ".join(by_n[n]["text"] for n in nums if n in by_n)
                if texts:
                    ratios.append(support_ratio(sentence, texts))
            rec["support"] = mean(ratios) if ratios else None
            if use_judge:
                rec["judge"] = judge(rt, row["question"], ans)
        results.append(rec)
        print(f"  answers {i}/{len(rows)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    ans_rows = [r for r in results if r["type"] == "answerable"]
    abstain_rows = [r for r in results if r["type"] in ABSTAIN_TYPES]
    answered = [r for r in ans_rows if not r["abstained"]]
    judged = [r["judge"] for r in answered if isinstance(r.get("judge"), dict) and "supported" in r["judge"]]
    summary = {
        "mode": rt.answerer.mode,
        "answerable": len(ans_rows),
        "answered_rate": len(answered) / len(ans_rows) if ans_rows else 0.0,
        "key_fact_recall": mean([r.get("key_facts") for r in answered]),
        "cited_gold_rate": mean([float(r.get("cited_gold", False)) for r in answered]),
        "grounded_rate": mean([float(r.get("grounded", False)) for r in answered]),
        "citation_support": mean([r.get("support") for r in answered]),
        "invalid_citation_rate": mean(
            [float(bool(r["invalid_citations"])) for r in results if not r["abstained"]]
        ),
        "should_abstain": len(abstain_rows),
        "abstention_rate": mean([float(r["abstained"]) for r in abstain_rows]),
        "p50_ms": p50([r["ms"] for r in results]),
        "p95_ms": percentile([r["ms"] for r in results], 0.95),
    }
    if judged:
        summary["judge_supported"] = mean([float(bool(j.get("supported"))) for j in judged])
        summary["judge_answers_question"] = mean([float(bool(j.get("answers_question"))) for j in judged])
    return {"summary": summary, "results": results}


# ---- abstention threshold sweep -------------------------------------------------------------


def sweep(rt: Runtime, rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal = "rerank" if rt.reranker is not None else "dense"
    if signal == "dense" and rt.embedder is None:
        return {"error": "the sweep needs a reranker or an embedding model"}
    tops: list[tuple[str, float]] = []
    for row in rows:
        res = rt.retriever.retrieve(
            row["question"], row["workspace"], rt.access.principals[row["as"]], k=rt.settings.top_k
        )
        vals = [(h.rerank_score if signal == "rerank" else h.dense_score) for h in res.hits]
        tops.append((row["type"], max([v for v in vals if v is not None], default=0.0)))
    if signal == "rerank":
        grid = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    else:
        lo, hi = min(v for _, v in tops), max(v for _, v in tops)
        grid = [round(lo + (hi - lo) * i / 20, 4) for i in range(21)]
    table = []
    for t in grid:
        ans = [v for typ, v in tops if typ == "answerable"]
        abst = [v for typ, v in tops if typ in ABSTAIN_TYPES]
        kept = sum(v >= t for v in ans) / len(ans) if ans else 0.0
        blocked = sum(v < t for v in abst) / len(abst) if abst else 0.0
        table.append({"threshold": t, "answerable_kept": kept, "should_abstain_blocked": blocked})
    ok = [r for r in table if r["answerable_kept"] >= 0.95]
    best = max(ok, key=lambda r: (r["should_abstain_blocked"], -r["threshold"])) if ok else None
    return {"signal": signal, "table": table, "recommended": best}


# ---- report -------------------------------------------------------------------------------


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report(
    rt: Runtime,
    retrieval: list[RetrievalScores],
    answers: dict[str, Any] | None,
    sw: dict[str, Any] | None,
    n_rows: int,
) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    info = rt.store.index_info()
    lines = [
        "# Citewise eval report",
        "",
        f"- Run: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Questions: {n_rows} | embeddings: `{info.get('embedding_model')}` | reranker: "
        f"`{rt.reranker.name if rt.reranker else 'none'}` | store: {info.get('backend')}",
        "",
        "## Retrieval (answerable questions, document-level)",
        "",
        "| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Leaks | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in retrieval:
        lines.append(
            f"| {s.config} | {pct(s.hit1)} | {pct(s.hit3)} | {pct(s.hit5)} | {pct(s.hit10)} | "
            f"{s.mrr10:.3f} | {s.ndcg10:.3f} | {s.leaks} | {s.p50_ms:.0f} | {s.p95_ms:.0f} |"
        )
    lines += [
        "",
        "Hit@5 by slice: questions that reuse the page's wording vs. paraphrased ones.",
        "",
        "| Config | Same wording | Paraphrased |",
        "|---|---|---|",
    ]
    lines += [f"| {s.config} | {pct(s.keyword_hit5)} | {pct(s.paraphrase_hit5)} |" for s in retrieval]
    if answers:
        a = answers["summary"]
        lines += [
            "",
            f"## Answers (mode: `{a['mode']}`)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Answered (answerable questions) | {pct(a['answered_rate'])} of {a['answerable']} |",
            f"| Key-fact recall | {pct(a['key_fact_recall'])} |",
            f"| Cites the right page | {pct(a['cited_gold_rate'])} |",
            f"| Has citations | {pct(a['grounded_rate'])} |",
            f"| Citation word support | {pct(a['citation_support'])} |",
            f"| Answers with invalid citation numbers | {pct(a['invalid_citation_rate'])} |",
            f"| Correct abstentions (unanswerable + restricted) | {pct(a['abstention_rate'])} of {a['should_abstain']} |",
            f"| Latency p50 / p95 | {a['p50_ms']:.0f} / {a['p95_ms']:.0f} ms |",
        ]
        if "judge_supported" in a:
            lines.append(f"| Judge: every claim supported | {pct(a['judge_supported'])} |")
            lines.append(f"| Judge: answers the question | {pct(a['judge_answers_question'])} |")
        wrong_abstain = [r for r in answers["results"] if r["type"] in ABSTAIN_TYPES and not r["abstained"]]
        if wrong_abstain:
            lines += ["", "Answered when it should have abstained:", ""]
            lines += [f"- `{r['id']}` {r['question']}" for r in wrong_abstain]
    if sw and "table" in sw:
        lines += [
            "",
            f"## Abstention threshold sweep (signal: {sw['signal']})",
            "",
            "| Threshold | Answerable kept | Should-abstain blocked |",
            "|---|---|---|",
        ]
        lines += [
            f"| {r['threshold']} | {pct(r['answerable_kept'])} | {pct(r['should_abstain_blocked'])} |"
            for r in sw["table"]
        ]
        if sw.get("recommended"):
            lines += [
                "",
                f"Recommended: **{sw['recommended']['threshold']}** (keeps at least 95% of answerable questions).",
            ]
    primary = next(
        (s for s in retrieval if s.config == primary_config(rt)), retrieval[-1] if retrieval else None
    )
    if primary and primary.misses:
        lines += ["", f"## Misses for `{primary.config}` (gold page not in top 5)", ""]
        lines += [
            f"- `{m['id']}` {m['question']} — expected `{m['gold'][0]}`, got `{(m['got'] or ['-'])[0]}`"
            for m in primary.misses
        ]
    path = REPORTS / "latest.md"
    path.write_text("\n".join(lines) + "\n")
    (REPORTS / "latest.json").write_text(
        json.dumps(
            {
                "retrieval": [s.__dict__ for s in retrieval],
                "answers": answers,
                "sweep": sw,
            },
            indent=2,
            default=str,
        )
    )
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--configs", help="comma-separated subset, e.g. 'lexical,hybrid+rerank'")
    ap.add_argument("--answers", action="store_true", help="also generate and score answers")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness check (needs an LLM)")
    ap.add_argument("--sweep", action="store_true", help="abstention threshold sweep")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-hit5", type=float, help="fail if the primary config's Hit@5 is below this")
    ap.add_argument("--min-mrr", type=float, help="fail if the primary config's MRR@10 is below this")
    ap.add_argument("--min-abstention", type=float, help="fail if correct-abstention rate is below this")
    ap.add_argument("--min-key-facts", type=float, help="fail if key-fact recall is below this")
    args = ap.parse_args(argv)

    rt = Runtime(get_settings())
    rows = load_dataset(args.dataset, args.limit)
    print(f"{len(rows)} questions | {rt.retriever.describe()}", file=sys.stderr)

    retrieval = []
    for cfg in configs_for(rt, args.configs.split(",") if args.configs else None):
        s = eval_retrieval(rt, rows, cfg)
        retrieval.append(s)
        print(f"  {cfg.name:<16} hit@5 {pct(s.hit5):>6}  mrr {s.mrr10:.3f}  leaks {s.leaks}", file=sys.stderr)
    answers = eval_answers(rt, rows, args.judge) if args.answers else None
    sw = sweep(rt, rows) if args.sweep else None
    path = write_report(rt, retrieval, answers, sw, len(rows))
    print(path.read_text())

    failures = []
    primary = next((s for s in retrieval if s.config == primary_config(rt)), None)
    if any(s.leaks for s in retrieval):
        failures.append("access-control leak: a principal retrieved a chunk they may not read")
    if primary and args.min_hit5 is not None and primary.hit5 < args.min_hit5:
        failures.append(f"{primary.config} Hit@5 {primary.hit5:.3f} < {args.min_hit5}")
    if primary and args.min_mrr is not None and primary.mrr10 < args.min_mrr:
        failures.append(f"{primary.config} MRR@10 {primary.mrr10:.3f} < {args.min_mrr}")
    if (
        answers
        and args.min_abstention is not None
        and answers["summary"]["abstention_rate"] < args.min_abstention
    ):
        failures.append(f"abstention {answers['summary']['abstention_rate']:.3f} < {args.min_abstention}")
    if (
        answers
        and args.min_key_facts is not None
        and answers["summary"]["key_fact_recall"] < args.min_key_facts
    ):
        failures.append(f"key-fact recall {answers['summary']['key_fact_recall']:.3f} < {args.min_key_facts}")
    for f in failures:
        print(f"EVAL GATE FAILED: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
