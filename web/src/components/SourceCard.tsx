import { useEffect, useRef, useState } from "react";
import type { Source } from "../types";
import { highlight, ordinal, tidy } from "../text";

interface Props {
  source: Source;
  claims: string[];
  cited: boolean;
  active: boolean;
  onSelect: () => void;
}

const PREVIEW_CHARS = 420;

function ranking(s: Source): string {
  const parts: string[] = [];
  if (s.lexical_rank) parts.push(`${ordinal(s.lexical_rank)} by keyword`);
  if (s.dense_rank) parts.push(`${ordinal(s.dense_rank)} by meaning`);
  if (!parts.length) return "";
  return `Ranked ${parts.join(", ")}`;
}

export function SourceCard({ source, claims, cited, active, onSelect }: Props) {
  const ref = useRef<HTMLElement>(null);
  const [expanded, setExpanded] = useState(false);
  const segments = highlight(tidy(source.text), claims);
  const hasMark = segments.some((s) => s.mark);
  const long = source.text.length > PREVIEW_CHARS;
  const open = expanded || active || !long;

  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  let shown = 0;
  const visible = open
    ? segments
    : segments.filter((s) => {
        const keep = shown < PREVIEW_CHARS || s.mark;
        shown += s.text.length;
        return keep;
      });

  return (
    <article
      ref={ref}
      className={`source${active ? " is-active" : ""}${cited ? " is-cited" : ""}`}
      onClick={onSelect}
      aria-current={active}
    >
      <div className="source-tab" aria-hidden="true">{source.n}</div>
      <div className="source-body">
        <h3 className="source-title">
          <a href={source.url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
            {source.title}
          </a>
        </h3>
        {source.headings.length > 0 && <p className="source-path">{source.headings.join(" / ")}</p>}
        <div className={`source-text${active && hasMark ? " draw" : ""}`}>
          {visible.map((seg, i) =>
            seg.mark ? <mark key={i}>{seg.text}</mark> : <span key={i}>{seg.text}</span>,
          )}
          {!open && <span className="fade">…</span>}
        </div>
        <div className="source-meta">
          {source.rerank_score !== null && (
            <span className="relevance" title="Cross-encoder relevance">
              <span className="relevance-bar" style={{ width: `${Math.max(4, source.rerank_score * 100)}%` }} />
            </span>
          )}
          <span>{ranking(source)}</span>
          {source.flags.includes("possible_injection") && (
            <span className="flag">Contains text addressed to AI systems, treated as a quote</span>
          )}
          {long && !active && (
            <button type="button" className="link" onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}>
              {expanded ? "Show less" : "Show all"}
            </button>
          )}
        </div>
      </div>
    </article>
  );
}
