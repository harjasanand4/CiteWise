import type { ReactNode } from "react";
import { parseBlocks, type Inline } from "../text";

interface Props {
  text: string;
  available: Set<number>;
  active: number | null;
  onCite: (n: number) => void;
}

export function AnswerText({ text, available, active, onCite }: Props) {
  const renderInline = (parts: Inline[]): ReactNode[] =>
    parts.map((p, i) => {
      if (p.kind === "strong") return <strong key={i}>{p.text}</strong>;
      if (p.kind === "code") return <code key={i}>{p.text}</code>;
      if (p.kind === "cite") {
        if (!available.has(p.n)) return null;
        return (
          <button
            key={i}
            type="button"
            className={`cite${active === p.n ? " is-active" : ""}`}
            onClick={() => onCite(p.n)}
            aria-label={`Show source ${p.n}`}
          >
            {p.n}
          </button>
        );
      }
      return <span key={i}>{p.text}</span>;
    });

  return (
    <div className="answer-text">
      {parseBlocks(text).map((b, i) =>
        b.kind === "p" ? (
          <p key={i}>{renderInline(b.inline)}</p>
        ) : b.kind === "ul" ? (
          <ul key={i}>{b.items.map((it, j) => <li key={j}>{renderInline(it)}</li>)}</ul>
        ) : (
          <ol key={i}>{b.items.map((it, j) => <li key={j}>{renderInline(it)}</li>)}</ol>
        ),
      )}
    </div>
  );
}
