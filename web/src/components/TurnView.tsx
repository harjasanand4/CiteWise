import type { Turn } from "../types";
import { seconds } from "../text";
import { AnswerText } from "./AnswerText";

interface Props {
  turn: Turn;
  workspaceName: string;
  selected: boolean;
  activeSource: number | null;
  onCite: (n: number) => void;
  onSelect: () => void;
  onFeedback: (rating: 1 | -1) => void;
}

function timingLine(t: Record<string, number>): string {
  const parts = [];
  if (t.retrieve !== undefined) parts.push(`found sources in ${seconds(t.retrieve)}`);
  if (t.first_token !== undefined) parts.push(`started answering at ${seconds(t.first_token)}`);
  if (t.total !== undefined) parts.push(`finished in ${seconds(t.total)}`);
  const line = parts.join(", ");
  return line.charAt(0).toUpperCase() + line.slice(1);
}

export function TurnView({ turn, workspaceName, selected, activeSource, onCite, onSelect, onFeedback }: Props) {
  const a = turn.answer;
  const available = new Set(turn.sources.map((s) => s.n));
  return (
    <section className={`turn${selected ? " is-selected" : ""}`} onClick={onSelect}>
      <h2 className="question">{turn.question}</h2>

      {turn.status === "searching" && <p className="status">Searching the {workspaceName}…</p>}

      {turn.status === "writing" && (
        <div className="answer is-writing" aria-live="polite">
          <AnswerText text={turn.draft} available={available} active={activeSource} onCite={onCite} />
        </div>
      )}

      {turn.status === "error" && <p className="error">{turn.error}</p>}

      {a && a.abstained && (
        <div className="answer is-abstained">
          <p className="abstain-title">Not in the {workspaceName}</p>
          <p>
            {a.abstain_reason === "no_sources"
              ? "Nothing you have access to mentions this."
              : "The closest pages are listed alongside, but none of them answer the question."}
          </p>
        </div>
      )}

      {a && !a.abstained && (
        <div className="answer">
          <AnswerText text={a.text} available={available} active={selected ? activeSource : null} onCite={onCite} />
          {!a.grounded && <p className="warning">This answer has no citations. Check the sources before relying on it.</p>}
        </div>
      )}

      {a && (
        <footer className="turn-meta">
          <span>
            {a.mode === "extractive" ? "Extractive mode, no language model" : a.mode}. {timingLine(a.timings_ms)}.
          </span>
          {!a.abstained && (
            <span className="feedback" role="group" aria-label="Was this answer helpful?">
              <button type="button" aria-pressed={turn.feedback === 1} onClick={(e) => { e.stopPropagation(); onFeedback(1); }}>
                Helpful
              </button>
              <button type="button" aria-pressed={turn.feedback === -1} onClick={(e) => { e.stopPropagation(); onFeedback(-1); }}>
                Not helpful
              </button>
            </span>
          )}
        </footer>
      )}
    </section>
  );
}
