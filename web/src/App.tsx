import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ApiError, askStream, getDemoUsers, getMe, sendFeedback } from "./api";
import { SourceCard } from "./components/SourceCard";
import { TurnView } from "./components/TurnView";
import { claimsFor } from "./text";
import type { DemoUser, Me, Turn } from "./types";

const USER_KEY = "citewise.user";

function remember(key: string, value?: string): string | null {
  try {
    if (value !== undefined) localStorage.setItem(key, value);
    return localStorage.getItem(key);
  } catch {
    return null; // storage can be unavailable (private windows, previews)
  }
}

export default function App() {
  const [users, setUsers] = useState<DemoUser[] | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [turns, setTurns] = useState<Record<string, Turn[]>>({});
  const [selectedTurn, setSelectedTurn] = useState<string | null>(null);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const [question, setQuestion] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const user = users?.find((u) => u.id === userId) ?? null;
  const apiKey = user?.api_key ?? null;

  useEffect(() => {
    getDemoUsers()
      .then((list) => {
        setUsers(list);
        if (list?.length) {
          const saved = remember(USER_KEY);
          setUserId(list.some((u) => u.id === saved) ? saved : list[0].id);
        } else {
          setUserId("local");
        }
      })
      .catch(() => setLoadError("Can't reach the Citewise API. Start it with `citewise serve`."));
  }, []);

  useEffect(() => {
    if (!userId) return;
    abort.current?.abort();
    setTurns({});
    setSelectedTurn(null);
    setActiveSource(null);
    getMe(apiKey)
      .then((m) => {
        setMe(m);
        setWorkspace((w) => (w && m.workspaces.some((x) => x.id === w) ? w : m.workspaces[0]?.id ?? null));
      })
      .catch((e) => setLoadError(e instanceof ApiError ? e.message : "Couldn't load your workspaces."));
  }, [userId, apiKey]);

  const ws = me?.workspaces.find((w) => w.id === workspace) ?? null;
  const thread = (workspace && turns[workspace]) || [];
  const busy = thread.some((t) => t.status === "searching" || t.status === "writing");
  const current = thread.find((t) => t.id === selectedTurn) ?? thread[thread.length - 1] ?? null;

  const update = useCallback((wid: string, id: string, patch: Partial<Turn>) => {
    setTurns((all) => ({ ...all, [wid]: (all[wid] ?? []).map((t) => (t.id === id ? { ...t, ...patch } : t)) }));
  }, []);

  const ask = async (text: string) => {
    const q = text.trim();
    if (!q || !workspace || busy) return;
    const wid = workspace;
    const id = crypto.randomUUID();
    const history = thread
      .filter((t) => t.answer && !t.answer.abstained)
      .slice(-3)
      .flatMap((t) => [
        { role: "user", content: t.question },
        { role: "assistant", content: t.answer!.text },
      ]);
    setTurns((all) => ({ ...all, [wid]: [...(all[wid] ?? []), { id, question: q, status: "searching", draft: "", sources: [] }] }));
    setSelectedTurn(id);
    setActiveSource(null);
    setQuestion("");
    abort.current = new AbortController();
    let draft = "";
    try {
      await askStream(apiKey, { workspace: wid, question: q, history }, {
        onSources: (sources) => update(wid, id, { sources, status: "writing" }),
        onToken: (t) => {
          draft += t;
          update(wid, id, { draft });
        },
        onAnswer: (answer) => {
          update(wid, id, { answer, sources: answer.sources, status: "done" });
          if (!answer.abstained && answer.cited.length) setActiveSource(answer.cited[0]);
        },
      }, abort.current.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      update(wid, id, { status: "error", error: e instanceof ApiError ? e.message : "The request failed. Try again." });
    }
  };

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, and React would
    // treat a returned value as the effect's cleanup function.
    bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [thread.length]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void ask(question);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void ask(question);
    }
  };

  const claims = useMemo(() => {
    const out: Record<number, string[]> = {};
    if (current?.answer && !current.answer.abstained) {
      for (const n of current.answer.cited) out[n] = claimsFor(current.answer.text, n);
    }
    return out;
  }, [current]);

  if (loadError) {
    return (
      <main className="fatal">
        <h1>Citewise</h1>
        <p>{loadError}</p>
      </main>
    );
  }

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="7" fill="currentColor" />
            <rect x="7" y="14" width="18" height="7" rx="1.5" fill="var(--highlight)" />
            <path d="M9 10h14M9 17.5h14M9 25h9" stroke="var(--surface)" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
          <span className="brand-name">Citewise</span>
        </div>
        {me && (
          <nav className="workspaces" aria-label="Handbooks">
            {me.workspaces.map((w) => (
              <button
                key={w.id}
                type="button"
                className={w.id === workspace ? "is-current" : ""}
                aria-current={w.id === workspace}
                onClick={() => {
                  setWorkspace(w.id);
                  setSelectedTurn(null);
                  setActiveSource(null);
                }}
              >
                {w.name}
              </button>
            ))}
          </nav>
        )}
        {users && users.length > 0 && (
          <label className="viewing-as">
            <span>Viewing as</span>
            <select
              value={userId ?? ""}
              onChange={(e) => {
                setUserId(e.target.value);
                remember(USER_KEY, e.target.value);
              }}
            >
              {users.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.name}
                </option>
              ))}
            </select>
            {user && (
              <small>{user.groups.length ? `Groups: ${user.groups.join(", ")}` : "No extra groups"}</small>
            )}
          </label>
        )}
      </header>

      <div className="layout">
        <main className="conversation">
          {ws && thread.length === 0 && (
            <div className="intro">
              <h1>Ask the {ws.name}</h1>
              <p className="intro-desc">{ws.description} Every answer cites the page it came from, and you only see pages your groups can read.</p>
              <ul className="examples">
                {ws.examples.map((ex) => (
                  <li key={ex}>
                    <button type="button" onClick={() => void ask(ex)}>
                      {ex}
                    </button>
                  </li>
                ))}
              </ul>
              <p className="license">Source text: {ws.license}.</p>
            </div>
          )}

          {thread.map((t) => (
            <TurnView
              key={t.id}
              turn={t}
              workspaceName={ws?.name ?? "handbook"}
              selected={t.id === current?.id}
              activeSource={t.id === current?.id ? activeSource : null}
              onSelect={() => setSelectedTurn(t.id)}
              onCite={(n) => {
                setSelectedTurn(t.id);
                setActiveSource(n);
              }}
              onFeedback={(rating) => {
                if (!t.answer || !workspace) return;
                update(workspace, t.id, { feedback: rating });
                void sendFeedback(apiKey, t.answer.id, rating).catch(() => update(workspace, t.id, { feedback: undefined }));
              }}
            />
          ))}
          <div ref={bottom} />

          <form className="composer" onSubmit={onSubmit}>
            <label htmlFor="q" className="sr-only">
              Your question
            </label>
            <textarea
              id="q"
              rows={1}
              value={question}
              placeholder={ws ? `Ask the ${ws.name}` : "Loading…"}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={onKey}
              disabled={!ws}
            />
            <button type="submit" disabled={!question.trim() || busy || !ws}>
              {busy ? "Answering" : "Ask"}
            </button>
          </form>
        </main>

        <aside className="sources" aria-label="Sources">
          {current && current.sources.length > 0 ? (
            <>
              <h2 className="sources-heading">
                {current.answer?.abstained ? "Closest pages" : `Sources for “${current.question}”`}
              </h2>
              {current.sources.map((s) => (
                <SourceCard
                  key={s.chunk_id}
                  source={s}
                  claims={claims[s.n] ?? []}
                  cited={!!current.answer?.cited.includes(s.n)}
                  active={activeSource === s.n}
                  onSelect={() => setActiveSource(s.n)}
                />
              ))}
            </>
          ) : (
            <p className="sources-empty">Sources appear here, with the supporting sentence highlighted.</p>
          )}
        </aside>
      </div>
    </div>
  );
}
