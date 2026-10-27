import type { Answer, DemoUser, Me, Source } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function headers(apiKey: string | null): HeadersInit {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) h.Authorization = `Bearer ${apiKey}`;
  return h;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/** Demo users exist only when the server runs with CITEWISE_AUTH_MODE=demo. */
export async function getDemoUsers(): Promise<DemoUser[] | null> {
  const res = await fetch("/v1/demo-users");
  if (res.status === 404) return null;
  return json<DemoUser[]>(res);
}

export async function getMe(apiKey: string | null): Promise<Me> {
  return json<Me>(await fetch("/v1/me", { headers: headers(apiKey) }));
}

export async function sendFeedback(apiKey: string | null, answerId: string, rating: 1 | -1): Promise<void> {
  await json(await fetch("/v1/feedback", {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ answer_id: answerId, rating }),
  }));
}

export interface StreamHandlers {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onAnswer: (answer: Answer) => void;
}

/**
 * POST /v1/ask/stream and parse the Server-Sent Events by hand
 * (EventSource can't send a POST body or an Authorization header).
 */
export async function askStream(
  apiKey: string | null,
  body: { workspace: string; question: string; history: { role: string; content: string }[] },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/v1/ask/stream", {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) await json(res);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = raw.split("\n").find((l) => l.startsWith("data: "));
      if (!data) continue;
      const event = JSON.parse(data.slice(6));
      if (event.type === "sources") handlers.onSources(event.sources);
      else if (event.type === "token") handlers.onToken(event.text);
      else if (event.type === "answer") handlers.onAnswer(event.answer);
      else if (event.type === "error") throw new ApiError(502, event.detail);
    }
  }
}
