/** Small, dependency-free text helpers: citation parsing, a tiny Markdown subset, highlighting. */

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "strong"; text: string }
  | { kind: "code"; text: string }
  | { kind: "cite"; n: number };

export type Block = { kind: "p"; inline: Inline[] } | { kind: "ul" | "ol"; items: Inline[][] };

const CITE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+(?:\s*,\s*\d+)*\])/g;

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  for (const part of text.split(INLINE)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**")) out.push({ kind: "strong", text: part.slice(2, -2) });
    else if (part.startsWith("`") && part.endsWith("`")) out.push({ kind: "code", text: part.slice(1, -1) });
    else if (/^\[\d/.test(part)) {
      for (const n of part.slice(1, -1).split(",")) out.push({ kind: "cite", n: Number(n.trim()) });
    } else out.push({ kind: "text", text: part });
  }
  return out;
}

/** Paragraphs and lists are all the model is told to produce, so that's all we render. */
export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  for (const chunk of text.trim().split(/\n{2,}/)) {
    const lines = chunk.split("\n").filter((l) => l.trim());
    const bullet = lines.every((l) => /^\s*[-*•]\s+/.test(l));
    const numbered = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));
    if (lines.length && (bullet || numbered)) {
      blocks.push({
        kind: bullet ? "ul" : "ol",
        items: lines.map((l) => parseInline(l.replace(/^\s*(?:[-*•]|\d+[.)])\s+/, ""))),
      });
    } else if (lines.length) {
      blocks.push({ kind: "p", inline: parseInline(lines.join(" ")) });
    }
  }
  return blocks;
}

const WORD = /[a-z0-9$£€%]+/g;
const STOP = new Set(
  "the and for you your are can with that this from have has will not but our out all any per how what when who".split(" "),
);

function words(s: string): Set<string> {
  return new Set((s.toLowerCase().match(WORD) ?? []).filter((w) => (w.length > 2 || /\d/.test(w)) && !STOP.has(w)));
}

// A sentence plus any citation markers that follow its full stop ("... 14 days. [2]").
const SENTENCE_WITH_CITES = /[^.!?\n]+(?:[.!?]+|$)(?:\s*\[\d+(?:\s*,\s*\d+)*\])*/g;

/** Sentences of the answer that cite source `n`, with the citation markers removed. */
export function claimsFor(answer: string, n: number): string[] {
  return (answer.match(SENTENCE_WITH_CITES) ?? [])
    .filter((s) => [...s.matchAll(CITE)].some((m) => m[1].split(",").map((x) => Number(x.trim())).includes(n)))
    .map((s) => s.replace(CITE, "").trim());
}

/** Light clean-up of Markdown residue in a source passage, for reading. */
export function tidy(passage: string): string {
  return passage
    .replace(/^\s*>\s?/gm, "")
    .replace(/^(\s*)[-*]\s+/gm, "$1• ")
    .replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,;:!?]|$)/g, "$1$2")
    .replace(/`([^`\n]+)`/g, "$1");
}

/**
 * Split a source passage into segments, marking the sentence(s) that best support the
 * answer's claims. This is what lets a reader check a citation in one glance.
 * Separators (spaces, line breaks) are kept as their own segments so layout survives.
 */
export function highlight(passage: string, claims: string[]): { text: string; mark: boolean }[] {
  const parts = passage.split(/((?<=[.!?:])[ \t]+(?=[A-Z0-9"'(•|])|\n+)/);
  const segments = parts.map((text, i) => ({ text, mark: false, sentence: i % 2 === 0 && text.trim() !== "" }));
  for (const claim of claims) {
    const cw = words(claim);
    let best = -1;
    let bestScore = 0;
    segments.forEach((seg, i) => {
      if (!seg.sentence) return;
      const sw = words(seg.text);
      let overlap = 0;
      cw.forEach((w) => sw.has(w) && overlap++);
      const score = overlap / Math.sqrt(sw.size + 1);
      if (overlap >= 2 && score > bestScore) {
        bestScore = score;
        best = i;
      }
    });
    if (best >= 0) segments[best].mark = true;
  }
  return segments.map(({ text, mark }) => ({ text, mark }));
}

export function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

export function seconds(ms: number | undefined): string {
  if (ms === undefined) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
