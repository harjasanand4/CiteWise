"""Turn handbook Markdown / MDX / Liquid pages into clean sections of text.

Real handbooks are not clean Markdown. The PostHog handbook is MDX (JSX components such as
`<CompensationCalculator />` inline with the text) and the TTS handbook is Eleventy with
Liquid tags (`{% alert %}`, `{% page "..." %}`, `{% comment %}`). Leaving that markup in
pollutes both keyword search and embeddings, so it is stripped here while headings, lists,
tables and code blocks keep their structure for the chunker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

LIQUID_COMMENT = re.compile(r"\{%-?\s*comment\s*-?%\}.*?\{%-?\s*endcomment\s*-?%\}", re.S)
LIQUID_SLACK = re.compile(r"\{%-?\s*slack_channel\s+[\"']([^\"']+)[\"'](?:\s+[\"'][^\"']*[\"'])?\s*-?%\}")
LIQUID_TAG = re.compile(r"\{%-?.*?-?%\}", re.S)
LIQUID_OUTPUT = re.compile(r"\{\{.*?\}\}", re.S)
NUNJUCKS_COMMENT = re.compile(r"\{#.*?#\}", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.S | re.I)

# JSX: <Component ... /> (possibly spanning lines) and paired <Component ...> ... </Component>
JSX_SELF_CLOSING = re.compile(r"<([A-Z][\w.]*)\b((?:[^<>]|\{[^{}]*\})*?)/>", re.S)
JSX_OPEN_CLOSE = re.compile(r"</?[A-Z][\w.]*\b(?:[^<>]|\{[^{}]*\})*?>", re.S)
JSX_LABEL_ATTR = re.compile(r"\b(?:title|label|name|slug|text)\s*=\s*[\"']([^\"']+)[\"']")
HTML_BR = re.compile(r"<br\s*/?>", re.I)
HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]*?>")
MDX_IMPORT_EXPORT = re.compile(r"^(import|export)\s.+$", re.M)

IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
LINK_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.M)
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")
ENTITY = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&rsquo;": "'"}


@dataclass
class Section:
    headings: list[str]  # path below the document title, e.g. ["Logistics", "Receipts"]
    blocks: list[str] = field(default_factory=list)


@dataclass
class ParsedDoc:
    title: str
    sections: list[Section]
    keywords: list[str] = field(default_factory=list)
    permalink: str | None = None
    skip_reason: str | None = None  # redirects / empty pages are not indexed

    @property
    def text(self) -> str:
        return "\n\n".join(b for s in self.sections for b in s.blocks)


def split_front_matter(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[m.end() :]


def _jsx_self_closing(m: re.Match[str]) -> str:
    label = JSX_LABEL_ATTR.search(m.group(2))
    return label.group(1).replace("-", " ") if label else ""


def strip_markup(text: str) -> str:
    """Remove templating and component markup. Runs on everything outside code fences."""
    text = LIQUID_COMMENT.sub("", text)
    text = NUNJUCKS_COMMENT.sub("", text)
    text = LIQUID_SLACK.sub(lambda m: "#" + m.group(1), text)
    text = LIQUID_TAG.sub("", text)
    text = LIQUID_OUTPUT.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = SCRIPT_STYLE.sub("", text)
    text = MDX_IMPORT_EXPORT.sub("", text)
    text = JSX_SELF_CLOSING.sub(_jsx_self_closing, text)
    text = JSX_OPEN_CLOSE.sub("", text)
    text = HTML_BR.sub("\n", text)
    text = HTML_TAG.sub("", text)
    return text


def clean_inline(line: str) -> str:
    line = IMAGE.sub("", line)
    line = LINK.sub(r"\1", line)
    line = REF_LINK.sub(r"\1", line)
    line = AUTOLINK.sub(r"\1", line)
    line = EMPHASIS.sub(r"\2", line)
    for k, v in ENTITY.items():
        line = line.replace(k, v)
    return line.rstrip()


def _blocks(lines: list[str]) -> list[str]:
    """Group lines into blocks: paragraphs, whole lists, whole tables, whole code fences."""
    blocks: list[list[str]] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal current
        if any(x.strip() for x in current):
            blocks.append(current)
        current = []

    for line in lines:
        if FENCE.match(line):
            if not in_fence:
                flush()
            current.append(line)
            in_fence = not in_fence
            if not in_fence:
                flush()
            continue
        if in_fence:
            current.append(line)
            continue
        if not line.strip():
            flush()
            continue
        current.append(line)
    flush()

    # A list whose items are separated by blank lines is still one list.
    merged: list[list[str]] = []
    for b in blocks:
        if merged and LIST_ITEM.match(b[0]) and LIST_ITEM.match(merged[-1][0]):
            merged[-1].extend(b)
        else:
            merged.append(b)
    return [_unwrap(b) for b in merged]


def _unwrap(lines: list[str]) -> str:
    """Join hard-wrapped lines (many handbooks wrap Markdown at 80 columns) so a sentence is
    one line again. Code fences and tables keep their line structure; list items stay one
    per line, with their wrapped continuation lines folded back in."""
    if FENCE.match(lines[0]) or all(x.lstrip().startswith("|") for x in lines if x.strip()):
        return "\n".join(lines).strip("\n")
    out: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        starts_item = bool(LIST_ITEM.match(line))
        if out and not starts_item:
            out[-1] = f"{out[-1]} {text.lstrip('> ') if out[-1].lstrip().startswith('>') else text}"
        else:
            out.append(line.rstrip())
    return "\n".join(out)


def parse_markdown(raw: str, fallback_title: str) -> ParsedDoc:
    meta, body = split_front_matter(raw.replace("\r\n", "\n"))
    title = str(meta.get("title") or "").strip()
    keywords = meta.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    permalink = meta.get("permalink") if isinstance(meta.get("permalink"), str) else None

    if meta.get("redirect_to"):
        return ParsedDoc(title or fallback_title, [], skip_reason="redirect")

    # Strip markup outside code fences only; code samples are content.
    pieces = re.split(r"(^\s*(?:```|~~~).*?^\s*(?:```|~~~)\s*$)", body, flags=re.S | re.M)
    body = "".join(p if p.lstrip().startswith(("```", "~~~")) else strip_markup(p) for p in pieces)
    body = LINK_DEF.sub("", body)

    sections: list[Section] = [Section(headings=[])]
    stack: list[tuple[int, str]] = []
    pending: list[str] = []
    in_fence = False

    def close_section() -> None:
        sections[-1].blocks.extend(_blocks(pending))
        pending.clear()

    for line in body.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            level, heading = len(m.group(1)), clean_inline(m.group(2)).strip()
            if not heading:
                continue
            if level == 1 and not title and not stack:
                title = heading  # a leading H1 is the document title, not a section
                continue
            close_section()
            stack = [(lvl, h) for lvl, h in stack if lvl < level]
            stack.append((level, heading))
            sections.append(Section(headings=[h for _, h in stack]))
            continue
        pending.append(line if in_fence else clean_inline(line))
    close_section()

    sections = [s for s in sections if s.blocks]
    doc = ParsedDoc(
        title=title or fallback_title, sections=sections, keywords=list(keywords), permalink=permalink
    )
    if sum(len(b.split()) for s in sections for b in s.blocks) < 15:
        doc.skip_reason = "too short"
    return doc


def slugify(heading: str) -> str:
    """GitHub-style anchor, so citations can deep-link to the section on the real site."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", "-", s)
