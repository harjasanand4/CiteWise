"""Heading-aware chunking.

Chunks never cross a section boundary (a chunk about "Receipts" should not end with the
first lines of "Reviewing expenses"), lists and tables are kept whole when they fit, and
tiny sections are merged with their neighbours so retrieval doesn't return a three-word
heading stub. Each chunk carries its heading path, which is embedded with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .markdown import ParsedDoc

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Piece:
    headings: list[str]
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def _split_long_block(block: str, max_words: int) -> list[str]:
    """Split a block that is too long on its own: lists/tables/code by line, prose by sentence."""
    lines = block.split("\n")
    units = lines if len(lines) > 1 else SENTENCE_END.split(block)
    out: list[str] = []
    current: list[str] = []
    count = 0
    joiner = "\n" if len(lines) > 1 else " "
    for unit in units:
        n = len(unit.split())
        if current and count + n > max_words:
            out.append(joiner.join(current))
            current, count = [], 0
        if n > max_words:  # one enormous sentence or line: hard split on words
            words = unit.split()
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i : i + max_words]))
            continue
        current.append(unit)
        count += n
    if current:
        out.append(joiner.join(current))
    return out


def chunk_document(
    doc: ParsedDoc, max_words: int = 220, min_words: int = 40, overlap_words: int = 50
) -> list[Piece]:
    pieces: list[Piece] = []
    for section in doc.sections:
        blocks: list[str] = []
        for b in section.blocks:
            blocks.extend(_split_long_block(b, max_words) if len(b.split()) > max_words else [b])

        current: list[str] = []
        count = 0
        for block in blocks:
            n = len(block.split())
            if current and count + n > max_words:
                pieces.append(Piece(section.headings, "\n\n".join(current)))
                # Carry the last block over when it is short, so a list doesn't lose its lead-in.
                tail = current[-1]
                tail_words = len(tail.split())
                current, count = (
                    ([tail], tail_words)
                    if tail_words <= overlap_words and tail_words + n <= max_words
                    else ([], 0)
                )
            current.append(block)
            count += n
        if current:
            pieces.append(Piece(section.headings, "\n\n".join(current)))

    return _merge_small(pieces, max_words, min_words)


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return out


def _merge_small(pieces: list[Piece], max_words: int, min_words: int) -> list[Piece]:
    merged: list[Piece] = []
    for p in pieces:
        prev = merged[-1] if merged else None
        if prev and (prev.words < min_words or p.words < min_words) and prev.words + p.words <= max_words:
            prefix = _common_prefix(prev.headings, p.headings)
            # Only merge siblings under the same top-level section (or doc intro + first section).
            if prefix or not prev.headings:

                def labelled(piece: Piece, prefix: list[str] = prefix) -> str:
                    label = " > ".join(piece.headings[len(prefix) :])
                    return f"{label}\n{piece.text}" if label else piece.text

                merged[-1] = Piece(prefix, f"{labelled(prev)}\n\n{labelled(p)}")
                continue
        merged.append(p)
    return merged
