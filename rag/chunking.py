"""
rag/chunking.py: split documents into retrievable pieces.

You rarely embed a whole document as one vector. A long page covers many topics,
and one vector can only point in one "average" direction, so it matches
everything vaguely and nothing well. Instead you split the document into
**chunks** and embed each chunk. Retrieval then pulls back the specific
paragraph that answers the question, not the whole file.

The two knobs that matter, and their tradeoffs:

  - chunk size: too big -> each chunk is unfocused and you waste context-window
    space on irrelevant text; too small -> a chunk may not contain enough to
    answer on its own, and you fragment ideas across many pieces.

  - overlap: chunks share a few words at their boundaries so a sentence that
    straddles a split isn't lost to both neighbours. Costs a little duplication.

There's no universally right setting. It depends on your documents and
questions, which is exactly why example 05 has you *measure* it. This module is
pure Python and makes no API calls, so it's free to explore.
"""

import re


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Split `text` into overlapping chunks of ~`chunk_size` words.

    A sliding window: take `chunk_size` words, then slide forward by
    `chunk_size - overlap` so consecutive chunks share `overlap` words.

    We count in *words*, not characters or tokens, because it's simple and
    provider-agnostic. (Production systems often chunk by tokens so a chunk maps
    cleanly to the model's limits: a refinement, not a different idea.)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break  # this window reached the end; don't emit a tiny tail-only chunk
        start += step
    return chunks


def chunk_paragraphs(text: str) -> list[str]:
    """Split on blank lines, one chunk per paragraph.

    The alternative to fixed-size windows: respect the document's own structure.
    Great when paragraphs are self-contained (FAQs, docs, articles); weaker when
    they vary wildly in length. A common production approach blends the two:
    split on paragraphs, then pack/sub-split toward a target size.
    """
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if p]


def chunk_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown on its headings, returning (heading, body) sections.

    The most *structure-aware* splitter here: instead of a fixed-size window that
    cuts wherever the word count runs out, happily gluing the tail of one topic
    onto the head of the next, this cuts only at the document's own `#`/`##`/...
    headings. Each section is about one thing, and the heading travels with it as
    metadata you can filter and cite ("getting-started.md > Exporting your notes").

    Example 13 shows *why* this matters: a fixed-size window merged a doc's
    "Importing" and "Exporting" sections into one chunk, so retrieval returned a
    chunk whose first sentence was about the wrong topic. Heading splitting keeps
    them apart. The catch: sections vary in length, so production pipelines often
    split on headings *first*, then size-cap the long ones with `chunk_text()`.
    """
    sections: list[tuple[str, str]] = []
    heading = "(intro)"
    buf: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = line.lstrip("#").strip()
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]
