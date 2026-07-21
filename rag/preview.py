"""
rag/preview.py: a display helper for the examples.

This is NOT a moving part of RAG. It's pure presentation, and the one file here
you can skip if you only care about retrieval. But it fixes a genuinely confusing
habit in every example that prints a retrieved chunk.

When you show the *first* N characters of a chunk, you often advertise the wrong
sentence: the words that actually matched the query can sit anywhere inside the
chunk, and a fixed-size chunk may even open on a different topic entirely (see
example 07, where a chunk that answers "how do I export?" opens on an unrelated
import-error code). `snippet()` centers a short preview on where the query hit, so
the printed line shows *why* the chunk came back. It falls back to the chunk's
start when no query word appears, which is itself a useful signal.

Pure Python, no API calls.
"""

import re

# Words too common to be worth centering a preview on; they'd match everywhere.
_STOP = {"how", "do", "i", "get", "my", "of", "the", "a", "an", "to", "is", "in",
         "on", "does", "what", "mean", "for", "your", "you", "and", "or", "it",
         "can", "how's", "with", "are", "was", "if"}


def snippet(text: str, query: str, width: int = 75) -> str:
    """A keyword-in-context preview: center a `width`-char window on the query hit.

    The matching phrase often sits in the *middle* of a chunk, so showing the
    first `width` chars would advertise the wrong sentence. We find where the
    query's content words land and center the window there, marking truncation
    with leading/trailing ellipses. If nothing matches, fall back to the start.
    """
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    low = text.lower()
    terms = {w for w in re.findall(r"[a-z0-9]+", query.lower())
             if len(w) > 2 and w not in _STOP}
    terms |= {w[:-1] for w in terms if w.endswith("s")}  # so "notes" matches "note"
    hits = [i for t in terms for i in range(len(low)) if low.startswith(t, i)]
    if not hits:
        return text[:width] + "..."
    # Of all the hit positions, center on the one with the most hits nearby.
    half = width // 2
    best = max(hits, key=lambda c: sum(c - half <= h <= c + half for h in hits))
    end = min(len(text), max(best + half, width))
    start = max(0, end - width)
    return f"{'...' if start else ''}{text[start:end]}{'...' if end < len(text) else ''}"
