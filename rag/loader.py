"""
rag/loader.py: read a folder of documents into (name, text) pairs.

Trivial on purpose: glob the `.md` / `.txt` files in a directory and read each
one. The rest of the pipeline takes `(source_name, text)` tuples, and this is
where they come from. Real systems ingest PDFs, HTML, databases, and Slack
exports, but every one of those reduces to "get me some text and a name for it,"
which is all the pipeline needs.
"""

import glob
import os


def load_corpus(corpus_dir: str) -> list[tuple[str, str]]:
    """Return `(filename, text)` for every .md/.txt file in `corpus_dir`."""
    paths = sorted(
        glob.glob(os.path.join(corpus_dir, "*.md"))
        + glob.glob(os.path.join(corpus_dir, "*.txt"))
    )
    docs: list[tuple[str, str]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs
