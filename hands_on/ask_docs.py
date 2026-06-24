#!/usr/bin/env python3
"""
ask_docs.py — the capstone: ask questions over your documents.
==============================================================

Everything in the repo comes together here. Point it at the `corpus/` folder and
ask a question; it chunks and embeds the documents (once — the index is cached to
disk), retrieves the most relevant chunks, and has the model answer using ONLY
those chunks, with [n] citations you can check against a table of sources.

It's a real, if small, "chat with your docs" tool — the thing people mean when
they say "RAG."

Examples
--------
  # Ask the built-in demo question
  python hands_on/ask_docs.py

  # Ask your own
  python hands_on/ask_docs.py "Can students get a discount?"

  # Retrieve more chunks, and show the exact text that was retrieved
  python hands_on/ask_docs.py "How is my data protected?" -k 6 --show-context

  # Re-embed from scratch (after editing corpus/, or to change chunking)
  python hands_on/ask_docs.py "What plans are there?" --rebuild --chunk-size 80

The index is cached in .rag_index.json. It records which provider and chunk
settings built it, and rebuilds automatically if those change — vectors from one
embedding model are meaningless to another.
"""

import argparse
import json
import os
import sys

# Make the repo root importable so `import rag` works from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

import rag

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(REPO_ROOT, "corpus")
CACHE_PATH = os.path.join(REPO_ROOT, ".rag_index.json")

DEFAULT_QUESTION = "How do I turn on two-factor authentication, and what if I lose my device?"


def build_or_load_index(chunk_size: int, overlap: int, rebuild: bool):
    """Return (store, was_built). Reuse the cached index when it matches the
    current provider and chunk settings; otherwise (re)build and cache it.

    Embedding is the slow, paid step — so we do it once and persist the result,
    exactly as a production system would. The cache header guards against using
    vectors built by a different embedding model.
    """
    meta = {
        "provider": rag.provider_name(),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }

    if not rebuild and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("meta") == meta:
            recs = blob["records"]
            store = rag.VectorStore()
            store.add(
                [r["text"] for r in recs],
                [r["vector"] for r in recs],
                [r["metadata"] for r in recs],
            )
            return store, False

    docs = rag.load_corpus(CORPUS_DIR)
    if not docs:
        sys.exit(f"No documents found in {CORPUS_DIR}. Add some .md/.txt files.")
    store = rag.index_documents(docs, chunk_size=chunk_size, overlap=overlap)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "records": [
                    {"text": r.text, "vector": r.vector, "metadata": r.metadata}
                    for r in store.records
                ],
            },
            f,
        )
    return store, True


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ask questions over the corpus/ documents, with citations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("question", nargs="?", default=DEFAULT_QUESTION,
                   help="Question to answer (default: a built-in demo question).")
    p.add_argument("-k", "--top-k", type=int, default=4,
                   help="How many chunks to retrieve and give the model (default 4).")
    p.add_argument("--chunk-size", type=int, default=120,
                   help="Chunk size in words for indexing (default 120).")
    p.add_argument("--overlap", type=int, default=20,
                   help="Chunk overlap in words (default 20).")
    p.add_argument("--rebuild", action="store_true",
                   help="Re-embed the corpus from scratch instead of using the cache.")
    p.add_argument("--show-context", action="store_true",
                   help="Print the full text of each retrieved chunk.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    load_dotenv()
    rag.ensure_ready()

    console = Console()
    console.print(f"[dim]Provider: {rag.describe()}[/dim]")

    store, built = build_or_load_index(args.chunk_size, args.overlap, args.rebuild)
    state = "built and cached" if built else "loaded from cache"
    console.print(f"[dim]Index {state}: {len(store)} chunks.[/dim]\n")

    console.print(f"[bold]Q:[/bold] {args.question}\n")

    text, hits = rag.answer(store, args.question, k=args.top_k)

    # The answer, rendered as Markdown (the model often replies with formatting).
    console.print(Markdown(text))

    # The sources behind it — [n] matches the citation markers in the answer.
    table = Table(title="\nSources", show_lines=False)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Document", style="green")
    table.add_column("Similarity", justify="right")
    for n, (score, rec) in enumerate(hits, start=1):
        table.add_row(str(n), rec.metadata["source"], f"{score:.3f}")
    console.print(table)

    if args.show_context:
        console.print("\n[bold]Retrieved chunks:[/bold]")
        for n, (_score, rec) in enumerate(hits, start=1):
            console.print(f"\n[cyan][{n}] {rec.metadata['source']}[/cyan]")
            console.print(" ".join(rec.text.split()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
