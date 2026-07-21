"""
Example 12: metadata filtering & parent-document retrieval.

Two retrieval upgrades that have nothing to do with embeddings. They're about
*what you store alongside each chunk* and *what you return*.

1. METADATA FILTERING. Every chunk already carries metadata (its source document).
   If you also store fields like category, date, or access-level, you can constrain
   the search: "only retrieve from billing docs," "only docs this user may see."
   This is precision (don't match the wrong document) AND security (don't leak one).

2. PARENT-DOCUMENT (small-to-big) RETRIEVAL. There's a tension in chunk size: small
   chunks *match* precisely (focused vectors) but are *too small to answer from*;
   big chunks answer well but match fuzzily. The fix: embed SMALL chunks for the
   match, but return the LARGER parent (section or document) for the model to read.
   Best of both: precise retrieval, complete context.

This script demonstrates each over the support corpus.

Run it:

    secrun python examples/12_metadata_and_parent.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import rag

load_dotenv()
rag.ensure_ready()
print(f"Provider: {rag.describe()}\n")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = rag.load_corpus(os.path.join(REPO_ROOT, "corpus"))

# Map each source file to a coarse category, the kind of metadata you'd attach at
# ingest time (here derived from the filename for simplicity).
def category_of(source: str) -> str:
    s = source.lower()
    if "billing" in s or "plan" in s:
        return "billing"
    if "security" in s or "privacy" in s:
        return "security"
    return "general"


def filtered_search(store, query: str, k: int, where) -> list:
    """Top-k cosine search, but only over records whose metadata passes `where`."""
    qv = rag.embed([query], input_type="query")[0]
    scored = [
        (rag.cosine_similarity(qv, r.vector), r)
        for r in store.records
        if where(r.metadata)
    ]
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored[:k]


def demo_metadata_filtering():
    print("1) METADATA FILTERING\n" + "-" * 40)
    # Index with a `category` field on every chunk.
    texts, metas = [], []
    for source, text in docs:
        for i, chunk in enumerate(rag.chunk_text(text, chunk_size=120, overlap=20)):
            texts.append(chunk)
            metas.append({"source": source, "chunk": i, "category": category_of(source)})
    store = rag.VectorStore()
    store.add(texts, rag.embed(texts, input_type="document"), metas)

    query = "how do I keep my account safe?"
    print(f'Query: "{query}"\n')
    print("Unfiltered top-3 (any category):")
    for score, rec in store.search(rag.embed([query], input_type="query")[0], k=3):
        print(f"  {score:.3f}  [{rec.metadata['category']}] {rec.metadata['source']}")
    print("\nFiltered to category == 'security' only:")
    for score, rec in filtered_search(store, query, 3, lambda m: m["category"] == "security"):
        print(f"  {score:.3f}  [{rec.metadata['category']}] {rec.metadata['source']}")
    print("\n-> The filter guarantees results come only from allowed/relevant docs,\n"
          "   precision and access-control in one move.\n")
    return store


def demo_parent_document():
    print("2) PARENT-DOCUMENT (SMALL-TO-BIG) RETRIEVAL\n" + "-" * 40)
    # Embed sentence-sized chunks for a precise match, but map each back to its
    # heading SECTION (example 13's split), the "parent" we actually return. Small
    # enough to match sharply, big enough to answer from.
    parents: dict[tuple[str, str], str] = {}
    texts, metas = [], []
    for source, text in docs:
        for heading, body in rag.chunk_markdown_sections(text):
            parents[(source, heading)] = body
            for chunk in rag.chunk_text(body, chunk_size=14, overlap=4):  # ~one sentence
                texts.append(chunk)
                metas.append({"source": source, "heading": heading})
    store = rag.VectorStore()
    store.add(texts, rag.embed(texts, input_type="document"), metas)

    query = "how do I turn on two-factor authentication?"
    print(f'Query: "{query}"\n')
    top_score, top_rec = store.search(rag.embed([query], input_type="query")[0], k=1)[0]
    key = (top_rec.metadata["source"], top_rec.metadata["heading"])
    parent = parents[key]
    chunk_len, parent_len = len(top_rec.text.split()), len(parent.split())
    print(f"Best-matching SMALL chunk ({chunk_len} words, score {top_score:.3f}) "
          f"from {key[0]} > {key[1]}:")
    print(f"  \"{top_rec.text}\"")
    print("  ^ a precise hit, but too thin to answer from. It says where to click and\n"
          "    nothing about the recovery codes you'd be locked out without.\n")
    print(f"-> Return its PARENT section instead ({parent_len} words, ~{parent_len // chunk_len}x "
          f"the chunk): same precise match, now a complete answer:")
    print("  " + " ".join(parent.split()) + "\n")


if __name__ == "__main__":
    print("=" * 70)
    demo_metadata_filtering()
    demo_parent_document()
    print("=" * 70)
    print(
        "Takeaway: retrieval quality isn't only about embeddings. Metadata lets you\n"
        "constrain WHERE you search (relevance + security); small-to-big lets you MATCH\n"
        "on focused chunks but ANSWER from full context. Both are cheap structural wins."
    )
