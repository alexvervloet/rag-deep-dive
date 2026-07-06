"""
Example 11 — contextual retrieval.
==================================

A chunk ripped out of its document often loses the context that makes it findable.
"The cap is 500 GB." — the cap on *what*, for *which plan*? Embedded alone, that
sentence is indistinguishable from every other plan's "the cap is X," so retrieval
can't tell which one answers "what's the cap on the Cedar plan?".

**Contextual retrieval** (popularized by Anthropic) fixes this cheaply: before
embedding each chunk, prepend a short, LLM-written sentence that *situates the chunk
within its document*. Now the chunk's vector carries "this is the Cedar plan's
storage cap," so the query finds the right one. The original chunk text is still
what you show the model; the context is only there to improve the *embedding*.

This is deliberately the hardest case for retrieval — several near-identical chunks
that differ only by an entity (the plan name) that the chunk text never mentions.
Modern embeddings are strong enough that a chunk which merely *implies* its context
usually still ranks; the place they genuinely fail is disambiguation like this,
which is exactly where contextual retrieval earns its keep.

We use a tiny purpose-built corpus below (not corpus/, which is written too self-
containedly to ever be under-specified). Each plan doc puts its bare "the cap is X"
fact in its own paragraph, naming neither the plan nor "storage".

(One LLM call per chunk at index time — a one-time ingest cost. A real system does
one context pass per document and caches it.)

Run it:

    secrun python examples/11_contextual_retrieval.py
    secrun python examples/11_contextual_retrieval.py "what is the storage cap on the Wren plan?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import rag

load_dotenv()
rag.ensure_ready()
print(f"Provider: {rag.describe()}\n")

# A purpose-built mini-corpus: three plans whose storage cap lives in its own
# paragraph, naming neither the plan nor the word "storage". Plain retrieval can't
# tell the three "the cap is X" chunks apart; only document context disambiguates.
MINI_CORPUS = [
    ("orchid.md", "Orchid is one of our three subscription plans.\n\n"
                  "The cap is 2 TB, pooled across the workspace, and it refreshes every year."),
    ("cedar.md", "Cedar is one of our three subscription plans.\n\n"
                 "The cap is 500 GB, pooled across the workspace, and it refreshes every year."),
    ("wren.md", "Wren is one of our three subscription plans.\n\n"
                "The cap is 50 GB, pooled across the workspace, and it refreshes every year."),
]

QUERY = sys.argv[1] if len(sys.argv) > 1 else "what is the storage cap on the Cedar plan?"
K = 3

# One (chunk, source) per paragraph — small enough that the bare fact stands alone.
CHUNKS = [(para, source) for source, doc in MINI_CORPUS for para in rag.chunk_paragraphs(doc)]


def build_plain():
    """Ordinary chunking: embed each chunk as-is."""
    store = rag.VectorStore()
    texts = [c for c, _ in CHUNKS]
    store.add(texts, rag.embed(texts, input_type="document"),
              [{"source": s} for _, s in CHUNKS])
    return store


def context_for(chunk: str, full_doc: str, source: str) -> str:
    """One short sentence situating this chunk in its document."""
    return rag.generate(
        "You write one short sentence that situates a text chunk within its document, "
        "to improve search. Mention the document topic and what the chunk is about. "
        "Reply with only the sentence.",
        f"Document '{source}':\n{full_doc}\n\nChunk:\n{chunk}\n\nContext sentence:",
        max_tokens=60,
    ).strip()


def build_contextual():
    """Prepend an LLM-written context sentence to each chunk *before embedding*.

    We embed `context + chunk` but keep the original `chunk` as the stored text, so
    the model still sees clean source text — only retrieval benefits.
    """
    docs = dict(MINI_CORPUS)
    to_embed = [f"{context_for(c, docs[s], s)}\n{c}" for c, s in CHUNKS]
    store = rag.VectorStore()
    store.add([c for c, _ in CHUNKS], rag.embed(to_embed, input_type="document"),
              [{"source": s} for _, s in CHUNKS])
    return store


def show(label: str, hits):
    print(f"[{label}]")
    for score, rec in hits:
        preview = rag.snippet(rec.text, QUERY, width=70)
        print(f"  {score:.3f}  [{rec.metadata.get('source', '?')}]  {preview}")
    print()


if __name__ == "__main__":
    print(f"Question: {QUERY}\n" + "=" * 70 + "\n")

    plain = build_plain()
    show("plain chunks", rag.retrieve(plain, QUERY, k=K))

    print("Building contextual index (one short LLM context call per chunk)...\n")
    contextual = build_contextual()
    show("contextual chunks (context prepended before embedding)",
         rag.retrieve(contextual, QUERY, k=K))

    print("=" * 70)
    print(
        "Takeaway: plain retrieval returns the WRONG plan's cap at #1 — the three\n"
        "'the cap is X' chunks are near-identical and none names its plan, so the\n"
        "embedding can't disambiguate. Prepending a one-sentence 'this is the Cedar\n"
        "plan's storage cap' before embedding (while still showing the model the clean\n"
        "chunk) puts the right one on top. That's the cheap, high-leverage win — and it\n"
        "matters most exactly where modern embeddings struggle: telling near-identical\n"
        "passages apart by an entity the text never states."
    )
