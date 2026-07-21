"""
Example 13: chunking strategies: fixed-size vs structure-aware.

Back in example 07 we found a subtle bug that wasn't in the *retrieval* code at
all. It was in the *chunking*. A fixed-size sliding window (120 words) had glued
the tail of one section onto the head of the next, merging a doc's "Importing"
troubleshooting text and its "Exporting your notes" section into a single chunk.
Retrieval still found that chunk, but its first sentence was about the wrong
topic (an "NN-413" import error), and its keyword profile was muddied.

This example makes the fix concrete. We chunk the same corpus two ways:

  1. FIXED-SIZE   (`chunk_text`, a blind word-window), and
  2. HEADING-AWARE (`chunk_markdown_sections`, split on the document's own
     `#`/`##` headings),

and watch the difference. Part 1 is **offline** (pure text). Part 2 embeds and
retrieves, so it needs a provider key like the other examples.

The honest ending matters: heading-aware chunking fixes the *structure* problem
(clean, single-topic, citable chunks), but it does **not** fix the *vocabulary*
problem from example 07, where the query says "get my notes out" and the answer
says "export." No chunking rescues that; that's what query transformation
(example 10) is for. Chunking and query understanding are different levers.

Run it:

    secrun python examples/13_chunking_strategies.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY = "how do I get my notes out of the app?"


def oneline(text: str, width: int = 88) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width] + "..."


docs = rag.load_corpus(os.path.join(REPO_ROOT, "corpus"))


# --- Part 1: the merge, side by side (offline) -------------------------------
print("=" * 78)
print("1) THE SAME DOC, CHUNKED TWO WAYS  (offline)\n")

# The getting-started doc is where example 07's merged chunk came from.
src, text = next((s, t) for s, t in docs if "getting-started" in s)

fixed = rag.chunk_text(text, chunk_size=120, overlap=20)
merged = next(c for c in fixed if "NN-413" in c and "Exporting" in c)
merged_flat = " ".join(merged.split())
seam = merged_flat.find("Exporting")
import_half = merged_flat[:seam].rstrip(" #")
export_half = merged_flat[seam:]
print(f"FIXED-SIZE (120-word window): {len(fixed)} chunks for {src}.")
print("The 120-word cut glued an import-error paragraph onto the start of the export")
print("section. Both lines below are the SAME chunk: one chunk, two subjects:")
print(f"    topic 1 · importing:  {oneline(import_half, 62)}")
print(f"    topic 2 · exporting:  {oneline(export_half, 62)}\n")

sections = rag.chunk_markdown_sections(text)
print(f"HEADING-AWARE: {len(sections)} sections for {src}. The same content, but")
print("'Importing' and 'Exporting' are now separate, single-topic chunks:")
for heading, body in sections:
    if "mport" in heading or "xport" in heading:
        print(f"  [{heading}]  {oneline(body, 62)}")
print()


# --- Part 2: what each one retrieves (needs a provider key) -------------------
print("=" * 78)
print("2) WHAT EACH STRATEGY RETRIEVES  (needs a provider key)\n")
rag.ensure_ready()


def build_store(chunks_with_meta):
    texts = [t for t, _ in chunks_with_meta]
    metas = [m for _, m in chunks_with_meta]
    store = rag.VectorStore()
    store.add(texts, rag.embed(texts, input_type="document"), metas)
    return store


fixed_chunks = [
    (c, {"source": s})
    for s, t in docs
    for c in rag.chunk_text(t, chunk_size=120, overlap=20)
]
heading_chunks = [
    (b, {"source": s, "heading": h})
    for s, t in docs
    for h, b in rag.chunk_markdown_sections(t)
]

q_vec = rag.embed([QUERY], input_type="query")[0]
print(f"Query: {QUERY!r}\n")

fixed_store = build_store(fixed_chunks)
score, rec = fixed_store.search(q_vec, k=1)[0]
print(f"  fixed-size    top hit  ({score:.3f})  [{rec.metadata['source']}]")
print(f"                {oneline(rec.text, 70)}")
print("                ^ right chunk, wrong first sentence, and no citable heading.\n")

heading_store = build_store(heading_chunks)
score, rec = heading_store.search(q_vec, k=1)[0]
cite = f"{rec.metadata['source']} > {rec.metadata['heading']}"
print(f"  heading-aware top hit  ({score:.3f})  [{cite}]")
print(f"                {oneline(rec.text, 70)}")
print("                ^ one clean topic, and a heading you can cite.\n")


# --- Part 3: the honest caveat, chunking isn't the whole story --------------
print("=" * 78)
print("3) BUT CHUNKING DOESN'T FIX EVERYTHING\n")

kw_index = rag.BM25Index([t for t, _ in heading_chunks])
kw_ranked = sorted(
    zip(kw_index.scores(QUERY), heading_chunks), key=lambda p: p[0], reverse=True
)
kw_top = kw_ranked[0][1][1]
print("Even with clean heading chunks, *keyword* search on the same query still")
print(f"ranks the wrong section first: [{kw_top['source']} > {kw_top['heading']}].")
print(
    "That's example 07's other lesson: the query ('get my notes out') and the\n"
    "answer ('export') share no rare words, so BM25 can't connect them, and no\n"
    "chunking strategy changes that. The fix for a vocabulary gap is to transform\n"
    "the query (example 10), not to re-cut the documents."
)

print("\n" + "=" * 78)
print(
    "Takeaway: match the cut to the document. A fixed-size window is fine for flat\n"
    "prose, but structured docs (Markdown, HTML, PDFs with headings) chunk far\n"
    "better on their own boundaries: one topic per chunk, a heading to cite, and\n"
    "no accidental topic-mixing. It's the cheapest retrieval win you can make at\n"
    "index time. Example 14 applies exactly this to messy real-world formats."
)
