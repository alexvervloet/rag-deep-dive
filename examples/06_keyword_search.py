"""
Example 06 — keyword search from scratch (BM25).
================================================

Example 03 built the *semantic* half of retrieval: embed the chunks, and a query
finds them by **meaning**, even when it shares no words with the answer. This
example builds the other half — **keyword** (lexical) search — which finds chunks
by the **words** they literally contain. It's the mirror image, and it's exactly
what embeddings are worst at: exact strings like product names, error codes, and
IDs, which carry little semantic signal but must match precisely.

We use **BM25**, the classic search-engine ranking function. It's smarter than
just counting shared words (see [rag/keyword.py](../rag/keyword.py)): it weighs
each query word by how *rare* it is (a shared "the" means nothing; a shared
"NN-413" means everything) and normalizes for chunk length. And because it's pure
word-counting arithmetic — no model, no embeddings — it's **completely offline**:
no key, no cost.

We run two deliberately different queries to see where keyword search shines and
where it falls down:

  1. an exact-term query ("what does error NN-413 mean?") — keyword search nails
     the rare code, and
  2. a paraphrase query ("how do I get my notes out of the app?") — where the
     answer talks about "export" and "Markdown", not the user's words, so keyword
     search whiffs.

That second failure is the whole motivation for example 07, which blends this
with vector search into *hybrid* retrieval.

Run it:

    python examples/06_keyword_search.py        # offline — no key, no cost
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = rag.load_corpus(os.path.join(REPO_ROOT, "corpus"))

# Chunk the corpus WITHOUT embedding it — keyword search needs no vectors, so
# there's no API call here at all. Each entry keeps its source for display.
chunks: list[tuple[str, str]] = []
for source, text in docs:
    for piece in rag.chunk_text(text, chunk_size=120, overlap=20):
        chunks.append((piece, source))

index = rag.BM25Index([text for text, _ in chunks])
print(f"Indexed {len(chunks)} chunks with BM25 (no embeddings, no API calls).\n")


def show_top(query: str, k: int = 3) -> None:
    ranked = sorted(zip(index.scores(query), chunks), key=lambda p: p[0], reverse=True)
    print(f"Query: {query!r}")
    for score, (text, source) in ranked[:k]:
        preview = rag.snippet(text, query)
        print(f"  {score:.3f}  [{source}]  {preview}")
    print()


show_top("what does error NN-413 mean?")
show_top("how do I get my notes out of the app?")

# Why BM25 beats naïve word-counting: it weighs each word by how *rare* it is
# (IDF). A shared common word barely moves the score; a shared rare one decides it.
# The exact code is rare, so one match dominates — compare it to a common word:
rare, common = "nn-413", "the"
print(f"IDF({rare!r}) = {index.idf.get(rare, 0.0):.2f}   vs   IDF({common!r}) = {index.idf.get(common, 0.0):.2f}")
print("(that gap is the whole edge BM25 has over just counting shared words)\n")

# The same rarity spread across the entire vocabulary, low to high:
ranked_idf = sorted(index.idf.items(), key=lambda kv: kv[1])
print("Lowest-IDF words (common):  " + ", ".join(repr(w) for w, _ in ranked_idf[:5]))
print("Highest-IDF words (rare):   " + ", ".join(repr(w) for w, _ in ranked_idf[-5:]))

print(
    "\nKeyword search is decisive on the exact code but whiffs on the paraphrase — "
    "the answer says 'export' and 'Markdown', words the query never uses. Vector "
    "search (example 03) is the exact opposite. Example 07 blends the two so you "
    "get both strengths at once — that's hybrid retrieval."
)
