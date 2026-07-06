"""
Example 08 — reranking: over-retrieve, then reorder.
====================================================

Fast retrieval (cosine over embeddings) is cheap but approximate — the truly best
chunk isn't always #1. A common fix is a two-stage pipeline:

  1. RETRIEVE a generous handful (say top 8) with the cheap vector search.
  2. RERANK those few with a slower, smarter scorer and keep the best 3.

Reranking only looks at a handful of candidates, so you can afford something more
careful. Production systems use a dedicated **cross-encoder reranker** (e.g.
Voyage's or Cohere's rerank endpoints) that scores each (query, chunk) pair
jointly. Here we demonstrate the *pattern* with a tool you already have: ask the
LLM itself to pick the most relevant passages (a "listwise" LLM reranker). Same
shape, no new dependency.

Run it:

    secrun python examples/08_reranking.py
    secrun python examples/08_reranking.py "I lost my phone with the authenticator app"
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import rag

load_dotenv()
rag.ensure_ready()
print(f"Provider: {rag.describe()}\n")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = rag.load_corpus(os.path.join(REPO_ROOT, "corpus"))
store = rag.index_documents(docs, chunk_size=120, overlap=20)

RERANK_SYSTEM = (
    "You are a search reranker. Given a question and numbered passages, identify "
    "which passages actually answer the question. Reply with ONLY the numbers of "
    "the most relevant passages, most relevant first, comma-separated. No prose."
)


def llm_rerank(query: str, hits, top_n: int = 3):
    """Reorder `hits` by asking the model which passages truly answer the query."""
    listing = "\n".join(
        f"[{i}] {' '.join(rec.text.split())[:200]}"
        for i, (_score, rec) in enumerate(hits, start=1)
    )
    reply = rag.generate(RERANK_SYSTEM, f"Question: {query}\n\nPassages:\n{listing}")

    # Parse the numbers the model returned, keep valid + unique ones in order.
    order: list[int] = []
    for n in (int(m) for m in re.findall(r"\d+", reply)):
        if 1 <= n <= len(hits) and n not in order:
            order.append(n)
    # Anything the model didn't mention falls to the back, original order kept.
    order += [i for i in range(1, len(hits) + 1) if i not in order]
    return [hits[i - 1] for i in order][:top_n]


query = sys.argv[1] if len(sys.argv) > 1 else "How do I get back in if I lost my 2FA device?"
print(f"Query: {query!r}\n")

# Stage 1: over-retrieve.
candidates = rag.retrieve(store, query, k=8)

print("Stage 1 — vector retrieval (top 8):")
for n, (score, rec) in enumerate(candidates, start=1):
    preview = rag.snippet(rec.text, query, width=70)
    print(f"  {n}. {score:.3f}  [{rec.metadata['source']}]  {preview}")

# Stage 2: rerank down to the best 3.
reranked = llm_rerank(query, candidates, top_n=3)

print("\nStage 2 — after LLM reranking (top 3):")
for n, (score, rec) in enumerate(reranked, start=1):
    preview = rag.snippet(rec.text, query, width=70)
    print(f"  {n}. [{rec.metadata['source']}]  {preview}")

print(
    "\nReranking promotes the passage that truly answers the question even when it "
    "wasn't the closest vector match. Feed the reranked top-k to the model and the "
    "answer improves — for the cost of one extra call over a few candidates.\n"
    "(The order is the model's judgment, so the lower ranks may shuffle run to run; "
    "what's stable is the strongest answer being pulled to the top.)"
)
