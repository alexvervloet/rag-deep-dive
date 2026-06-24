"""
Example 06 — hybrid retrieval (keyword + semantic).
===================================================

Pure vector search is great at *meaning* ("get my notes out" ≈ "export") but it
can fumble exact strings — product names, error codes, IDs — because those carry
little semantic signal. Old-fashioned **keyword** search is the opposite: it nails
exact terms but is blind to paraphrase.

**Hybrid retrieval** runs both and combines the scores, so you get each one's
strength. This example scores every chunk three ways — vector-only, keyword-only,
and hybrid — on two deliberately different queries:

  1. a paraphrase query ("how do I get my notes out of the app?") where keyword
     search struggles but vectors shine, and
  2. an exact-term query ("what does error NN-413 mean?") where vectors are weak
     but keyword search is decisive.

Watch how hybrid does well on both.

Run it:

    python examples/06_hybrid_retrieval.py
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
print(f"Indexed {len(store)} chunks.\n")


def tokenize(s: str) -> set[str]:
    """Lowercase word/number/hyphen tokens — enough for a teaching keyword score."""
    return set(re.findall(r"[a-z0-9\-]+", s.lower()))


def keyword_scores(query: str, records) -> list[float]:
    """Score each chunk by how many distinct query terms it contains."""
    q_terms = tokenize(query)
    return [float(len(q_terms & tokenize(rec.text))) for rec in records]


def normalize(scores: list[float]) -> list[float]:
    """Scale scores to 0–1 so vector and keyword scores are comparable."""
    hi = max(scores) if scores else 0.0
    return [s / hi for s in scores] if hi else [0.0 for _ in scores]


def show_top(label: str, scores: list[float], records, k: int = 3) -> None:
    ranked = sorted(zip(scores, records), key=lambda p: p[0], reverse=True)[:k]
    print(f"  {label}")
    for score, rec in ranked:
        preview = " ".join(rec.text.split())[:75]
        print(f"    {score:.3f}  [{rec.metadata['source']}]  {preview}...")


for query in [
    "how do I get my notes out of the app?",
    "what does error NN-413 mean?",
]:
    records = store.records
    q_vec = rag.embed([query], input_type="query")[0]

    vec = normalize([rag.cosine_similarity(q_vec, r.vector) for r in records])
    kw = normalize(keyword_scores(query, records))
    # Hybrid: a simple weighted blend. 0.5 weights both equally; tune per workload.
    hybrid = [0.5 * v + 0.5 * k for v, k in zip(vec, kw)]

    print(f"Query: {query!r}")
    show_top("vector-only :", vec, records)
    show_top("keyword-only:", kw, records)
    show_top("hybrid      :", hybrid, records)
    print()

print(
    "Keyword search alone can't match the paraphrase; vector search alone is shaky "
    "on the exact code 'NN-413'. Hybrid ranks the right chunk on top in both — "
    "which is why most production retrieval is hybrid."
)
