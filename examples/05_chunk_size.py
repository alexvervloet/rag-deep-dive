"""
Example 05 — chunk size changes what you retrieve.
==================================================

Example 02 showed chunk size changing the *number* of chunks. This one shows it
changing *retrieval quality* — the thing you actually care about. We index the
same corpus twice, with small chunks and with large chunks, then run the same
query through both and compare what comes back.

The intuition to confirm by running it:

  - Small chunks are precise — a hit is a tight, on-topic snippet — but a single
    chunk may be too small to fully answer, and a relevant idea can be split
    across several.
  - Large chunks carry more context per hit, but each is less focused, so an
    irrelevant sentence can drag a chunk up the rankings (or a relevant one down).

The lesson isn't "small good, large bad" — it's that this is an empirical knob
you tune for *your* data by looking at results, which is exactly what example 09
turns into real numbers.

Run it:

    secrun python examples/05_chunk_size.py
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

query = "What happens to my notes if I delete them?"
print(f"Query: {query!r}\n")

for label, size, overlap in [("small", 40, 10), ("large", 250, 40)]:
    store = rag.index_documents(docs, chunk_size=size, overlap=overlap)
    hits = rag.retrieve(store, query, k=3)
    print(f"--- {label} chunks (size={size}, overlap={overlap}; {len(store)} total) ---")
    for score, rec in hits:
        preview = rag.snippet(rec.text, query, width=100)
        print(f"  {score:.3f}  [{rec.metadata['source']}]  {preview}")
    print()

print(
    "Compare the snippets: small chunks pinpoint the exact sentence; large chunks "
    "hand the model more surrounding context but less precisely. Neither is "
    "universally better — measure it (example 09)."
)
