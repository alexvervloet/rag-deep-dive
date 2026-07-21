"""
Example 01: embeddings & cosine, the foundation (recap).

RAG is built on one capability: turning text into vectors that capture *meaning*,
so you can measure how related two pieces of text are. If you worked through the
sibling repos (openai-api-deep-dive / claude-api-deep-dive), this is review, so skim
it and move on. If not, this is the one idea everything else depends on.

  - An *embedding* turns a string into a list of numbers (a vector).
  - Texts with similar meaning get vectors pointing in similar directions.
  - *Cosine similarity* measures that: 1.0 = same direction, 0 = unrelated.

The punchline that makes retrieval work: two texts can score high even when they
share **no words**, because the match is on meaning, not spelling.

Run it:

    secrun python examples/01_embeddings_recap.py
"""

import os
import sys

# Make the repo root importable so `import rag` works from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag
from dotenv import load_dotenv

load_dotenv()
rag.ensure_ready()
print(f"Provider: {rag.describe()}\n")

query = "How do I get my notes out of the app?"
candidates = [
    "Any notebook can be exported to Markdown, PDF, or HTML.",  # relevant, no shared words
    "Support operates Monday to Friday, 9 to 5.",  # unrelated
    "Deleted notes stay in Trash for 30 days.",  # near-miss, edged out the export result by ~0.001
]

# Embed the documents and the query. (We pass input_type so the Voyage stack can
# optimize each side; the OpenAI stack ignores it. See rag/providers.py.)
doc_vectors = rag.embed(candidates, input_type="document")
query_vector = rag.embed([query], input_type="query")[0]

print(f"Query: {query!r}\n")
print("Ranked by cosine similarity:")
ranked = sorted(
    zip(candidates, doc_vectors),
    key=lambda pair: rag.cosine_similarity(query_vector, pair[1]),
    reverse=True,
)
for text, vec in ranked:
    print(f"  {rag.cosine_similarity(query_vector, vec):.3f}  {text}")

print(
    "\nNotice the second result shares no words with the query yet nearly ties "
    "for first: 'export' is close to 'get my notes out'. That semantic match is what "
    "makes retrieval possible."
)
