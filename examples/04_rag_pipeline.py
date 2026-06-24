"""
Example 04 — the RAG pipeline, end to end.
==========================================

Now we put retrieval and generation together — the whole point of the repo.

  index_documents(docs)  -> chunk + embed the corpus into a VectorStore
  answer(store, question) -> retrieve the closest chunks, paste them into a
                             grounded prompt, generate, and return the answer
                             *with* the sources it used

The one idea, restated: a model can only answer from what's in its context
window. RAG retrieves the right chunks and puts them there — and the grounding
instruction (see GROUNDED_SYSTEM in rag/pipeline.py) tells the model to answer
ONLY from them and to cite which it used. Ask about something *not* in the corpus
and a well-grounded system should say it doesn't know rather than invent.

Run it:

    python examples/04_rag_pipeline.py
    python examples/04_rag_pipeline.py "Can students get a discount?"
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

# Index once, then answer. (A real app saves the index and reloads it — see the
# ask_docs.py capstone. Here we rebuild each run to keep it simple.)
store = rag.index_documents(docs, chunk_size=120, overlap=20)
print(f"Indexed {len(store)} chunks from {len(docs)} documents.\n")

question = sys.argv[1] if len(sys.argv) > 1 else "How do I turn on two-factor authentication?"
print(f"Question: {question}\n")

text, hits = rag.answer(store, question, k=4)

print("=" * 70)
print(text)
print("=" * 70)

# The sources behind the answer — this is what makes a RAG answer *checkable*.
print("\nSources the answer drew on:")
for n, (score, rec) in enumerate(hits, start=1):
    print(f"  [{n}] {rec.metadata['source']}  (similarity {score:.3f})")

print(
    "\nTry a question the corpus can't answer (e.g. \"What's the CEO's name?\") "
    "and watch a grounded system decline instead of guessing."
)
