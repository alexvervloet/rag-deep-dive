"""
Example 03 — a vector store from scratch.
=========================================

Example 02 showed that larger chunks "bundle several topics." This one shows why
that matters: embed the chunks, load them into a store, and ask a question —
you'll see which shape of chunk actually surfaces the right answer.

A "vector store" is just a list of (text, vector) pairs plus a way to find the
vectors closest to a query. Here we build one by hand to see every step:

  1. chunk the corpus,
  2. embed the chunks,
  3. load them into a VectorStore with their source as metadata,
  4. embed a query and ask the store for the top-k closest chunks.

This is the "retrieval" half of RAG, with no generation yet — just finding the
right text. Example 04 adds the model on top.

A chunk that spans two topics has a diluted vector: it points in an "average"
direction between them, so neither query lands a clean hit. The answer may
technically be present, but buried in a chunk whose averaged vector won't rank it
well. chunk_size=50 is small enough to keep each section in its own chunk, so the
right one floats to the top. Example 05 puts small and large chunks side by side
and measures the difference.

Run it:

    secrun python examples/03_vector_store.py
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

# Build the store by hand (example 04 will use the index_documents() shortcut).
store = rag.VectorStore()
texts, metas = [], []
for source, text in docs:
    for i, chunk in enumerate(rag.chunk_text(text, chunk_size=50, overlap=10)):
        texts.append(chunk)
        metas.append({"source": source, "chunk": i})

vectors = rag.embed(texts, input_type="document")
store.add(texts, vectors, metas)
print(f"Indexed {len(store)} chunks from {len(docs)} documents.\n")

query = "How long can I recover a note after deleting it?"
query_vector = rag.embed([query], input_type="query")[0]
hits = store.search(query_vector, k=3)

print(f"Query: {query!r}\n")
print("Top 3 chunks by similarity:")
for score, rec in hits:
    preview = rag.snippet(rec.text, query, width=90)
    print(f"  {score:.3f}  [{rec.metadata['source']}]  {preview}")

print(
    "\nThe right chunk floats to the top — focused text, precise vector. "
    "That ranked list IS retrieval; everything else is deciding what to do with it."
)
