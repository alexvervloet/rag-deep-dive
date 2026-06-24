"""
Example 02 — chunking (offline, no API call).
=============================================

Before you can retrieve, you have to decide what the "pieces" are. You don't
embed a whole document as one vector — you split it into **chunks** and embed
each one, so retrieval can pull back the specific paragraph that answers a
question instead of a whole file.

This example is completely offline — chunking is just string slicing — so it
costs nothing and needs no key. Run it and watch how the two knobs (chunk size
and overlap) change the number and shape of the chunks.

Run it:

    python examples/02_chunking.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO_ROOT, "corpus", "security-and-privacy.md")

text = open(DOC, encoding="utf-8").read()
print(f"Document: {os.path.basename(DOC)} ({len(text.split())} words)\n")

# Same document, three different fixed-size settings. Smaller chunks = more, more
# focused pieces (but each holds less context); larger chunks = fewer, broader
# pieces (but each may bundle several topics).
print("Fixed-size chunking (size / overlap in words):")
for size, overlap in [(40, 5), (120, 20), (300, 40)]:
    chunks = rag.chunk_text(text, chunk_size=size, overlap=overlap)
    print(f"  size={size:>3} overlap={overlap:>2}  ->  {len(chunks)} chunks")

# Now the structure-aware alternative: one chunk per paragraph.
paras = rag.chunk_paragraphs(text)
print(f"\nParagraph chunking          ->  {len(paras)} chunks")

# Peek at a chunk so the abstraction feels concrete.
sample = rag.chunk_text(text, chunk_size=40, overlap=5)[1]
print(f"\nExample chunk (size=40):\n  {sample!r}")

print(
    "\nThere's no universally correct setting — it depends on your documents and "
    "questions. Example 05 has you *measure* the effect on retrieval instead of "
    "guessing."
)
