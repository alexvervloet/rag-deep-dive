"""
rag: a small, from-scratch retrieval-augmented-generation library.

Everything here is built to be *read*, not just used. Five modules, each one
moving part of RAG:

  providers.py  the ONLY file that talks to an LLM provider (embed + generate)
  chunking.py   splitting documents into retrievable pieces
  store.py      an in-memory vector store (brute-force cosine search)
  keyword.py    keyword search (BM25), the lexical counterpart to the store
  pipeline.py   tying it together: index -> retrieve -> answer

plus two helpers: loader.py (read a folder of docs) and preview.py (a display
helper the examples use to print retrieved chunks readably, not RAG itself).

Import the pieces you need, e.g.:

    from rag import index_documents, answer

or reach into a specific module to see how it works.
"""

from .ann import IVFIndex, recall_at_k
from .chunking import chunk_markdown_sections, chunk_paragraphs, chunk_text
from .keyword import BM25Index, tokenize
from .loader import load_corpus
from .pipeline import GROUNDED_SYSTEM, answer, build_prompt, index_documents, retrieve
from .preview import snippet
from .providers import describe, embed, ensure_ready, generate, provider_name
from .store import VectorStore, cosine_similarity

__all__ = [
    "chunk_text",
    "chunk_paragraphs",
    "chunk_markdown_sections",
    "BM25Index",
    "tokenize",
    "load_corpus",
    "VectorStore",
    "cosine_similarity",
    "IVFIndex",
    "recall_at_k",
    "embed",
    "generate",
    "provider_name",
    "describe",
    "ensure_ready",
    "index_documents",
    "retrieve",
    "answer",
    "build_prompt",
    "GROUNDED_SYSTEM",
    "snippet",
]
