"""
rag/pipeline.py: the RAG loop: index -> retrieve -> answer.

This module wires the other three together into the actual flow:

  index_documents(docs)  chunk every document, embed the chunks, load a store
  retrieve(store, q)     embed the question, return the closest chunks
  answer(store, q)       retrieve, paste the chunks into a grounded prompt,
                         generate, and hand back the answer *and* its sources

The whole idea of RAG in one sentence: a model can only answer from what's in its
context window, so we put the most relevant chunks there, and we instruct the
model to answer ONLY from them and to cite which chunk it used. That grounding +
citation discipline is what separates RAG from "the model guessing."
"""

from .chunking import chunk_text
from .providers import embed, generate
from .store import Record, VectorStore

# The grounding instruction. This is doing a lot of work: it forbids outside
# knowledge (so wrong-but-confident answers become "I don't know") and asks for
# [n] citations that point back at the numbered context blocks.
GROUNDED_SYSTEM = (
    "You answer questions using ONLY the numbered context provided in the user "
    "message. Cite the sources you use with bracketed numbers like [1] or [2]. "
    "If the context does not contain the answer, say you don't know. Do not "
    "guess or rely on outside knowledge."
)


def index_documents(
    docs: list[tuple[str, str]],
    chunk_size: int = 120,
    overlap: int = 20,
) -> VectorStore:
    """Build a VectorStore from `(source_name, text)` documents.

    Chunk every document, embed all the chunks in as few calls as the provider
    allows, and load them into a store tagged with their source. This is the
    "ingest" step: the expensive one, run once.
    """
    texts: list[str] = []
    metadatas: list[dict] = []
    for source, text in docs:
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    vectors = embed(texts, input_type="document")
    store = VectorStore()
    store.add(texts, vectors, metadatas)
    return store


def retrieve(store: VectorStore, query: str, k: int = 4) -> list[tuple[float, Record]]:
    """Embed the query and return the top-k (score, record) chunks."""
    query_vector = embed([query], input_type="query")[0]
    return store.search(query_vector, k)


def build_prompt(query: str, hits: list[tuple[float, Record]]) -> str:
    """Assemble the user message: numbered context blocks, then the question.

    Numbering each block is what makes citations possible: the model can refer
    to "[2]" and you (or the reader) can map it back to a real source.
    """
    blocks = []
    for n, (_score, rec) in enumerate(hits, start=1):
        source = rec.metadata.get("source", "?")
        blocks.append(f"[{n}] (source: {source})\n{rec.text}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def answer(
    store: VectorStore,
    query: str,
    k: int = 4,
    max_tokens: int = 512,
) -> tuple[str, list[tuple[float, Record]]]:
    """The full RAG answer: retrieve, ground, generate.

    Returns `(answer_text, hits)` so the caller can show the answer alongside the
    exact sources it was built from, the basis of trustworthy, checkable output.
    """
    hits = retrieve(store, query, k)
    prompt = build_prompt(query, hits)
    text = generate(GROUNDED_SYSTEM, prompt, max_tokens=max_tokens)
    return text, hits
