"""
rag/store.py: a tiny vector store, built from scratch.

A "vector store" sounds fancy; at heart it's just:

  1. a list of (text, vector) pairs, and
  2. a way to find the vectors closest in meaning to a query vector.

That's all this class is. We store each chunk with its embedding and some
metadata (which document it came from), and `search()` does a brute-force scan:
compute cosine similarity against every stored vector and return the top-k.

Brute force is O(n) per query. For a few hundred or few thousand chunks that's
instant, and it keeps the logic completely transparent. When you have millions
of vectors you switch to an approximate-nearest-neighbour index (FAISS, hnswlib)
or a hosted vector database (pgvector, Pinecone, Weaviate): same idea, cleverer
data structure. That swap is the main thing a "production RAG" course adds on top
of this file; the *concept* doesn't change.
"""

import json
import math
from dataclasses import dataclass, field


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two vectors: 1.0 = same direction (meaning),
    0 = unrelated. The same by-hand formula from the sibling repos' embeddings
    examples, no math libraries needed."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class Record:
    """One stored chunk: its text, its embedding, and where it came from."""

    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """An in-memory store of embedded chunks with top-k cosine search."""

    def __init__(self) -> None:
        self.records: list[Record] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add a batch of chunks. `texts[i]` must line up with `vectors[i]`."""
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must be the same length")
        metadatas = metadatas or [{} for _ in texts]
        for text, vector, meta in zip(texts, vectors, metadatas):
            self.records.append(Record(text=text, vector=vector, metadata=meta))

    def search(self, query_vector: list[float], k: int = 5) -> list[tuple[float, Record]]:
        """Return the top-k (score, record) pairs, highest cosine first."""
        scored = [
            (cosine_similarity(query_vector, rec.vector), rec) for rec in self.records
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]

    # --- Persistence. Embedding is the one step that costs money and time, so a
    #     real app embeds once and saves the result, then loads it on startup.
    #     JSON keeps the saved index human-readable so you can peek inside. ---

    def save(self, path: str) -> None:
        data = [
            {"text": r.text, "vector": r.vector, "metadata": r.metadata}
            for r in self.records
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store.records = [
            Record(text=d["text"], vector=d["vector"], metadata=d.get("metadata", {}))
            for d in data
        ]
        return store
