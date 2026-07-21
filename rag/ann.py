"""
rag/ann.py: an approximate nearest-neighbour index, from scratch.

`store.py` does brute force: score the query against *every* vector, sort, return
the top-k. That's exact and instant for thousands of chunks, but O(n) per query,
so at millions of vectors it's too slow. Production systems switch to an
**approximate** index (FAISS, hnswlib, pgvector's IVFFlat/HNSW): trade a little
recall for a large speedup by *not* looking at every vector.

This file builds the simplest such index, an **IVF** ("inverted file"), the same
idea behind pgvector's IVFFlat, by hand, so the tradeoff is visible rather than
imported:

  1. Pick `n_clusters` centroids and assign every vector to its nearest one. Now
     the vectors are bucketed into clusters (done once, at build time).
  2. To search, score the query against the *centroids* only, then brute-force
     within the `n_probe` closest clusters, skipping the rest of the data.

Probe every cluster (`n_probe == n_clusters`) and you get exact results by a
slower route. Probe fewer and you scan a fraction of the data: much faster, but
you may miss a neighbour that sat just over a cluster border. That miss rate is
**recall**, and `examples/15_approximate_index.py` measures it against the exact
brute-force answer. The lesson is the dial, not this toy index: real ANN libraries
are far cleverer, but they all trade recall for speed the same way.

Dependency-free and deterministic (seeded), so the recall/speed numbers reproduce.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .store import Record, cosine_similarity


@dataclass
class IVFIndex:
    """A from-scratch IVF (inverted-file) approximate index over stored vectors."""

    n_clusters: int = 32
    seed: int = 0
    centroids: list[list[float]] = field(default_factory=list)
    # postings[c] = indices of records assigned to cluster c
    postings: list[list[int]] = field(default_factory=list)
    records: list[Record] = field(default_factory=list)

    def build(self, records: list[Record]) -> "IVFIndex":
        """Assign every record to a cluster. One-time cost, amortized over queries."""
        self.records = list(records)
        rng = random.Random(self.seed)
        n = len(self.records)
        k = min(self.n_clusters, n)
        # Seed centroids from random data points (a cheap stand-in for k-means; the
        # recall/speed tradeoff is the lesson, not the clustering algorithm).
        seeds = rng.sample(range(n), k)
        self.centroids = [list(self.records[i].vector) for i in seeds]
        self.postings = [[] for _ in range(k)]
        for idx, rec in enumerate(self.records):
            c = max(range(k), key=lambda ci: cosine_similarity(rec.vector, self.centroids[ci]))
            self.postings[c].append(idx)
        return self

    def search(self, query_vector: list[float], k: int = 5, n_probe: int = 4) -> list[tuple[float, Record]]:
        """Score the query against centroids, then brute-force the `n_probe` closest
        clusters only. Lower `n_probe` = fewer vectors scanned = faster, lower recall."""
        ranked_clusters = sorted(
            range(len(self.centroids)),
            key=lambda ci: cosine_similarity(query_vector, self.centroids[ci]),
            reverse=True,
        )
        candidates: list[int] = []
        for ci in ranked_clusters[: max(1, n_probe)]:
            candidates.extend(self.postings[ci])
        scored = [(cosine_similarity(query_vector, self.records[i].vector), self.records[i]) for i in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]

    def last_scanned(self, query_vector: list[float], n_probe: int = 4) -> int:
        """How many vectors a search with this `n_probe` would scan (for the demo)."""
        ranked = sorted(
            range(len(self.centroids)),
            key=lambda ci: cosine_similarity(query_vector, self.centroids[ci]),
            reverse=True,
        )
        return sum(len(self.postings[ci]) for ci in ranked[: max(1, n_probe)])


def recall_at_k(approx: list[tuple[float, Record]], exact: list[tuple[float, Record]], k: int) -> float:
    """Fraction of the exact top-k that the approximate search also returned.

    We match on vector identity (same list object) since the demo's synthetic
    vectors have no text. 1.0 = the approximate index missed nothing."""
    exact_ids = {id(rec.vector) for _, rec in exact[:k]}
    approx_ids = {id(rec.vector) for _, rec in approx[:k]}
    if not exact_ids:
        return 1.0
    return len(exact_ids & approx_ids) / len(exact_ids)
