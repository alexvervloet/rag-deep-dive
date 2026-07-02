"""
Example 15 — approximate search: trading a little recall for a lot of speed.
============================================================================

Our vector store (§4) does brute force: score the query against EVERY vector,
sort, return the top-k. Exact, and instant for the few thousand chunks in a normal
corpus. But it's O(n) per query — at millions of vectors it's too slow, and that's
when production RAG switches to an **approximate nearest-neighbour (ANN)** index:
FAISS, hnswlib, or pgvector's IVFFlat/HNSW.

ANN buys speed by *not looking at every vector*. To see the tradeoff instead of
importing it, we build the simplest such index by hand — an **IVF** (the idea
behind pgvector's IVFFlat), in `rag/ann.py`: bucket the vectors into clusters once,
then at query time only scan the few clusters nearest the query. Scan fewer
clusters (`n_probe`) and you touch a fraction of the data — faster, but you may
miss a neighbour sitting just over a cluster border. That miss rate is **recall**.

This one is fully OFFLINE and needs no key: we generate a pile of synthetic
*clustered* vectors (the effect only shows at a scale — thousands of vectors — the
tiny demo corpus can't reach), then measure the approximate index against the exact
brute-force answer as we turn the `n_probe` dial.

Run it:

    python examples/15_approximate_index.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag
from rag.store import Record

# --- A synthetic dataset large enough for the tradeoff to show. ------------
# Real embeddings aren't uniform noise — related documents cluster together in
# vector space. We mimic that: scatter a set of "topic" centers, then place each
# vector as a center plus a little noise. That structure is exactly what an IVF
# index exploits, so the recall-for-speed tradeoff looks like it does in practice.
N_VECTORS = 3000
DIM = 64
N_TOPICS = 60        # ground-truth blobs the data is drawn from
N_CLUSTERS = 60      # IVF buckets (independent of the true topics)
N_QUERIES = 40
K = 10

rng = random.Random(42)


def random_vector() -> list[float]:
    return [rng.gauss(0, 1) for _ in range(DIM)]


# Topic centers, spread far apart so blobs are distinguishable.
_CENTERS = [[x * 6.0 for x in random_vector()] for _ in range(N_TOPICS)]


def near_center(center: list[float], noise: float = 1.0) -> list[float]:
    return [c + rng.gauss(0, noise) for c in center]


def main() -> None:
    print(f"Building a synthetic index: {N_VECTORS} clustered vectors, dim {DIM}, "
          f"{N_CLUSTERS} IVF buckets. (Offline — no embeddings, no key.)\n")

    records = [
        Record(text=f"vec-{i}", vector=near_center(_CENTERS[i % N_TOPICS]), metadata={})
        for i in range(N_VECTORS)
    ]
    exact = rag.VectorStore()
    exact.records = records  # reuse the brute-force store as ground truth

    index = rag.IVFIndex(n_clusters=N_CLUSTERS, seed=0).build(records)
    # Queries land near a random topic, like a real question near its answer's cluster.
    queries = [near_center(_CENTERS[rng.randrange(N_TOPICS)]) for _ in range(N_QUERIES)]

    print(f"Averaged over {N_QUERIES} queries, recall@{K} vs the exact top-{K}:\n")
    print(f"  {'n_probe':>8}  {'clusters scanned':>16}  {'vectors scanned':>16}  {'recall@'+str(K):>10}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*16}  {'-'*10}")

    for n_probe in (1, 2, 4, 8, 16, N_CLUSTERS):
        recalls, scanned = [], []
        for q in queries:
            approx_hits = index.search(q, k=K, n_probe=n_probe)
            exact_hits = exact.search(q, k=K)
            recalls.append(rag.recall_at_k(approx_hits, exact_hits, K))
            scanned.append(index.last_scanned(q, n_probe=n_probe))
        avg_recall = sum(recalls) / len(recalls)
        avg_scanned = sum(scanned) / len(scanned)
        pct = 100 * avg_scanned / N_VECTORS
        label = f"{n_probe}" + (" (all)" if n_probe == N_CLUSTERS else "")
        print(f"  {label:>8}  {n_probe:>16}  {avg_scanned:>10.0f} ({pct:>3.0f}%)  {avg_recall:>10.2f}")

    print(
        "\nRead the dial: n_probe=1 scans a sliver of the data (~2%) but misses roughly a\n"
        "quarter of the true neighbours; crank it to all clusters and you're back to\n"
        "exact results — having done MORE work than plain brute force, because you also\n"
        "scored the centroids. The sweet spot is in between: a few probes (here ~7% of\n"
        "the vectors) recover almost all the recall. That is the whole ANN\n"
        "bargain. Real indexes (HNSW, IVFFlat) are far cleverer than this toy, but they\n"
        "all sell the same thing — a recall-for-speed knob you tune against an eval\n"
        "(§10), never on faith. Reach for one only when brute force is genuinely too\n"
        "slow; for a few thousand chunks, `store.py` is the right answer."
    )


if __name__ == "__main__":
    main()
