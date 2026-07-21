"""
Example 07: hybrid retrieval, and why fusion is not a free lunch.

You've built both halves of retrieval separately: vector search by *meaning*
(example 03) and keyword search by *words* (BM25, example 06). Each has a blind
spot. Vectors fumble exact strings like error codes; keyword search is deaf to
paraphrase. **Hybrid retrieval** runs both and combines the results, hoping to
get each one's strength.

The folk wisdom is "hybrid is strictly better." It isn't. Combining two rankers
only helps when they *disagree in useful ways*; when one is confidently wrong, a
naive combination can drag a good result *down*. This example makes that visible
by scoring the same corpus five ways on two deliberately opposite queries:

  1. a paraphrase query ("how do I get my notes out of the app?"), where the answer
     talks about "export," a word the query never uses. Vectors nail it; BM25 is
     not just weak but actively misled, ranking a keyword-dense intro chunk above
     the answer. Watch a 50/50 blend make things *worse* than vector-only.
  2. an exact-term query ("what does error NN-413 mean?"), a rare token both
     halves agree on, so every fusion method safely ranks it first.

The five methods per query: vector-only, keyword-only, then three ways to fuse:
a 50/50 score blend, Reciprocal Rank Fusion (RRF), and a vector-weighted blend.
The keyword half reuses the BM25 index from example 06 (see
[rag/keyword.py](../rag/keyword.py)).

Run it:

    secrun python examples/07_hybrid_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag
from dotenv import load_dotenv

load_dotenv()
rag.ensure_ready()
print(f"Provider: {rag.describe()}\n")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = rag.load_corpus(os.path.join(REPO_ROOT, "corpus"))
store = rag.index_documents(docs, chunk_size=120, overlap=20)
print(f"Indexed {len(store)} chunks.\n")
print(
    "Reading the output: 'rel' is each score rescaled so the best match in that\n"
    "list = 1.000, a relative rank rather than an absolute similarity. The parentheses\n"
    "show where the score came from (cos = cosine, bm25 = keyword, v/k = the\n"
    "normalized halves, v#/k# = each half's rank for RRF).\n"
)

# The keyword half: a BM25 index over the same chunks, in the same order, so its
# scores line up one-to-one with the store's records (example 06 built this).
records = store.records
keyword_index = rag.BM25Index([r.text for r in records])


def normalize(scores: list[float]) -> list[float]:
    """Scale scores to 0–1 so vector and keyword scores are comparable."""
    hi = max(scores) if scores else 0.0
    return [s / hi for s in scores] if hi else [0.0 for _ in scores]


def ranks(scores: list[float]) -> list[int]:
    """Rank position (1 = best) of each item under `scores`."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = [0] * len(scores)
    for position, i in enumerate(order):
        out[i] = position + 1
    return out


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion: blend *ranks*, not raw scores.

    Each method contributes 1 / (k + rank) for every item, and we sum those. It
    only cares about ordering, so it's immune to the scale mismatch that forces
    us to normalize score blends, but it still trusts every method equally, so a
    result one half buries deep (a big rank) is hard to rescue. k=60 is the
    common default; it softens the gap between the very top ranks.
    """
    return [
        sum(1.0 / (k + rl[i]) for rl in rank_lists) for i in range(len(rank_lists[0]))
    ]


def show_top(
    label: str, scores: list[float], records, annot: list[str], query: str, k: int = 3
) -> None:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    print(f"  {label}")
    for i in order:
        # rag.snippet centers the preview on the query match, so the printed line
        # shows *why* the chunk ranked, not just its first words. (See rag/preview.py.)
        preview = rag.snippet(records[i].text, query)
        # `scores[i]` is the rescaled rank-score (best = 1.000); `annot[i]` shows
        # what it came from, so the reader can see 1.000 != perfect match.
        print(
            f"    rel {scores[i]:.3f}  ({annot[i]})  "
            f"[{records[i].metadata['source']}]  {preview}"
        )


for query in [
    "how do I get my notes out of the app?",
    "what does error NN-413 mean?",
]:
    q_vec = rag.embed([query], input_type="query")[0]

    # Keep the raw scores around: the normalized "rel" number always makes the top
    # match 1.000, so we print the raw score beside it to show what it really was.
    vec_raw = [rag.cosine_similarity(q_vec, r.vector) for r in records]
    kw_raw = keyword_index.scores(query)
    vec = normalize(vec_raw)
    kw = normalize(kw_raw)

    # Three ways to fuse the two halves:
    blend_50 = [0.5 * v + 0.5 * k for v, k in zip(vec, kw)]  # equal-weight score blend
    blend_90 = [0.9 * v + 0.1 * k for v, k in zip(vec, kw)]  # vector-weighted blend
    fused = rrf([ranks(vec), ranks(kw)])  # rank-based fusion

    print(f"Query: {query!r}")
    show_top("vector-only    :", vec, records, [f"cos {x:.2f}" for x in vec_raw], query)
    show_top("keyword-only   :", kw, records, [f"bm25 {x:.2f}" for x in kw_raw], query)
    show_top(
        "blend 50/50    :",
        blend_50,
        records,
        [f"v {v:.2f} k {k:.2f}" for v, k in zip(vec, kw)],
        query,
    )
    show_top(
        "RRF            :",
        fused,
        records,
        [f"v#{vr} k#{kr}" for vr, kr in zip(ranks(vec), ranks(kw))],
        query,
    )
    show_top(
        "blend 90/10    :",
        blend_90,
        records,
        [f"v {v:.2f} k {k:.2f}" for v, k in zip(vec, kw)],
        query,
    )
    print()

print(
    "The paraphrase query is the cautionary tale. Vector search ranks the right "
    "chunk #1; BM25 ranks it #11, because the query's only keywords ('notes', "
    "'app') describe the product, not 'export', so BM25 confidently prefers a "
    "keyword-dense intro chunk. The 50/50 blend inherits that mistake and drops "
    "the answer to ~#8, *worse than vector-only*. RRF is more robust (it fuses "
    "ranks, not noisy scores) but still can't undo an #11, landing around #5. "
    "Only a heavily vector-weighted blend (90/10) recovers #1, at which point "
    "you've nearly switched keyword off."
)
print(
    "\nThe exact-code query is the easy case: both halves rank the NN-413 chunk "
    "#1, so every fusion method agrees. That's the real rule of thumb: hybrid "
    "shines when the two halves *agree*, and needs weighting or query-routing "
    "when they don't. It is a tool to tune, not a free lunch."
)
print(
    "\n(Aside: the answer to *both* queries lives in the same chunk. Fixed-size "
    "chunking merged the 'Importing' error text and the 'Exporting' section into "
    "one. Previews are centered on the match, so you can see the relevant "
    "sentence even when it starts mid-chunk. Chunk boundaries matter too.)"
)
