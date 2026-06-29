"""
Example 09 — query transformation: HyDE & multi-query.
======================================================

Retrieval embeds the user's *question* and looks for nearby chunks. But a question
and its answer often don't look alike: "can I get my money back?" shares few words
(and little embedding-space proximity) with a doc that says "refunds are issued
within 30 days." When the raw query retrieves poorly, you *transform* it first.

Two classic, complementary techniques — both just an extra LLM call before retrieval:

  - HyDE (Hypothetical Document Embeddings): ask the model to *draft a hypothetical
    answer*, then embed THAT instead of the question. A fake answer lives in
    "answer space," much closer to the real passage than the question is.

  - Multi-query: ask the model for several paraphrases of the question, retrieve for
    each, and union the results. More shots on goal — robust to one bad phrasing.

This script compares direct retrieval vs. HyDE vs. multi-query on a deliberately
oblique question, and prints which chunks each one surfaces.

Run it:

    python examples/09_query_transformation.py
    python examples/09_query_transformation.py "can I get my money back?"
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
store = rag.index_documents(docs, chunk_size=120, overlap=20)
print(f"Indexed {len(store)} chunks.\n")

QUERY = sys.argv[1] if len(sys.argv) > 1 else "can I get my money back?"
K = 3


def show(label: str, hits):
    print(f"[{label}]")
    for score, rec in hits:
        snippet = " ".join(rec.text.split()[:14])
        print(f"  {score:.3f}  ({rec.metadata.get('source','?')}) {snippet}...")
    print()


def direct():
    return rag.retrieve(store, QUERY, k=K)


def hyde():
    # Draft a hypothetical answer, then retrieve with ITS embedding.
    hypo = rag.generate(
        "You draft a short, plausible answer to help a search system. Be specific.",
        f"Write a 2-sentence hypothetical answer to: {QUERY}",
        max_tokens=120,
    )
    hypo_vec = rag.embed([hypo], input_type="document")[0]
    print(f"[HyDE] hypothetical answer used for retrieval:\n  {hypo.strip()[:160]}...\n")
    return store.search(hypo_vec, k=K)


def multi_query():
    # Generate paraphrases, retrieve for each, union by best score per chunk.
    variants_raw = rag.generate(
        "You rewrite a search query into diverse alternative phrasings.",
        f"Give 3 alternative phrasings of this question, one per line, no numbering:\n{QUERY}",
        max_tokens=120,
    )
    variants = [QUERY] + [v.strip("-• ").strip() for v in variants_raw.splitlines() if v.strip()]
    print("[multi-query] phrasings:")
    for v in variants:
        print(f"  - {v}")
    print()
    best: dict[int, tuple[float, object]] = {}
    for v in variants:
        for score, rec in rag.retrieve(store, v, k=K):
            key = id(rec)
            if key not in best or score > best[key][0]:
                best[key] = (score, rec)
    return sorted(best.values(), key=lambda p: p[0], reverse=True)[:K]


if __name__ == "__main__":
    print(f"Question: {QUERY}\n" + "=" * 70 + "\n")
    show("direct (embed the question as-is)", direct())
    show("HyDE (embed a hypothetical answer)", hyde())
    show("multi-query (union of paraphrases)", multi_query())
    print("=" * 70)
    print(
        "Takeaway: when a question and its answer don't 'look alike,' transforming the\n"
        "query first — drafting a hypothetical answer (HyDE) or fanning out into\n"
        "paraphrases (multi-query) — pulls the right chunk up the ranking. The cost is\n"
        "one extra LLM call before retrieval."
    )
