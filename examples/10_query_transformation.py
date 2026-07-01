"""
Example 10 — query transformation: HyDE & multi-query.
======================================================

Retrieval embeds the user's *question* and looks for nearby chunks. But a question
and its answer often don't look alike. Ask "someone keeps signing into my stuff"
and the answer lives under a heading called "Two-factor authentication" — they
share no words, and sit far apart in embedding space, so direct retrieval buries
the right chunk beneath vaguely-related ones. When the raw query retrieves poorly,
you *transform* it first.

Two classic, complementary techniques — both just an extra LLM call before retrieval:

  - HyDE (Hypothetical Document Embeddings): ask the model to *draft a hypothetical
    answer*, then embed THAT instead of the question. A fake answer lives in
    "answer space," much closer to the real passage than the question is.

  - Multi-query: ask the model for several paraphrases of the question, retrieve for
    each, and union the results. More shots on goal — robust to one bad phrasing.

This script compares direct retrieval vs. HyDE vs. multi-query on a deliberately
oblique question, printing each chunk's `source > heading` so you can watch the
2FA section climb. We chunk on headings (example 13) so each chunk is one clean,
citable topic.

An honest note on what you'll see: direct retrieval scores the 2FA section low
(~0.38) and near the bottom of the top-k. The reliable effect of transforming the
query is a big jump in that *score* — HyDE roughly doubles it — which drags the
answer from "ignorable" into contention. Exact ranks wobble (the LLM calls are
non-deterministic, and on this tiny corpus password/recovery chunks are genuinely
related too), so sometimes the 2FA section climbs a rank, sometimes it just closes
the gap. On a real corpus of thousands of chunks, that score lift is the whole
game: it's the difference between the answer landing in your top-k or never being
seen.

Run it:

    python examples/10_query_transformation.py
    python examples/10_query_transformation.py "can I get my money back?"
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

# Heading-aware chunks (example 13): one topic per chunk, plus a heading to cite.
# That keeps each chunk's vector focused *and* makes the previews below readable.
texts, metas = [], []
for source, text in docs:
    for heading, body in rag.chunk_markdown_sections(text):
        texts.append(" ".join(body.split()))
        metas.append({"source": source, "heading": heading})
store = rag.VectorStore()
store.add(texts, rag.embed(texts, input_type="document"), metas)
print(f"Indexed {len(store)} heading-based chunks.\n")

QUERY = sys.argv[1] if len(sys.argv) > 1 else "someone keeps signing into my stuff"
K = 3


def cite(rec) -> str:
    """`source > heading` — the clean citation heading-aware chunking buys us."""
    return f"{rec.metadata['source']} > {rec.metadata.get('heading', '?')}"


def show(label: str, hits):
    print(f"[{label}]")
    for score, rec in hits:
        # Preview is centered on the original question; when the question shares no
        # words with the chunk (the whole point here) it falls back to the chunk's
        # start — which is fine now that each chunk is a single clean topic.
        preview = rag.snippet(rec.text, QUERY, width=60)
        print(f"  {score:.3f}  ({cite(rec)})  {preview}")
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
    print(f"[HyDE] hypothetical answer used for retrieval:\n  {' '.join(hypo.split())[:160]}...\n")
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
        "paraphrases (multi-query) — pulls the answer closer in embedding space. Look at\n"
        "the 'Two-factor authentication' score: a weak ~0.38 on the raw question, but\n"
        "far higher once transformed. On a large corpus that lift is what lands the\n"
        "right chunk in your top-k. The cost is one extra LLM call before retrieval."
    )
