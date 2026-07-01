"""
rag/keyword.py — keyword search from scratch (BM25).
====================================================

Vector search (store.py) matches on *meaning*. This module is its opposite
number: **lexical** search, which matches on the actual *words*. That sounds
primitive next to embeddings, but it's exactly what you want for the things
embeddings are worst at — product names, error codes, IDs, rare jargon — tokens
that carry little semantic signal but must match *exactly*.

The algorithm is **BM25**, the workhorse behind decades of search engines. It
scores a chunk for a query by asking three questions about each query word:

  1. Does the chunk contain it, and how often?      (term frequency)
  2. How rare is the word across the whole corpus?   (inverse document frequency)
  3. Is the chunk short or long?                      (length normalization)

The intuition each one encodes:

  - **More occurrences → higher score**, but with *diminishing returns* — the 5th
    mention of a word doesn't matter as much as the 1st (that's the `k1` knob).
  - **Rare words matter more.** A chunk sharing the word "the" with your query
    tells you nothing; a chunk sharing "NN-413" tells you almost everything. IDF
    makes common words nearly worthless and rare ones decisive.
  - **Shorter chunks win ties.** A word match in a focused 20-word chunk is a
    stronger signal than the same match buried in a 300-word one (the `b` knob).

That's the whole difference from naïvely counting shared words — and it's why a
real keyword scorer beats term-counting. No model, no embeddings, no API call:
BM25 is pure arithmetic over word counts, so it's free and instant.
"""

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens, keeping hyphenated codes like "nn-413" whole.

    A real engine would also stem ("running" -> "run") and drop stop words; we
    keep it minimal so the BM25 scoring stays the star, not the tokenizer.
    """
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class BM25Index:
    """An in-memory BM25 keyword index over a list of documents (chunks).

    Build it once from the chunk texts; then `scores(query)` returns a BM25 score
    for every chunk (aligned to the input order, so you can zip it back to your
    records) and `search(query, k)` returns the top-k. `k1` and `b` are the
    standard BM25 knobs — the defaults are the usual, well-behaved starting point.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(t) for t in texts]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.docs) if self.docs else 0.0
        self.tf = [Counter(d) for d in self.docs]  # word -> count, per chunk

        # Document frequency: in how many chunks does each word appear at least once.
        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))

        # Inverse document frequency. Rare words get a big weight, common words a
        # tiny one. The `+ 1` inside the log keeps IDF non-negative even for a word
        # that appears in more than half the chunks (the standard BM25+ form).
        n_docs = len(self.docs)
        self.idf = {
            word: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for word, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score for every chunk, in the order the chunks were added."""
        q_terms = tokenize(query)
        results: list[float] = []
        for i, counts in enumerate(self.tf):
            # Length normalization: dividing by the average means a match in a
            # short chunk outweighs the same match in a long, padded one.
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl) if self.avgdl else self.k1
            score = 0.0
            for term in q_terms:
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                # Term frequency with diminishing returns (saturates at k1 + 1),
                # weighted by how rare the term is across the corpus.
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / (freq + norm)
            results.append(score)
        return results

    def search(self, query: str, k: int = 5) -> list[tuple[float, int]]:
        """Return the top-k `(score, index)` pairs, highest score first."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda p: p[1], reverse=True)
        return [(score, i) for i, score in ranked[:k]]
