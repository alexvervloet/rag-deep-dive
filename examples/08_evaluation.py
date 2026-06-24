"""
Example 08 — evaluation: is your RAG actually working?
======================================================

Every knob so far — chunk size, hybrid weighting, reranking — is a guess until you
*measure*. RAG has two things to evaluate, and they fail independently:

  - RETRIEVAL: did the right chunk come back at all? If not, the model never had a
    chance. Two standard metrics:
      * hit rate @ k — fraction of questions whose correct source is in the top k.
      * MRR (mean reciprocal rank) — 1/rank of the first correct hit, averaged.
        Rewards ranking the right chunk *higher*, not just including it.
  - ANSWER correctness: given good context, did the model produce the right
    answer? Here we use a simple check — does the expected fact appear in the
    answer? (Production setups often use an "LLM judge" for fuzzier grading.)

We score against a tiny labelled set: a question, which document should answer it,
and a fact the answer must contain. A real eval set is bigger and harder, but the
mechanics are identical — and even five questions catch regressions when you
change a knob.

Run it:

    python examples/08_evaluation.py
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

# The labelled eval set: (question, document that should answer it, fact the
# answer must contain). Keep expected facts short and unambiguous.
EVALSET = [
    ("How long are deleted notes kept?", "security-and-privacy.md", "30 days"),
    ("How much does the Plus plan cost per month?", "plans-and-billing.md", "$4"),
    ("Where is my data stored?", "security-and-privacy.md", "Frankfurt"),
    ("Can I get a refund on an annual subscription?", "plans-and-billing.md", "14 days"),
    ("Which app can I import my notes from?", "getting-started.md", "Evernote"),
]

K = 4
hit_count = 0
rr_total = 0.0
answer_correct = 0

print(f"Evaluating {len(EVALSET)} questions at k={K}...\n")
print(f"{'retr':>4} {'rank':>4} {'ans':>3}  question")
print("-" * 60)

for question, expected_source, expected_fact in EVALSET:
    hits = rag.retrieve(store, question, k=K)
    sources = [rec.metadata["source"] for _score, rec in hits]

    # Retrieval metrics.
    if expected_source in sources:
        rank = sources.index(expected_source) + 1  # 1-based
        hit_count += 1
        rr_total += 1.0 / rank
    else:
        rank = 0  # not found in top-k

    # Answer correctness: generate, then check the expected fact is present.
    text, _ = rag.answer(store, question, k=K)
    correct = expected_fact.lower() in text.lower()
    answer_correct += int(correct)

    retr = "yes" if expected_source in sources else "NO"
    rank_str = str(rank) if rank else "-"
    ans = "ok" if correct else "X"
    print(f"{retr:>4} {rank_str:>4} {ans:>3}  {question}")

n = len(EVALSET)
print("-" * 60)
print(f"\nRetrieval hit rate @ {K}: {hit_count}/{n} = {hit_count / n:.0%}")
print(f"Mean reciprocal rank:    {rr_total / n:.3f}  (1.0 = right chunk always #1)")
print(f"Answer correctness:      {answer_correct}/{n} = {answer_correct / n:.0%}")

print(
    "\nThese three numbers are your dashboard. Change a knob (chunk size, k, hybrid "
    "weighting, reranking) and re-run — if a metric drops, you just caught a "
    "regression you'd otherwise have shipped."
)
