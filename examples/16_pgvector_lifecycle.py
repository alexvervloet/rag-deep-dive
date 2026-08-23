"""
Example 16: the index lifecycle, in a real database (Postgres + pgvector).

Every other example in this repo builds an index and throws it away, or caches
it to `.rag_index.json` and rebuilds the whole thing when anything changes. That
is fine for learning retrieval and hopeless as an operating model: a cache file
has exactly one verb, "rebuild everything", and rebuilding everything means
re-embedding everything, which is the one step that costs money.

This example runs the same pipeline against a real vector database and walks the
lifecycle that a durable index forces you to handle:

  1. an empty database becomes a schema and a first index          (create)
  2. the same corpus syncs again and embeds NOTHING                (idempotent)
  3. one edited document re-embeds one document                    (incremental)
  4. a removed document takes its chunks with it                   (delete)
  5. an HNSW index gets built, and the planner ignores it          (honest ANN)
  6. a changed embedding model is caught, not silently obeyed      (migration)

The corpus is never edited on disk; steps 3 and 4 use in-memory variants of it,
so you can run this repeatedly and get the same story.

Setup (two commands and a few cents of embeddings):

    docker compose up -d
    pip install -r requirements-postgres.txt

Run it:

    secrun python examples/16_pgvector_lifecycle.py

Then stop the database when you're done:  docker compose down
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import rag
from rag.pgstore import DEFAULT_DSN, PgVectorStore

load_dotenv()
rag.ensure_ready()

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")
DSN = os.getenv("RAG_DATABASE_URL", DEFAULT_DSN)
QUESTION = "How do I turn on two-factor authentication?"


def show(report, label: str) -> None:
    print(f"  {label:<34} {report.summary()}")
    if report.rebuilt_reason:
        print(f"  {'':<34} rebuilt: {report.rebuilt_reason}")


def top_hit(store: PgVectorStore, question: str) -> str:
    hits = rag.retrieve(store, question, k=1)
    if not hits:
        return "(nothing indexed)"
    score, record = hits[0]
    return f"{record.metadata['source']} #{record.metadata['chunk']}  (score {score:.3f})"


def main() -> None:
    print(f"Provider: {rag.describe()}")
    print(f"Database: {DSN}\n")

    docs = rag.load_corpus(CORPUS_DIR)
    print(f"Corpus: {len(docs)} documents\n")

    store = PgVectorStore.connect(DSN)
    # Start from nothing so the lesson is repeatable rather than cumulative.
    store.drop_all()

    # --- 1. Create ---------------------------------------------------------
    print("1. First sync: an empty database")
    show(store.sync(docs), "everything is new")
    print(f"  {'':<34} {len(store)} chunks across {len(store.documents())} documents")
    print(f"  {'':<34} top hit: {top_hit(store, QUESTION)}\n")

    # --- 2. Idempotence ----------------------------------------------------
    print("2. Sync the same corpus again")
    show(store.sync(docs), "nothing changed")
    print(
        "   Not one embedding call. The store compares a content hash per document,\n"
        "   so a deploy that changes no documents costs nothing. This is the single\n"
        "   biggest practical difference from a cache file.\n"
    )

    # --- 3. Incremental update --------------------------------------------
    print("3. Edit one document")
    edited = [
        (name, text + "\n\nNew in this release: passkeys are now supported.")
        if name == "security-and-privacy.md"
        else (name, text)
        for name, text in docs
    ]
    show(store.sync(edited), "one document differs")
    print(
        "   Only the edited document was re-chunked and re-embedded; the other three\n"
        "   were skipped on a hash comparison. Its old chunks were deleted rather than\n"
        "   appended to, so a document that gets SHORTER cannot leave stale chunks\n"
        "   behind, still retrievable, still citable.\n"
    )

    # --- 4. Delete ---------------------------------------------------------
    print("4. Remove a document from the corpus")
    without = [(name, text) for name, text in edited if name != "plans-and-billing.md"]
    before = len(store)
    show(store.sync(without), "one document is gone")
    print(f"  {'':<34} chunks: {before} -> {len(store)}")
    print(
        "   The chunk table has ON DELETE CASCADE, so deleting the document row takes\n"
        "   its vectors with it. This is the failure a cache file makes easy to ship:\n"
        "   delete a page from your docs site, forget to reindex, and the assistant\n"
        "   keeps answering from it, with a citation that 404s.\n"
    )

    # --- 5. The ANN index --------------------------------------------------
    print("5. Build the approximate index")
    store.create_ann_index()
    query_vector = rag.embed([QUESTION], input_type="query")[0]
    print(f"  {'':<34} index built: {store.has_ann_index()}")
    for key, value in store.storage().items():
        print(f"  {'':<34} {key}: {value}")

    print("   Query plan as the planner chooses it:")
    for line in store.explain_search(query_vector, k=4).splitlines():
        print(f"     {line}")
    print("   Query plan with the sequential scan disabled:")
    for line in store.explain_search(query_vector, k=4, force_index=True).splitlines():
        print(f"     {line}")

    exact = store.search(query_vector, k=4)
    approximate = store.search(query_vector, k=4, force_index=True)
    same = [r.metadata for _, r in exact] == [r.metadata for _, r in approximate]
    print(f"  {'':<34} exact and approximate top-4 agree: {same}")
    print(
        "   Read those two plans honestly. On a corpus this small Postgres ignores the\n"
        "   index and scans, and it is right to: scanning a few hundred rows beats\n"
        "   walking a graph. Disabling the scan is the only way to see the index run,\n"
        "   and here it returns the same four chunks, because HNSW recall is near\n"
        "   perfect when the graph holds every vector you own. That agreement is a\n"
        "   property to MEASURE at your scale (§10, §15), not to assume: the index is\n"
        "   approximate by construction, it stores its own copy of every vector, and it\n"
        "   slows every insert. Build it when brute force is measurably too slow.\n"
    )

    # --- 6. The migration --------------------------------------------------
    print("6. Change the embedding model")

    def different_model(texts, input_type="document"):
        """Stand-in for a different embedding model: same call, narrower vectors."""
        return [vector[:256] for vector in rag.embed(texts, input_type=input_type)]

    settings = store.settings()
    assert settings is not None
    print(f"  {'':<34} indexed with: {settings.embedding_model} ({settings.dimensions}d)")
    show(
        store.sync(without, embed_fn=different_model, model_name="demo-embed-256"),
        "a different model",
    )
    after = store.settings()
    assert after is not None
    print(f"  {'':<34} now: {len(store)} chunks, {after.dimensions} dimensions")
    print(
        "   Nothing in the corpus changed, and everything was re-embedded anyway. The\n"
        "   store compares the model id it recorded against the one now in use, which\n"
        "   is the reason that id is stored beside the vectors. A fixed-width vector\n"
        "   column cannot hold a narrower vector at all, and even when two models\n"
        "   happen to share a width, similarity between their vectors measures nothing:\n"
        "   they put their axes in different places. Changing the embedding model is a\n"
        "   migration with a re-embedding bill attached, not a config edit.\n"
    )

    print("Done. Stop the database with:  docker compose down")
    store.close()


if __name__ == "__main__":
    main()
