"""Integration tests for the Postgres/pgvector store.

These need the local database, so they are skipped unless you point them at one:

    docker compose up -d
    RAG_TEST_DATABASE_URL=postgresql://rag:rag_local_only@localhost:54331/rag \\
      python -m unittest tests.test_pgstore -v

They make no API calls. The embedder is a deterministic stand-in, because what
is under test is the *lifecycle*, and the lifecycle does not care what the
numbers in a vector are.

Every test here exists because the behaviour it asserts was wrong once. They are
mostly assertions that something is *absent* after a sync: no stale chunk, no
half-written index, no document quietly dropped. Retrieval bugs of this kind do
not raise; they return a plausible answer citing a page that no longer exists,
so the only way to catch them is to look for what should not be there.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pgstore import PgVectorStore

DSN = os.getenv("RAG_TEST_DATABASE_URL")

DOCS = [
    ("a.md", "alpha " * 300),
    ("b.md", "beta " * 300),
    ("c.md", "gamma " * 300),
]


def embedder(dimensions: int):
    """A deterministic stand-in for a real embedding model."""

    def embed(texts, input_type="document"):
        vectors = []
        for text in texts:
            rng = random.Random(hash(text) % 10**6)
            vectors.append([rng.gauss(0, 1) for _ in range(dimensions)])
        return vectors

    return embed


@unittest.skipUnless(DSN, "set RAG_TEST_DATABASE_URL to run the Postgres tests")
class PgVectorStoreTests(unittest.TestCase):
    def setUp(self):
        assert DSN is not None
        self.store = PgVectorStore.connect(DSN)
        self.store.drop_all()
        self.store.sync(DOCS, embed_fn=embedder(8), model_name="m8")

    def tearDown(self):
        self.store.close()

    # --- incremental sync --------------------------------------------------

    def test_an_unchanged_corpus_embeds_nothing(self):
        report = self.store.sync(DOCS, embed_fn=embedder(8), model_name="m8")
        self.assertEqual(report.embedded_chunks, 0)
        self.assertEqual(len(report.unchanged), 3)

    def test_only_the_edited_document_is_re_embedded(self):
        edited = [(n, t + " extra") if n == "b.md" else (n, t) for n, t in DOCS]
        report = self.store.sync(edited, embed_fn=embedder(8), model_name="m8")
        self.assertEqual(report.updated, ["b.md"])
        self.assertEqual(len(report.unchanged), 2)

    def test_a_deleted_document_takes_its_chunks_with_it(self):
        remaining = [(n, t) for n, t in DOCS if n != "b.md"]
        self.store.sync(remaining, embed_fn=embedder(8), model_name="m8")
        self.assertEqual(self.store._chunk_count("b.md"), 0)
        self.assertNotIn("b.md", dict(self.store.documents()))
        self.assertEqual(len(self.store), 6)

    # --- the awkward cases -------------------------------------------------

    def test_an_emptied_document_loses_its_chunks(self):
        """Emptying a page is deleting it, as far as retrieval is concerned."""
        emptied = [(n, "   " if n == "b.md" else t) for n, t in DOCS]
        self.store.sync(emptied, embed_fn=embedder(8), model_name="m8")
        self.assertEqual(dict(self.store.documents())["b.md"], 0)
        self.assertEqual(len(self.store), 6)

    def test_an_emptied_document_stays_emptied(self):
        """The stored hash has to advance, or the edit is re-reported forever."""
        emptied = [(n, "   " if n == "b.md" else t) for n, t in DOCS]
        self.store.sync(emptied, embed_fn=embedder(8), model_name="m8")
        report = self.store.sync(emptied, embed_fn=embedder(8), model_name="m8")
        self.assertEqual(report.updated, [])
        self.assertEqual(len(report.unchanged), 3)

    def test_a_new_empty_document_is_recorded(self):
        with_empty = DOCS + [("d.md", "")]
        report = self.store.sync(with_empty, embed_fn=embedder(8), model_name="m8")
        self.assertEqual(report.added, ["d.md"])
        self.assertEqual(dict(self.store.documents())["d.md"], 0)

    # --- model changes -----------------------------------------------------

    def test_a_model_change_reindexes_every_document(self):
        """Not just the ones that happened to change in the same sync."""
        edited = [(n, t + " extra") if n == "b.md" else (n, t) for n, t in DOCS]
        report = self.store.sync(edited, embed_fn=embedder(4), model_name="m4")
        self.assertIsNotNone(report.rebuilt_reason)
        self.assertEqual(sorted(dict(self.store.documents())), ["a.md", "b.md", "c.md"])
        self.assertEqual(self.store._chunk_table_dimensions(), 4)

    def test_a_width_change_under_the_same_model_id_is_caught(self):
        """OpenAI's `dimensions=` narrows the output and keeps the model name."""
        edited = [(n, t + " extra") if n == "b.md" else (n, t) for n, t in DOCS]
        report = self.store.sync(edited, embed_fn=embedder(4), model_name="m8")
        self.assertIn("dimensions changed", report.rebuilt_reason or "")
        self.assertEqual(sorted(dict(self.store.documents())), ["a.md", "b.md", "c.md"])

    def test_a_rebuild_does_not_report_documents_as_added(self):
        report = self.store.sync(DOCS, embed_fn=embedder(4), model_name="m4")
        self.assertEqual(report.added, [])
        self.assertEqual(len(report.reindexed), 3)
        self.assertIn("reindexed", report.summary())

    def test_a_query_from_the_wrong_model_is_refused(self):
        """The backstop for a swap no sync could see: same id, same corpus."""
        with self.assertRaises(ValueError) as caught:
            self.store.search(embedder(512)(["q"])[0], k=2)
        self.assertIn("dimensions", str(caught.exception))

    # --- transactionality ---------------------------------------------------

    def test_a_crash_mid_sync_leaves_the_index_untouched(self):
        """Including the table rebuild, which is DDL and still rolls back."""

        class Crashing(PgVectorStore):
            def _write_settings(self, cur, settings):
                raise RuntimeError("simulated crash")

        assert DSN is not None
        crashing = Crashing.connect(DSN)
        with self.assertRaises(RuntimeError):
            crashing.sync(DOCS, embed_fn=embedder(4), model_name="m4")
        crashing.close()

        settings = self.store.settings()
        assert settings is not None
        self.assertEqual(settings.embedding_model, "m8")
        self.assertEqual(self.store._chunk_table_dimensions(), 8)
        self.assertEqual(len(self.store), 9)

    def test_a_short_embedding_batch_is_an_error_not_a_truncation(self):
        def short(texts, input_type="document"):
            return embedder(8)(texts)[:-1]

        edited = [(n, t + " extra") for n, t in DOCS]
        with self.assertRaises(ValueError):
            self.store.sync(edited, embed_fn=short, model_name="m8")
        self.assertEqual(len(self.store), 9)

    # --- retrieval ----------------------------------------------------------

    def test_search_scores_match_the_from_scratch_cosine(self):
        from rag.store import cosine_similarity

        query = embedder(8)(["a query"])[0]
        hits = self.store.search(query, k=3, include_vectors=True)
        for score, record in hits:
            self.assertAlmostEqual(score, cosine_similarity(query, record.vector), places=6)

    def test_vectors_are_omitted_unless_asked_for(self):
        query = embedder(8)(["a query"])[0]
        self.assertEqual(self.store.search(query, k=1)[0][1].vector, [])


if __name__ == "__main__":
    unittest.main()
