"""
rag/pgstore.py: the same vector store, in a real database.

[rag/store.py](store.py) keeps chunks in a Python list and caches them to a JSON
file. That is the right way to *learn* retrieval, and it quietly skips the part
of the job that actually takes the time in production: the **lifecycle** of an
index that outlives the process.

A cache file has one operation, "rebuild everything." A real store has five, and
this module implements each one against Postgres with the `pgvector` extension:

  1. **Schema.** The vector column has a fixed width, so your embedding model is
     part of your schema. Changing the model is a migration, not a config tweak.
  2. **Incremental sync.** Only documents whose content changed get re-embedded.
     Embedding is the step that costs money, so "what changed?" is the question
     the whole ingest path is built around.
  3. **Deletes.** A document removed from the corpus must lose its chunks, or
     retrieval keeps citing a page that no longer exists. Deleting the document
     row cascades to its chunks, so an orphaned vector is not possible.
  4. **Index maintenance.** The approximate index (§ "Approximate
     nearest-neighbour") is a real object here: build it after loading, and know
     that the planner will ignore it when a sequential scan is cheaper.
  5. **Provenance.** One row records which provider, model, dimensionality, and
     chunk settings built the index, so vectors from a different model are
     detected instead of silently returning nonsense.

The retrieval maths does not change at all. `search()` returns exactly what
`VectorStore.search()` returns, ranked by exactly the same cosine similarity, so
the pipeline in [rag/pipeline.py](pipeline.py) cannot tell them apart. That is
the point: the database is an operational upgrade, not a different idea.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .chunking import chunk_text
from .providers import embed as default_embed
from .providers import embed_model, provider_name
from .store import Record

# The throwaway container in compose.yaml. Local credentials for a disposable
# development database, which is why they can sit in the source.
DEFAULT_DSN = "postgresql://rag:rag_local_only@localhost:54331/rag"

ANN_INDEX_NAME = "rag_chunks_embedding_hnsw_idx"

# Tables and the extension that do not depend on the embedding width. The chunk
# table is created separately, because it cannot exist until we know how wide a
# vector this embedding model produces.
BASE_SCHEMA = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS rag_index (
        only_row boolean PRIMARY KEY DEFAULT true CHECK (only_row),
        provider text NOT NULL,
        embedding_model text NOT NULL,
        dimensions integer NOT NULL,
        chunk_size integer NOT NULL,
        overlap integer NOT NULL,
        built_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rag_documents (
        source text PRIMARY KEY,
        content_hash text NOT NULL,
        chunk_count integer NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now()
    )
    """,
)


def content_hash(text: str) -> str:
    """The fingerprint that answers "did this document change?".

    A hash of the bytes, nothing cleverer. Comparing hashes is free; re-embedding
    a document that did not change is not.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_literal(values: Sequence[float]) -> str:
    """pgvector's text input format: `[0.1,0.2,0.3]`."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


@dataclass
class IndexSettings:
    """What built the current index. Stored in `rag_index`, one row."""

    provider: str
    embedding_model: str
    dimensions: int
    chunk_size: int
    overlap: int

    def conflicts_with(self, other: "IndexSettings") -> str | None:
        """Return why `other` cannot reuse this index, or None if it can.

        Dimensionality is a hard schema conflict; the rest are semantic ones. A
        vector from a different model is not "slightly off", it is meaningless:
        the two models put their axes in different places, so cosine similarity
        between them measures nothing.
        """
        if self.embedding_model != other.embedding_model:
            return (
                f"embedding model changed: {self.embedding_model} -> "
                f"{other.embedding_model}"
            )
        if self.provider != other.provider:
            return f"provider changed: {self.provider} -> {other.provider}"
        if self.dimensions != other.dimensions:
            return f"dimensions changed: {self.dimensions} -> {other.dimensions}"
        if (self.chunk_size, self.overlap) != (other.chunk_size, other.overlap):
            return (
                f"chunk settings changed: {self.chunk_size}/{self.overlap} -> "
                f"{other.chunk_size}/{other.overlap}"
            )
        return None


@dataclass
class SyncReport:
    """What one sync actually did. Print it; it is the whole lesson."""

    added: list[str]
    updated: list[str]
    unchanged: list[str]
    deleted: list[str]
    embedded_chunks: int
    skipped_chunks: int
    rebuilt_reason: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.deleted)

    def summary(self) -> str:
        parts = [
            f"{len(self.added)} added",
            f"{len(self.updated)} updated",
            f"{len(self.unchanged)} unchanged",
            f"{len(self.deleted)} deleted",
        ]
        return (
            ", ".join(parts)
            + f"; embedded {self.embedded_chunks} chunks, "
            + f"skipped {self.skipped_chunks}"
        )


class PgVectorStore:
    """A vector store backed by Postgres + pgvector.

    Same two operations as `VectorStore` (put chunks in, get the nearest ones
    out), plus the lifecycle a durable store forces you to think about.
    """

    def __init__(self, connection: Any) -> None:
        self.conn = connection
        self._ensure_base_schema()

    # --- connection ---------------------------------------------------------

    @classmethod
    def connect(cls, dsn: str = DEFAULT_DSN) -> "PgVectorStore":
        """Open a connection and make sure the base schema exists.

        Import is local so the rest of the repo never needs psycopg installed:
        this path is optional, and `pip install -r requirements-postgres.txt`
        is what turns it on.
        """
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "The Postgres path needs psycopg. Install it with:\n"
                "  pip install -r requirements-postgres.txt"
            ) from exc

        try:
            connection = psycopg.connect(dsn)
        except psycopg.OperationalError as exc:
            raise RuntimeError(
                f"Could not connect to Postgres at {dsn}.\n"
                "Start the local service with:  docker compose up -d"
            ) from exc
        return cls(connection)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.count_chunks()

    # --- schema -------------------------------------------------------------

    def _ensure_base_schema(self) -> None:
        with self.conn.cursor() as cur:
            for statement in BASE_SCHEMA:
                cur.execute(statement)
        self.conn.commit()

    def _ensure_chunk_table(self, dimensions: int) -> None:
        """Create the chunk table for a specific embedding width.

        `vector(1536)` is a typed column like `varchar(20)`: the width is fixed
        at creation. This is the concrete reason an embedding-model change is a
        migration. Nothing here can quietly adapt to a 1024-dimensional vector
        once the column says 1536.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    source text NOT NULL
                        REFERENCES rag_documents (source) ON DELETE CASCADE,
                    ordinal integer NOT NULL,
                    text text NOT NULL,
                    embedding vector({int(dimensions)}) NOT NULL,
                    PRIMARY KEY (source, ordinal)
                )
                """
            )
        self.conn.commit()

    def _chunk_table_dimensions(self) -> int | None:
        """The width the chunk table was created with, or None if it is absent."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'rag_chunks' AND a.attname = 'embedding'
                """
            )
            row = cur.fetchone()
        return int(row[0]) if row else None

    def settings(self) -> IndexSettings | None:
        """What built the current index, or None if nothing has been indexed."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT provider, embedding_model, dimensions, chunk_size, overlap
                FROM rag_index WHERE only_row
                """
            )
            row = cur.fetchone()
        if row is None:
            return None
        return IndexSettings(row[0], row[1], int(row[2]), int(row[3]), int(row[4]))

    def _write_settings(self, settings: IndexSettings) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_index
                    (only_row, provider, embedding_model, dimensions,
                     chunk_size, overlap, built_at)
                VALUES (true, %s, %s, %s, %s, %s, now())
                ON CONFLICT (only_row) DO UPDATE SET
                    provider = EXCLUDED.provider,
                    embedding_model = EXCLUDED.embedding_model,
                    dimensions = EXCLUDED.dimensions,
                    chunk_size = EXCLUDED.chunk_size,
                    overlap = EXCLUDED.overlap,
                    built_at = now()
                """,
                (
                    settings.provider,
                    settings.embedding_model,
                    settings.dimensions,
                    settings.chunk_size,
                    settings.overlap,
                ),
            )

    def drop_all(self) -> None:
        """Remove every table this module owns. The reset button for the lesson."""
        with self.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS rag_chunks")
            cur.execute("DROP TABLE IF EXISTS rag_documents")
            cur.execute("DROP TABLE IF EXISTS rag_index")
        self.conn.commit()
        self._ensure_base_schema()

    # --- the lifecycle ------------------------------------------------------

    def sync(
        self,
        docs: Iterable[tuple[str, str]],
        chunk_size: int = 120,
        overlap: int = 20,
        embed_fn: Callable[..., list[list[float]]] | None = None,
        model_name: str | None = None,
    ) -> SyncReport:
        """Make the database match `docs`, doing the least work possible.

        This is the whole point of the module. For each document we compare a
        content hash against what is stored:

          * absent from the database  -> chunk, embed, insert   (added)
          * hash differs              -> chunk, embed, replace  (updated)
          * hash matches              -> do nothing at all      (unchanged)
          * absent from `docs`        -> delete, cascade        (deleted)

        Only the first two cost an embedding call. On a corpus where one page in
        forty changed, that is the difference between a cent and a dollar, and
        between a second and a minute, every single time you deploy.

        `embed_fn` and `model_name` exist so a lesson (or a test) can stand in a
        different embedding model without an account for one; production code
        passes neither and gets the active provider's.

        Everything that changes is written in **one transaction**. A crash
        halfway leaves the index as it was, not half-updated: a half-updated
        index is worse than a stale one, because it retrieves chunks of an old
        document beside chunks of the new one and cites both.
        """
        embed_fn = embed_fn or default_embed
        docs = list(docs)

        wanted = IndexSettings(
            provider=provider_name(),
            embedding_model=model_name or embed_model(),
            dimensions=-1,  # filled in once we have seen a vector
            chunk_size=chunk_size,
            overlap=overlap,
        )
        stored = self.settings()

        # A settings change we can see *before* embedding forces a full rebuild:
        # every stored vector was made by different rules.
        rebuilt_reason: str | None = None
        if stored is not None:
            comparable = IndexSettings(**{**wanted.__dict__, "dimensions": stored.dimensions})
            rebuilt_reason = stored.conflicts_with(comparable)

        existing = self._document_hashes()
        if rebuilt_reason:
            existing = {}

        added, updated, unchanged = [], [], []
        pending: list[tuple[str, int, str]] = []  # (source, ordinal, chunk text)
        skipped_chunks = 0

        for source, text in docs:
            digest = content_hash(text)
            if not rebuilt_reason and existing.get(source) == digest:
                unchanged.append(source)
                skipped_chunks += self._chunk_count(source)
                continue
            (added if source not in existing else updated).append(source)
            for ordinal, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                pending.append((source, ordinal, chunk))

        current_sources = {source for source, _ in docs}
        deleted = sorted(set(self._document_hashes()) - current_sources)

        vectors: list[list[float]] = []
        if pending:
            # One embedding call for everything that changed, not one per
            # document: batching is the difference between N round trips and 1.
            vectors = embed_fn([chunk for _, _, chunk in pending], input_type="document")
            wanted.dimensions = len(vectors[0])
        elif stored is not None:
            wanted.dimensions = stored.dimensions

        table_dimensions = self._chunk_table_dimensions()
        if pending and table_dimensions not in (None, wanted.dimensions):
            # A vector(1024) value cannot go in a vector(1536) column. The table
            # is rebuilt, which is exactly what a real migration would do.
            rebuilt_reason = rebuilt_reason or (
                f"dimensions changed: {table_dimensions} -> {wanted.dimensions}"
            )
            with self.conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS rag_chunks")
            self.conn.commit()
            table_dimensions = None
        if pending and table_dimensions is None:
            self._ensure_chunk_table(wanted.dimensions)

        try:
            with self.conn.cursor() as cur:
                if rebuilt_reason:
                    # `TRUNCATE ... CASCADE` clears the chunk table with it.
                    cur.execute("TRUNCATE rag_documents CASCADE")
                for source in deleted:
                    # One statement, and the chunks go too: the foreign key says
                    # ON DELETE CASCADE, so orphaned vectors are unrepresentable.
                    cur.execute("DELETE FROM rag_documents WHERE source = %s", (source,))
                by_source: dict[str, list[tuple[int, str, list[float]]]] = {}
                for (source, ordinal, chunk), vector in zip(pending, vectors):
                    by_source.setdefault(source, []).append((ordinal, chunk, vector))
                for source, text in docs:
                    rows = by_source.get(source)
                    if rows is None:
                        continue
                    cur.execute(
                        """
                        INSERT INTO rag_documents (source, content_hash, chunk_count)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (source) DO UPDATE SET
                            content_hash = EXCLUDED.content_hash,
                            chunk_count = EXCLUDED.chunk_count,
                            updated_at = now()
                        """,
                        (source, content_hash(text), len(rows)),
                    )
                    # Replace, never append: a shorter new version of a document
                    # would otherwise leave its extra old chunks behind.
                    cur.execute("DELETE FROM rag_chunks WHERE source = %s", (source,))
                    cur.executemany(
                        """
                        INSERT INTO rag_chunks (source, ordinal, text, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        """,
                        [
                            (source, ordinal, chunk, _vector_literal(vector))
                            for ordinal, chunk, vector in rows
                        ],
                    )
                if wanted.dimensions > 0:
                    self._write_settings(wanted)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        if rebuilt_reason:
            added, updated, unchanged = sorted(current_sources), [], []
            deleted = []

        return SyncReport(
            added=sorted(added),
            updated=sorted(updated),
            unchanged=sorted(unchanged),
            deleted=deleted,
            embedded_chunks=len(pending),
            skipped_chunks=0 if rebuilt_reason else skipped_chunks,
            rebuilt_reason=rebuilt_reason,
        )

    def delete_document(self, source: str) -> int:
        """Delete one document and its chunks. Returns the chunks removed."""
        removed = self._chunk_count(source)
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM rag_documents WHERE source = %s", (source,))
        self.conn.commit()
        return removed

    # --- retrieval ----------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        force_index: bool = False,
    ) -> list[tuple[float, Record]]:
        """Top-k by cosine similarity: the same contract as `VectorStore.search`.

        `<=>` is pgvector's cosine *distance* operator, so similarity is
        `1 - distance`. The database does the arithmetic `cosine_similarity()`
        does by hand in store.py, over data that never has to fit in memory.

        Ordering by the operator is also what makes an index usable: pgvector's
        HNSW index is defined over `<=>`, and a query that computed the distance
        any other way could not use it.

        `force_index=True` tells the planner to stop preferring a sequential scan,
        which is the only way to see what the approximate index actually returns
        on a small corpus. Use it to *measure* the index, never in production: a
        planner overruled by hand is a bug waiting for the data to grow.
        """
        if self._chunk_table_dimensions() is None:
            return []
        with self.conn.cursor() as cur:
            if force_index:
                cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                """
                SELECT text, source, ordinal, 1 - (embedding <=> %s::vector) AS score
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vector_literal(query_vector), _vector_literal(query_vector), k),
            )
            rows = cur.fetchall()
        self.conn.commit()
        return [
            (
                float(score),
                Record(text=text, vector=[], metadata={"source": source, "chunk": ordinal}),
            )
            for text, source, ordinal, score in rows
        ]

    def explain_search(
        self, query_vector: list[float], k: int = 5, force_index: bool = False
    ) -> str:
        """The query plan for a search. Shows whether the ANN index was used.

        Worth running once. On a corpus this small Postgres will choose a
        sequential scan even with an HNSW index present, because scanning a few
        hundred rows is cheaper than walking a graph, and it is right.
        """
        with self.conn.cursor() as cur:
            if force_index:
                cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                """
                EXPLAIN (COSTS OFF)
                SELECT text FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vector_literal(query_vector), k),
            )
            plan = "\n".join(row[0] for row in cur.fetchall())
        self.conn.commit()
        # The plan inlines the whole query vector, which is a thousand numbers
        # of noise around the one word that matters (Seq Scan or Index Scan).
        return re.sub(r"'\[[^\]]{40,}\]'::vector", "'[...]'::vector", plan)

    # --- the approximate index ---------------------------------------------

    def create_ann_index(self, m: int = 16, ef_construction: int = 64) -> None:
        """Build an HNSW index over the embedding column.

        Two things worth knowing before you copy this into production. Build the
        index **after** bulk-loading, not before: every insert into an existing
        index pays graph-maintenance cost. And the index is approximate by
        construction, so its recall belongs in your eval (§10), not in your
        assumptions. `m` and `ef_construction` buy recall with build time and
        memory; `hnsw.ef_search` buys it back at query time.
        """
        dimensions = self._chunk_table_dimensions()
        if dimensions is None:
            raise RuntimeError("Nothing indexed yet: run sync() first.")
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {ANN_INDEX_NAME}
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = {int(m)}, ef_construction = {int(ef_construction)})
                """
            )
            # A fresh index has no statistics until the planner is told to look.
            cur.execute("ANALYZE rag_chunks")
        self.conn.commit()

    def drop_ann_index(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {ANN_INDEX_NAME}")
        self.conn.commit()

    def has_ann_index(self) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_indexes WHERE indexname = %s", (ANN_INDEX_NAME,)
            )
            return cur.fetchone() is not None

    # --- introspection ------------------------------------------------------

    def count_chunks(self) -> int:
        if self._chunk_table_dimensions() is None:
            return 0
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM rag_chunks")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def documents(self) -> list[tuple[str, int]]:
        """`(source, chunk_count)` for everything indexed, alphabetically."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT source, chunk_count FROM rag_documents ORDER BY source"
            )
            return [(row[0], int(row[1])) for row in cur.fetchall()]

    def storage(self) -> dict[str, str]:
        """On-disk size of the chunk table and its ANN index, if any.

        Vectors are large: 1536 floats is 6 KB per chunk before any index. This
        is the number people are surprised by, and it is why the ANN index in
        §15 is not free either, it stores its own copy of the graph.
        """
        if self._chunk_table_dimensions() is None:
            return {}
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_size_pretty(pg_total_relation_size('rag_chunks'))")
            row = cur.fetchone()
            total = row[0] if row else "0 bytes"
            index = "none"
            if self.has_ann_index():
                cur.execute(
                    "SELECT pg_size_pretty(pg_relation_size(%s::regclass))",
                    (ANN_INDEX_NAME,),
                )
                index_row = cur.fetchone()
                index = index_row[0] if index_row else "0 bytes"
        return {"table_total": total, "ann_index": index}

    # --- internals ----------------------------------------------------------

    def _document_hashes(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT source, content_hash FROM rag_documents")
            return {row[0]: row[1] for row in cur.fetchall()}

    def _chunk_count(self, source: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_count FROM rag_documents WHERE source = %s", (source,)
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0
