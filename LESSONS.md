# Lessons

## A model change is detected by its name, not by its vectors

- **Expected:** the Postgres store would notice a switched embedding model when the
  new vectors arrived, since a narrower vector cannot go into a fixed-width
  `vector(n)` column.
- **Actual:** it noticed nothing. The corpus had not changed, so every content hash
  matched, so `sync()` had no documents to embed and never saw a vector at all. The
  index stayed as it was, and a query embedded by the new model would have been
  ranked against vectors from the old one, scoring plausible nonsense with no error
  anywhere.
- **Next time:** compare the recorded model id *before* deciding what work to do,
  not the vectors afterwards. Dimensionality is a backstop that only fires when the
  widths differ, and two models sharing a width is common. Anything that persists
  embeddings has to store what produced them.

## `EXPLAIN` on a vector query prints the whole query vector

- **Expected:** a query plan that fits on a few lines, showing `Seq Scan` or
  `Index Scan`.
- **Actual:** roughly 1,024 floats inlined into the `Sort Key` line, burying the one
  word the reader needs in several screens of numbers.
- **Next time:** collapse long `'[...]'::vector` literals before printing a plan.
  Worth doing anywhere a lesson shows real database output: the point of printing a
  plan is that someone reads it.

## Skipping a document that chunked to nothing left its old chunks behind

- **Expected:** "replace a changed document's chunks" covered every kind of
  change, including a page emptied upstream.
- **Actual:** the write loop skipped any document with no new chunks, so an
  emptied document kept all of its old ones, kept its old hash, and reported
  "1 updated" on every future sync while doing nothing. The module docstring and
  the README table both promised the opposite, which is worse than saying
  nothing: the claim was in front of the bug.
- **Next time:** treat zero as a value, not as absence. The loop should skip
  documents that did not change, and a document that changed to nothing is not
  one of them.

## DDL committed before the transaction it belonged to

- **Expected:** `DROP TABLE` and `CREATE TABLE` had to be committed separately,
  the way a migration usually is, before the rows could be written.
- **Actual:** Postgres does DDL transactionally, so the separate commit bought
  nothing and cost the guarantee the docstring made. A crash after it left an
  empty chunk table beside a full document table whose hashes all said
  "unchanged": a permanently empty index that every later sync agreed was fine.
- **Next time:** in Postgres, put the schema change in the same transaction as
  the data it enables. And when a docstring claims atomicity, test it by crashing
  on purpose, which is a three-line subclass and the only way to know.

## A comparison the caller made impossible

- **Expected:** the settings check compared dimensionality, so a narrower vector
  under the same model id would be caught.
- **Actual:** the caller built the comparison object with the stored width copied
  in, so the two were equal by construction and the branch could never run. The
  case is real: OpenAI's `dimensions=` parameter narrows the output while the
  model id stays `text-embedding-3-small`. Worse, the check that did eventually
  fire (the column width) fired *after* embedding only the changed documents, and
  the rebuild path then wrote only those, dropping the rest of the corpus while
  reporting success.
- **Next time:** a check that cannot fail is a check you do not have. Write the
  test that makes it fail first. And when a late discovery changes the scope of
  the work, widen the work, not just the log line.

## `argparse` defaults are evaluated before `load_dotenv()`

- **Expected:** `default=os.getenv("RAG_DATABASE_URL")` would pick up the value
  documented in `.env.example`.
- **Actual:** `parse_args()` runs before `load_dotenv()`, so the default was read
  from an environment that had not been loaded yet. The flag silently fell back
  to the compose service and the run looked like a success against the wrong
  database.
- **Next time:** resolve environment-backed defaults inside the function that
  runs after the environment is loaded, and keep `argparse` defaults to literals.
