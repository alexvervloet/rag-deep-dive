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
