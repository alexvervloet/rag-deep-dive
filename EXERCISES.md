# Exercises — make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts: a thing to predict, a thing to change, and a question to
answer from memory.

How to use it: work the section first, then come back. **Commit to an answer
before you run or reveal** — the prediction is where the learning happens, even
(especially) when you're wrong. Answers are hidden behind ▸ toggles.

> The chunking exercises are **(offline)** — no API call, no cost. The rest make
> small, cheap calls.

---

## Section 2 — Embeddings (recap)

**Recall.** Two sentences share no words. Can they still score a high cosine
similarity? Why does that matter for retrieval?

<details><summary>▸ Answer</summary>

Yes — embeddings capture *meaning*, not spelling, so "get my notes out" lands near
"export to Markdown." That's the whole basis of retrieval: you can find the right
passage even when the user's words don't appear in it.
</details>

---

## Section 3 — Chunking **(offline)**

**Predict, then run.** In `examples/02_chunking.py`, you chunk the same document
at size 40 and size 300. Which produces more chunks? Which produces more *focused*
ones?

<details><summary>▸ Answer</summary>

Size 40 produces many more, smaller, tightly-focused chunks; size 300 produces a
few broad ones. More focused isn't automatically better — a too-small chunk may
not contain enough to answer on its own. That tension is the whole reason chunk
size is a tunable knob.
</details>

**Do (offline).** Set `overlap` equal to `chunk_size` in a call to `chunk_text`.
What happens, and why does the code forbid it?

<details><summary>▸ Answer</summary>

It raises `ValueError`. With overlap == size the window would never advance
(step = size − overlap = 0), so it'd loop forever. Overlap must be *smaller* than
the chunk so the window slides forward while still sharing a boundary.
</details>

---

## Section 4 — The vector store

**Recall.** `VectorStore.search` is described as "brute force." What does it
actually do for each query, and why is that fine here but not at a billion
vectors?

<details><summary>▸ Answer</summary>

It computes cosine similarity against *every* stored vector, then sorts — O(n) per
query. For hundreds or thousands of chunks that's instant. At millions/billions
you switch to an approximate-nearest-neighbour index (FAISS, hnswlib) or a vector
database; same idea, cleverer data structure.
</details>

---

## Section 5 — The RAG pipeline

**Predict, then run.** Ask `examples/04_rag_pipeline.py` something the corpus
does NOT cover (e.g. `"Who founded the company?"`). What should a well-grounded
system do?

<details><summary>▸ Answer</summary>

It should say it doesn't know, rather than invent an answer. That's the job of the
grounding instruction (`GROUNDED_SYSTEM`): answer only from the provided context.
Hallucinating a confident wrong answer is the failure RAG is meant to prevent.
</details>

**Do.** Open `rag/pipeline.py` and read `build_prompt`. Why are the retrieved
chunks *numbered* `[1] [2] ...`?

<details><summary>▸ Answer</summary>

So the model can cite them — "[2]" in the answer maps back to a real source the
reader can check. Numbering the context is what makes citations possible.
</details>

---

## Section 6 — Chunk size & retrieval quality

**Do.** In `examples/05_chunk_size.py`, the same query is run against small and
large chunks. Pick a query where you'd expect them to differ a lot, change it, and
see. When might *large* chunks retrieve worse?

<details><summary>▸ Answer</summary>

When a large chunk bundles several topics, an off-topic sentence can pull it up
(or push the right chunk down) in the ranking, and you spend context-window space
on irrelevant text. Small chunks are more precise but can fragment an answer. The
only way to know for your data is to measure (Section 10).
</details>

---

## Section 7 — Keyword search, BM25 **(offline)**

**Predict, then run.** `examples/06_keyword_search.py` scores chunks for "what does
error NN-413 mean?" on words alone — no embeddings. Which of BM25's three
ingredients (term frequency, IDF, or length normalization) does the most work in
ranking the right chunk first?

<details><summary>▸ Answer</summary>

IDF. "nn-413" appears in just one chunk, so its inverse-document-frequency weight
is huge — one match is decisive. Term frequency barely matters (the code occurs
once) and length only breaks ties. That's BM25's whole edge over counting shared
words: rare terms are worth far more than common ones. And it's all **offline** —
keyword search needs no model, no API call, no cost.
</details>

---

## Section 8 — Hybrid retrieval

**Predict, then run.** `examples/07_hybrid_retrieval.py` asks "what does error
NN-413 mean?" Which scorer — vector or keyword — do you expect to nail it, and
why?

<details><summary>▸ Answer</summary>

Keyword. An exact code like "NN-413" carries little semantic meaning for the
embedding model, but a keyword match finds it instantly. The paraphrase query
("get my notes out") is the reverse — vectors win. Hybrid combines both, which is
why production retrieval is usually hybrid.
</details>

---

## Section 9 — Reranking

**Recall.** Reranking re-scores only the top ~8 chunks, not the whole corpus. Why
is it affordable to use a slower, smarter method at that stage?

<details><summary>▸ Answer</summary>

Because you've already narrowed millions/thousands down to a handful with cheap
vector search. A method too slow to run over everything is fine over 8 candidates.
That's the point of a two-stage retrieve-then-rerank pipeline.
</details>

---

## Section 10 — Evaluation

**Recall.** What's the difference between "hit rate @ k" and "MRR," and why track
both?

<details><summary>▸ Answer</summary>

Hit rate @ k asks *did the right chunk appear in the top k at all?* MRR asks *how
high?* (1/rank of the first correct hit). Two systems can have the same hit rate
but different MRR — the one that ranks the right chunk #1 instead of #4 gives the
model better context. Retrieval can also succeed while the *answer* is still
wrong, which is why the example measures answer correctness too.
</details>

**Do.** Change `K` in `examples/09_evaluation.py` from 4 to 1, then to 8. What
happens to the hit rate, and what's the catch with just cranking k up?

<details><summary>▸ Answer</summary>

Hit rate usually rises with k (more chances to include the right chunk). But every
extra chunk costs context-window space and tokens, dilutes the prompt with
less-relevant text, and can *lower* answer quality. Bigger k is not free — it's a
tradeoff you measure, not maximize.
</details>

---

## Going further — more retrieval techniques

**Recall (query transformation, `10`).** HyDE embeds a *hypothetical answer*
instead of the question. Why does that help retrieval?

<details><summary>▸ Answer</summary>

A question and its answer often share few words, so the question's vector sits far
from the passage that answers it. A drafted answer lives in "answer space" — much
closer to the real passage — so embedding it pulls the right chunk up the ranking.
Multi-query gets there differently: more phrasings = more chances to match.
</details>

**Predict (contextual retrieval, `11`).** The example embeds `context + chunk` but
stores only `chunk`. Why embed one thing and show the model another?

<details><summary>▸ Answer</summary>

The prepended context exists only to make the **embedding** findable (it carries the
"which document / what plan" words a bare chunk lacks). The model should still read
the **clean** chunk, not the synthetic context — so you embed the augmented text but
store and display the original.
</details>

**Recall (metadata & parent-doc, `12`).** Name one relevance reason and one security
reason to filter retrieval by metadata. And what tension does small-to-big resolve?

<details><summary>▸ Answer</summary>

Relevance: only search the docs that could answer (e.g. billing). Security: never
return a doc this user isn't allowed to see. Small-to-big resolves the chunk-size
tension — **small** chunks match precisely, but you return the **parent** so the
model reads complete context.
</details>

**Do (chunking strategies, `13`).** The first part runs offline. Why does splitting on
Markdown headings beat a blind word-window — and what's the one thing it *doesn't* fix?

<details><summary>▸ Answer</summary>

Heading-split sections are each about **one topic**, so a chunk doesn't straddle two
ideas the way a fixed window can (that merged "Import/Export" chunk that tripped up
hybrid search in Section 8). The **heading** also becomes metadata you can filter on
and cite ("Billing > Refunds"). What it *doesn't* fix: a **vocabulary** gap — if the
query says "get my notes out" and the doc says "export," better chunking can't connect
them. That's query transformation's job (`10`).
</details>

**Predict (ingestion, `14`).** Real corpora are PDFs and HTML, not tidy Markdown. What
single shape does every parser reduce a document to, so the rest of the pipeline never
has to care about the original format?

<details><summary>▸ Answer</summary>

`(source, text)` — clean text plus where it came from. HTML through an stdlib parser,
PDFs through pdfplumber/pypdf, Word through python-docx: each just produces text, which
then flows through the *same* heading-split → embed → retrieve path as everything else.
Ingestion is format-in, `(source, text)`-out.
</details>

**Predict (approximate index, `15`).** The IVF index scans only the clusters nearest
the query. Set `n_probe=1` (one cluster) and it scans ~2% of the vectors. Will recall
be near-perfect, near-zero, or somewhere in between — and why?

<details><summary>▸ Answer</summary>

In between (~0.76 on the run). Because the data clusters, most of a query's true
neighbours live in its single nearest cluster — so one probe already finds the bulk of
them. But some genuine neighbours sit just over a cluster border, in the *second*
nearest bucket you didn't scan, so you miss them. That miss rate is recall, and it
climbs fast as you probe more (~0.97 at ~7% scanned). If the vectors had **no** cluster
structure (uniform noise), one probe would miss almost everything — ANN only works
because real embeddings cluster.
</details>

**Recall.** When should you *not* reach for an approximate index?

<details><summary>▸ Answer</summary>

When brute force is fast enough — which, for the few thousand chunks in a normal
corpus, it is. ANN adds a recall risk and a dial to tune (against an eval, §10) in
exchange for speed you may not need. It earns its place at millions of vectors, not
before. Reach for `store.py` first.
</details>

---

## Capstone — `ask_docs.py`

**Do.** Run `secrun python hands_on/ask_docs.py` once, then again. The second run is much
faster — why? Then add a fact to a file in `corpus/`, run with `--rebuild`, and
ask about your new fact.

<details><summary>▸ Answer</summary>

The first run embeds the corpus and caches the index to `.rag_index.json`; later
runs load it instead of re-embedding (embedding is the slow, paid step). Editing
the corpus means the cache is stale — `--rebuild` re-embeds. The cache also auto-
rebuilds if you change provider or chunk settings, since vectors aren't comparable
across embedding models.
</details>

**Stretch.** Ask a question, then run again with `--show-context` and read the
chunks the answer was built from. Do the citations `[n]` in the answer actually
match the right sources? This is how you audit a RAG system for trust.

---

### Where to take it next

Invent your own. Pick a real folder of your own notes or docs, drop them in
`corpus/`, `--rebuild`, and ask. The moment it answers something only *your*
documents know — with a citation you can check — RAG has clicked.
