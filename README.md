# RAG: A Guided Deep Dive

A hands-on playground for learning **retrieval-augmented generation (RAG)** from
the ground up. You'll build a real "chat with your documents" tool, and along the
way understand every moving part by building each one from scratch: chunking,
embeddings, a vector store, retrieval, hybrid search, reranking, grounding with
citations, and evaluation. No LangChain, no LlamaIndex, no vector database, just
enough code to see how it works.

This is the fourth of eight core repos in the series. The sibling repos
([openai-api-deep-dive](https://github.com/alexvervloet/openai-api-deep-dive),
[claude-api-deep-dive](https://github.com/alexvervloet/claude-api-deep-dive)) teach the underlying API
calls: embeddings and chat. This one assumes you can make those calls and asks
the next question: **how do you get a model to answer accurately from *your*
documents?**

Like its siblings, it's meant to be walked through, not just read. Each section
ends with something to run. Do the running; that's where the learning is. And
[EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for each section.

---

## 0. The one big idea

RAG sounds like a lot of machinery, but it all hangs off a single sentence:

> **A model can only answer from what's in its context window. RAG is the
> discipline of putting the *right* text there.**

That's it. Chunking is how you cut documents into put-able pieces. Embeddings and
the vector store are how you find the right pieces. Hybrid search and reranking
are how you find them better. Grounding and citations are how you use them
honestly. Evaluation is how you know it's working. Every section below is a
variation on that one sentence.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Choose your provider (set PROVIDER in .env); your key(s) load separately
cp .env.example .env
#    Your API key(s) do NOT go in .env. Store them in your OS keychain and run
#    lessons with `secrun`; 2-minute setup in ../docs/SECRETS.md. Default is openai
#    (one key); claude needs ANTHROPIC_API_KEY + VOYAGE_API_KEY.

# 4. Confirm everything is wired up (makes no API call, costs nothing)
secrun python check_setup.py       # secrun injects your key so the check can see it
```

**RAG is provider-agnostic, so this repo is too.** Pick whichever stack you set up
in the sibling repos with `PROVIDER` in `.env`:

| `PROVIDER` | Embeddings | Chat | Keys needed |
|------------|-----------|------|-------------|
| `openai` (default) | OpenAI `text-embedding-3-small` | OpenAI `gpt-5.4-nano` | `OPENAI_API_KEY` |
| `claude` | Voyage AI `voyage-3.5` | Claude `claude-haiku-4-5` | `ANTHROPIC_API_KEY` + `VOYAGE_API_KEY` |

Every example and the capstone work the same way on either. The only file that
knows which provider you picked is [rag/providers.py](rag/providers.py). That's
the whole point: RAG is an architecture, not a provider feature.

> **You can start before spending much.** Example 02 (chunking) is fully
> offline, no key and no cost. The rest make small, cheap embedding/chat calls.
> Nothing needs Docker except the optional database section (§12).

---

## 2. Prerequisites: embeddings, recapped

RAG is built on one capability from the sibling repos: an **embedding** turns text
into a vector that captures its meaning, and **cosine similarity** measures how
close two vectors are (1.0 = same meaning, 0 = unrelated). The magic that makes
retrieval possible: two texts can match strongly even when they share **no words**.

```bash
secrun python examples/01_embeddings_recap.py
```

If that's already familiar, skim it and move on. If not, the embeddings examples
in the sibling repos go slower. Everything else here builds on this one idea.

---

## 3. Chunking: cutting documents into pieces

You don't embed a whole document as one vector. A long page covers many topics,
and one vector can only point in one averaged direction, matching everything
vaguely and nothing well. Instead you split documents into **chunks** and embed
each one, so retrieval can return the specific paragraph that answers a question.

```bash
python examples/02_chunking.py        # offline, no key, no cost
```

Two knobs, both tradeoffs:
- **chunk size**: too big and each chunk is unfocused and wastes context space;
  too small and a chunk may not say enough on its own.
- **overlap**: chunks share a few words at the seams so an idea split across a
  boundary isn't lost to both neighbours.

There's no universally correct setting. It depends on your documents and
questions, which is why Section 6 has you *measure* it. How you cut matters too,
not just how big: see [rag/chunking.py](rag/chunking.py) for `chunk_text()`
(sliding window), `chunk_paragraphs()` (split on blank lines), and
`chunk_markdown_sections()` (split on headings), compared head-to-head in the
"Chunking strategies" example under [Going further](#going-further-six-more-retrieval-upgrades).

---

## 4. The vector store: finding the right chunks

A "vector store" is just a list of `(text, vector)` pairs plus a way to find the
vectors closest to a query. We build one by hand so there's no mystery: store each
chunk with its embedding, then for a query, score every chunk by cosine similarity
and return the top-k.

```bash
secrun python examples/03_vector_store.py
```

This is the *retrieval* half of RAG: finding the right text, no model answer yet.
Our [rag/store.py](rag/store.py) does a brute-force O(n) scan, which is instant for
thousands of chunks and completely transparent. Swapping in an approximate index
(FAISS) or a hosted vector database (pgvector, Pinecone) for millions of vectors
is the main thing "production RAG" adds. Same idea, cleverer data structure, and
§12 does exactly that swap against a real Postgres so you can see what changes
(the lifecycle) and what does not (the retrieval).

---

## 5. The RAG pipeline: putting it together

Now retrieval meets generation, the actual point of the repo:

```bash
secrun python examples/04_rag_pipeline.py
secrun python examples/04_rag_pipeline.py "Can students get a discount?"
```

The flow (see [rag/pipeline.py](rag/pipeline.py)):

1. `index_documents()` chunks and embeds the corpus into a vector store.
2. `retrieve()` embeds the question and returns the closest chunks.
3. `answer()` pastes those chunks into a **grounded prompt**, generates, and hands
   back the answer with its sources.

Two disciplines make a RAG answer trustworthy rather than a guess, both in
`GROUNDED_SYSTEM`:
- **Grounding**: the model is told to answer ONLY from the provided context, and
  to say "I don't know" otherwise. Ask about something not in the corpus and watch
  it decline instead of hallucinate.
- **Citations**: the context blocks are numbered, so the model can cite `[2]` and
  you can check the claim against the real source.

---

## 6. Tuning retrieval I: chunk size

Section 3 showed chunk size changing the *number* of chunks; this shows it
changing retrieval quality, which is what you actually care about. Same corpus,
same query, indexed small vs large:

```bash
secrun python examples/05_chunk_size.py
```

Small chunks pinpoint the exact sentence but can fragment an answer; large chunks
carry more context per hit but less precisely. The lesson isn't "small good". It's
that this is an empirical knob you tune by looking at results (and, in Section 10,
at numbers).

---

## 7. Keyword search: the other half of retrieval

The vector store (Section 4) matches on *meaning*. Its opposite number is
**keyword** (lexical) search, which matches on the actual words: exactly what
embeddings are worst at, like product names, error codes, and IDs. We build it
from scratch with **BM25**, the classic search-engine ranking function, which
weighs each query word by how rare it is (a shared "the" is worthless; a shared
"NN-413" is decisive) and normalizes for chunk length. No model, no embeddings,
just arithmetic over word counts, so it's **free and offline**.

```bash
python examples/06_keyword_search.py   # offline, no key, no cost
```

The example runs an exact-code query (keyword nails it) against a paraphrase query
(keyword whiffs, because the answer uses different words than the question). That
second failure is the mirror image of vector search's, and the whole reason the
next section blends the two. See [rag/keyword.py](rag/keyword.py) for the BM25 code.

---

## 8. Tuning retrieval II: hybrid search

You've now built both halves of retrieval: vector search by meaning (Section 4)
and keyword search by words (Section 7). Each has a blind spot the other covers.
**Hybrid retrieval** runs both and blends the scores, getting each one's strength.

```bash
secrun python examples/07_hybrid_retrieval.py
```

The example contrasts a paraphrase query (vectors win) with an exact-code query
(keyword wins) and shows hybrid handling both, which is why most production
retrieval is hybrid.

---

## 9. Tuning retrieval III: reranking

Vector search is cheap but approximate; the best chunk isn't always #1. A two-stage
pipeline fixes this: **retrieve** a generous handful cheaply, then **rerank** those
few with a slower, smarter scorer and keep the best.

```bash
secrun python examples/08_reranking.py
```

Production systems use a dedicated **cross-encoder reranker** (Voyage and Cohere
both offer rerank endpoints). The example demonstrates the same pattern with a
tool you already have, asking the model to pick the most relevant passages, so
you see the shape without a new dependency.

---

## 10. Evaluation: is it actually working?

Every knob above is a guess until you measure. RAG fails in two independent places,
so you evaluate both:

```bash
secrun python examples/09_evaluation.py
```

- **Retrieval**: did the right chunk come back? `hit rate @ k` (was it in the top
  k?) and `MRR` (how high?).
- **Answer**: given good context, was the final answer right?

The example scores a tiny labelled set and prints all three numbers. Change a knob
(chunk size, k, hybrid weighting, reranking) and re-run. If a number drops, you
caught a regression you'd otherwise have shipped. **This is the habit that
separates a RAG demo from a RAG system.**

---

## Going further: six more retrieval upgrades

Once the core pipeline works and you can measure it (§10), these are the highest-
leverage upgrades. Each is a small, self-contained example you can run and score.

### Query transformation: HyDE and multi-query
A question and its answer often don't "look alike" in embedding space. Transform the
query first: draft a hypothetical answer and embed that (**HyDE**), or fan the
question out into several paraphrases and union the results (**multi-query**). One
extra LLM call before retrieval, and oblique questions start finding the right chunk.
```bash
secrun python examples/10_query_transformation.py
```

### Contextual retrieval
A chunk embedded in isolation loses the words that make it findable ("the limit is
5 GB", of what?). Prepend a one-sentence, LLM-written context that situates the
chunk in its document *before embedding*, while still showing the model the clean
chunk. A cheap, high-leverage win for short, under-specified passages.
```bash
secrun python examples/11_contextual_retrieval.py
```

### Metadata filtering & parent-document retrieval
Retrieval quality isn't only embeddings. **Metadata filtering** constrains *where*
you search (category, date, access-level), giving you relevance and security in one
move. **Parent-document (small-to-big)** embeds small chunks for a precise match but
returns the larger parent for complete context, resolving the chunk-size tension.
```bash
secrun python examples/12_metadata_and_parent.py
```

### Chunking strategies: fixed-size vs structure-aware
A fixed-size word window cuts wherever the count runs out, sometimes mid-topic,
gluing the tail of one section onto the head of the next (exactly the merged chunk
that tripped up §8's hybrid search). Splitting on the document's own headings instead
gives one topic per chunk and a heading you can cite. The honest limit: it fixes
*structure*, not the vocabulary gap that query transformation handles.
```bash
secrun python examples/13_chunking_strategies.py
```

### Document ingestion: from messy sources to clean chunks
Real corpora are PDFs and HTML, not tidy Markdown. Parse each format down to clean
text, apply the heading-aware split from the previous section, attach metadata, and
tidy whitespace. Ingestion is where retrieval quality is won or lost.
```bash
secrun python examples/14_document_ingestion.py
```

### Approximate nearest-neighbour: recall for speed
Our store (§4) does exact brute force: perfect for thousands of chunks, too slow at
millions. Production then switches to an **approximate** index (FAISS, hnswlib,
pgvector's IVFFlat/HNSW): bucket the vectors and only scan the buckets nearest the
query, skipping most of the data. You trade a little **recall** for a large speedup.
The example builds a tiny IVF index by hand ([rag/ann.py](rag/ann.py)) over synthetic
clustered vectors and turns the "how many buckets to probe" dial. Scanning ~7% of
the data recovers ~97% of the exact top-10. Offline, no key. The honest caveat:
don't reach for ANN until brute force is genuinely too slow, and always tune the dial
against an eval (§10), never on faith.
```bash
python examples/15_approximate_index.py       # offline
```

---

## 11. The capstone: `ask_docs.py`

Everything comes together in a real "chat with your docs" tool. Point it at the
[corpus/](corpus/) folder and ask; it indexes once (caching the embeddings to
disk), retrieves, and answers with `[n]` citations plus a table of sources.

```bash
# Ask the built-in demo question:
secrun python hands_on/ask_docs.py

# Ask your own:
secrun python hands_on/ask_docs.py "Can students get a discount?"

# Retrieve more chunks and show the exact text retrieved:
secrun python hands_on/ask_docs.py "How is my data protected?" -k 6 --show-context

# Re-embed from scratch with different chunking:
secrun python hands_on/ask_docs.py "What plans are there?" --rebuild --chunk-size 80
```

Read the source in [hands_on/ask_docs.py](hands_on/ask_docs.py): the index is
cached in `.rag_index.json` and rebuilt automatically when the provider or chunk
settings change (vectors from one embedding model are meaningless to another).
**Suggested exercise:** drop your own notes or docs into `corpus/`, run with
`--rebuild`, and ask. When it answers something only your documents know, with a
citation you can check, RAG has clicked.

---

## 12. A real vector store: Postgres + pgvector

Everything so far keeps the index in memory and caches it to `.rag_index.json`.
That is the right shape for learning retrieval, and it hides the half of the job
that takes the time in production. A cache file has exactly one verb, *rebuild
everything*, and rebuilding everything means re-embedding everything, which is
the one step that costs money.

This section runs the same pipeline against **Postgres with the pgvector
extension** so the index outlives the process, and walks the lifecycle a durable
index forces on you:

| The lifecycle event | What the JSON cache does | What the database does |
|---|---|---|
| Nothing changed since last run | Rebuild everything, or trust a cache with no idea what is in the corpus | Compare a content hash per document; embed nothing |
| One document edited | Re-embed the whole corpus | Re-embed that document; replace its chunks in one transaction |
| A document deleted from the corpus | Nothing, unless you remember to rebuild | Delete the row; `ON DELETE CASCADE` takes its vectors with it |
| Embedding model changed | Silently compare vectors that mean different things, unless the cache header catches it | Detect it against the recorded model id, and migrate: the vector column has a fixed width |
| Corpus outgrows a brute-force scan | Nothing on offer | `CREATE INDEX ... USING hnsw`, built after loading |

Two commands and a few cents of embeddings:

```bash
docker compose up -d                          # pinned pgvector 0.8.6 on Postgres 18
pip install -r requirements-postgres.txt      # psycopg, the Postgres driver

secrun python examples/16_pgvector_lifecycle.py
```

The example runs the six events above in order and prints what each one cost, so
the incremental-sync argument arrives as numbers rather than a claim: a second
sync of an unchanged corpus embeds **0 chunks**, and editing one document of four
embeds **3 of 12**.

The capstone speaks to it too, with the same retrieval code and a different store:

```bash
secrun python hands_on/ask_docs.py --store pg
secrun python hands_on/ask_docs.py --store pg --rebuild        # start the index over
```

Read [rag/pgstore.py](rag/pgstore.py) next to [rag/store.py](rag/store.py). The
retrieval maths is unchanged: `<=>` is pgvector's cosine distance operator, so
`1 - (embedding <=> query)` is the same cosine similarity `store.py` computes by
hand, and `pipeline.py` cannot tell the two stores apart. **The database is an
operational upgrade, not a different idea.**

Three things the example is deliberately honest about:

- **The planner ignores your index, and it is right to.** With a few hundred
  chunks, Postgres chooses a sequential scan over the HNSW index, because
  scanning a few hundred rows beats walking a graph. The example prints both
  plans, forcing the index with `enable_seqscan = off` to show what it returns.
  An index whose recall you have not measured at your own scale is a guess; §15
  is the same dial, turned by hand.
- **An index is not free before it pays.** HNSW stores its own copy of every
  vector and slows every insert. Build it when brute force is measurably too
  slow, not in advance.
- **A database is not automatically the right answer.** For a corpus of a few
  thousand chunks on one machine, `store.py` plus a cache file is genuinely
  better: no service to run, no migration to write, no schema to keep in step
  with your embedding model. What the database buys you is the lifecycle, so
  reach for it when documents *change*, not when they are merely numerous.

Stop the service when you are finished; the data survives in a named volume.

```bash
docker compose down
```

---

## RAG, fine-tuning, or something else?

RAG is the right tool for a specific problem: the model lacks **knowledge** it
needs right now. It's not the only tool, and reaching for it reflexively is a
common mistake. The honest framing comes straight from this repo's one big idea.
RAG changes *what's in the context window*; fine-tuning changes *how the model
behaves by default*. Different problems.

| What you actually need | Reach for | Why |
|------------------------|-----------|-----|
| Facts that change, are private, or must be **cited** | **RAG** | Retrieve the source at request time; update the corpus, not the model |
| A consistent **format, tone, or narrow skill** done the same way every time | **Fine-tuning** | You're teaching *behavior*, not facts, and that's what training adjusts |
| The relevant material is small enough to just **include** | **Long context** | If it fits in the prompt, retrieval is machinery you don't need |
| The model must **act** or fetch live data | **Tools / agents** | The gap is *capability*, not knowledge. See the [agents repo](https://github.com/alexvervloet/agents-deep-dive) |
| Lower **latency/cost** on a fixed, high-volume task | **Fine-tune a smaller model** | Distill known-good behavior into a cheaper model |

They're complementary, not either/or: a common production shape is **fine-tune for
format, RAG for facts**. Train the model to always answer in your house style, and
retrieve the facts it cites.

Two rules of thumb. **Don't fine-tune first.** It's the slow, expensive,
provider-specific option, and it can't add knowledge that changes. Exhaust
prompting, better context, and RAG before you reach for it. And **don't decide by
vibes.** The only way to know whether fine-tuning beat your RAG baseline (or made
things worse) is to measure both on the same gold set. That's what the
[evals repo](https://github.com/alexvervloet/evals-deep-dive) is for; the evaluation in
Section 10 is the same method, pointed at a different decision.

---

## Where to go next

You've built a complete small RAG system. The road to production is mostly about
scale and robustness on top of these same ideas:

- **A real vector database**: pgvector, Pinecone, Weaviate, or a local FAISS /
  hnswlib index, for fast approximate search over millions of vectors instead of
  our brute-force scan. §15 builds a toy IVF index by hand so you can see the
  recall-for-speed dial these tools all expose, and **§12 runs the whole pipeline
  against a real pgvector database**, where the interesting part turns out to be
  the index lifecycle rather than the search itself.
- **Smarter chunking**: token-based sizing, structure-aware splitting, and
  attaching metadata (titles, dates, sections) for filtering.
- **Dedicated rerankers**: cross-encoder rerank endpoints (Voyage, Cohere) in
  place of the LLM reranker pattern in Section 9.
- **Query transformation**: rewriting or expanding the user's question, or
  generating multiple sub-queries, before retrieval.
- **Serious evaluation**: bigger labelled sets, LLM-as-judge for faithfulness,
  and frameworks like Ragas; plus tracing what was retrieved on every request.
- **Agentic RAG**: letting the model decide *when* to retrieve and issue its own
  searches, instead of always retrieving once up front.

Each slots on top of the one idea you started with: put the right text in the
context window.

---

## From teaching code to production

The "Where to go next" section above is about scaling RAG itself. This one is
about the operational layer every RAG system needs once people rely on it. It's
orthogonal to retrieval quality, and the same for any LLM app:

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| The pipeline `print()`s its answer and sources | One **structured trace** per query, including *which chunks* were retrieved and their scores |
| `embed()` / `generate()` called bare | The calls wrapped in **retries + backoff** so a flaky provider doesn't fail the query |
| No ceiling on embedding a big corpus | An enforced **cost budget** on indexing and querying |
| Index cached to a local JSON file | A **shared cache** (Redis / a real vector DB) across servers, plus caching repeat *answers* |
| The grounding/citation prompt is inline | A **versioned prompt** promoted only past an **eval gate**, so retrieval-prompt changes can't silently regress |
| Retrieved documents are trusted text | **Guardrails.** Retrieved content is untrusted input and a classic *indirect injection* vector |

These shortcuts are right for learning and wrong for production. All seven
concerns (observability, cost, reliability, caching, guardrails, prompt
versioning, and eval gates) are built from scratch and wired into one running
app in **[Production](https://github.com/alexvervloet/ai-in-production-deep-dive)** (#8 in the
series). It runs **offline on a mock provider**, so you can see the whole ops
machinery with no key and no cost.

---

## File map

```
check_setup.py              ← run first: verifies Python, packages, provider, keys
README.md                   ← this guide
EXERCISES.md                ← predict-then-run prompts, one per section
rag/                        ← the from-scratch library (read it!)
  providers.py              ← the ONLY provider-specific file: embed() + generate()
  chunking.py               ← split documents into chunks
  store.py                  ← in-memory vector store + cosine search
  pgstore.py                ← the same store in Postgres/pgvector: the index lifecycle
  ann.py                    ← a from-scratch approximate (IVF) index: recall-for-speed
  keyword.py                ← keyword search (BM25), the lexical counterpart
  loader.py                 ← read a folder of docs into (name, text)
  pipeline.py               ← index -> retrieve -> answer (grounding + citations)
  preview.py                ← display helper: print retrieved chunks centered on the match
corpus/                     ← a small fictional document set to retrieve over
hands_on/
  ask_docs.py               ← capstone: chat with your docs, with citations
examples/
  01_embeddings_recap.py    ← embeddings & cosine (recap from the sibling repos)
  02_chunking.py            ← splitting documents (offline, no key)
  03_vector_store.py        ← build a store, retrieve top-k
  04_rag_pipeline.py        ← the full loop: retrieve -> ground -> answer
  05_chunk_size.py          ← how chunk size changes what you retrieve
  06_keyword_search.py      ← keyword search from scratch, BM25 (offline, no key)
  07_hybrid_retrieval.py    ← keyword + semantic, combined
  08_reranking.py           ← over-retrieve, then reorder
  09_evaluation.py          ← hit rate, MRR, and answer correctness
  10_query_transformation.py ← HyDE & multi-query: transform the query first
  11_contextual_retrieval.py ← prepend a context sentence before embedding
  12_metadata_and_parent.py ← metadata filtering + small-to-big retrieval
  13_chunking_strategies.py ← fixed-size vs heading-aware chunking (partly offline)
  14_document_ingestion.py  ← HTML/PDF parsing into clean, structured chunks (partly offline)
  15_approximate_index.py   ← approximate (IVF) search: recall-for-speed tradeoff (offline, no key)
  16_pgvector_lifecycle.py  ← create, re-sync, delete, index, migrate: a real database
compose.yaml                ← pinned local pgvector service for §12
requirements-postgres.txt   ← psycopg, needed only for §12
```

---

## Troubleshooting

Run `secrun python check_setup.py` first; it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `PROVIDER=... needs ... in the environment` | Set `PROVIDER` in `.env`, then load the key(s) from your keychain by running under `secrun`. See [SECRETS.md](../docs/SECRETS.md). |
| `ModuleNotFoundError` (openai / anthropic / voyageai / rich) | Dependencies aren't installed or the venv isn't active. `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| `AuthenticationError` / 401 | A key is present but wrong. For `claude`, remember you need **two** keys (Anthropic *and* Voyage). |
| Answers look wrong or "I don't know" for facts that ARE in the corpus | A retrieval problem, not a model problem. Raise `-k`, try `--rebuild` after editing the corpus, or check Section 10's metrics. |
| Switched provider and results went haywire | Stale index. The capstone auto-rebuilds, but if you cached elsewhere, delete `.rag_index.json`; vectors aren't comparable across embedding models. |
| `Could not connect to Postgres at ...` | The optional §12 database isn't running. `docker compose up -d`, or drop `--store pg` to use the JSON cache. |
| `The Postgres path needs psycopg` | `pip install -r requirements-postgres.txt`. Only the §12 path needs it. |
| `expected 1536 dimensions, not 1024` from Postgres | You switched embedding models against an existing index. `sync()` handles this itself; if you hit it from your own code, the column width is part of the schema. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained. Open it, read the docstring
at the top, and run it directly.

---

## The series

This is one of the standalone, hands-on deep dives into building with LLM APIs: eight core, plus the bonus dives listed below.
Each one stands on its own, with its own setup, examples, and capstone, and they
all share the same house style: provider-agnostic, built from scratch (no
frameworks), offline-first examples, and a real capstone. Do them in any order;
this sequence builds naturally:

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive): the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive): the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive): shape model behavior with better prompts (zero/few-shot, chain-of-thought, roles)
4. [RAG](https://github.com/alexvervloet/rag-deep-dive): answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive): measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive): give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive): attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive): operate one app end to end: observability, cost, reliability, caching, guardrails, prompt versioning, eval gates

**Bonus dives**, standalone and slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive): manage what's in the window: memory, compaction, assembly
- [AI Data Engineering](https://github.com/alexvervloet/ai-data-engineering-deep-dive): the corpus behind the index: versions, lineage, ACLs, deletes
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive): images & audio, not just text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive): teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive): serve tools, data & prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive): run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive): build on the loop: hooks, permissions, sandboxing, subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive): low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive): watch a running app over time: drift, quality, alerting, the flywheel
- [Architecture](https://github.com/alexvervloet/architecture-deep-dive): the seams between the components, each decision measured rather than asserted
- [GenAI Security](https://github.com/alexvervloet/genai-security-deep-dive): treat the model as an untrusted principal: identity, supply chain, isolation, budgets, release gates
- [Inference Platform Engineering](https://github.com/alexvervloet/inference-platform-deep-dive): turn finite GPU memory and a request queue into latency, throughput, and a fleet size you can defend
- [Testing & Delivery](https://github.com/alexvervloet/testing-and-delivery-deep-dive): decide whether a build has earned promotion: evidence, gates, staged rollout, rollback
- [Professional Tools](https://github.com/alexvervloet/professional-tools-deep-dive): rebuild each from-scratch primitive with the tool professionals reach for, and measure both

And the whole series lands in one codebase in the
[capstone](https://github.com/alexvervloet/deep-dive-capstone): a codebase Q&A tool
built step by step, one tag per dive.

**You are here: #4, RAG.**
