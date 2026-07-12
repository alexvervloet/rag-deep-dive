# Chapter 4: Retrieval, or Teaching a Model What It Never Learned

*This is the textbook chapter for the RAG deep dive. The [README](README.md) is the lab manual; this is the lecture. It covers where retrieval-augmented generation came from, why every piece of the pipeline exists, and the sixty years of search-engine research the AI industry rediscovered in about eighteen months.*

---

## 4.1 The problem RAG was invented for

Every language model has a peculiar relationship with knowledge. It knows an astonishing amount, and it cannot learn anything new. Training baked the world into its weights up to some cutoff date, and from that moment the model is frozen. It has never seen your company's documentation, yesterday's outage report, or the contract on your desk. Worse, when asked about things it does not know, it does not reliably say so. It produces the most plausible-sounding text, and the most plausible-sounding text about an unfamiliar topic is often a confident fabrication.

This is not a bug to be patched out; it follows from what the machine is. A model is a compression of its training data into statistical tendencies. Facts it saw thousands of times are represented well. Facts it saw once are barely there. Facts it never saw are, of course, not there at all, but the machinery for producing fluent text works regardless.

For a while, people assumed the fix was retraining: your company's model, trained on your company's data. That turns out to be slow, expensive, and, most damningly, wrong for the job. Training changes a model's habits far more reliably than its facts, a distinction Chapter 13 makes precise. Meanwhile a much cheaper observation was sitting in plain sight. The model may be frozen, but the context window is not. Whatever text you place in the prompt, the model can read, reason over, and quote, right now, with no training at all. If the model does not know your refund policy, paste the refund policy into the prompt and ask again.

Retrieval-augmented generation is that observation, industrialized:

> **A model can only answer from what is in its context window. RAG is the discipline of putting the right text there.**

The name comes from a 2020 research paper out of Facebook AI Research, which proposed coupling a retriever to a generator so the system could look things up instead of relying on memorized parameters. The paper's version was an exotic research architecture. The version that conquered the industry after 2022 is almost embarrassingly simpler: search for relevant text, paste it into the prompt, ask the question. Everything in this dive, and every commercial "chat with your data" product you have seen, elaborates on that paste.

One more piece of context worth having: RAG is the technology behind most enterprise AI deployments you encounter. When a bank's assistant answers questions about its own products, when a support bot cites the actual help article, when an internal tool answers from a wiki, that is RAG. It won by default, because it is the only approach that handles facts which change daily, respects document-level access permissions, and can show its sources. Keep those three properties in mind; they explain several design choices below.

## 4.2 Meaning as geometry

RAG's central act is search, and its search rests on a genuinely beautiful idea: meaning can be represented as position in space.

An embedding model takes a piece of text and returns a vector, a long list of numbers, produced so that texts with similar meaning land near each other. "How do I get my money back?" and "What is your refund policy?" share almost no vocabulary, yet their vectors sit close together, because the model that produced them was trained on the relationships between texts, not their spelling. Closeness is measured with cosine similarity, which you can read as "how aligned do these two directions point?" and leave the trigonometry to the library.

The idea has a longer pedigree than the current boom. Linguists in the 1950s proposed that a word's meaning is characterized by the company it keeps, a claim usually credited to J.R. Firth. Decades of work turned that slogan into arithmetic, through word vectors like word2vec in 2013 (famous for stunts like king minus man plus woman landing near queen) and on to today's models that embed whole passages. What changed recently is quality and availability: for fractions of a cent, an API call now gives you a vector good enough that nearest-neighbor lookup finds paraphrases, translations of intent, and oblique references that no keyword index could.

Once documents are points in space, retrieval becomes geometry: embed the question, find the nearest stored chunks, return them. The lab has you build the entire "vector store" by hand, and it is deliberately anticlimactic: a list of text-and-vector pairs and a loop that scores every one against the query. That transparency is the point. Commercial vector databases are that loop with better data structures, and meeting the naive version first inoculates you against believing the database is where the intelligence lives. It is not. It lives in the embeddings, and in the decisions this chapter turns to next.

## 4.3 Chunking: the unglamorous decision that decides everything

You do not embed whole documents. A fifty-page manual covers dozens of topics, and a single vector can only point in one direction, an average of all of them, which matches every query weakly and none well. So documents get cut into chunks, each embedded separately, so retrieval can return the specific passage that answers the question.

How you cut turns out to matter enormously, and it is the least glamorous, most consequential decision in the pipeline. Two knobs frame the tradeoff. Size: big chunks carry context but arrive diluted and eat your context budget; small chunks are precise but may not say enough to be understood alone. Overlap: adjacent chunks share a seam of words so that an idea straddling a boundary is not lost to both neighbors.

But the deeper lesson in the lab is that where you cut matters as much as how much. A fixed-size window slices wherever the word count runs out, and this repo's own examples got burned by it in a way instructive enough to be preserved in [AUTHORING-LESSONS.md](../AUTHORING-LESSONS.md): a 120-word window glued the tail of a document's "Importing" section onto the head of its "Exporting" section, producing a chunk about neither, which then polluted retrieval for both topics. Documents with structure (headings, sections, paragraphs) chunk far better along their own boundaries: one topic per chunk, and a heading you can cite. The general principle travels well beyond RAG: respect the structure the author already gave the data before imposing your own.

There is no universally correct setting, and be suspicious of anyone who quotes one. The honest answer is that chunking is an empirical knob, tuned by measurement against your documents and your users' questions, which is why the evaluation section below is not optional decoration.

## 4.4 What the older search technology still knows

Embeddings match meaning, and that is also their weakness. Ask a corpus "what does error NN-413 mean?" and semantic similarity is precisely the wrong instrument: "NN-413" does not mean anything. It is an arbitrary string whose only virtue is being rare and exact, and embedding models blur exactly that kind of token. Product names, error codes, ticket IDs, function names, part numbers: the queries where users most expect an exact hit are the queries where pure vector search most reliably whiffs.

The technology that handles those queries was waiting in the library stacks. Keyword search ranks documents by the words they actually share with the query, weighting each word by rarity. A shared "the" is worthless; a shared "NN-413" is decisive. That insight, that a term's importance is inverse to its frequency across the corpus, was formalized by Karen Spärck Jones in 1972, and the specific function this lab builds, BM25, was refined through decades of information-retrieval research and still powers serious search engines today. It requires no model, no API, no key: arithmetic over word counts. There is a certain comedy in the AI industry's rediscovery of it around 2023, when teams that had gone all-in on embeddings found their users typing part numbers.

Each method's blind spot is the other's strength, so production systems run both and blend the results: hybrid retrieval. And here the lab teaches its most honest lesson, one that got the example rewritten (see AUTHORING-LESSONS.md again). The tidy claim "hybrid is strictly better" is false. On a paraphrase query, keyword search is not merely unhelpful; it is confidently wrong, and a naive 50/50 blend of scores can drag the right answer down the ranking, below where vector search alone had it. Fusion is a dial to tune, not a free lunch: rank-based fusion (RRF) is more robust than score-blending, weights need tuning against measurements, and the exercise has you watch each failure mode happen. If you remember one meta-lesson from this dive, make it this shape: almost nothing in retrieval is strictly better, and the systems that work are the ones whose builders measured.

Reranking completes the retrieval toolkit with a two-stage pattern borrowed from web search: retrieve a generous candidate set with the cheap method, then re-score just those few with a slower, smarter judge and keep the best. The economics are the whole idea. A scorer that reads the query and the passage together is far more accurate than one that must summarize each into a vector before they ever meet, but it is far too slow to run against a whole corpus. Running it against twenty candidates costs almost nothing. Cheap-and-broad feeding expensive-and-narrow is a pattern you will see across engineering, and here it routinely buys the largest quality jump per line of code in the pipeline.

## 4.5 Grounding: the honesty layer

Everything so far finds text. What the model does with it is a separate discipline, and it is where RAG earns or loses its reputation.

A grounded prompt tells the model to answer only from the provided context, and to say "I don't know" when the context does not contain the answer. Recall from Chapter 3 why the second clause is not decoration: a model with no permitted path to uncertainty will manufacture an answer, because refusal was not on the menu. The lab's pipeline asks about things absent from the corpus specifically so you can watch the model decline, which, run against everything you know about model behavior by now, is a small and hard-won victory.

Citations complete the arrangement. The context chunks are numbered, the model cites [2], and a human can check the claim against the actual source. This matters beyond user trust. Citations make the system auditable (which passage produced this wrong answer?), debuggable (was the failure in retrieval or in generation?), and honest in a way no amount of model quality can substitute for. It is worth being clear-eyed that grounding is an instruction, not a guarantee: a model can still misread or embellish the context it was given. Faithfulness, whether the answer actually follows from the retrieved text, is itself measurable, and Chapter 5 builds the judge that measures it.

RAG also changes the hallucination conversation in a way worth stating plainly. Without retrieval, a wrong answer is unfalsifiable to the user; with retrieval and citations, wrongness has a paper trail. The system does not just answer more accurately; it fails more visibly. In production software, failing visibly is a feature of enormous value.

## 4.6 Two places to fail, two instruments to watch

A RAG system is a pipeline of two stages, and it fails in two independent ways. Retrieval can return the wrong chunks, in which case no model, however brilliant, can answer correctly; it is working from the wrong page. Or retrieval can succeed and generation can still botch the answer: misread the context, merge two passages into a claim neither makes, over-hedge into uselessness.

Because the failures are independent, one score cannot diagnose the system, and this is the single most common analytical mistake in RAG work. The lab therefore measures the stages separately. Retrieval gets hit rate (did a correct chunk appear in the top k?) and MRR, mean reciprocal rank (how high did it appear?). Generation gets answer correctness given the retrieved context. When quality drops, the metric that moved names the stage that broke, and the fix differs completely: retrieval failures send you to chunking, hybrid weights, or the embedder; generation failures send you to the prompt or the model.

This diagnostic habit has real teeth. This series' own capstone hit a case where a local model's citation score collapsed while its correctness held, and the tidy conclusion ("small models can't ground answers") turned out to be false: the model was citing real sources in a format the strict parser rejected. The number that moved, examined closely, named a formatting bug, not a grounding failure. Metrics do not just tell you that something broke; read carefully, they tell you what.

Every knob this chapter has introduced (chunk size, overlap, k, hybrid weighting, reranking, and the upgrades below) is a guess until these numbers move. Change one thing, rerun the eval, keep what measured better. The habit is the difference between a RAG demo and a RAG system, and it is the bridge to the next chapter, which makes a full discipline of it.

## 4.7 The upgrade path, honestly labeled

Past the core pipeline, a set of well-established upgrades earns most of the remaining quality. Each exists to fix one nameable failure, and each comes with an honest caveat.

**Query transformation** fixes the vocabulary gap between questions and answers. A question and its answer often do not look alike in embedding space ("why did my payment bounce?" versus a passage about transaction declines). HyDE has the model draft a hypothetical answer and embeds that instead, on the logic that a fake answer resembles the real one more than the question does; multi-query fans the question into several paraphrases and unions the results. The caveat, learned the hard way in this repo: on well-matched questions these techniques change nothing, because there is nothing to fix. They earn their extra LLM call on oblique queries specifically.

**Contextual retrieval** fixes chunks that lose their identity when cut loose. "The limit is 5 GB" is unfindable in isolation: the limit of what, on which plan? Prepending a one-sentence, model-written orientation ("This chunk is from the storage section of the Pro plan documentation...") before embedding makes the chunk findable while the model still sees clean text. Anthropic published the technique in 2024 with strong numbers. The caveat: it shines exactly where chunks are under-specified and near-duplicates need disambiguating; on self-contained, well-written passages it may change nothing, a fact this repo verified the awkward way.

**Metadata filtering** does with a database predicate what no embedding can: restrict search by date, category, or, critically, access level. Security by retrieval scope (the user's clearance filters which chunks are even searchable) is both a relevance and a safety feature. **Parent-document retrieval** resolves the chunk-size tension by embedding small for precise matching but returning the chunk's larger parent for complete context: match narrow, read wide.

**Approximate nearest-neighbor indexes** are the scale story. Brute-force scanning is exact and, for thousands of chunks, instant; at millions of vectors it becomes the bottleneck, and production systems switch to approximate indexes (FAISS, HNSW, pgvector's index types) that cluster the space and search only promising regions, trading a little recall for a large speedup. The lab builds a toy one so you can turn the dial yourself: scanning roughly 7% of the data recovers roughly 97% of the true top ten. The caveat is a career's worth of avoided complexity in one sentence: do not reach for ANN until brute force is measurably too slow, and tune the tradeoff against your eval, never on faith.

**Ingestion**, finally, is where more real-world RAG quality is won or lost than anywhere glamorous: real corpora are PDFs and HTML and export sludge, and parsing them into clean, structured, metadata-tagged text is unheroic work that everything downstream inherits. Garbage in, confidently cited garbage out.

## 4.8 When not to use RAG

The strongest sign you understand a tool is knowing when to put it down.

If the relevant material simply fits in the prompt, include it and skip the machinery. Context windows now run to hundreds of thousands of tokens, and "just paste the manual" is a legitimate architecture for corpora that fit. The long-context-versus-RAG debate is live and the boundary keeps moving, but three things keep retrieval alive at scale: cost (re-sending a million tokens per question is real money, even with caching), the well-documented tendency of models to lose facts buried in the middle of very long contexts, and the fact that no window holds a corpus that grows forever.

If the problem is behavior rather than knowledge (the model knows enough but answers in the wrong format, tone, or style), that is fine-tuning's territory, Chapter 13. If the model needs to act or fetch live state, that is tools, Chapter 6. The recurring triad from [CHOOSING.md](../CHOOSING.md) applies: RAG changes what the model knows right now; fine-tuning changes how it behaves by default; tools change what it can do. Most "we need to fine-tune on our docs" instincts are knowledge problems wearing the wrong label, and the common production shape is a combination: fine-tune for the house style, retrieve for the facts.

One forward reference with teeth: retrieved documents are untrusted input. A RAG system reads text written by whoever wrote the documents, and a document that contains instructions ("ignore your previous instructions and...") is read by a model that has trouble telling data from commands. That makes RAG a classic vector for indirect prompt injection, and it is Chapter 7's opening exhibit.

## 4.9 Where this chapter leaves you

The capstone is a working "chat with your documents" tool: index a folder once, cache the embeddings, retrieve, answer with checkable citations. The suggested exercise is the one that makes it real: drop in your own notes, ask something only they contain, and check the citation. A model just answered from knowledge that did not exist when it was trained, because you put the right text in front of it.

You leave this dive with the pipeline (chunk, embed, store, retrieve, blend, rerank, ground, cite), with the two-instrument diagnostic habit, and with a healthy suspicion of "strictly better." What you do not yet have is the measurement discipline to wield all these knobs responsibly at scale. That is not a gap in this chapter; it is the next one.

---

*Lab manual: [README.md](README.md) · Exercises: [EXERCISES.md](EXERCISES.md) · Previous: [Prompt Engineering](../prompt-engineering-deep-dive/TEXTBOOK.md) · Next: [Evals](../evals-deep-dive/TEXTBOOK.md)*
