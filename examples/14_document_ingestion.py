"""
Example 14 — document ingestion: from messy sources to clean, structured chunks.
================================================================================

The corpus here is tidy Markdown, but real documents arrive as PDFs, HTML pages,
Word files, and wiki exports. Ingestion is the unglamorous step that turns any of
them into the `(source, text)` the pipeline wants — and doing it *well* (respecting
the document's structure) improves every downstream retrieval.

This script shows three ingestion ideas you can apply to real corpora:

  1. STRUCTURE-AWARE CHUNKING. Instead of a blind sliding window, split on the
     document's own headings — the same `chunk_markdown_sections()` that example
     13 introduced. Each section becomes a chunk that's about one thing, and its
     heading becomes metadata — so retrieval can cite "Billing > Refunds."

  2. PARSING NON-MARKDOWN. A quick HTML→text pass with the standard library shows
     that "ingest a web page" reduces to the same `(source, text)` shape. (PDFs are
     the same idea with `pdfplumber`/`pypdf`; we note it rather than add a dep.)

  3. CLEANING. Collapse whitespace, drop boilerplate — small hygiene that stops
     junk tokens from polluting your chunks and your bill.

The first two parts run **offline** (pure parsing). The indexing demo at the end
needs a provider key like the other examples.

Run it:

    secrun python examples/14_document_ingestion.py
"""

import os
import re
import sys
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import rag

load_dotenv()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Structure-aware splitting lives in the library now (rag/chunking.py), shared
# with example 13 — ingestion's job is to *feed* it clean text from any format.
split_markdown_sections = rag.chunk_markdown_sections


# --- 2. Parse non-Markdown: HTML -> text with the standard library -----------
class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return clean(" ".join(parser.parts))


SAMPLE_HTML = """
<html><head><style>.x{color:red}</style></head><body>
  <h1>Refund Policy</h1>
  <p>Refunds are issued within <b>30 days</b> of purchase.</p>
  <script>track('view')</script>
  <p>Contact support@example.com to request one.</p>
</body></html>
"""


# --- 3. Cleaning -------------------------------------------------------------
def clean(text: str) -> str:
    """Collapse runs of whitespace; trim. The cheapest quality win in ingestion."""
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    print("=" * 70)
    print("1) STRUCTURE-AWARE CHUNKING (split on Markdown headings)\n")
    docs = rag.load_corpus(os.path.join(REPO_ROOT, "corpus"))
    source, text = docs[0]
    sections = split_markdown_sections(text)
    print(f"{source} -> {len(sections)} heading-based sections:")
    for heading, body in sections[:6]:
        print(f"  • [{heading}]  {clean(body)[:55]}...")
    print("\n-> Each section is about ONE thing, and its heading becomes metadata you\n"
          "   can retrieve on and cite. Compare to a blind word-window that splits\n"
          "   mid-topic.\n")

    print("=" * 70)
    print("2) PARSING NON-MARKDOWN (HTML -> text, standard library)\n")
    print("Raw HTML in, clean text out (scripts/styles dropped):")
    print(f"  {html_to_text(SAMPLE_HTML)}")
    print("\n-> A web page became the same (source, text) the pipeline ingests. PDFs are\n"
          "   the same move with pdfplumber/pypdf; Word with python-docx. The pipeline\n"
          "   doesn't care where the text came from.\n")

    print("=" * 70)
    print("3) INDEXING SECTION-CHUNKS WITH HEADING METADATA (needs a provider key)\n")
    rag.ensure_ready()
    texts, metas = [], []
    for src, doc in docs:
        for heading, body in split_markdown_sections(doc):
            texts.append(clean(body))
            metas.append({"source": src, "heading": heading})
    store = rag.VectorStore()
    store.add(texts, rag.embed(texts, input_type="document"), metas)
    print(f"Indexed {len(store)} section-chunks.\n")

    query = "how long do refunds take?"
    print(f'Query: "{query}"  ->  top sections (note the heading citations):')
    for score, rec in store.search(rag.embed([query], input_type="query")[0], k=3):
        print(f"  {score:.3f}  {rec.metadata['source']} > {rec.metadata['heading']}")

    print("\n" + "=" * 70)
    print(
        "Takeaway: ingestion is where retrieval quality is won or lost. Respect the\n"
        "document's structure (headings -> sections -> metadata), parse each format\n"
        "down to clean text, and tidy whitespace. Garbage in, garbage retrieval out."
    )
