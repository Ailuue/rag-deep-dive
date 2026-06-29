"""
Example 10 — contextual retrieval.
==================================

A chunk ripped out of its document often loses the context that makes it findable.
"The limit is 5 GB." — the limit on *what*, for *which plan*? Embedded alone, that
chunk matches almost nothing useful, because the words that would connect it to the
query ("Free plan storage") live in a different chunk.

**Contextual retrieval** (popularized by Anthropic) fixes this cheaply: before
embedding each chunk, prepend a short, LLM-written sentence that *situates the chunk
within its document*. Now the chunk's vector carries that context, so the query
finds it. The original chunk text is what you still show the model; the context is
only there to improve the *embedding*.

This script builds two stores over the same corpus — plain chunks vs.
context-prefixed chunks — and compares retrieval on a query whose answer lives in an
under-specified chunk.

(One LLM call per chunk at index time. That's a one-time ingest cost; a real system
caches the document so all its chunks share one cheap context pass.)

Run it:

    python examples/10_contextual_retrieval.py
    python examples/10_contextual_retrieval.py "how much storage on the free plan?"
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

QUERY = sys.argv[1] if len(sys.argv) > 1 else "how much storage on the free plan?"
K = 3


def build_plain():
    """Ordinary chunking: embed each chunk as-is."""
    return rag.index_documents(docs, chunk_size=120, overlap=20)


def context_for(chunk: str, full_doc: str, source: str) -> str:
    """One short sentence situating this chunk in its document."""
    return rag.generate(
        "You write one short sentence that situates a text chunk within its document, "
        "to improve search. Mention the document topic and what the chunk is about. "
        "Reply with only the sentence.",
        f"Document '{source}':\n{full_doc[:1500]}\n\nChunk:\n{chunk}\n\nContext sentence:",
        max_tokens=60,
    ).strip()


def build_contextual():
    """Prepend an LLM-written context sentence to each chunk *before embedding*.

    We embed `context + chunk` but keep the original `chunk` as the stored text, so
    the model still sees clean source text — only retrieval benefits.
    """
    texts, to_embed, metas = [], [], []
    for source, doc in docs:
        for i, chunk in enumerate(rag.chunk_text(doc, chunk_size=120, overlap=20)):
            ctx = context_for(chunk, doc, source)
            texts.append(chunk)                       # what we SHOW the model
            to_embed.append(f"{ctx}\n{chunk}")        # what we EMBED
            metas.append({"source": source, "chunk": i, "context": ctx})
    vectors = rag.embed(to_embed, input_type="document")
    store = rag.VectorStore()
    store.add(texts, vectors, metas)
    return store


def show(label: str, hits):
    print(f"[{label}]")
    for score, rec in hits:
        snippet = " ".join(rec.text.split()[:16])
        print(f"  {score:.3f}  ({rec.metadata.get('source','?')}) {snippet}...")
    print()


if __name__ == "__main__":
    print(f"Question: {QUERY}\n" + "=" * 70 + "\n")

    plain = build_plain()
    show("plain chunks", rag.retrieve(plain, QUERY, k=K))

    print("Building contextual index (one short LLM context call per chunk)...\n")
    contextual = build_contextual()
    show("contextual chunks (context prepended before embedding)",
         rag.retrieve(contextual, QUERY, k=K))

    print("=" * 70)
    print(
        "Takeaway: a chunk embedded in isolation can lose the words that make it\n"
        "findable. Prepending a one-sentence 'where this came from' before embedding\n"
        "(but still showing the model the clean chunk) is a cheap, high-leverage win —\n"
        "especially for short, under-specified passages."
    )
