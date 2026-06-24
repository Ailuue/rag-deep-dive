"""
rag — a small, from-scratch retrieval-augmented-generation library.

Everything here is built to be *read*, not just used. Four modules, each one
moving part of RAG:

  providers.py  — the ONLY file that talks to an LLM provider (embed + generate)
  chunking.py   — splitting documents into retrievable pieces
  store.py      — an in-memory vector store (brute-force cosine search)
  pipeline.py   — tying it together: index -> retrieve -> answer

Import the pieces you need, e.g.:

    from rag import index_documents, answer

or reach into a specific module to see how it works.
"""

from .chunking import chunk_paragraphs, chunk_text
from .loader import load_corpus
from .pipeline import GROUNDED_SYSTEM, answer, build_prompt, index_documents, retrieve
from .providers import describe, embed, ensure_ready, generate, provider_name
from .store import VectorStore, cosine_similarity

__all__ = [
    "chunk_text",
    "chunk_paragraphs",
    "load_corpus",
    "VectorStore",
    "cosine_similarity",
    "embed",
    "generate",
    "provider_name",
    "describe",
    "ensure_ready",
    "index_documents",
    "retrieve",
    "answer",
    "build_prompt",
    "GROUNDED_SYSTEM",
]
