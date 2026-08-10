"""
WHAT
----
A thin retriever wrapper around Chroma's similarity_search() (Steps 6-7),
returning the top-k chunks most relevant to a student's question, with
their metadata intact.

WHY
---
Steps 6-7 already proved Chroma + incremental indexing work correctly. This
doesn't change any of that - it just gives the RAG chain (chain.py) a
single, simple function to call, instead of reaching into vectorstore.py
directly. Keeping this as its own thin module means the retrieval STRATEGY
(how many chunks, re-ranking, filtering by chapter, etc.) can evolve later
without touching the prompt or generation logic in chain.py/prompts.py.

FLOW
----
  1. Open the persisted Chroma collection (get_vectorstore(), Step 6).
  2. retrieve_with_scores(question, k) -> vectorstore.similarity_search_with_relevance_scores() --
     embeds the question with the SAME model used to embed every chunk
     (Step 4's text-embedding-005, enforced by vectorstore.py's
     VertexAIEmbeddings adapter), so the comparison is meaningful.
  3. Return (Document, relevance_score) pairs -- chain.py uses these scores
     only loosely (see prompts.py's WHY for why NOT as a hard threshold).

LOGIC / MECHANISM
------------------
This file is a deliberately thin seam between "how retrieval works" (Chroma,
already built and verified) and "how retrieval is USED" (chain.py), so each
can be reasoned about and changed independently.

Optional chapter_number FILTER: the chat UI lets a student pick one chapter
to focus on instead of searching the whole book. This uses Chroma's own
metadata filter (`filter={"chapter_number": N}`) -- every chunk already
carries chapter_number in its metadata (chunking.py, Step 5), so no new
indexing work is needed, just passing a `where` clause through to Chroma.
"""

from langchain_core.documents import Document

from rag_ncert_biology_teacher.indexing.vectorstore import get_vectorstore

DEFAULT_K = 4


def retrieve_with_scores(
    question: str, k: int = DEFAULT_K, chapter_number: int | None = None
) -> list[tuple[Document, float]]:
    """Return the top-k (chunk, relevance_score) pairs for a question.
    If chapter_number is given, only that chapter's chunks are searched --
    otherwise the whole book is searched.
    """
    vectorstore = get_vectorstore()
    filter_ = {"chapter_number": chapter_number} if chapter_number is not None else None
    return vectorstore.similarity_search_with_relevance_scores(question, k=k, filter=filter_)


def retrieve(question: str, k: int = DEFAULT_K, chapter_number: int | None = None) -> list[Document]:
    """Return just the top-k chunks (no scores) most relevant to a question."""
    return [doc for doc, _score in retrieve_with_scores(question, k=k, chapter_number=chapter_number)]


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.rag.retriever
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    questions = [
        "How does tubal ligation work?",
        "What is natural selection?",
        "How does a biogas plant work?",
    ]
    for question in questions:
        print(f"\nQuestion: {question!r}")
        for doc, score in retrieve_with_scores(question):
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  score={score:.3f}  Ch.{doc.metadata['chapter_number']} p.{doc.metadata['page_number']}: {preview}")
