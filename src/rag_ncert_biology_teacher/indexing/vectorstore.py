"""
WHAT
----
Persist Step 5's chunks + Step 4's text embeddings into a real, on-disk
vector DATABASE (Chroma) that supports efficient similarity search, instead
of the plain numpy arrays + hand-written argsort loop used in image_handling/.

WHY
---
Everything before this point (image_handling/) re-loaded vectors into memory
every run and searched them with a hand-written loop
(caption_vectors @ query_vector, np.argsort) - fine for a single chapter's
handful of images, but nothing was actually PERSISTED as a real database, and
it wouldn't scale to searching across all 13 chapters at once. Chroma stores
vectors + their metadata on disk (CHROMA_DIR) and gives us real similarity
search out of the box. We use LangChain's Chroma integration specifically
because Step 7's SQLRecordManager (incremental re-indexing without
duplicates) is built to work with exactly this Document + vectorstore
interface - using it now sets up Step 7 for free.

FLOW
----
  1. Wrap Step 4's embed_texts() in a small adapter class implementing
     LangChain's Embeddings interface (embed_documents/embed_query) - Chroma
     expects that interface, not a plain function, so this bridges the two
     without duplicating any embedding logic.
  2. get_vectorstore() opens (or creates) a Chroma collection persisted at
     CHROMA_DIR, using that adapter.
  3. add_chunks(vectorstore, chunks) -- assigns each chunk a DETERMINISTIC id
     (book/chapter/page + a content hash), so re-running this script doesn't
     blindly pile up duplicate copies of the same content. (Real
     duplicate-safe incremental indexing, e.g. "re-index only what changed
     in the PDF," is Step 7's job via SQLRecordManager - this is a
     reasonable stopgap so this step's own testing stays clean.)
  4. similarity_search(vectorstore, query, k) -- embed the query, search the
     persisted collection, return top-k chunks with their metadata intact.

LOGIC / MECHANISM
------------------
"Persisted" means: once chunks are added, they're written to files under
CHROMA_DIR/. A brand-new Python process that opens the SAME persist
directory sees the SAME data immediately - no re-embedding, no re-adding.

The deterministic id uses hashlib (NOT Python's built-in hash()) on purpose
-- Python randomizes hash() per process run for security, so the same text
would get a DIFFERENT id every time the script restarts, defeating the whole
point of de-duplication. hashlib.md5() always gives the same digest for the
same input, in any process, forever.
"""

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_ncert_biology_teacher.config import CHROMA_DIR
from rag_ncert_biology_teacher.indexing.embeddings import embed_texts

COLLECTION_NAME = "ncert_biology"


class VertexAIEmbeddings(Embeddings):
    """Adapts Step 4's embed_texts() to the Embeddings interface Chroma/LangChain expect."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0].tolist()


def get_vectorstore() -> Chroma:
    """Open (or create) the persisted Chroma collection at CHROMA_DIR."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=VertexAIEmbeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _chunk_id(chunk: Document) -> str:
    """A deterministic id so re-adding the SAME chunk doesn't create a
    duplicate -- built from what makes a chunk unique: which book/chapter/
    page it's from, plus a hash of its actual text (two chunks from the same
    page are still different chunks of that page's text).
    """
    meta = chunk.metadata
    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
    return f"{meta['book']}_ch{meta['chapter_number']:02d}_p{meta['page_number']:03d}_{content_hash}"


def add_chunks(vectorstore: Chroma, chunks: list[Document]) -> int:
    """Add chunks to the vectorstore. Returns how many were actually NEW
    (already-present ids are skipped, not re-embedded/re-added).
    """
    if not chunks:
        return 0
    ids = [_chunk_id(c) for c in chunks]
    existing_ids = set(vectorstore.get(ids=ids)["ids"])
    new_pairs = [(chunk, cid) for chunk, cid in zip(chunks, ids) if cid not in existing_ids]

    if new_pairs:
        new_chunks, new_ids = zip(*new_pairs)
        vectorstore.add_documents(list(new_chunks), ids=list(new_ids))

    return len(new_pairs)


if __name__ == "__main__":
    import json

    from rag_ncert_biology_teacher.config import RAW_PDF_DIR
    from rag_ncert_biology_teacher.ingestion.chunking import build_chapter_chunks

    # Usage: uv run python -m rag_ncert_biology_teacher.indexing.vectorstore
    book = "class12_biology"
    manifest = json.loads((RAW_PDF_DIR / book / "chapters.json").read_text())
    chapter_03 = next(c for c in manifest["chapters"] if c["number"] == 3)

    chunks = build_chapter_chunks(book, chapter_03)
    vectorstore = get_vectorstore()

    added = add_chunks(vectorstore, chunks)
    print(f"Chapter 3: {len(chunks)} chunks, {added} newly added to Chroma at {CHROMA_DIR}")

    print("\n--- Re-adding the SAME chunks (should add 0 -- dedup test) ---")
    added_again = add_chunks(vectorstore, chunks)
    print(f"Added: {added_again}")
    assert added_again == 0, "FAIL: re-adding identical chunks should be a no-op"
    print("PASS")

    print("\n--- Similarity search: 'How does tubal ligation work?' ---")
    for doc in vectorstore.similarity_search("How does tubal ligation work?", k=3):
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"\nPage {doc.metadata['page_number']} (Ch.{doc.metadata['chapter_number']}): {preview}")
