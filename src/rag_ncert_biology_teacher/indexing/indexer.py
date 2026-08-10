"""
WHAT
----
Wrap Step 6's Chroma writes with LangChain's official index() API +
SQLRecordManager, so re-indexing a chapter is genuinely INCREMENTAL --
unchanged chunks are skipped (no re-embedding, no re-add), new/changed
chunks are added, and chunks that no longer exist for that chapter (e.g. a
page got removed or reworded in the PDF) are actively DELETED from Chroma.
This becomes the REAL indexing entrypoint from here on, superseding Step 6's
add_chunks() (which was only ever a stopgap to test vectorstore.py in isolation).

WHY
---
Step 6's add_chunks() used a manual stopgap: deterministic ids so re-running
with IDENTICAL content doesn't duplicate. That's real, but limited -- it can
only ever ADD or skip; it has no way to notice "this chunk used to exist for
this chapter but doesn't anymore" and remove it. If a chapter's PDF were
re-extracted with different chunking, Step 6's approach would leave the OLD,
now-wrong chunks sitting in Chroma forever alongside the new ones.
SQLRecordManager fixes this: it keeps a content-hash ledger, per source, in
its own SQLite database (RECORD_MANAGER_DB, separate from Chroma), so
index() can tell the difference between "new," "unchanged," and "stale --
delete it."

FLOW
----
  1. SQLRecordManager tracks, per SOURCE (one chapter = one source, via each
     chunk's `source` metadata field set in chunking.py), a content hash +
     timestamp for every document it has ever indexed.
  2. index(chunks, record_manager, vectorstore, cleanup="incremental",
     source_id_key="source") compares the incoming chunks' hashes against
     what it already knows for that exact source:
       - unseen hash            -> added to Chroma, recorded
       - seen, unchanged hash   -> skipped entirely (no re-embed, no re-add)
       - recorded for this source but NOT in this call -> DELETED from Chroma
  3. Returns counts (num_added, num_updated, num_skipped, num_deleted) -- an
     honest, exact audit of what actually changed on this run.

LOGIC / MECHANISM
------------------
"incremental" cleanup only ever compares/deletes WITHIN the same source_id
being re-indexed in THIS call -- it will never touch or delete another
chapter's chunks, because every chapter's chunks share one `source` value
distinct from every other chapter's (chunking.py sets this to
"{book}/{chapter_key}"). LangChain also offers "full" cleanup, which treats
the WHOLE call as authoritative for the ENTIRE Chroma collection -- passing
just one chapter's chunks with "full" cleanup would DELETE every other
chapter's chunks. We deliberately never use that mode per-chapter; it would
only be safe if every chapter were indexed together in one single call.
"""

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index

from rag_ncert_biology_teacher.config import RECORD_MANAGER_DB
from rag_ncert_biology_teacher.indexing.vectorstore import COLLECTION_NAME, get_vectorstore

_record_manager: SQLRecordManager | None = None


def get_record_manager() -> SQLRecordManager:
    """Lazily create the record manager once (and its SQLite schema on first
    use), reused for every index_chunks() call after that.
    """
    global _record_manager
    if _record_manager is None:
        _record_manager = SQLRecordManager(
            namespace=f"chroma/{COLLECTION_NAME}",
            db_url=f"sqlite:///{RECORD_MANAGER_DB}",
        )
        _record_manager.create_schema()
    return _record_manager


def index_chunks(chunks: list[Document]):
    """Incrementally index one chapter's worth of chunks. All chunks passed
    in ONE call must share the same `source` metadata value (see
    chunking.py) -- that's the boundary "incremental" cleanup respects.
    Returns LangChain's IndexingResult: {num_added, num_updated, num_skipped, num_deleted}.
    """
    vectorstore = get_vectorstore()
    record_manager = get_record_manager()

    return index(
        chunks,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source",
        key_encoder="sha256",  # LangChain warns SHA-1 (its default) isn't collision-resistant
    )


if __name__ == "__main__":
    import json

    from rag_ncert_biology_teacher.config import RAW_PDF_DIR
    from rag_ncert_biology_teacher.ingestion.chunking import build_chapter_chunks

    # Usage: uv run python -m rag_ncert_biology_teacher.indexing.indexer
    book = "class12_biology"
    manifest = json.loads((RAW_PDF_DIR / book / "chapters.json").read_text())
    chapter_03 = next(c for c in manifest["chapters"] if c["number"] == 3)

    chunks = build_chapter_chunks(book, chapter_03)
    print(f"Chapter 3 -> {len(chunks)} chunks\n")

    print("--- Run 1: first index of chapter 3 ---")
    result1 = index_chunks(chunks)
    print(result1)
    assert result1["num_added"] == len(chunks), "FAIL: first run should add every chunk"
    print("PASS: every chunk added\n")

    print("--- Run 2: SAME chunks again (should be a total no-op) ---")
    result2 = index_chunks(chunks)
    print(result2)
    assert result2["num_added"] == 0 and result2["num_updated"] == 0, "FAIL: identical re-run was not a no-op"
    assert result2["num_skipped"] == len(chunks)
    print("PASS: nothing added/updated, everything correctly skipped\n")

    print("--- Run 3: FEWER chunks for the SAME source (proves real deletion, not just add/skip) ---")
    fewer_chunks = chunks[:20]
    result3 = index_chunks(fewer_chunks)
    print(result3)
    assert result3["num_deleted"] == len(chunks) - len(fewer_chunks), "FAIL: dropped chunks should be deleted"
    print(f"PASS: {result3['num_deleted']} stale chunks deleted from Chroma\n")

    print("--- Run 4: back to the FULL chunk set (proves the deleted ones come back) ---")
    result4 = index_chunks(chunks)
    print(result4)
    assert result4["num_added"] == len(chunks) - len(fewer_chunks), "FAIL: previously-deleted chunks should re-add"
    print("PASS: previously-deleted chunks are back")
