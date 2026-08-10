"""Index the entire NCERT Class 12 Biology book (all 13 chapters) into
Chroma, via SQLRecordManager's incremental indexing (Step 7).

This is the real production data-loading step: captions every extracted
diagram (Step 3), chunks every chapter's text + captions (Step 5), and
indexes everything chapter by chapter (Step 7). Resumable at every layer:
captions/page-text are cached to disk per chapter (so a crash mid-chapter
loses at most one image's captioning work, not the whole run), and
SQLRecordManager itself tracks what's already in Chroma, so simply
re-running this script after an interruption picks up where it left off
instead of starting from zero or re-paying for anything already done.

Usage: uv run python scripts/index_all_chapters.py
"""

import json
import sys

from rag_ncert_biology_teacher.config import RAW_PDF_DIR
from rag_ncert_biology_teacher.indexing.indexer import index_chunks
from rag_ncert_biology_teacher.ingestion.chunking import build_chapter_chunks

# Windows' default console codepage (cp1252) can't print every character Gemini
# legitimately outputs in a caption (e.g. u2640 "female sign") -- this crashed a
# real run partway through Chapter 4. Force UTF-8 stdout so printing never crashes
# a caption that was already successfully generated (and saved) beforehand.
sys.stdout.reconfigure(encoding="utf-8")


def index_book(book: str = "class12_biology") -> None:
    manifest = json.loads((RAW_PDF_DIR / book / "chapters.json").read_text())
    chapters = manifest["chapters"]

    totals = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}
    for chapter in chapters:
        print(f"\n{'=' * 60}")
        print(f"Chapter {chapter['number']}: {chapter['title']}")
        print(f"{'=' * 60}", flush=True)

        chunks = build_chapter_chunks(book, chapter)
        result = index_chunks(chunks)
        print(f"  -> {len(chunks)} chunks | {result}", flush=True)

        for key in totals:
            totals[key] += result[key]

    print(f"\n{'=' * 60}")
    print(f"WHOLE BOOK DONE: {totals}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    index_book()
