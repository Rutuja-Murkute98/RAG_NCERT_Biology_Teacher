"""
WHAT
----
Turn each chapter's page-level text (+ that page's image captions) into
small, retrieval-sized CHUNKS, each tagged with metadata (book, chapter,
page number, and that page's image paths) so a retrieved chunk can always be
traced back to exactly where it came from - and to its diagrams, if it has any.

WHY
---
Embedding a whole page squashes multiple unrelated topics into one blurry
vector that doesn't sharply match any single question, and retrieval would
hand the LLM entire pages of mostly-irrelevant text instead of just the
relevant paragraph. Splitting into smaller chunks fixes both: each chunk
covers one focused idea, and only the genuinely relevant piece gets
retrieved. Each page's USEFUL captions (Step 4's caption_retrieval.py,
already dropping noise Step 3's marker flagged) are folded in as extra text,
so a chunk can be found by a question about its diagram too, even outside
the dedicated image_handling/ pipeline.

FLOW
----
  1. Load a chapter's cached page text (text_extraction.build_text_index)
     and cached captions (caption_retrieval.build_caption_index, cache-only
     once a chapter's already been captioned -- makes zero new Gemini calls).
  2. Build ONE Document per PAGE: page text + "[Figure: caption]" for every
     image on that page, tagged with metadata. A page can have zero, one, or
     several diagrams -- unlike a "one image per page" scheme, so images are
     grouped by page number (parsed back out of each filename via
     pdf_loader.image_page_number()) rather than assumed to line up 1:1 with pages.
  3. Run RecursiveCharacterTextSplitter over those page-Documents -- it tries
     to break at paragraph boundaries first, then sentences, then words, only
     falling back to a hard character cut as a last resort. LangChain
     automatically COPIES each page-Document's metadata onto every chunk it
     produces, so every chunk still knows its book/chapter/page/images.
  4. Return the full list of chunks across all pages of the chapter.

LOGIC / MECHANISM
------------------
CHUNK_SIZE=1000 / CHUNK_OVERLAP=150 (config.py): aim for ~1000-character
chunks, but let each new chunk start 150 characters BEFORE the previous one
ended, so a sentence that lands right at a chunk boundary isn't orphaned
from its surrounding context in both neighboring chunks.

image_paths metadata is stored as a JSON-encoded STRING, not a raw Python
list -- Chroma's metadata only supports scalar values (str/int/float/bool),
not lists, so a page's (possibly several) image paths are json.dumps()'d
here and json.loads()'d back wherever they're consumed later (the RAG chain,
the UI).
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_ncert_biology_teacher.config import CHUNK_OVERLAP, CHUNK_SIZE, EXTRACTED_DIR, RAW_PDF_DIR
from rag_ncert_biology_teacher.image_handling.caption_retrieval import build_caption_index
from rag_ncert_biology_teacher.ingestion.pdf_loader import image_page_number
from rag_ncert_biology_teacher.ingestion.text_extraction import build_text_index


def build_page_documents(book: str, chapter: dict) -> list[Document]:
    """One Document per PAGE (not yet chunked): page text + its captions,
    tagged with metadata. This is the input chunk_documents() below splits.
    """
    pdf_path = RAW_PDF_DIR / book / chapter["file"]
    chapter_key = pdf_path.stem
    chapter_dir = EXTRACTED_DIR / book / chapter_key

    texts = build_text_index(pdf_path, cache_path=chapter_dir / "text_index.json")
    image_paths, captions, _vectors = build_caption_index(
        book, chapter_key, cache_path=chapter_dir / "caption_index.json"
    )

    images_by_page: dict[int, list[tuple[Path, str]]] = {}
    for image_path, caption in zip(image_paths, captions):
        images_by_page.setdefault(image_page_number(image_path), []).append((image_path, caption))

    documents = []
    for page_number, text in enumerate(texts, start=1):
        page_images = images_by_page.get(page_number, [])
        combined_text = text
        for _image_path, caption in page_images:
            combined_text += f"\n\n[Figure on this page: {caption}]"

        documents.append(
            Document(
                page_content=combined_text,
                metadata={
                    "book": book,
                    "chapter_number": chapter["number"],
                    "chapter_title": chapter["title"],
                    "page_number": page_number,
                    "image_paths": json.dumps([str(p) for p, _c in page_images]),
                    # One stable id per CHAPTER (not per page/chunk) -- what
                    # SQLRecordManager's incremental cleanup groups by later.
                    "source": f"{book}/{chapter_key}",
                },
            )
        )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split page-level Documents into retrieval-sized chunks. Metadata from
    each source page-Document is automatically copied onto every chunk it produces.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)


def build_chapter_chunks(book: str, chapter: dict) -> list[Document]:
    """The full chunking pipeline for ONE chapter: pages -> page-Documents -> chunks."""
    page_documents = build_page_documents(book, chapter)
    return chunk_documents(page_documents)


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.ingestion.chunking
    book = "class12_biology"
    manifest = json.loads((RAW_PDF_DIR / book / "chapters.json").read_text())
    chapter_03 = next(c for c in manifest["chapters"] if c["number"] == 3)

    chunks = build_chapter_chunks(book, chapter_03)

    print(f"\nChapter 3 -> {len(chunks)} chunks\n")
    with_images = [c for c in chunks if json.loads(c.metadata["image_paths"])]
    print(f"{len(with_images)} chunks have at least one figure folded in\n")

    print("--- Sample chunk with a figure ---")
    sample = with_images[0]
    print(f"Text ({len(sample.page_content)} chars):\n{sample.page_content[:500]}")
    print(f"\nMetadata:\n{sample.metadata}")
