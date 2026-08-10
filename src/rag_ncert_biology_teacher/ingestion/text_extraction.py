"""Extract each PDF page's raw TEXT and cache it to disk, page-by-page - the
text-side counterpart to pdf_loader.py's image-side pipeline, kept
deliberately separate so chunking doesn't have to re-pay for image
extraction just to get text.

Why a separate module instead of reusing pdf_loader.load_pdf()'s
PageContent.text: load_pdf() also runs the parallel, subprocess-guarded
vector-diagram detection (pdf_loader.PAGE_TIMEOUT_SECONDS per page) - by far
the expensive part of that pipeline (the whole-book run took ~15 minutes).
Chunking only needs plain page text, which is pure local PyMuPDF parsing
with no rate limits and no per-page timeout risk - re-running the expensive
half just to get text again would be pure waste.
"""

import json
from pathlib import Path

import fitz  # PyMuPDF


def extract_page_texts(pdf_path: Path) -> list[str]:
    """Pull raw text for every page, in order. Page N's text is texts[N-1]."""
    texts = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            texts.append(page.get_text().strip())
    finally:
        doc.close()
    return texts


def build_text_index(pdf_path: Path, cache_path: Path | None = None) -> list[str]:
    """Extract every page's text, cached to a JSON file for reuse.
    Returns a list of strings, one per page (1-indexed page N is texts[N-1]).
    """
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        # Sanity check: only trust the cache if its page count matches the
        # PDF's actual page count -- a stale cache silently returning
        # wrong-length text would corrupt every downstream step.
        doc = fitz.open(pdf_path)
        n_pages = doc.page_count
        doc.close()
        if len(cached["texts"]) == n_pages:
            print(f"Loaded cached text for {len(cached['texts'])} pages from {cache_path}")
            return cached["texts"]

    print(f"Extracting text from {pdf_path.name}...")
    texts = extract_page_texts(pdf_path)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"page_numbers": list(range(1, len(texts) + 1)), "texts": texts}, indent=2)
        )
        print(f"Saved text cache to {cache_path} ({len(texts)} pages)")

    return texts


if __name__ == "__main__":
    from rag_ncert_biology_teacher.config import EXTRACTED_DIR, RAW_PDF_DIR

    # Usage: uv run python -m rag_ncert_biology_teacher.ingestion.text_extraction
    pdf_path = RAW_PDF_DIR / "class12_biology" / "chapter_03.pdf"
    cache_path = EXTRACTED_DIR / "class12_biology" / "chapter_03" / "text_index.json"

    texts = build_text_index(pdf_path, cache_path=cache_path)
    print(f"\nPages: {len(texts)}")
    print("--- Page 1 preview (first 300 chars) ---")
    print(texts[0][:300])
