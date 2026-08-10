import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from rag_ncert_biology_teacher.config import EXTRACTED_DIR, RAW_PDF_DIR


@dataclass
class PageContent:
    page_number: int
    text: str
    image_paths: list[str] = field(default_factory=list)


def list_chapter_images(book: str, chapter_key: str) -> list[Path]:
    """Every already-extracted diagram/photo for one chapter, sorted by filename
    (which sorts by page number first, since files are named page_NNN_...).
    Used by image_handling/ - none of it re-derives images from the PDF.
    """
    images_dir = EXTRACTED_DIR / book / chapter_key / "images"
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file())


def image_page_number(image_path: Path) -> int:
    """Parse the page number back out of an extracted image's filename
    (e.g. "page_012_diagram_0.png" -> 12) - every image is named this way
    by load_pdf() below.
    """
    return int(image_path.stem.split("_")[1])


def render_page_thumbnail(book: str, chapter_key: str, page_number: int, dpi: int = 150) -> Path:
    """Render ONE whole PDF page as a flat image and cache it to disk, purely
    for use as decorative background art in the UI (app/server.py's
    _thumbnail_urls()) - never used for the actual diagrams shown in answers,
    which stay precisely cropped (pdf_loader.load_pdf() above). Matches the
    reference project's own whole-page thumbnail look exactly.
    """
    out_path = EXTRACTED_DIR / book / chapter_key / "pages" / f"page_{page_number:03d}.png"
    if out_path.exists():
        return out_path

    pdf_path = RAW_PDF_DIR / book / f"{chapter_key}.pdf"
    zoom = dpi / 72  # PDF's native unit is 72 dpi; scale up for a sharper render
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(out_path)
    finally:
        doc.close()

    return out_path


@dataclass
class ChapterStats:
    total_pages: int = 0
    raster_found: int = 0
    raster_kept: int = 0
    vector_regions_extracted: int = 0
    vector_pages_with_diagrams: int = 0
    vector_pages_timed_out: int = 0

    @property
    def raster_discarded(self) -> int:
        return self.raster_found - self.raster_kept

    @property
    def total_kept(self) -> int:
        return self.raster_kept + self.vector_regions_extracted


MIN_IMAGE_DIMENSION = 200  # px, both sides - filters small icons/logos
MAX_ASPECT_RATIO = 4.0  # filters thin banner/header strips
MIN_BYTES_PER_PIXEL = 0.02  # filters near-solid-color background fills, N/A to GIF (see below)


def _looks_like_real_diagram(width: int, height: int, num_bytes: int, ext: str) -> bool:
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return False
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        return False
    # GIF's palette+LZW compression makes simple line art shrink as much as a flat background
    # fill does, so the density check would wrongly reject real GIF diagrams - skip it for GIF.
    if ext != "gif" and num_bytes / (width * height) < MIN_BYTES_PER_PIXEL:
        return False
    return True


# --- Pure vector diagrams (no embedded raster image) ---
# NCERT diagrams are often drawn as vector paths, not raster pictures, so get_images() alone
# misses them. We cluster nearby vector paths into regions and rasterize each region only -
# never the whole page - so captions and retrieval stay focused on the actual diagram.
VECTOR_GRID_CELL_SIZE = 12.0  # points - coarser merges nearby fragments (gradients, shading) into one region
MIN_REGION_SIZE = 60.0  # points, both sides - filters stray rules/underlines/bullets
MAX_REGION_ASPECT_RATIO = 6.0  # filters full-width banner/divider bars
MIN_DRAWING_OPACITY = 0.5  # filters faint watermark washes, which can bridge unrelated regions
REGION_PADDING = 12.0  # points, margin so nearby text labels aren't cropped off
REGION_RENDER_DPI = 120
MAX_TEXT_COVERAGE_RATIO = 0.35  # rejects info/callout boxes (mostly text) posing as diagrams
MAX_REGIONS_PER_PAGE = 4  # a real page rarely has more distinct figures than this; caps render cost
MIN_REGION_PNG_DENSITY = 0.10  # bytes/pixel after render - rejects flat decorative badges/banners

# PyMuPDF's get_drawings() can hang indefinitely on certain malformed pages, and clustering
# cost grows with path count (some pages have 50,000+ paths) - run each page in a killable
# subprocess so one bad page can only ever cost us that one page, never stall the book.
PAGE_TIMEOUT_SECONDS = 75
MAX_PARALLEL_WORKERS = 4


def _find_diagram_regions(page: fitz.Page) -> list[fitz.Rect]:
    drawings = [
        d
        for d in page.get_drawings()
        if (d.get("fill_opacity") or 1.0) >= MIN_DRAWING_OPACITY
        and (d.get("stroke_opacity") or 1.0) >= MIN_DRAWING_OPACITY
    ]

    # Some diagrams are assembled from many tiny raster tiles (print-style gradient shading)
    # rather than vector paths - individually each tile is too small to keep, but together
    # they form a real figure. Feed small raster fragments into the same clustering as ink.
    raster_fragment_rects = []
    for info in page.get_images(full=True):
        bbox = page.get_image_bbox(info)
        if bbox.width < MIN_REGION_SIZE or bbox.height < MIN_REGION_SIZE:
            raster_fragment_rects.append(bbox)

    if not drawings and not raster_fragment_rects:
        return []

    page_rect = page.rect
    cols = int(page_rect.width // VECTOR_GRID_CELL_SIZE) + 1
    rows = int(page_rect.height // VECTOR_GRID_CELL_SIZE) + 1

    # Some PDFs also render body text as vector paths - mask real text words out so those can't bridge into diagrams.
    text_occupied = [[False] * cols for _ in range(rows)]
    for x0, y0, x1, y1, *_rest in page.get_text("words"):
        c0 = max(0, int(x0 // VECTOR_GRID_CELL_SIZE))
        c1 = min(cols - 1, int(x1 // VECTOR_GRID_CELL_SIZE))
        r0 = max(0, int(y0 // VECTOR_GRID_CELL_SIZE))
        r1 = min(rows - 1, int(y1 // VECTOR_GRID_CELL_SIZE))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                text_occupied[r][c] = True

    # Prefix-sum table so any region's text-cell count is an O(1) lookup, not a per-region rescan.
    text_prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            text_prefix[r + 1][c + 1] = (
                text_prefix[r][c + 1]
                + text_prefix[r + 1][c]
                - text_prefix[r][c]
                + (1 if text_occupied[r][c] else 0)
            )

    def _text_cell_count(r0: int, c0: int, r1: int, c1: int) -> int:
        return (
            text_prefix[r1 + 1][c1 + 1]
            - text_prefix[r0][c1 + 1]
            - text_prefix[r1 + 1][c0]
            + text_prefix[r0][c0]
        )

    occupied = [[False] * cols for _ in range(rows)]
    ink_rects = [d["rect"] for d in drawings] + raster_fragment_rects
    for rect in ink_rects:
        c0 = max(0, int(rect.x0 // VECTOR_GRID_CELL_SIZE))
        c1 = min(cols - 1, int(rect.x1 // VECTOR_GRID_CELL_SIZE))
        r0 = max(0, int(rect.y0 // VECTOR_GRID_CELL_SIZE))
        r1 = min(rows - 1, int(rect.y1 // VECTOR_GRID_CELL_SIZE))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not text_occupied[r][c]:
                    occupied[r][c] = True

    visited = [[False] * cols for _ in range(rows)]
    neighbor_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    regions: list[fitz.Rect] = []

    for start_r in range(rows):
        for start_c in range(cols):
            if not occupied[start_r][start_c] or visited[start_r][start_c]:
                continue

            stack = [(start_r, start_c)]
            visited[start_r][start_c] = True
            min_r = max_r = start_r
            min_c = max_c = start_c

            while stack:
                r, c = stack.pop()
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)
                for dr, dc in neighbor_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if occupied[nr][nc] and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

            # Info/callout boxes have a vector border but are mostly real text inside it;
            # real diagrams are mostly drawn linework with only sparse text labels.
            bbox_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
            text_cells = _text_cell_count(min_r, min_c, max_r, max_c)
            if text_cells / bbox_cells > MAX_TEXT_COVERAGE_RATIO:
                continue

            regions.append(
                fitz.Rect(
                    min_c * VECTOR_GRID_CELL_SIZE,
                    min_r * VECTOR_GRID_CELL_SIZE,
                    (max_c + 1) * VECTOR_GRID_CELL_SIZE,
                    (max_r + 1) * VECTOR_GRID_CELL_SIZE,
                )
            )

    return regions


def _render_page_worker(
    pdf_path: str, page_index: int, page_number: int, images_dir: str, result_queue: mp.Queue
) -> None:
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    candidates = []
    for region in _find_diagram_regions(page):
        width, height = region.width, region.height
        if width < MIN_REGION_SIZE or height < MIN_REGION_SIZE:
            continue
        if max(width, height) / min(width, height) > MAX_REGION_ASPECT_RATIO:
            continue
        candidates.append(region)

    # Over-fragmentation (gradient shading splitting into many small pieces) is far more
    # common than a page genuinely holding many distinct figures - keep only the largest.
    candidates.sort(key=lambda r: r.width * r.height, reverse=True)
    candidates = candidates[:MAX_REGIONS_PER_PAGE]

    saved_paths: list[str] = []
    for region_position, region in enumerate(candidates):
        padded = fitz.Rect(
            region.x0 - REGION_PADDING,
            region.y0 - REGION_PADDING,
            region.x1 + REGION_PADDING,
            region.y1 + REGION_PADDING,
        ) & page.rect

        pixmap = page.get_pixmap(dpi=REGION_RENDER_DPI, clip=padded)
        png_bytes = pixmap.tobytes("png")

        # Flat decorative badges/banners compress to almost nothing per pixel; real
        # diagrams (lines, labels, color) don't - same principle as the raster noise filter.
        if len(png_bytes) / (pixmap.width * pixmap.height) < MIN_REGION_PNG_DENSITY:
            continue

        image_path = Path(images_dir) / f"page_{page_number:03d}_diagram_{region_position}.png"
        image_path.write_bytes(png_bytes)
        saved_paths.append(str(image_path))

    doc.close()
    result_queue.put(saved_paths)


def _render_pages_parallel(
    pdf_path: Path, page_numbers: list[int], images_dir: Path
) -> tuple[dict[int, list[str]], int]:
    ctx = mp.get_context("spawn")
    results: dict[int, list[str]] = {}
    timed_out = 0
    pending = list(page_numbers)

    while pending:
        batch = pending[:MAX_PARALLEL_WORKERS]
        pending = pending[MAX_PARALLEL_WORKERS:]

        queues: dict[int, mp.Queue] = {}
        processes: dict[int, mp.Process] = {}
        for page_number in batch:
            queue: mp.Queue = ctx.Queue()
            process = ctx.Process(
                target=_render_page_worker,
                args=(str(pdf_path), page_number - 1, page_number, str(images_dir), queue),
            )
            process.start()
            queues[page_number] = queue
            processes[page_number] = process

        deadline = time.time() + PAGE_TIMEOUT_SECONDS
        for page_number, process in processes.items():
            process.join(timeout=max(0.0, deadline - time.time()))
            if process.is_alive():
                process.terminate()
                process.join()
                timed_out += 1
                print(f"  [warning] page {page_number}: diagram check timed out, skipped", flush=True)
                continue

            queue = queues[page_number]
            result = queue.get() if not queue.empty() else []
            if result:
                results[page_number] = result

    return results, timed_out


def load_pdf(relative_pdf_path: str) -> tuple[list[PageContent], ChapterStats]:
    pdf_path = RAW_PDF_DIR / relative_pdf_path
    book_key = Path(relative_pdf_path).parent.name
    chapter_key = Path(relative_pdf_path).stem

    images_dir = EXTRACTED_DIR / book_key / chapter_key / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageContent] = []
    stats = ChapterStats()
    page_numbers: list[int] = []

    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        # Chapter opening pages are title/QR/TOC decoration, not study diagrams - skip them.
        if page_number != 1:
            page_numbers.append(page_number)
        text = page.get_text().strip()

        image_paths: list[str] = []
        for image_position, image_info in enumerate(page.get_images(full=True)):
            xref = image_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            stats.raster_found += 1

            if not _looks_like_real_diagram(
                base_image["width"], base_image["height"], len(image_bytes), base_image["ext"]
            ):
                continue

            image_filename = f"page_{page_number:03d}_img_{image_position}.{base_image['ext']}"
            image_path = images_dir / image_filename
            image_path.write_bytes(image_bytes)
            image_paths.append(str(image_path))
            stats.raster_kept += 1

        pages.append(PageContent(page_number=page_number, text=text, image_paths=image_paths))
    doc.close()

    diagram_results, timed_out = _render_pages_parallel(pdf_path, page_numbers, images_dir)
    stats.vector_pages_with_diagrams = len(diagram_results)
    stats.vector_regions_extracted = sum(len(paths) for paths in diagram_results.values())
    stats.vector_pages_timed_out = timed_out
    stats.total_pages = len(pages)

    pages_by_number = {page.page_number: page for page in pages}
    for page_number, image_paths_found in diagram_results.items():
        pages_by_number[page_number].image_paths.extend(image_paths_found)

    return pages, stats


def load_book(book_key: str) -> dict[str, tuple[list[PageContent], ChapterStats]]:
    book_dir = RAW_PDF_DIR / book_key
    chapter_files = sorted(book_dir.glob("*.pdf"))

    book_pages: dict[str, tuple[list[PageContent], ChapterStats]] = {}
    for chapter_file in chapter_files:
        relative_path = f"{book_key}/{chapter_file.name}"
        book_pages[chapter_file.stem] = load_pdf(relative_path)
    return book_pages


def _print_stats_row(chapter_key: str, stats: ChapterStats) -> None:
    print(
        f"  {chapter_key}: {stats.total_pages} pages | "
        f"raster found {stats.raster_found}, kept {stats.raster_kept}, "
        f"discarded as noise {stats.raster_discarded} | "
        f"vector diagrams {stats.vector_regions_extracted} on "
        f"{stats.vector_pages_with_diagrams} pages (timed out {stats.vector_pages_timed_out}) | "
        f"total images extracted {stats.total_kept}",
        flush=True,
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "class12_biology"

    if target.endswith(".pdf"):
        pages, stats = load_pdf(target)
        print(f"Loaded {len(pages)} pages from {target}")
        _print_stats_row(Path(target).stem, stats)
    else:
        book_dir = RAW_PDF_DIR / target
        chapter_files = sorted(book_dir.glob("*.pdf"))
        totals = ChapterStats()

        for chapter_file in chapter_files:
            relative_path = f"{target}/{chapter_file.name}"
            pages, stats = load_pdf(relative_path)
            _print_stats_row(chapter_file.stem, stats)

            totals.total_pages += stats.total_pages
            totals.raster_found += stats.raster_found
            totals.raster_kept += stats.raster_kept
            totals.vector_regions_extracted += stats.vector_regions_extracted
            totals.vector_pages_with_diagrams += stats.vector_pages_with_diagrams
            totals.vector_pages_timed_out += stats.vector_pages_timed_out

        print(f"\nTOTAL: {len(chapter_files)} chapters, {totals.total_pages} pages")
        print(
            f"Raster images: {totals.raster_found} found, {totals.raster_kept} kept, "
            f"{totals.raster_discarded} discarded as noise"
        )
        print(
            f"Vector diagrams: {totals.vector_regions_extracted} extracted on "
            f"{totals.vector_pages_with_diagrams} pages "
            f"({totals.vector_pages_timed_out} pages timed out)"
        )
        print(f"Total images extracted: {totals.total_kept}")
