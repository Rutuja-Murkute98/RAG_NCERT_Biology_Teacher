"""Turn each extracted diagram/photo into a text caption, using Gemini
(via Vertex AI) - so a figure becomes findable by normal text search too,
not just by looking at it.

Unlike the reference implementation this project started from (which
rendered and captioned whole PDF pages, to sidestep diagrams built from many
tiny fragments), our pdf_loader.py already solves that fragmentation problem
at extraction time - every file under data/extracted/.../images/ is already
a single, complete diagram or photo, precisely cropped, with no surrounding
page text or unrelated figures mixed in. So captioning here runs directly on
those files; there is no separate "render whole page" step to build first.

Auth: Application Default Credentials (`gcloud auth application-default
login`) - no API key file needed, only GOOGLE_CLOUD_PROJECT/LOCATION from
config.py (verified working in Step 1).
"""

from pathlib import Path

from google import genai
from google.genai import types

from rag_ncert_biology_teacher.config import (
    GEMINI_CAPTIONING_MODEL,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
)
from rag_ncert_biology_teacher.retry_utils import call_with_retry

CAPTION_PROMPT = (
    "You are captioning a figure extracted from an NCERT Class 12 Biology textbook, "
    "for a search index. This image has already been cropped to show ONLY the figure "
    "itself (no surrounding page text).\n\n"
    "If it shows real, identifiable biological content (a diagram, labeled structures, "
    "an organism, a process, a photograph of something specific), describe it in 1-2 "
    "sentences - name the labeled structures, organisms, or process shown, not just "
    "'a diagram' or 'a photo'.\n\n"
    "If it does NOT show any identifiable biological content (e.g. a plain texture, a "
    "flat color/gradient fragment, an unreadable scrap left over from extraction), say "
    "so briefly instead.\n\n"
    "End your reply with exactly one line: \"USEFUL: yes\" or \"USEFUL: no\"."
)


def _split_useful_marker(response_text: str) -> tuple[str, bool]:
    """Split the trailing "USEFUL: yes/no" marker off a caption. Returns
    (caption_without_marker, is_useful). Defaults to useful=True on a parse
    failure -- safer to keep a real diagram than to silently drop one.
    """
    lines = response_text.strip().splitlines()
    is_useful = True
    if lines and lines[-1].strip().upper().startswith("USEFUL:"):
        is_useful = lines[-1].split(":", 1)[1].strip().lower() == "yes"
        lines = lines[:-1]
    return "\n".join(lines).strip(), is_useful

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily create the Vertex AI client once, and reuse it after that."""
    global _client
    if _client is None:
        if not GOOGLE_CLOUD_PROJECT:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set -- add it to your .env file.")
        _client = genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
    return _client


def caption_image(image_path: Path, prompt: str = CAPTION_PROMPT) -> tuple[str, bool]:
    """Generate a caption for one extracted diagram/photo using Gemini.
    Returns (caption, is_useful) -- is_useful is False when Gemini itself
    recognizes the image has no real biological content (a leftover texture
    or gradient fragment our geometric noise filters let through), so
    callers can drop it instead of indexing/showing a meaningless caption.
    """
    client = _get_client()
    image_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    response = call_with_retry(
        lambda: client.models.generate_content(
            model=GEMINI_CAPTIONING_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        )
    )
    return _split_useful_marker(response.text.strip())


if __name__ == "__main__":
    from rag_ncert_biology_teacher.config import EXTRACTED_DIR

    # Usage: uv run python -m rag_ncert_biology_teacher.ingestion.captioning
    # Three real extracted images, on purpose: a vector-cropped line diagram, a
    # raster photo, and a raster-fragment-clustered diagram (see pdf_loader.py) -
    # to confirm captioning works well across every extraction path we built.
    base = EXTRACTED_DIR / "class12_biology"
    sample_images = [
        base / "chapter_03" / "images" / "page_005_diagram_0.png",  # vector diagram
        base / "chapter_08" / "images" / "page_002_img_12.jpeg",  # raster photo
        base / "chapter_02" / "images" / "page_012_diagram_0.png",  # raster-fragment cluster
    ]

    print(f"Model: {GEMINI_CAPTIONING_MODEL} (Vertex AI, project={GOOGLE_CLOUD_PROJECT})\n")
    for image_path in sample_images:
        if not image_path.exists():
            print(f"{image_path.name}: SKIPPED (file not found)")
            continue
        caption, is_useful = caption_image(image_path)
        tag = "USEFUL" if is_useful else "NOT USEFUL - would be dropped"
        print(f"{image_path.relative_to(EXTRACTED_DIR)} [{tag}]:\n  {caption}\n")
