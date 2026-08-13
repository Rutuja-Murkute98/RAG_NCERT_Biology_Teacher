"""
WHAT
----
Approach 3 of 3 for handling images in this RAG pipeline: hand the actual
RETRIEVED image to the LLM at answer time, and have it generate a teacher-
style explanation that directly reasons about what's really in the picture -
not a pre-written caption computed once, ahead of time, during indexing.

WHY
---
Approaches 1 and 2 both solve FINDING the right image (retrieval). Neither
solves EXPLAINING it well for a SPECIFIC question. A caption is written once
with no idea what a student will actually ask, so it can easily omit the
exact detail a particular question needs. By passing the real image into the
LLM's context at answer time, alongside the student's real question, the
model looks at the picture FRESH with that question in mind, and can report
details no generic caption ever wrote down.
  - Approach 1 or 2 finds WHICH image is relevant     (retrieval)
  - Approach 3 looks at THAT image and explains it     (generation)
  - The real image path is returned too, so a UI can display the actual
    diagram next to the text answer.

FLOW
----
  1. Take a student's question.
  2. RETRIEVE the most relevant image (reuses caption_retrieval.py's index +
     retrieve() -- Approach 1 finds candidates cheaply/fast).
  3. Load the ACTUAL image bytes for the winning candidate.
  4. Send the image + the question + a teacher instruction to Gemini in ONE
     generative call -- a different kind of API call than Approaches 1/2,
     which only ever called an EMBEDDING endpoint (returns a vector, used
     for comparison, never "understands" anything).
  5. Return (answer_text, image_path) together.

LOGIC / MECHANISM
------------------
Embedding calls and generative calls are fundamentally different jobs, even
though both can accept an image as input:
  - Embed  : image -> fixed-length vector, built for fast similarity search
             across many candidates. Cheap, purely geometric.
  - Generate: image + prompt -> free-form text, built for deep, one-off
             reasoning about ONE specific image for ONE specific question.
             Slower and pricier per call, but genuinely reasons, not just compares.
This is why a real pipeline uses embeddings to CHEAPLY narrow candidates
down first, then spends a slower/pricier generative call only on those few -
running a generative call over every image in the book as the primary
search mechanism would be far too slow and expensive at scale.
"""

from pathlib import Path

from google import genai
from google.genai import types

from rag_ncert_biology_teacher.config import GEMINI_API_KEY, GEMINI_CAPTIONING_MODEL
from rag_ncert_biology_teacher.image_handling.caption_retrieval import build_caption_index, retrieve
from rag_ncert_biology_teacher.retry_utils import call_with_retry

TEACHER_PROMPT_TEMPLATE = (
    "You are a patient NCERT Biology teacher. A student asked the question below. "
    "Look carefully at the attached textbook figure and answer using what is ACTUALLY "
    "visible in it -- name the specific labeled structures or steps shown, not just a "
    "generic description. If the image doesn't actually answer the question, say so "
    "honestly instead of guessing.\n\n"
    "Student's question: {question}"
)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Lazily create the client once, reused for every answer call."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def answer_with_image(question: str, image_path: Path) -> str:
    """The core of Approach 3: ONE generative call given the real image AND
    the real question together, so the model reasons about both jointly -
    not a caption that was written in isolation, ahead of time.
    """
    client = get_client()
    image_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    response = call_with_retry(
        lambda: client.models.generate_content(
            model=GEMINI_CAPTIONING_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                TEACHER_PROMPT_TEMPLATE.format(question=question),
            ],
        )
    )
    return response.text.strip()


def answer_question(question: str, image_paths, captions, caption_vectors):
    """Full two-stage pipeline for one question:
      1. RETRIEVE the most relevant image  (Approach 1 -- cheap, fast, ranks ALL images)
      2. GENERATE a grounded answer        (Approach 3 -- slower, runs on ONE image only)
    Returns (answer_text, winning_image_path) so a caller can show both.
    """
    if not image_paths:
        return "I couldn't find any diagram in this chapter to check.", None

    top_image_path, top_caption, score = retrieve(question, image_paths, captions, caption_vectors, top_k=1)[0]
    print(f"  [retrieval] best match: {top_image_path.name} (score={score:.3f})")
    print(f"  [retrieval] its caption: {top_caption[:100]}")

    answer = answer_with_image(question, top_image_path)
    return answer, top_image_path


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.image_handling.multimodal_answer
    from rag_ncert_biology_teacher.config import EXTRACTED_DIR

    book, chapter_key = "class12_biology", "chapter_03"
    cache_path = EXTRACTED_DIR / book / chapter_key / "caption_index.json"

    # Reuse Approach 1's index-building (cache-only here -- no new Gemini calls
    # for captioning since chapter_03 was already indexed above).
    image_paths, captions, caption_vectors = build_caption_index(book, chapter_key, cache_path=cache_path)

    # Questions chosen to test whether Gemini adds detail BEYOND the caption's
    # own wording, by actually looking at the image again for this question.
    sample_questions = [
        "Which duct is cut and tied during a vasectomy?",
        "What shape is the Copper T device, and what is it made of?",
    ]

    for question in sample_questions:
        print(f"\n=== Question: {question!r} ===")
        answer, image_path = answer_question(question, image_paths, captions, caption_vectors)
        print(f"  [answer] {answer}")
        print(f"  [image]  {image_path}")
