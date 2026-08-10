"""
WHAT
----
The COMBINED image pipeline: ties Approaches 1, 2 and 3 together into one
retrieval + answer flow, instead of trusting any single approach alone.

WHY
---
Each approach has a real blind spot:
  - Approach 1 (caption_retrieval.py): bottlenecked by caption wording -
    near-synonym topics can embed close together in caption text and get
    confused (e.g. "vasectomy" vs. a tubectomy page, both "sterilization").
  - Approach 2 (embedding_retrieval.py): raw visual similarity doesn't
    capture precise wording the way text does.
  - Approach 3 (multimodal_answer.py) alone has no way to pick WHICH image
    to look at in the first place - it needs a candidate handed to it.
Combining them plays to each strength: cast a wide net cheaply with BOTH
retrieval signals (1 and 2), then use a generative call to actually VERIFY
whether the top candidate answers the question - falling through to the
next candidate if not, instead of committing blindly to retrieval's #1 result.

FLOW
----
  1. Get candidate images from Approach 1 (caption/text embedding search).
  2. Get candidate images from Approach 2 (direct image embedding search).
  3. Merge into one de-duplicated candidate list (interleaved, best-first).
  4. For each candidate, IN ORDER:
       a. Ask Gemini, in ONE call, to (i) say YES/NO whether this image
          actually answers the question, and (ii) answer it if YES.
       b. If YES -> return this answer + this image. Done.
       c. If NO  -> try the next candidate.
  5. If no candidate passes, say so honestly instead of forcing an answer
     out of an irrelevant image.

LOGIC / MECHANISM
------------------
This is a simplified version of "retrieve-then-verify" RAG: instead of
trusting retrieval's #1 result blindly, a generative call is used a second
time as a cheap judge of its own input before committing to showing it to
the student. Worst case, it costs a few extra generative calls (checking
2-4 candidates instead of 1) - but that turns "usually right" retrieval into
"verified right, or honestly unsure," which matters more for a student-facing
teacher chatbot than saving a fraction of a second.
"""

from pathlib import Path

from google.genai import types

from rag_ncert_biology_teacher.config import GEMINI_CAPTIONING_MODEL
from rag_ncert_biology_teacher.image_handling.caption_retrieval import build_caption_index
from rag_ncert_biology_teacher.image_handling.caption_retrieval import retrieve as caption_retrieve
from rag_ncert_biology_teacher.image_handling.embedding_retrieval import build_image_index
from rag_ncert_biology_teacher.image_handling.embedding_retrieval import retrieve as embedding_retrieve
from rag_ncert_biology_teacher.image_handling.multimodal_answer import get_client
from rag_ncert_biology_teacher.retry_utils import call_with_retry

GROUNDING_PROMPT_TEMPLATE = (
    "You are a patient NCERT Biology teacher checking whether a textbook figure actually "
    "answers a student's question, before explaining it to them.\n\n"
    "Student's question: {question}\n\n"
    "First line of your reply must be exactly YES (this image genuinely shows or explains "
    "something that answers the question) or exactly NO (it does not).\n"
    "Then, only if YES, add 1-3 sentences answering the question using specific labeled "
    "structures/steps ACTUALLY visible in the image."
)


def _merge_candidates(caption_hits, embedding_hits, max_candidates: int = 4) -> list[Path]:
    """Interleave both ranked lists (caption[0], embedding[0], caption[1], ...),
    keeping only the first occurrence of each image. A simple, transparent
    merge, not a fancy fusion formula.
    """
    merged: list[Path] = []
    seen: set[Path] = set()
    max_len = max(len(caption_hits), len(embedding_hits))
    for i in range(max_len):
        for hits in (caption_hits, embedding_hits):
            if i < len(hits):
                image_path = hits[i][0]
                if image_path not in seen:
                    seen.add(image_path)
                    merged.append(image_path)
        if len(merged) >= max_candidates:
            break
    return merged[:max_candidates]


def answer_with_grounding_check(question: str, image_path: Path) -> tuple[bool, str]:
    """ONE generative call that both CHECKS relevance and ANSWERS, so we
    don't pay for two separate calls per candidate. Returns (is_relevant, text).
    """
    client = get_client()
    image_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    response = call_with_retry(
        lambda: client.models.generate_content(
            model=GEMINI_CAPTIONING_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                GROUNDING_PROMPT_TEMPLATE.format(question=question),
            ],
        )
    )
    text = response.text.strip()
    is_relevant = text.upper().startswith("YES")
    return is_relevant, text


def answer_question_combined(question: str, caption_index, embedding_index, max_candidates: int = 4):
    """The full combined pipeline: cast a wide net with BOTH retrieval
    signals, then verify candidates in order with Approach 3 until one
    actually grounds an answer - or admit honestly that none did.
    """
    image_paths, captions, caption_vectors = caption_index
    embed_image_paths, image_vectors = embedding_index

    caption_hits = caption_retrieve(question, image_paths, captions, caption_vectors, top_k=3)
    embedding_hits = embedding_retrieve(question, embed_image_paths, image_vectors, top_k=3)
    candidates = _merge_candidates(caption_hits, embedding_hits, max_candidates=max_candidates)
    print(f"  [candidates, best-first] {[p.name for p in candidates]}")

    for candidate_path in candidates:
        is_relevant, text = answer_with_grounding_check(question, candidate_path)
        verdict = "RELEVANT" if is_relevant else "not relevant, trying next"
        print(f"  [check] {candidate_path.name}: {verdict}")
        if is_relevant:
            return text, candidate_path

    return "I couldn't find a diagram in this chapter that answers that question.", None


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.image_handling.combined_pipeline
    from rag_ncert_biology_teacher.config import EXTRACTED_DIR

    book, chapter_key = "class12_biology", "chapter_03"
    chapter_dir = EXTRACTED_DIR / book / chapter_key

    print("--- Building Approach 1 index (captions, cache-only) ---")
    caption_index = build_caption_index(book, chapter_key, cache_path=chapter_dir / "caption_index.json")
    print("\n--- Building Approach 2 index (image embeddings, cache-only) ---")
    embedding_index = build_image_index(book, chapter_key, cache_path=chapter_dir / "image_embeddings.npz")

    # The tricky question that broke a SINGLE approach in this project's
    # reference build (near-synonym confusion between vasectomy/tubectomy).
    tricky_questions = [
        "Which duct is cut and tied during a vasectomy?",
        "How does a Copper-T prevent pregnancy?",
    ]

    for question in tricky_questions:
        print(f"\n=== Question: {question!r} ===")
        answer, image_path = answer_question_combined(question, caption_index, embedding_index)
        print(f"  [final answer] {answer}")
        print(f"  [final image]  {image_path}")
