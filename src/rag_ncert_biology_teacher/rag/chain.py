"""
WHAT
----
The actual RAG chain -- ties retriever.py (finds relevant chunks) and
prompts.py (teacher persona + conversation history) together with a real
Gemini call, so `ask(question, history)` -> a grounded, pedagogical,
context-aware answer is one function call away. This is what a chat UI calls
directly.

WHY
---
Retrieval and prompt-shaping each solve one piece, but neither alone
produces an answer. This file is the seam where retrieval becomes
generation -- the same "retrieve cheaply, then spend one generative call"
pattern already used successfully in image_handling/combined_pipeline.py,
applied here to plain text chunks instead of images. `history` lets the
model resolve what a follow-up like "explain that more simply" refers to,
while the FACTS still come from fresh retrieval every turn (prompts.py rule 5).

FLOW
----
  1. ALWAYS retrieve_with_scores(question) -> top-k chunks.
  2. format_context(chunks) + format_history(history), fill
     TEACHER_SYSTEM_PROMPT, send to Gemini. The model itself decides,
     reading the actual retrieved content, whether this was a real
     question or just a greeting.
  3. Parse the trailing "GROUNDED: yes/no" then "SHOW_IMAGE: yes/no"
     markers off the response -- GROUNDED decides whether to show
     citations/sources at all, SHOW_IMAGE decides whether to additionally
     show the top source's diagram(s).
  4. Return (answer_text, chunks, show_image) -- chunks is [] whenever
     GROUNDED was "no" (nothing to cite for a greeting).

A retrieved chunk's `image_paths` metadata (chunking.py, Step 5) is a
JSON-encoded list -- a page can carry zero, one, or several diagrams, unlike
a "one image per page" scheme, and EVERY chunk split from that page carries
the SAME full list (chunking.py attaches images per-PAGE, before splitting).
When a page has more than one diagram, that list alone can't say which one
the question is actually about -- `select_best_image_path()` below resolves
that by re-ranking that page's (deduplicated) candidate images against the
QUESTION using each image's own cached caption, and returns just the one
best match, so a caller (the Flask UI) never has to guess by taking
"whichever path happens to be listed first."

LOGIC / MECHANISM
------------------
`history` is a plain list of (question, answer) tuples, kept by the CALLER
(the chat UI, client-side) -- this file stays stateless itself, it just
formats whatever history it's handed into the prompt.

chapter_number (optional): lets a caller (the chat UI's chapter selector)
restrict retrieval to ONE chapter instead of the whole book -- passed
straight through to retriever.py's Chroma metadata filter. A real question
that isn't covered by the SELECTED chapter still gets GROUNDED: yes (the
model DID use real retrieved context to determine it's not covered here) --
that's intentional, so its sources still show, letting the student see which
pages were actually checked.

ask_stream() is the actual fix for perceived response time -- text starts
appearing as Gemini generates it instead of the UI waiting silently for the
full answer. ask() still exists, built on top of ask_stream(), for callers
that just want the final result (tests, scripts).
"""

import json

from google import genai

from rag_ncert_biology_teacher.config import EXTRACTED_DIR, GEMINI_API_KEY, LLM_MODEL, RAW_PDF_DIR
from rag_ncert_biology_teacher.indexing.embeddings import embed_texts
from rag_ncert_biology_teacher.rag.prompts import (
    TEACHER_SYSTEM_PROMPT,
    format_chapter_list,
    format_context,
    format_history,
    split_trailing_markers,
)
from rag_ncert_biology_teacher.rag.retriever import retrieve_with_scores
from rag_ncert_biology_teacher.retry_utils import call_with_retry

_client: genai.Client | None = None
_chapter_list_block: str | None = None


def _get_chapter_list_block() -> str:
    """Cached, loaded once -- chapters.json doesn't change while the app runs."""
    global _chapter_list_block
    if _chapter_list_block is None:
        manifest = json.loads((RAW_PDF_DIR / "class12_biology" / "chapters.json").read_text())
        _chapter_list_block = format_chapter_list(manifest["chapters"])
    return _chapter_list_block


def get_client() -> genai.Client:
    """Lazily create the client once, reused for every ask() call."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _generate_stream(prompt: str):
    """Yield text deltas from Gemini as they arrive, instead of blocking
    until the whole response is ready -- the actual fix for "the chat feels
    slow": the first words appear in ~1-2s instead of waiting several
    seconds for a complete grounded answer.

    generate_content_stream() returns a LAZY generator -- the actual HTTP
    request (and any rate limit) only fires once the first chunk is pulled
    from it, not when the generator object is created. Wrapping just the
    call that CREATES the generator in call_with_retry (an earlier version
    of this function did that) protects nothing: the real request happens
    later, outside that wrapper, and a 429 there went unretried (found for
    real running deepeval/run_eval.py, which fires enough requests to
    reliably hit it). Retrying START-STREAM-PLUS-FIRST-CHUNK as one atomic
    unit is safe to redo from scratch on failure, since nothing has reached
    the caller yet. Once any chunk goes out, retrying would risk duplicating
    already-shown text, so later chunks are deliberately left unprotected --
    an error there surfaces to the caller rather than silently restarting.
    """
    client = get_client()

    def _start_stream_and_get_first_chunk():
        stream = client.models.generate_content_stream(model=LLM_MODEL, contents=prompt)
        iterator = iter(stream)
        return iterator, next(iterator, None)

    iterator, first_chunk = call_with_retry(_start_stream_and_get_first_chunk)
    if first_chunk is not None and first_chunk.text:
        yield first_chunk.text
    for chunk in iterator:
        if chunk.text:
            yield chunk.text


# How many trailing characters we always hold back from the client while
# streaming -- enough margin to contain BOTH marker lines ("GROUNDED: yes"
# + "SHOW_IMAGE: yes", prompts.py rules 7-8) plus their leading newlines,
# so neither line ever actually reaches the UI.
_MARKER_HOLD_BACK = 60


# {chapter_number: {image_path_str: caption}} -- loaded once per chapter from
# the same caption_index.json cache captioning.py/chunking.py already write,
# reused here purely for local lookup (no new Gemini captioning calls).
_caption_cache: dict[int, dict[str, str]] = {}


def _load_chapter_captions(chapter_number: int) -> dict[str, str]:
    if chapter_number not in _caption_cache:
        chapter_key = f"chapter_{chapter_number:02d}"
        cache_path = EXTRACTED_DIR / "class12_biology" / chapter_key / "caption_index.json"
        captions: dict[str, str] = {}
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            captions = {
                path: entry["caption"] for path, entry in zip(data["image_paths"], data["entries"])
            }
        _caption_cache[chapter_number] = captions
    return _caption_cache[chapter_number]


def select_best_image_path(question: str, chunks) -> str | None:
    """Among every diagram attached to the retrieved chunks, pick the ONE
    whose own caption is the best semantic match for the question -- not
    just the first path listed on the top chunk's page.

    Why this exists: a page can carry several unrelated diagrams (e.g.
    chapter 3 page 4 has both a condom diagram AND a Copper T diagram), and
    chunking.py attaches that PAGE's full image list to every chunk split
    from it. Naively taking image_paths[0] means whichever file sorts first
    alphabetically always wins, regardless of the question -- found for real
    asking about the Copper T and getting shown the condom instead. Ranking
    each candidate's own caption against the question (reusing the same
    embed_texts()/cosine-similarity approach already proven in
    image_handling/caption_retrieval.py) fixes that directly.
    """
    candidates: list[tuple[str, str]] = []  # [(image_path, caption), ...], deduplicated
    seen: set[str] = set()
    for chunk in chunks:
        paths = json.loads(chunk.metadata.get("image_paths", "[]"))
        if not paths:
            continue
        chapter_captions = _load_chapter_captions(chunk.metadata["chapter_number"])
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            candidates.append((path, chapter_captions.get(path, "")))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    # One batched call: [question, caption_1, caption_2, ...] -- cheap and
    # only fires at all when a page genuinely has more than one candidate.
    vectors = embed_texts([question] + [caption or path for path, caption in candidates])
    question_vector, caption_vectors = vectors[0], vectors[1:]
    best_index = int((caption_vectors @ question_vector).argmax())
    return candidates[best_index][0]


# How many chunks to consider as IMAGE candidates -- deliberately wider than
# the k=4 default used for the text answer's own context. A page's dedicated
# diagram can rank just outside the tight top-4 window even when it's
# clearly the right image, because the CHUNK that carries it is often mostly
# a folded-in figure caption (chunking.py), not dense explanatory prose --
# found for real asking "explain the process of spermatogenesis": the
# correct diagram's chunk ranked 8th by retrieval score, comfortably outside
# k=4, while several chunks about the SAME topic but without that image
# ranked higher.
_IMAGE_CANDIDATE_K = 15


def ask_stream(
    question: str,
    history: list[tuple[str, str]] | None = None,
    k: int = 4,
    chapter_number: int | None = None,
):
    """Generator version of ask(): yields ("text", chunk) pieces as they
    stream in from Gemini, ending with
    ("meta", {"chunks": [...], "show_image": bool, "image_paths": [...]})
    once the full answer (and its trailing markers) is known. Lets a chat UI
    render progressively instead of waiting for one long response.
    """
    scored_chunks = retrieve_with_scores(question, k=k, chapter_number=chapter_number)
    chunks = [chunk for chunk, _score in scored_chunks]

    # Widen the IMAGE candidate pool, but ONLY within chapter(s) the tight
    # top-k already established as relevant -- NOT a flat book-wide search.
    # An earlier version of this widened across the whole book and broke for
    # real: asking about the menstrual cycle pulled in a Copper T photo from
    # a totally different chapter, whose own chunk barely scraped into a
    # book-wide k=15 (rank 12, a genuinely weak match), yet its caption
    # scored HIGHEST in the re-ranking below purely on reproductive-health
    # vocabulary overlap ("uterus" etc.), not real relevance --
    # select_best_image_path()'s caption re-ranking is good at picking
    # correctly AMONG legitimately relevant candidates (proven by the Copper
    # T/condom same-page fix), but that's not the same as being safe to feed
    # it a candidate from an unrelated chapter that only looks relevant on
    # thin vocabulary overlap. Restricting the widened search to chapters
    # ALREADY confirmed relevant by the top-k keeps catching same-chapter
    # near-misses (the spermatogenesis case above) without that failure mode.
    relevant_chapters = (
        {chapter_number} if chapter_number is not None else {c.metadata["chapter_number"] for c in chunks}
    )
    image_candidate_chunks = list(chunks)
    for ch in relevant_chapters:
        image_candidate_chunks += [c for c, _score in retrieve_with_scores(question, k=_IMAGE_CANDIDATE_K, chapter_number=ch)]

    context = format_context(chunks) if chunks else "(No relevant content was retrieved.)"
    # TEMPORARY diagnostic -- every real question was coming back "ungrounded"
    # on the live Render deploy (sources: [] for genuinely on-topic biology
    # questions), but the exact same code+data reproduced correctly on a
    # local machine. No existing log line showed WHAT retrieval actually
    # returned in production, so this pins that down directly instead of
    # guessing further -- remove once the real cause is found.
    print(f"[diag] retrieved {len(chunks)} chunks for {question!r}; context chars={len(context)}")
    history_block = format_history(history or [])
    prompt = TEACHER_SYSTEM_PROMPT.format(
        context=context,
        question=question,
        history_block=history_block,
        chapter_list=_get_chapter_list_block(),
    )

    held = ""
    for piece in _generate_stream(prompt):
        held += piece
        if len(held) > _MARKER_HOLD_BACK:
            to_send, held = held[: -_MARKER_HOLD_BACK], held[-_MARKER_HOLD_BACK:]
            yield "text", to_send

    # Stream ended -- `held` is at most _MARKER_HOLD_BACK chars, containing
    # both marker lines (if the model included them, as instructed).
    final_text, grounded, show_image = split_trailing_markers(held)
    if final_text:
        yield "text", final_text

    # GROUNDED: no (a greeting/small talk) -> no citations, no image, even
    # though we DID retrieve chunks (that's fine -- retrieval is cheap;
    # what matters is not showing sources for a reply that never used them).
    # "sources" shown to the student stay the TIGHT top-k actually used in
    # the prompt -- but image selection draws from the WIDER pool
    # (image_candidate_chunks, see _IMAGE_CANDIDATE_K above), since the
    # right diagram's chunk doesn't have to be one of the chunks that made
    # the cut for the text answer itself.
    shown_chunks = chunks if grounded else []
    best_image_path = (
        select_best_image_path(question, image_candidate_chunks) if (show_image and grounded) else None
    )
    yield "meta", {
        "chunks": shown_chunks,
        "show_image": show_image and grounded,
        "image_paths": [best_image_path] if best_image_path else [],
    }


def ask(
    question: str,
    history: list[tuple[str, str]] | None = None,
    k: int = 4,
    chapter_number: int | None = None,
):
    """Non-streaming convenience wrapper around ask_stream() -- collects the
    full answer before returning, for callers that don't need progressive
    display (tests, scripts). Returns (answer_text, retrieved_chunks, show_image).
    """
    parts: list[str] = []
    chunks, show_image = [], False
    for kind, payload in ask_stream(question, history=history, k=k, chapter_number=chapter_number):
        if kind == "text":
            parts.append(payload)
        else:
            chunks, show_image = payload["chunks"], payload["show_image"]
    return "".join(parts).strip(), chunks, show_image


if __name__ == "__main__":
    import sys

    # Windows' default console codepage (cp1252) can't print some characters
    # Gemini legitimately outputs -- force UTF-8 stdout so this demo doesn't
    # crash on the model's own correct output (found for real in Step 8).
    sys.stdout.reconfigure(encoding="utf-8")

    # Usage: uv run python -m rag_ncert_biology_teacher.rag.chain

    print("=== Test: greeting should NOT retrieve/cite/show an image ===")
    greeting_answer, greeting_chunks, greeting_show_image = ask("hi")
    print(f"Answer: {greeting_answer}")
    print(f"Chunks: {len(greeting_chunks)}, show_image: {greeting_show_image}")
    assert greeting_chunks == [] and greeting_show_image is False, "greeting gate failed"
    print("PASS\n")

    print("=== Regression test: misspelled real question should NOT be treated as small talk ===")
    typo_answer, typo_chunks, typo_show_image = ask("show me the images of the codom")
    print(f"Answer: {typo_answer}")
    print(f"Chunks: {len(typo_chunks)}, show_image: {typo_show_image}")
    assert len(typo_chunks) > 0, "REGRESSION: misspelled real question treated as greeting!"
    print("PASS\n")

    print("=== Test: a question explicitly asking to SEE a diagram -> show_image should be True ===")
    img_answer, img_chunks, img_show_image = ask("Show me what the Copper T looks like.")
    print(f"Answer: {img_answer}")
    print(f"show_image: {img_show_image}")
    print()

    print("=== Turn 1: 'What is natural selection?' ===")
    answer1, chunks1, show_image1 = ask("What is natural selection?")
    print(f"Answer:\n{answer1}\nshow_image: {show_image1}")

    history = [("What is natural selection?", answer1)]
    print("\n=== Turn 2 (follow-up, tests conversation memory): 'Explain that in one simple sentence.' ===")
    answer2, chunks2, show_image2 = ask("Explain that in one simple sentence.", history=history)
    print(f"Answer:\n{answer2}\nshow_image: {show_image2}")
