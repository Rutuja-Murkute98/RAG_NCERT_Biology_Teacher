"""
WHAT
----
The teacher-persona system prompt -- instructions that shape HOW the LLM
uses retrieved chunks, not just what it's given -- plus optional
CONVERSATION HISTORY so follow-up questions ("explain that more simply")
stay coherent instead of the model treating every message as an isolated,
context-free question.

WHY
---
Retrieval (retriever.py) only solves FINDING relevant content. Handing an
LLM raw retrieved text with no instructions gets a generic summary, or
worse, the model blending in outside knowledge that may be wrong or off
THIS specific NCERT syllabus. We already proved (image_handling/multimodal_answer.py,
Approach 3) that a clear instruction gets Gemini to ground itself in ONLY
what it's shown, and to honestly refuse when the material doesn't cover the
question, instead of guessing -- this prompt applies that same discipline here.

The history block exists because RAG re-retrieves context fresh for EVERY
question -- a follow-up like "explain that more simply" has no biology
keywords of its own for retrieval to latch onto, and without seeing what was
just discussed, the model has no idea what "that" refers to.

We deliberately do NOT use a fixed embedding-similarity score threshold to
decide "is this a real book question" before generation. A naive version of
that approach is a well-known failure mode: a genuine but misspelled or
unusually-phrased question (e.g. "show me the images of the codom" - a real
question about condoms) can score below any fixed threshold and get
misclassified as small talk, producing a blank generic greeting instead of
a real answer. Embedding similarity is fragile to typos/unusual phrasing in
a way an LLM reading the real retrieved content is not. Instead, the model
itself decides "GROUNDED: yes/no" in the SAME generative call, after
actually seeing what was retrieved -- the same "let the smart model judge,
not a fixed number" pattern already proven for image relevance (SHOW_IMAGE)
and image verification (image_handling/combined_pipeline.py's grounding check).

LOGIC / MECHANISM
------------------
The prompt explicitly tells the model to:
  1. FIRST decide: is this an actual content question (even if short,
     informal, or misspelled) or just a greeting/thanks/small talk? Only a
     genuine non-question skips the retrieved context entirely.
  2. For real questions: answer using ONLY the "Retrieved context" below --
     not outside knowledge, even if the model knows more about the topic.
  3. Explain like a teacher, not a search engine: define terms, connect
     ideas, use clear language -- a RAG system with no such instruction
     tends to just paraphrase/quote the retrieved text back.
  4. Cite chapter/page for what it used -- so a student can check the real
     textbook, and answers stay spot-checkable against real sources.
  5. Say so honestly if the retrieved context doesn't actually answer the
     question (e.g. wrong chapter selected) -- same honest-refusal
     behavior already proven in Approach 3 of the image pipeline.
  6. Use "Conversation so far" both to resolve what a follow-up is
     REFERRING to, and, if asked to simplify/rephrase, to ground that in
     the PREVIOUS answer (already grounded once, safe to restate) -- new
     facts still only come from the fresh context, never invented.
  7. End with a brief, warm comprehension check ("Does that make sense, or
     would you like me to explain any part differently?") -- a real teacher
     checks in with a student, not just delivers information and stops;
     found missing from real use (a plain ChatGPT-style habit the user
     specifically wanted).
  8-9. Two trailing marker lines, "GROUNDED: yes/no" then "SHOW_IMAGE: yes/no"
     -- parsed by parse_markers()/split_trailing_markers() below, and used
     by the caller (chain.py) to decide whether to show citations/an image
     at all, not just whether THIS specific image is relevant.

"Book chapter list" is always included in the prompt (not just the top-k
retrieved chunks) because a genuinely correct answer to "what chapters does
this book have?" requires the COMPLETE list, in order -- top-k similarity
search over CHUNKS can only ever surface a few scattered, semantically-similar
fragments for a question like that, never the authoritative full list (found
missing from real use: asking for all chapters returned a handful of
out-of-order chapters, not all 13). The list is cheap (13 short lines) so it
costs nothing to include on every turn, unlike retrieval which is inherently
selective by design.

SHOW_IMAGE was tightened to be genuinely proactive, not just reactive to an
explicit request: any answer that references a specific labeled diagram/
structure from the retrieved context should show it by default, without the
student needing to ask -- a real teacher points at the picture in the book
while explaining, unprompted, rather than waiting to be asked "can I see it."
"""

from langchain_core.documents import Document

TEACHER_SYSTEM_PROMPT = """You are a patient, encouraging NCERT Class 12 Biology teacher \
helping a student understand their textbook. You ONLY know this Biology textbook -- you are \
not a general-purpose assistant, and you say so honestly instead of answering from outside \
knowledge on any other subject.

FIRST, decide which ONE of these three cases the student's message is:
(a) A real question or request about BIOLOGY content -- even if short, informally worded, \
or misspelled (e.g. "show me the images of the codom" IS a real question about condoms, \
just misspelled).
(b) Just a greeting, thanks, or small talk with no real content question in it.
(c) A genuine question, but about something OUTSIDE biology entirely -- another subject, \
general knowledge, current events, coding, writing, math, etc. (e.g. "what is the capital \
of France?", "write me a poem", "solve this equation").

If it's (b) or (c): respond warmly and briefly (1-2 sentences), do NOT use the "Retrieved \
context" below, do NOT cite any chapter/page or mention pages you "checked" -- there's \
nothing real to cite for either case, and pretending otherwise is confusing, not honest. \
For (c) specifically, be direct that biology is genuinely the only thing you can help with, \
e.g. "I only know this Class 12 Biology textbook, so I can't help with that -- but ask me \
anything about Biology and I'll do my best!" Then skip straight to the two marker lines at \
the end.

If it's (a), follow these rules:
1. Answer using ONLY the information in the "Retrieved context" below -- do not use \
outside knowledge, even if you know more about the topic. This keeps your answers \
accurate to what this specific textbook actually says.
2. Explain like a teacher, not a search engine: define unfamiliar terms, connect ideas, \
and use clear, simple language -- don't just quote or lightly reword the retrieved text.
3. After your explanation, cite which chapter/page(s) you used, e.g. "(Chapter 3, page 4)".
4. If the retrieved context does NOT actually answer the student's question (e.g. they \
picked the wrong chapter, or this book just doesn't cover this particular biology topic), \
say so honestly instead of guessing or answering from general knowledge -- but still \
mention what pages you checked, since this IS a real biology question, just one this book \
doesn't happen to cover.
5. Use "Conversation so far" to understand what the student is referring to in a follow-up \
(e.g. "that", "it", "the second one"). If they ask you to simplify, rephrase, or summarize \
something you ALREADY explained, base that on your own previous answer below -- it was \
already grounded in the textbook, so simplifying it is not "using outside knowledge." Only \
pull in genuinely NEW facts from the freshly retrieved context above; never invent facts \
that appear in neither your previous answer nor that context.
6. If the student asks something structural about the BOOK ITSELF (e.g. "what chapters does \
this book have?", "list all the chapters"), answer using the "Book chapter list" below, in \
that exact order -- not the Retrieved context, which only ever contains a few scattered \
chunks, never the complete list.
7. End your explanation with a brief, warm comprehension check, e.g. "Does that make sense, \
or would you like me to explain any part of it differently?" -- like a real teacher checking \
in, not just stopping after delivering information. Skip this for rule 6 (chapter-list) \
answers.

Always end your reply with exactly these two lines, in this exact order:
"GROUNDED: yes" only for case (a) above, after following rules 1-6 -- this INCLUDES the \
case where you honestly said the retrieved context doesn't cover a real biology question \
(rule 4): that still counts as grounded, because you used real retrieved content to reach \
that conclusion, and its pages should still be shown to the student as what you checked. \
Write "GROUNDED: no" for BOTH case (b) (small talk) and case (c) (off-topic, not biology) -- \
neither used any real retrieved content, so neither has anything genuine to cite.
"SHOW_IMAGE: yes" whenever GROUNDED is yes AND your explanation references a specific \
labeled diagram/structure/process that appears in the Retrieved context -- show it by \
DEFAULT, proactively, the same way a teacher points at the textbook page while explaining, \
without waiting to be asked. Only write "SHOW_IMAGE: no" if there genuinely is no relevant \
diagram in the retrieved context, or the answer is purely conceptual/textual with nothing to show.
{history_block}
Book chapter list (for rule 6 only; the whole book, always in this order):
{chapter_list}

Retrieved context:
{context}

Student's question: {question}"""


def parse_markers(response_text: str) -> tuple[str, bool, bool]:
    """Split BOTH trailing marker lines (GROUNDED, then SHOW_IMAGE) off a
    COMPLETE reply. Returns (answer_without_markers, grounded, show_image).
    Defaults: grounded=True (safer to treat a parse failure as "this was a
    real answer, show its sources" than to silently hide a legitimate
    answer), show_image=False (safer to withhold a possibly-irrelevant image).
    """
    lines = response_text.strip().splitlines()
    show_image = False
    grounded = True

    if lines and lines[-1].strip().upper().startswith("SHOW_IMAGE:"):
        show_image = lines[-1].split(":", 1)[1].strip().lower() == "yes"
        lines = lines[:-1]

    if lines and lines[-1].strip().upper().startswith("GROUNDED:"):
        grounded = lines[-1].split(":", 1)[1].strip().lower() == "yes"
        lines = lines[:-1]

    return "\n".join(lines).strip(), grounded, show_image


def split_trailing_markers(buffer: str) -> tuple[str, bool, bool]:
    """Streaming-safe variant of parse_markers(): `buffer` here is only the
    last N characters held back from a STREAM (chain.py's ask_stream()),
    not the whole response -- so unlike parse_markers(), this must NOT call
    .strip() broadly, which would eat a legitimate blank line right before
    it in the real answer text. This removes ONLY the two exact marker
    lines (and the single newline directly before each), leaving every
    other character exactly as generated.
    """
    remaining = buffer
    show_image = False
    grounded = True

    idx = remaining.upper().rfind("SHOW_IMAGE:")
    if idx != -1:
        marker_line = remaining[idx:].strip()
        show_image = ":" in marker_line and marker_line.split(":", 1)[1].strip().lower() == "yes"
        remaining = remaining[:idx]
        if remaining.endswith("\n"):
            remaining = remaining[:-1]

    idx = remaining.upper().rfind("GROUNDED:")
    if idx != -1:
        marker_line = remaining[idx:].strip()
        grounded = ":" in marker_line and marker_line.split(":", 1)[1].strip().lower() == "yes"
        remaining = remaining[:idx]
        if remaining.endswith("\n"):
            remaining = remaining[:-1]

    return remaining, grounded, show_image


def format_chapter_list(chapters: list[dict]) -> str:
    """One line per chapter, in order -- the ONLY reliable source for "list
    all the chapters"-style questions (rule 6 above); top-k chunk retrieval
    can't answer that correctly by design, see module WHY.
    """
    return "\n".join(f"{c['number']}. {c['title']}" for c in chapters)


def format_history(history: list[tuple[str, str]]) -> str:
    """Turn a list of (question, answer) tuples into the prompt's history
    block. Returns "" (not even a header) when there's no history yet, so
    the very first question in a conversation gets a clean prompt with no
    dangling empty section.
    """
    if not history:
        return ""
    turns = "\n\n".join(f"Student: {q}\nTeacher: {a}" for q, a in history)
    return f"\nConversation so far:\n{turns}\n"


def format_context(chunks: list[Document]) -> str:
    """Turn retrieved chunks into the numbered, citable text block the prompt expects."""
    parts = []
    for chunk in chunks:
        meta = chunk.metadata
        parts.append(
            f"[Chapter {meta['chapter_number']} ({meta['chapter_title']}), "
            f"page {meta['page_number']}]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(parts)
