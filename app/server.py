"""Flask web server for the NCERT Biology Teacher chatbot.

WHY Flask, not Streamlit: Streamlit renders through its own component
system -- you can inject CSS, but you're still skinning a Streamlit app, not
building an actual website. Flask serves real HTML/CSS/JS the browser
renders directly. This is a straight adaptation of the reference project's
own Flask app (which itself replaced an earlier Streamlit version per user
request), kept in this project for the exact same reason.

FLOW
----
  GET  /                    -> render templates/index.html (the whole app --
                                one page), passing the chapter list so the
                                chat screen's chapter-selector can be
                                populated with real chapter names.
  POST /api/chat/stream     -> {question, chapter?, history?} -> streams
                                newline-delimited JSON as Gemini generates
                                the answer (rag.chain.ask_stream()), ending
                                with one {"type":"meta", sources, image_url}
                                line.
  GET  /api/image/<path>    -> serves ONE specific extracted diagram/photo
                                from data/extracted/ (NOT Flask's static/
                                folder -- these are real book content files,
                                not app assets). <path> is the file's path
                                relative to EXTRACTED_DIR, e.g.
                                "class12_biology/chapter_03/images/page_005_diagram_0.png".

Conversation history lives on the CLIENT (chat.js), not a Flask session --
a streaming response's headers (including the Set-Cookie header Flask
session relies on) are finalized BEFORE the generator function's body
actually runs, so writing to `session` from inside a streaming generator
silently doesn't persist. History is sent EXPLICITLY by the client in each
request instead (it already renders the full conversation anyway).

Image serving differs from a "one image per page" scheme: a page here can
carry ZERO, ONE, or SEVERAL cropped diagrams (pdf_loader.py's precise
per-diagram extraction, not a whole-page render), so chain.py's
first_image_paths() returns a LIST of real file paths. To match the
reference UI's one-image-per-answer design, this file shows only the first
of those, turned into a URL by path rather than by reconstructing one from
chapter+page.
"""

import json
import logging
import random
from pathlib import Path

import fitz  # PyMuPDF
from flask import Flask, Response, render_template, request, send_file

from rag_ncert_biology_teacher.config import EXTRACTED_DIR, GOOGLE_CLOUD_PROJECT, RAW_PDF_DIR
from rag_ncert_biology_teacher.ingestion.pdf_loader import render_page_thumbnail
from rag_ncert_biology_teacher.rag.chain import ask_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BOOK = "class12_biology"

EXAMPLE_QUESTIONS = [
    "How does tubal ligation work?",
    "What is natural selection?",
    "How does a biogas plant work?",
    "What is the Copper T (CuT) and how does it work?",
]

app = Flask(__name__)

_EXTRACTED_ROOT = EXTRACTED_DIR.resolve()

# Fail fast and clearly at startup, not with a deep, confusing traceback from
# inside chain.py on the FIRST chat request -- runs at import time (not just
# under `if __name__ == "__main__"`) so it also covers `gunicorn app.server:app`.
if not GOOGLE_CLOUD_PROJECT:
    raise RuntimeError(
        "GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and fill in your GCP "
        "project id (see README.md's Setup section) before starting this app."
    )


def _to_image_url(path) -> str:
    """Turn a real extracted-image file path into a URL this app can serve."""
    rel = Path(path).resolve().relative_to(_EXTRACTED_ROOT)
    return f"/api/image/{rel.as_posix()}"


def _thumbnail_urls(max_images: int = 6) -> list[str]:
    """A handful of REAL whole-page URLs, one per randomly chosen chapter,
    used as faded background decoration (see templates/index.html) --
    genuine book content, since there's no internet access here for stock
    photography. Whole PAGES here (not the precisely-cropped diagrams used
    in actual answers) purely to match the reference project's decorative
    look exactly -- pdf_loader.render_page_thumbnail() renders + caches one
    page per chosen chapter on first use.
    """
    manifest = json.loads((RAW_PDF_DIR / BOOK / "chapters.json").read_text())
    chapters = manifest["chapters"]
    random.seed(11)
    chosen = random.sample(chapters, k=min(max_images, len(chapters)))

    urls = []
    for chapter in chosen:
        chapter_key = f"chapter_{chapter['number']:02d}"
        doc = fitz.open(RAW_PDF_DIR / BOOK / chapter["file"])
        middle_page = (doc.page_count // 2) + 1
        doc.close()

        thumbnail_path = render_page_thumbnail(BOOK, chapter_key, middle_page)
        urls.append(_to_image_url(thumbnail_path))
    return urls


def _chapter_list() -> list[dict]:
    """The book's chapters, for the chat screen's chapter-selector dropdown."""
    manifest = json.loads((RAW_PDF_DIR / BOOK / "chapters.json").read_text())
    return manifest["chapters"]


@app.route("/")
def index():
    return render_template(
        "index.html",
        example_questions=EXAMPLE_QUESTIONS,
        thumbnail_urls=_thumbnail_urls(),
        chapters=_chapter_list(),
    )


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    chapter_number = data.get("chapter") or None
    # history comes from the CLIENT (it already has the full conversation
    # rendered) as [[question, answer], ...] -- see module docstring.
    history = [tuple(turn) for turn in data.get("history") or []]

    if not question:
        return Response(json.dumps({"type": "error", "message": "Empty question"}) + "\n",
                         mimetype="application/x-ndjson"), 400

    def generate():
        # ask_stream() can genuinely raise mid-stream (e.g. Gemini's rate
        # limit outlasting retry_utils.py's backoff, confirmed to happen for
        # real on this project's Vertex AI quota, see chain.py) -- without
        # this try/except, that crashes the generator and the connection
        # just drops with no explanation. chat.js already handles a
        # {"type":"error"} event; this is what actually sends one, instead
        # of leaving the student staring at a bubble that stopped mid-sentence.
        try:
            for kind, payload in ask_stream(question, history=history, chapter_number=chapter_number):
                if kind == "text":
                    yield json.dumps({"type": "text", "content": payload}) + "\n"
                else:  # "meta" -- always the last line
                    chunks = payload["chunks"]

                    # De-duplicate (top-k retrieval often returns several chunks
                    # from the SAME page), preserving first-seen (most relevant) order.
                    seen_pages = dict.fromkeys(
                        (c.metadata["chapter_number"], c.metadata["page_number"]) for c in chunks
                    )
                    sources = [{"chapter": ch, "page": pg} for ch, pg in seen_pages]
                    image_paths = payload["image_paths"]
                    image_url = _to_image_url(image_paths[0]) if image_paths else None

                    yield json.dumps({"type": "meta", "sources": sources, "image_url": image_url}) + "\n"
        except Exception:
            logger.exception("chat_stream failed for question=%r", question)
            yield json.dumps({
                "type": "error",
                "message": "Something went wrong generating that answer. Please try again in a moment.",
            }) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/api/image/<path:rel_path>")
def get_image(rel_path: str):
    """Serve one real extracted image. Resolves the path and checks it's
    still inside EXTRACTED_DIR before serving -- rel_path comes from a URL,
    so ".." segments must never be allowed to escape that directory.
    """
    full_path = (EXTRACTED_DIR / rel_path).resolve()
    if not full_path.is_relative_to(_EXTRACTED_ROOT) or not full_path.is_file():
        return "Not found", 404
    return send_file(full_path)


if __name__ == "__main__":
    import os

    # Usage: uv run python app/server.py
    # Render (and most PaaS hosts) set $PORT; default to 5000 for local dev.
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
