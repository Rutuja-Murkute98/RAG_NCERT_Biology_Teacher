"""Central place for paths, env vars, and model settings.

Every other module should import paths/settings from here instead of
hardcoding strings, so we only have one place to change things.

Migrated from Vertex AI to the Gemini Developer API (Google AI Studio) --
Vertex AI requires a GCP project with active billing, and this project's
billing account was suspended (payment declined) mid-project. The Developer
API is a genuinely separate product surface: authenticated by a single free
API key (aistudio.google.com/apikey), not tied to GCP project billing at
all, with its own free-tier rate limits instead of pay-per-call. Image
extraction (ingestion/pdf_loader.py) is untouched by any of this -- it never
called any Google API in the first place, only PyMuPDF.

Model names verified working against a REAL freshly-created API key (not
assumed from memory) -- "gemini-2.5-flash" turned out to be blocked for new
API keys ("no longer available to new users"), so LLM_MODEL defaults to the
"-latest" alias instead of a specific version, deliberately, so a future
model retirement doesn't silently break this again. gemini-embedding-2 was
confirmed to handle text AND image input in the same 3072-dim space.

One side effect worth knowing: Vertex AI's text embedding (text-embedding-005,
768-dim) and the Developer API's embedding model (gemini-embedding-2,
3072-dim, multimodal) are DIFFERENT vector spaces -- switching required a
full re-index (see indexing/embeddings.py), not just an auth change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DATA_DIR defaults to <repo>/data for local dev (unchanged behaviour), but is
# overridable via an env var for deployment: a host like Render mounts a
# persistent disk at a path IT chooses, not necessarily inside the git
# checkout -- without this override, data/chroma and data/extracted would
# have to land at a guessed internal path to be found, and silently vanish
# on every redeploy otherwise (the rest of the container filesystem is
# ephemeral). Set DATA_DIR to the disk's mount path in production.
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHROMA_DIR = DATA_DIR / "chroma"
RECORD_MANAGER_DB = DATA_DIR / "record_manager.sqlite3"

# --- Gemini Developer API --------------------------------------------------
# The ONE thing every Google API call in this project needs -- get a free
# key at https://aistudio.google.com/apikey (no credit card, not tied to any
# GCP project's billing status). Every genai.Client(...) in this codebase
# now takes api_key=GEMINI_API_KEY instead of vertexai=True/project/location.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Model settings ------------------------------------------------------
# ONE embedding model for everything -- chunks, captions, AND raw images,
# all in the same vector space. This is a genuine simplification over the
# old Vertex setup (which needed text-embedding-005 for chunks/captions and
# a SEPARATE gemini-embedding-2 for images, two incompatible spaces that
# could never be compared): gemini-embedding-2 on the Developer API is
# natively multimodal, so text-only and image inputs already share one
# space -- no separate "multimodal" model/config needed anymore.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-2")

# Chat LLM (used once the RAG chain is built) -- "-latest" alias, not a
# pinned version, so this doesn't break again the way gemini-2.5-flash did
# (retired for new API keys mid-project).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")

GEMINI_CAPTIONING_MODEL = os.getenv("GEMINI_CAPTIONING_MODEL", "gemini-flash-latest")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ensure_directories() -> None:
    for directory in (RAW_PDF_DIR, EXTRACTED_DIR, CHROMA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"PROJECT_ROOT              = {PROJECT_ROOT}")
    print(f"RAW_PDF_DIR                = {RAW_PDF_DIR}")
    print(f"EXTRACTED_DIR              = {EXTRACTED_DIR}")
    print(f"CHROMA_DIR                 = {CHROMA_DIR}")
    print(f"GEMINI_API_KEY set?        = {bool(GEMINI_API_KEY)}")
    print(f"EMBEDDING_MODEL_NAME       = {EMBEDDING_MODEL_NAME}")
    print(f"GEMINI_CAPTIONING_MODEL    = {GEMINI_CAPTIONING_MODEL}")
    print(f"LLM_PROVIDER               = {LLM_PROVIDER}")
    print(f"LLM_MODEL                  = {LLM_MODEL}")
    print("Config loaded OK.")
