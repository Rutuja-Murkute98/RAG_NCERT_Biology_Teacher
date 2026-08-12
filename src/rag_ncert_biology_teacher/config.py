"""Central place for paths, env vars, and model settings.

Every other module should import paths/settings from here instead of
hardcoding strings, so we only have one place to change things.

Updated to GCP/Vertex AI throughout (previously referenced local BLIP
captioning and local sentence-transformers embeddings, from before this
project's pipeline was built) -- image extraction (ingestion/pdf_loader.py)
is untouched; everything downstream of it now runs on Vertex AI.
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

# --- Model settings ------------------------------------------------------
TEXT_EMBEDDING_MODEL_NAME = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "text-embedding-005")

# Chat LLM (used once the RAG chain is built) -- Gemini, matching the rest
# of this project's GCP-only infrastructure.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# --- GCP / Vertex AI -------------------------------------------------------
# Auth uses Application Default Credentials (`gcloud auth application-default
# login`) -- no API key file needed, so only project + region live here.
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_CAPTIONING_MODEL = os.getenv("GEMINI_CAPTIONING_MODEL", "gemini-2.5-flash")

# Multimodal (image) embeddings -- a joint text+image vector space, for Approach 2's
# direct image-embedding retrieval. multimodalembedding@001 (the older vertexai SDK
# model) loses SDK access June 2026 and retires April 2027 - gemini-embedding-2 is
# the current replacement, but only reachable via the "global" endpoint right now,
# not GOOGLE_CLOUD_LOCATION - verified directly against this project before picking it.
MULTIMODAL_EMBEDDING_MODEL_NAME = os.getenv("MULTIMODAL_EMBEDDING_MODEL_NAME", "gemini-embedding-2")
MULTIMODAL_EMBEDDING_LOCATION = os.getenv("MULTIMODAL_EMBEDDING_LOCATION", "global")

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
    print(f"TEXT_EMBEDDING_MODEL_NAME  = {TEXT_EMBEDDING_MODEL_NAME}")
    print(f"GOOGLE_CLOUD_PROJECT       = {GOOGLE_CLOUD_PROJECT}")
    print(f"GOOGLE_CLOUD_LOCATION      = {GOOGLE_CLOUD_LOCATION}")
    print(f"GEMINI_CAPTIONING_MODEL    = {GEMINI_CAPTIONING_MODEL}")
    print(f"MULTIMODAL_EMBEDDING_MODEL_NAME = {MULTIMODAL_EMBEDDING_MODEL_NAME}")
    print(f"MULTIMODAL_EMBEDDING_LOCATION   = {MULTIMODAL_EMBEDDING_LOCATION}")
    print(f"LLM_PROVIDER               = {LLM_PROVIDER}")
    print(f"LLM_MODEL                  = {LLM_MODEL}")
    print("Config loaded OK.")
