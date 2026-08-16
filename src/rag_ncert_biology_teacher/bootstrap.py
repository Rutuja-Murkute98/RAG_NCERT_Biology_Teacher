"""
WHAT
----
On startup, download and extract a pre-built data.zip (data/chroma +
data/extracted + record_manager.sqlite3) from a GitHub Release asset, if it
isn't already on disk.

WHY
---
Render's Free tier has no persistent disk -- the container's filesystem is
wiped on every cold start and every redeploy. The built index is small
(~36MB zipped: captions, cropped diagrams, the Chroma vector DB) compared to
what it cost to build (Gemini captioning + embedding calls for the whole
book), so re-downloading it at each cold start is a fine trade for staying
on the free tier -- a few extra seconds on the first request after a cold
start, not minutes.

Previously hosted on Google Cloud Storage -- switched to a GitHub Release
asset after discovering, live in production, that the GCS bucket stopped
serving ANY downloads (even with a public bucket + anonymous client, no
credentials involved at all) once the owning GCP project's billing account
went delinquent: "The billing account for the owning project is disabled in
state delinquent" -- Google still attributes egress bandwidth cost to the
BUCKET-OWNING project regardless of who's requesting it or whether the
object is public, so an unpaid GCP billing account blocks Cloud Storage
downloads project-wide, not just Vertex AI. A GitHub Release asset on this
repo has no such dependency at all -- public, free, no billing account of
any kind involved, no credentials needed to download it.

The ~160MB raw chapter PDFs are deliberately EXCLUDED from this zip --
they're only used for decorative page thumbnails (see app/server.py's
_thumbnail_urls(), which degrades gracefully when they're missing), not for
answering questions, so downloading them every cold start would be pure
waste. chapters.json (small, needed for the real chapter list) still ships
with the deployed code itself -- it's the one raw_pdfs file NOT git-ignored
(see .gitignore) -- so it doesn't need downloading at all.

LOGIC / MECHANISM
------------------
DATA_ZIP_URL unset (the default -- true for local dev, see .env.example) ->
this is a complete no-op, since data/ already exists on a local machine.
Only set it in production.

Guarded by "does CHROMA_DIR already have files in it" so a gunicorn worker
restart within the SAME still-warm container (not a full cold start) doesn't
re-download anything. That guard is exactly why the extraction below is
staged, not direct: found for real in production -- retrieval was
consistently returning 0 results on Render despite the exact same code and
data reproducing correctly on a local machine every time, and boot-time
diagnostics showed CHROMA_DIR not existing right after a boot that had
printed "Index ready." with no error. The likely cause: Render can kill and
restart a slow-starting worker (free tier, cold start, ~250 small files to
write), and if that happens mid-extraction, a LATER worker's guard check
("does CHROMA_DIR have ANY files") sees the partially-written directory
from the killed attempt and treats it as "already downloaded" -- skipping
the real download forever after and silently serving retrieval against a
broken, incomplete index. Extracting into a staging directory first, then
moving each top-level item into place with "chroma" moved LAST (the one
thing the guard actually checks), means the guard can only ever see
CHROMA_DIR in one of two states: fully absent, or fully populated -- never
partial.
"""

import os
import shutil
import zipfile
from urllib.request import urlretrieve

from rag_ncert_biology_teacher.config import CHROMA_DIR, DATA_DIR


def ensure_data_present() -> None:
    zip_url = os.getenv("DATA_ZIP_URL")
    if not zip_url:
        return  # local dev -- data/ is already on disk, nothing to do

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return  # already downloaded this boot

    print(f"[bootstrap] No local index found -- downloading {zip_url} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    staging_dir = DATA_DIR / "_bootstrap_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    zip_path = DATA_DIR / "_bootstrap_data.zip"
    urlretrieve(zip_url, zip_path)

    print("[bootstrap] Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging_dir)
    zip_path.unlink()

    # Move everything into place, "chroma" LAST -- see WHY above. Sorting by
    # "is this the chroma entry" (False < True) puts it at the end of the
    # iteration without needing to special-case the loop body.
    for item in sorted(staging_dir.iterdir(), key=lambda p: p.name == "chroma"):
        dest = DATA_DIR / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    staging_dir.rmdir()

    print("[bootstrap] Index ready.")


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.bootstrap
    # (requires DATA_ZIP_URL set -- otherwise this is a documented no-op)
    ensure_data_present()
