"""
WHAT
----
On startup, download the already-built index (data/chroma + data/extracted +
record_manager.sqlite3) from a GCS bucket if it isn't already on disk.

WHY
---
Render's Free tier has no persistent disk -- the container's filesystem is
wiped on every cold start and every redeploy. Paying for a Starter-tier disk
would solve that directly, but the built index is small (~30MB: captions,
cropped diagrams, the Chroma vector DB) compared to what it cost to build
(Gemini captioning + embedding calls for the whole book), so re-downloading
it at each cold start is a fine trade for staying on the free tier -- a few
extra seconds on the first request after a cold start, not minutes.

The ~160MB raw chapter PDFs are deliberately EXCLUDED from this download --
they're only used for decorative page thumbnails (see app/server.py's
_thumbnail_urls(), which degrades gracefully when they're missing), not for
answering questions, so downloading them every cold start would be pure
waste. chapters.json (small, needed for the real chapter list) still ships
with the deployed code itself -- it's the one raw_pdfs file NOT git-ignored
(see .gitignore) -- so it doesn't need downloading at all.

LOGIC / MECHANISM
------------------
GCS_DATA_BUCKET unset (the default -- true for local dev, see .env.example)
-> this is a complete no-op, since data/ already exists on a local machine.
Only set it in production.

Guarded by "does CHROMA_DIR already have files in it" so a gunicorn worker
restart within the SAME still-warm container (not a full cold start) doesn't
re-download anything.

Downloads happen in parallel (ThreadPoolExecutor) -- sequential per-file GCS
calls for ~800 small files would otherwise noticeably lengthen the delay.

AUTH: uses an ANONYMOUS client, deliberately, not the default authenticated
one. storage.Client() (no args) tries Application Default Credentials --
fine on a local machine with `gcloud auth login`, but Render's container has
no gcloud, no service account, nothing -- found for real, crashed the app at
import time with DefaultCredentialsError. The whole point of migrating this
project off Vertex AI was to stop needing any GCP credential on the deploy
host at all, so the fix isn't to bring a service account key back -- it's to
make the bucket's objects public (read-only, non-sensitive textbook content)
and use create_anonymous_client(), which needs no credentials whatsoever.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_ncert_biology_teacher.config import CHROMA_DIR, DATA_DIR

_DOWNLOAD_WORKERS = 16


def ensure_data_present() -> None:
    bucket_name = os.getenv("GCS_DATA_BUCKET")
    if not bucket_name:
        return  # local dev -- data/ is already on disk, nothing to do

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return  # already downloaded this boot

    from google.cloud import storage

    print(f"[bootstrap] No local index found -- downloading from gs://{bucket_name}/data/ ...")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    def _download(blob) -> None:
        relative = Path(blob.name).relative_to("data")
        dest = DATA_DIR / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

    to_fetch = []
    for blob in bucket.list_blobs(prefix="data/"):
        if blob.name.endswith("/"):
            continue
        relative = Path(blob.name).relative_to("data")
        # Skip the big raw PDFs -- decorative thumbnails only, see WHY above.
        if relative.parts[0] == "raw_pdfs" and relative.suffix == ".pdf":
            continue
        to_fetch.append(blob)

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        list(pool.map(_download, to_fetch))

    print(f"[bootstrap] Downloaded {len(to_fetch)} files -- index ready.")


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.bootstrap
    # (requires GCS_DATA_BUCKET set -- otherwise this is a documented no-op)
    ensure_data_present()
