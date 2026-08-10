"""
WHAT
----
Approach 2 of 3 for handling images in this RAG pipeline: embed each
extracted image's raw PIXELS directly into a joint image+text vector space
(gemini-embedding-2), using Vertex AI's multimodal embedding model. There is
no caption step and no text description anywhere in this file - pixels go
in, a vector comes out.

WHY
---
Approach 1 (caption_retrieval.py) can only find an image if Gemini's caption
happened to use words related to the question - retrieval is bottlenecked by
captioning wording. Here, the image's actual pixels are embedded, so we can
retrieve a diagram even for questions whose exact wording never appears in
any caption, because we're comparing visual meaning, not words about it.
This also means images could be indexed WITHOUT ever running (paid)
captioning at all, if we wanted to skip that cost.

FLOW
----
  1. List every already-extracted image for a chapter (pdf_loader.list_chapter_images).
  2. Embed each image's PIXELS with embed_image() (embeddings.py, gemini-embedding-2)
     -> a 3072-dimensional vector living in a shared image+text space.
  3. At query time, embed the student's question TEXT with embed_text_multimodal()
     - the SAME model, not a separate text embedder - -> a vector in the
     SAME 3072-dim space.
  4. Rank image vectors by cosine similarity to the question vector.
  5. Return the winning image directly - no caption text is involved
     anywhere in this matching step.

LOGIC / MECHANISM
------------------
A "joint embedding space" model is trained so that an image and a text
description of that same scene land close together in vector space. Because
our images and the student's question both pass through the SAME model into
the SAME space, we can directly compare "does this picture visually match
this sentence" via cosine similarity - exactly like Approach 1's text-vs-text
comparison, except one side is now real pixels instead of a written caption.
embed_image()/embed_text_multimodal() already return unit-normalized
vectors, so the dot product below is already cosine similarity.

Rate limits: this calls the embedding endpoint once per image, in a tight
loop, which reliably risks the same per-minute quota wall every other Google
API caller in this project has hit - call_with_retry() (inside embeddings.py)
handles that; a small extra sleep between calls here keeps normal runs from
even reaching that wall.
"""

import time
from pathlib import Path

import numpy as np

from rag_ncert_biology_teacher.indexing.embeddings import embed_image, embed_text_multimodal
from rag_ncert_biology_teacher.ingestion.pdf_loader import list_chapter_images

_EMBEDDING_DIM = 3072  # gemini-embedding-2's output size


def build_image_index(book: str, chapter_key: str, cache_path: Path | None = None):
    """Embed every extracted image's raw pixels (no captioning involved).

    cache_path (optional): a .npz file to save/load (image_paths, vectors) -
    these Vertex AI calls are both rate-limited and the slowest step in this
    project, so caching matters even more here than for captions.

    RESUMABLE by design, same reasoning as build_caption_index(): progress
    is saved after EVERY image, so an interrupted run resumes instead of
    re-embedding (and re-paying for) images already done.
    """
    image_paths = list_chapter_images(book, chapter_key)

    vectors: list = []
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["image_paths"]) == [str(p) for p in image_paths]:
            vectors = list(cached["vectors"])

    if len(vectors) < len(image_paths):
        remaining = image_paths[len(vectors) :]
        if vectors:
            print(f"Resuming embedding: {len(vectors)}/{len(image_paths)} already cached, {len(remaining)} to go")
        elif image_paths:
            print(f"Embedding {len(image_paths)} images directly (no captions used)...")

        for image_path in remaining:
            vector = embed_image(image_path)
            vectors.append(vector)
            # Save BEFORE printing -- a paid-for embedding must survive even if
            # something downstream (e.g. console encoding) blows up.
            if cache_path:  # save after EVERY image -- a crash loses at most one image's work
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, image_paths=[str(p) for p in image_paths], vectors=np.array(vectors))
            print(f"  {image_path.name}: embedded -> vector length {len(vector)}")
            time.sleep(1)  # stay gentle on the per-minute quota
    else:
        print(f"Loaded {len(vectors)} cached image embeddings from {cache_path} (skipping Vertex AI calls)")

    return image_paths, np.array(vectors) if vectors else np.zeros((0, _EMBEDDING_DIM))


def retrieve(query: str, image_paths, image_vectors: np.ndarray, top_k: int = 3):
    """Embed the query as TEXT (multimodal model), rank images by cosine similarity."""
    if len(image_paths) == 0:
        return []
    query_vector = embed_text_multimodal(query)
    similarities = image_vectors @ query_vector

    ranked_indices = np.argsort(-similarities)[:top_k]
    return [(image_paths[i], float(similarities[i])) for i in ranked_indices]


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.image_handling.embedding_retrieval
    from rag_ncert_biology_teacher.config import EXTRACTED_DIR

    book, chapter_key = "class12_biology", "chapter_03"
    cache_path = EXTRACTED_DIR / book / chapter_key / "image_embeddings.npz"

    image_paths, image_vectors = build_image_index(book, chapter_key, cache_path=cache_path)
    print(f"\n{len(image_paths)} images indexed for {chapter_key}")

    # Same questions as Approach 1, on purpose -- so results are directly comparable.
    sample_questions = [
        "Which duct is cut and tied during a vasectomy?",
        "What is population stabilisation?",
        "Tell me about condoms as a contraceptive method.",
    ]

    for question in sample_questions:
        print(f"\n=== Question: {question!r} ===")
        for image_path, score in retrieve(question, image_paths, image_vectors):
            print(f"  score={score:.3f}  {image_path.name}")
