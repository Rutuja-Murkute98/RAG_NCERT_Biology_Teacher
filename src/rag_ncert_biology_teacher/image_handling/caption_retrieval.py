"""
WHAT
----
Approach 1 of 3 for handling images in this RAG pipeline: CAPTION each
extracted diagram/photo with Gemini, then retrieve it purely as TEXT. The
image's pixels are never embedded or looked at during search - only the
caption's words are.

WHY
---
This is the cheapest and simplest of the three strategies, and it reuses the
same text-embedding machinery the rest of the pipeline uses for normal
paragraph text - no extra infrastructure needed. It's also the foundation
Approach 3 (multimodal_answer.py) builds on for picking a candidate image.

FLOW
----
  1. List every already-extracted image for a chapter (pdf_loader.list_chapter_images
     - our extraction already solved the fragmentation problem this approach
     had to work around in the reference project, so there's no separate
     "render whole page" step here).
  2. Caption each image with Gemini -> text, dropping any Gemini itself
     flagged NOT useful (captioning.py's USEFUL marker, Step 3).
  3. Embed each surviving caption's TEXT with embed_texts() (Step 4's
     embeddings.py, gemini-embedding-2).
  4. At query time, embed the student's question with the SAME embed_texts().
  5. Rank captions by cosine similarity to the question.
  6. Return the image attached to the winning caption.

LOGIC / MECHANISM
------------------
embed_texts() returns unit-normalized vectors, so a plain dot product
between two vectors already equals their cosine similarity - no extra
division needed.

Real limitation to remember: retrieval quality here is bottlenecked by the
CAPTION's wording. If a caption never uses words related to the question,
this approach won't surface it, even if the diagram is visually relevant.
That gap is what Approach 2 (embedding_retrieval.py) fixes.
"""

import json
from pathlib import Path

import numpy as np

from rag_ncert_biology_teacher.indexing.embeddings import embed_texts
from rag_ncert_biology_teacher.ingestion.captioning import caption_image
from rag_ncert_biology_teacher.ingestion.pdf_loader import list_chapter_images


def build_caption_index(book: str, chapter_key: str, cache_path: Path | None = None):
    """Caption every extracted image for one chapter, embed the useful ones.

    cache_path (optional): a .json file to save captions to / resume from.
    Captioning costs real money and time (one Gemini call per image) -- a
    crash partway through a 13-chapter batch run should lose at most one
    image's work, not the whole chapter, so the cache is saved after EVERY
    image, not just at the end.

    Returns three parallel lists/arrays (same order, same length, USEFUL
    images only - see captioning.py): image_paths, captions, caption_vectors.
    """
    all_image_paths = list_chapter_images(book, chapter_key)

    entries: list[dict] = []  # {"caption": str, "useful": bool}, aligned with all_image_paths
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        # Only trust the cache if it's for these EXACT images, in this exact
        # order -- otherwise a reused cache_path could attach the wrong
        # caption to the wrong image.
        if cached["image_paths"] == [str(p) for p in all_image_paths]:
            entries = cached["entries"]

    if len(entries) < len(all_image_paths):
        remaining = all_image_paths[len(entries) :]
        if entries:
            print(f"Resuming captioning: {len(entries)}/{len(all_image_paths)} already cached, {len(remaining)} to go")
        elif all_image_paths:
            print(f"Captioning {len(all_image_paths)} images with Gemini...")

        for image_path in remaining:
            caption, useful = caption_image(image_path)
            entries.append({"caption": caption, "useful": useful})
            # Save BEFORE printing -- a paid-for caption must survive even if the
            # print itself blows up (e.g. Windows' console can't encode every
            # character Gemini legitimately outputs, see __main__ below).
            if cache_path:  # save after EVERY image -- a crash loses at most one image's work
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"image_paths": [str(p) for p in all_image_paths], "entries": entries}, indent=2)
                )
            tag = "" if useful else "  [dropped - not useful]"
            print(f"  {image_path.name}: {caption[:80]}{tag}")
    else:
        print(f"Loaded {len(entries)} cached captions from {cache_path} (skipping Gemini calls)")

    image_paths = [p for p, e in zip(all_image_paths, entries) if e["useful"]]
    captions = [e["caption"] for e in entries if e["useful"]]

    if not captions:
        return [], [], np.zeros((0, 3072))

    print(f"\nEmbedding {len(captions)} captions (gemini-embedding-2)...")
    caption_vectors = embed_texts(captions)

    return image_paths, captions, caption_vectors


def retrieve(query: str, image_paths, captions, caption_vectors, top_k: int = 3):
    """Embed the query the SAME way as the captions, rank by cosine similarity."""
    if len(image_paths) == 0:
        return []
    query_vector = embed_texts([query])[0]
    similarities = caption_vectors @ query_vector  # dot product of unit vectors = cosine sim

    ranked_indices = np.argsort(-similarities)[:top_k]
    return [(image_paths[i], captions[i], float(similarities[i])) for i in ranked_indices]


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.image_handling.caption_retrieval
    from rag_ncert_biology_teacher.config import EXTRACTED_DIR

    book, chapter_key = "class12_biology", "chapter_03"
    cache_path = EXTRACTED_DIR / book / chapter_key / "caption_index.json"

    image_paths, captions, caption_vectors = build_caption_index(book, chapter_key, cache_path=cache_path)
    print(f"\n{len(image_paths)} useful images indexed for {chapter_key}")

    sample_questions = [
        "Which duct is cut and tied during a vasectomy?",
        "What is population stabilisation?",
        "Tell me about condoms as a contraceptive method.",
    ]

    for question in sample_questions:
        print(f"\n=== Question: {question!r} ===")
        for image_path, caption, score in retrieve(question, image_paths, captions, caption_vectors):
            print(f"  score={score:.3f}  {image_path.name}: {caption[:110]}")
