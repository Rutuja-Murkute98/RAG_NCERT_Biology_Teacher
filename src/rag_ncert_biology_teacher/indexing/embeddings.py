"""Embedding wrappers, backed entirely by Vertex AI (via the same google-genai
SDK/client pattern already proven in captioning.py) - no local model anywhere.

Brought forward ahead of chunking/Chroma (its usual place in a RAG build)
because image_handling/'s Approach 1 (rank captions) and Approach 2 (rank raw
image pixels) both need a working embedding function before they can rank
anything at all - chunking doesn't need to exist first for that.

Two SEPARATE embedding spaces live here, and they must never be compared to
each other (different models -> incompatible vector spaces, like comparing
miles to kilograms):
  - embed_texts()             -> text-embedding-005, TEXT ONLY (chunks/queries)
  - embed_image()/embed_text_multimodal() -> gemini-embedding-2, a JOINT
    image+text space (Approach 2's whole point: compare an image's vector
    directly against a text query's vector)

gemini-embedding-2 is Step 1's verified replacement for the deprecated
multimodalembedding@001, and only reachable via the "global" endpoint, not
GOOGLE_CLOUD_LOCATION - see config.py's MULTIMODAL_EMBEDDING_LOCATION.
"""

import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from rag_ncert_biology_teacher.config import (
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    MULTIMODAL_EMBEDDING_LOCATION,
    MULTIMODAL_EMBEDDING_MODEL_NAME,
    TEXT_EMBEDDING_MODEL_NAME,
)
from rag_ncert_biology_teacher.retry_utils import call_with_retry

_text_client: genai.Client | None = None
_multimodal_client: genai.Client | None = None
_BATCH_SIZE = 50  # Vertex AI caps texts-per-call; loop internally so callers pass any-length lists


def _get_text_client() -> genai.Client:
    global _text_client
    if _text_client is None:
        _text_client = genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
    return _text_client


def _get_multimodal_client() -> genai.Client:
    global _multimodal_client
    if _multimodal_client is None:
        _multimodal_client = genai.Client(
            vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=MULTIMODAL_EMBEDDING_LOCATION
        )
    return _multimodal_client


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a LIST of texts via text-embedding-005, batched, rate-limit
    protected. Returns a 2D array: one unit-normalized row per input text,
    same order. Unit-normalized so cosine similarity anywhere downstream is
    just a plain dot product.
    """
    client = _get_text_client()
    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = call_with_retry(
            lambda b=batch: client.models.embed_content(model=TEXT_EMBEDDING_MODEL_NAME, contents=b)
        )
        all_vectors.extend(e.values for e in response.embeddings)
        if start + _BATCH_SIZE < len(texts):
            time.sleep(1)  # stay gentle on the per-minute quota

    vectors = np.array(all_vectors)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_image(image_path: Path) -> np.ndarray:
    """Embed one image's raw pixels via gemini-embedding-2 - Approach 2's
    joint image+text space. No caption/text is involved at all here.
    """
    client = _get_multimodal_client()
    image_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    response = call_with_retry(
        lambda: client.models.embed_content(
            model=MULTIMODAL_EMBEDDING_MODEL_NAME,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        )
    )
    vector = np.array(response.embeddings[0].values)
    return vector / np.linalg.norm(vector)


def embed_text_multimodal(text: str) -> np.ndarray:
    """Embed a text QUERY into the SAME space as embed_image()'s output.
    Must use gemini-embedding-2, never text-embedding-005 - comparing
    vectors from two different models would be meaningless.
    """
    client = _get_multimodal_client()
    response = call_with_retry(
        lambda: client.models.embed_content(model=MULTIMODAL_EMBEDDING_MODEL_NAME, contents=[text])
    )
    vector = np.array(response.embeddings[0].values)
    return vector / np.linalg.norm(vector)


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.indexing.embeddings
    sample_texts = [
        "The Fallopian tubes are cut and tied during tubal ligation.",
        "Photosynthesis occurs in the chloroplasts of plant cells.",
    ]
    text_vectors = embed_texts(sample_texts)
    print(f"embed_texts ({TEXT_EMBEDDING_MODEL_NAME}): shape {text_vectors.shape}")

    from rag_ncert_biology_teacher.config import EXTRACTED_DIR

    sample_image = EXTRACTED_DIR / "class12_biology" / "chapter_03" / "images" / "page_005_diagram_0.png"
    image_vector = embed_image(sample_image)
    text_multimodal_vector = embed_text_multimodal("surgical sterilisation of the male reproductive system")
    print(f"embed_image ({MULTIMODAL_EMBEDDING_MODEL_NAME}): shape {image_vector.shape}")
    print(f"embed_text_multimodal: shape {text_multimodal_vector.shape}")
    print(f"cosine similarity (image vs. related-topic text query): {image_vector @ text_multimodal_vector:.4f}")
