"""Embedding wrappers, backed by the Gemini Developer API (Google AI Studio) -
no local model anywhere.

Brought forward ahead of chunking/Chroma (its usual place in a RAG build)
because image_handling/'s Approach 1 (rank captions) and Approach 2 (rank raw
image pixels) both need a working embedding function before they can rank
anything at all - chunking doesn't need to exist first for that.

ONE embedding model, ONE vector space, for everything -- gemini-embedding-2
is natively multimodal (verified directly: text and raw image bytes both
embed successfully into the same 3072-dim space), so text chunks, image
captions, and raw image pixels all land in the same space already. This
replaces an earlier two-model Vertex AI setup (text-embedding-005 for text,
a separate gemini-embedding-2 via Vertex's "global" endpoint for images)
that deliberately kept those as two SEPARATE, never-comparable spaces --
that split existed only because Vertex's text-embedding-005 had no
multimodal counterpart in the same space. The Developer API doesn't have
that limitation, so embed_image() and embed_text() now genuinely return
directly-comparable vectors, which is what Approach 2 (embedding_retrieval.py)
wants in the first place.
"""

import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from rag_ncert_biology_teacher.config import EMBEDDING_MODEL_NAME, GEMINI_API_KEY
from rag_ncert_biology_teacher.retry_utils import call_with_retry

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a LIST of texts, rate-limit protected. Returns a 2D array: one
    unit-normalized row per input text, same order. Unit-normalized so
    cosine similarity anywhere downstream is just a plain dot product.

    ONE call per text, deliberately -- unlike Vertex AI's embed_content
    (which took a list and returned one embedding per item), the Developer
    API's embed_content treats a list of strings as a SINGLE multi-part
    input and returns exactly ONE embedding for the whole batch. Found for
    real: passing 3 texts silently came back as 1 embedding, not 3 --
    something that only surfaced as a downstream IndexError deep inside
    Chroma, not an error from this call itself, so the batched version
    looked like it worked right up until something tried to use the result.
    """
    client = _get_client()
    all_vectors: list[list[float]] = []
    for i, text in enumerate(texts):
        response = call_with_retry(lambda t=text: client.models.embed_content(model=EMBEDDING_MODEL_NAME, contents=t))
        all_vectors.extend(e.values for e in response.embeddings)
        if (i + 1) % 20 == 0:
            time.sleep(1)  # stay gentle on the free tier's per-minute quota

    vectors = np.array(all_vectors)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_image(image_path: Path) -> np.ndarray:
    """Embed one image's raw pixels -- same model/space as embed_texts(), so
    this can be compared directly against a text query's vector (Approach 2's
    whole point).
    """
    client = _get_client()
    image_bytes = Path(image_path).read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    response = call_with_retry(
        lambda: client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        )
    )
    vector = np.array(response.embeddings[0].values)
    return vector / np.linalg.norm(vector)


def embed_text_multimodal(text: str) -> np.ndarray:
    """Embed a text QUERY -- kept as a separate name (rather than just
    reusing embed_texts) for callers that are specifically comparing against
    embed_image() output, since that was the meaningful distinction under
    the old two-model setup. Now it's the same model/call either way.
    """
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.indexing.embeddings
    sample_texts = [
        "The Fallopian tubes are cut and tied during tubal ligation.",
        "Photosynthesis occurs in the chloroplasts of plant cells.",
    ]
    text_vectors = embed_texts(sample_texts)
    print(f"embed_texts ({EMBEDDING_MODEL_NAME}): shape {text_vectors.shape}")

    from rag_ncert_biology_teacher.config import EXTRACTED_DIR

    sample_image = EXTRACTED_DIR / "class12_biology" / "chapter_03" / "images" / "page_005_diagram_0.png"
    if sample_image.exists():
        image_vector = embed_image(sample_image)
        text_multimodal_vector = embed_text_multimodal("surgical sterilisation of the male reproductive system")
        print(f"embed_image ({EMBEDDING_MODEL_NAME}): shape {image_vector.shape}")
        print(f"embed_text_multimodal: shape {text_multimodal_vector.shape}")
        print(f"cosine similarity (image vs. related-topic text query): {image_vector @ text_multimodal_vector:.4f}")
    else:
        print(f"(skipping image embedding demo -- {sample_image} not found)")
