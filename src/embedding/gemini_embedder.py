"""
Gemini Embedding 2 — unified 768-dim vector space for text and images.

Key design decision: text chunks and image patches are embedded into the SAME
768-dim vector space using the same model. A single ANN search over Qdrant
retrieves both modalities simultaneously — no separate indexes, no alignment tricks.

Model: gemini-embedding-2-preview
Dimensions: 768 (MRL truncated from 3072 — fits Qdrant Cloud Free 1 GB RAM)
Task types:
  - RETRIEVAL_DOCUMENT at index time
  - RETRIEVAL_QUERY at query time
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIM = 768


class GeminiEmbedder:
    """
    Embeds text chunks and image patches into a unified 768-dim vector space.

    At index time: task_type=RETRIEVAL_DOCUMENT
    At query time: task_type=RETRIEVAL_QUERY

    Free tier rate limit: ~1500 RPM for embedding. Batch where possible.
    """

    def __init__(self, api_key: str | None = None, dimensions: int = EMBED_DIM):
        from google import genai

        self._client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._model = EMBED_MODEL
        self._dim = dimensions
        logger.info("GeminiEmbedder ready (model=%s, dim=%d)", self._model, self._dim)

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Embed a single text string."""
        from google.genai import types

        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dim,
            ),
        )
        return result.embeddings[0].values

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Embed multiple text strings. Gemini supports up to 100 per batch."""
        from google.genai import types

        # Batch in groups of 100
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            result = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dim,
                ),
            )
            all_vectors.extend(emb.values for emb in result.embeddings)
        return all_vectors

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> list[float]:
        """Embed a raw image patch into the same 768-dim space as text."""
        from google.genai import types

        result = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dim,
            ),
        )
        return result.embeddings[0].values

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query for retrieval."""
        return self.embed_text(query, task_type="RETRIEVAL_QUERY")
