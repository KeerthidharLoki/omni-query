"""
CPU fallback embedder using all-MiniLM-L6-v2.

Activated when FALLBACK_EMBED=local in .env.
Text-only — cannot embed images. 384 dimensions (incompatible with Gemini vectors).

IMPORTANT: The Qdrant collection must be fully re-indexed when switching between
embedding models. Gemini (768-dim) and MiniLM (384-dim) vectors live in
incompatible spaces and cannot coexist in the same collection.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "all-MiniLM-L6-v2"
FALLBACK_DIM = 384


class FallbackEmbedder:
    """
    Local CPU-based text embedder using sentence-transformers.

    Requires no API key. Useful for testing without Gemini quota.
    Does NOT support image embedding — image chunks will be skipped
    in fallback mode and only text/table chunks will be indexed.
    """

    def __init__(self, model_name: str = FALLBACK_MODEL):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = FALLBACK_DIM
        logger.info("FallbackEmbedder ready (model=%s, dim=%d)", model_name, self._dim)

    def embed_text(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)

    @property
    def dimensions(self) -> int:
        return self._dim
