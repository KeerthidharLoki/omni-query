"""
Gemini 2.5 Flash image description for ingestion.

At index time, each image patch extracted by PyMuPDF is sent to Gemini 2.5 Flash
once. The generated natural language description is stored in the Qdrant payload
and used for both BM25 keyword search and cross-encoder reranking.

If the MMDocRAG dataset already provides an img_description for a quote,
use that directly and skip the Gemini call.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DESCRIPTION_PROMPT = (
    "Describe this image in detail for a document retrieval system. "
    "Include: what type of visual this is (chart, table, diagram, photo), "
    "all visible labels, values, axes, legend entries, and the key finding "
    "or information conveyed. Be specific and complete."
)


class ImageDescriber:
    """
    Generates natural language descriptions of image patches using Gemini 2.5 Flash.

    Rate limits: Gemini free tier allows 10 RPM. The describer includes
    an optional rate-limit delay to avoid 429 errors during bulk ingestion.
    """

    def __init__(self, api_key: str | None = None, rate_limit_delay: float = 6.5):
        from google import genai

        self._client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._model = "gemini-2.5-flash"
        self._delay = rate_limit_delay
        logger.info("ImageDescriber ready (model=%s, delay=%.1fs)", self._model, self._delay)

    def describe(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Generate a description for a single image patch."""
        import time
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                DESCRIPTION_PROMPT,
            ],
        )
        if self._delay > 0:
            time.sleep(self._delay)
        return response.text.strip()

    def describe_with_fallback(self, image_bytes: bytes, existing_description: str = "") -> str:
        """
        Use existing description if available (MMDocRAG pre-computed),
        otherwise call Gemini. Avoids unnecessary API calls.
        """
        if existing_description and len(existing_description) > 20:
            return existing_description
        return self.describe(image_bytes)
