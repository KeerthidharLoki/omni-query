"""
Gemini 2.5 Flash multimodal answer generation per PRD Layer 8.

Receives: user query + Top-10 reranked chunks (text + base64 images).
Returns: structured JSON with answer, cited quote_ids, and optional bbox coordinates.

The structured prompt enforces citation — every claim must reference a quote_id.
If the evidence does not contain the answer, the model is instructed to say so.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise document question-answering assistant. "
    "Answer the question using ONLY the provided evidence. "
    "For each claim in your answer, cite the quote_id of the source. "
    "If an image is relevant, cite its quote_id and describe what specific region answers the question. "
    "If the evidence does not contain the answer, say so explicitly. "
    'Return your response as JSON: {"answer": str, "citations": [{"quote_id": str, "type": str, "relevant_region": str}]}'
)


class GeminiGenerator:
    """
    Multimodal answer generator using Gemini 2.5 Flash.

    Supports mixed text + image context. Each image chunk is sent as base64
    JPEG bytes alongside its description text. Gemini's native multimodal
    capability aligns the visual and text evidence before generating an answer.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        from google import genai

        self._client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._model = model
        logger.info("GeminiGenerator ready (model=%s)", model)

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate a grounded answer from retrieved chunks.

        Args:
            query: User's natural language question.
            chunks: List of chunk dicts with keys:
                quote_id, page_id, chunk_type, modality,
                text_content, [image_bytes] (for image chunks)

        Returns:
            {answer, citations: [{quote_id, type, relevant_region, page_id}]}
        """
        from google.genai import types

        parts = [SYSTEM_PROMPT]

        for chunk in chunks:
            header = (
                f"\n--- Quote {chunk['quote_id']} "
                f"(page {chunk['page_id']}, {chunk['chunk_type']}) ---\n"
            )
            parts.append(header)

            if chunk.get("modality") == "image" and chunk.get("image_bytes"):
                parts.append(
                    types.Part.from_bytes(
                        data=chunk["image_bytes"], mime_type="image/jpeg"
                    )
                )
            parts.append(chunk.get("text_content", ""))

        parts.append(f"\nQuestion: {query}")

        response = self._client.models.generate_content(
            model=self._model, contents=parts
        )

        raw = response.text.strip()
        # Strip markdown code block if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: wrap plain text answer
            return {"answer": raw, "citations": []}
