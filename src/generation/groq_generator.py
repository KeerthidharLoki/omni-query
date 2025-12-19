"""
Groq Llama 4 Scout fallback generator per PRD Layer 8.

Activated when FALLBACK_LLM=groq in .env.
Text-only context (no image bytes). Zero code change at the call site —
LlamaIndex LLM abstraction handles the swap.

Model: meta-llama/llama-4-scout-17b-16e-instruct
  17B active parameters, 16 experts, 109B total parameters.
  Groq free tier: 30 RPM.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = (
    "You are a precise document question-answering assistant. "
    "Answer the question using ONLY the provided evidence (text only — images not available in fallback mode). "
    "For each claim cite the quote_id. "
    'Return JSON: {"answer": str, "citations": [{"quote_id": str, "type": "text", "relevant_region": ""}]}'
)


class GroqGenerator:
    """
    Text-only fallback generator using Groq's Llama 4 Scout.

    Image chunks are skipped in fallback mode — only text and table chunks
    are included in the context. The answer may be less accurate for
    image-evidence questions when running in fallback mode.
    """

    def __init__(self, api_key: str | None = None, model: str = GROQ_MODEL):
        from groq import Groq

        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self._model = model
        logger.info("GroqGenerator ready (model=%s)", model)

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate answer using text-only context (image chunks excluded)."""
        context_parts = []
        for chunk in chunks:
            if chunk.get("modality") == "image":
                continue  # skip image chunks in text-only fallback
            context_parts.append(
                f"[{chunk['quote_id']}, page {chunk['page_id']}]\n{chunk.get('text_content', '')}"
            )

        user_message = "\n\n".join(context_parts) + f"\n\nQuestion: {query}"

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"answer": raw, "citations": []}
