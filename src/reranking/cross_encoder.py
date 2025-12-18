"""
Cross-encoder reranker per PRD Layer 6.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
Input: Top-50 candidates from hybrid retriever
Output: Top-10 reranked candidates for generation

Why cross-encoder over bi-encoder for reranking:
  Bi-encoder (Gemini Embedding 2): embeds query and document independently.
  Efficient for recall over 100K+ vectors but less precise.

  Cross-encoder: takes (query, document) together with full attention.
  Far more accurate but cannot scale to the full corpus — used only on
  the Top-50 shortlist from the retriever.

CPU-capable. ~150ms for 50 pairs on a modern CPU.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks retrieval candidates using a cross-encoder relevance model.

    Image chunks are scored using their text_content (Gemini description + caption)
    as the document side of the (query, document) pair. This is less accurate than
    a vision-capable reranker but practical for a portfolio system.
    """

    def __init__(self, model_name: str = RERANKER_MODEL, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, max_length=max_length)
        logger.info("CrossEncoderReranker ready (model=%s)", model_name)

    def rerank(self, query: str, nodes: list[Any], top_k: int = 10) -> list[tuple[float, Any]]:
        """
        Score all (query, node) pairs and return Top-K by descending score.

        Returns list of (score, node) tuples to preserve score for citations.
        """
        if not nodes:
            return []

        pairs = [(query, self._get_text(node)) for node in nodes]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores, nodes), key=lambda x: float(x[0]), reverse=True)
        return [(float(score), node) for score, node in ranked[:top_k]]

    @staticmethod
    def _get_text(node: Any) -> str:
        """Extract text for the cross-encoder from any node type."""
        # LlamaIndex nodes
        if hasattr(node, "get_content"):
            return node.get_content()[:512]
        # Plain dict (MMDocRAG quotes)
        if isinstance(node, dict):
            return (node.get("text") or node.get("img_description") or "")[:512]
        return str(node)[:512]
