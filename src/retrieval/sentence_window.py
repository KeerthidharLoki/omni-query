"""
Sentence-window expansion per PRD Layer 5.

After retrieval, each retrieved child chunk (single sentence) is expanded
to its parent window (3 sentences on each side). This gives the LLM more
context for generation without inflating the retrieval index with large chunks.

Uses LlamaIndex MetadataReplacementPostProcessor.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SentenceWindowExpander:
    """
    Expands retrieved child sentence nodes to their parent window context.

    The SentenceWindowNodeParser stores the surrounding window in the
    'window' metadata key. MetadataReplacementPostProcessor replaces the
    child text with the parent window text at generation time.
    """

    def __init__(self, target_metadata_key: str = "window"):
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor

        self._processor = MetadataReplacementPostProcessor(
            target_metadata_key=target_metadata_key
        )
        logger.info("SentenceWindowExpander ready (key=%s)", target_metadata_key)

    def expand(self, nodes: list[Any]) -> list[Any]:
        """
        Replace each node's text with its surrounding window context.

        Image chunks are not affected — their text_content (description) is kept as-is.
        """
        text_nodes = [n for n in nodes if n.metadata.get("modality") != "image"]
        image_nodes = [n for n in nodes if n.metadata.get("modality") == "image"]

        expanded_text = self._processor.postprocess_nodes(text_nodes)
        return expanded_text + image_nodes
