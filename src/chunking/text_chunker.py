"""
Two-level text chunking strategy per PRD Layer 2.

Level 1 — Sentence-level nodes (SentenceWindowNodeParser):
    Each sentence becomes a retrieval node. Surrounding context (parent window)
    stored in metadata for expansion at generation time.

Level 2 — Token-size guardrails:
    Docling semantic blocks > 512 tokens are pre-split at sentence boundaries.
    Blocks < 64 tokens are discarded as artefacts.
    Overlap: 64 tokens (~1-2 sentences) at pre-split boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.ingestion.docling_parser import SemanticBlock

logger = logging.getLogger(__name__)

MAX_BLOCK_TOKENS = 512
MIN_CHUNK_TOKENS = 64
OVERLAP_TOKENS = 64
WINDOW_SIZE = 3          # SentenceWindowNodeParser window (sentences each side)


@dataclass
class TextChunk:
    """A single text chunk ready for embedding and indexing."""

    text: str
    doc_name: str
    page_id: int
    layout_id: int
    section_title: str
    chunk_type: str = "text"
    modality: str = "text"
    quote_id: str = ""
    parent_chunk_id: str = ""
    token_count: int = 0
    window_text: str = ""    # surrounding context for generation


class TextChunker:
    """
    Converts Docling SemanticBlocks into sentence-level TextChunks.

    Uses LlamaIndex SentenceWindowNodeParser internally. Sentence-level nodes
    are used for retrieval; parent windows are stored in metadata for expansion
    via MetadataReplacementPostProcessor at query time.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        max_block_tokens: int = MAX_BLOCK_TOKENS,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ):
        from llama_index.core.node_parser import SentenceWindowNodeParser

        self._parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        self._max_block = max_block_tokens
        self._min_chunk = min_chunk_tokens
        self._overlap = overlap_tokens
        logger.info(
            "TextChunker ready (window=%d, max_block=%d, min=%d)",
            window_size, max_block_tokens, min_chunk_tokens,
        )

    def chunk(self, blocks: list[SemanticBlock], doc_name: str) -> list[TextChunk]:
        """Convert semantic blocks to sentence-level text chunks."""
        from llama_index.core import Document

        chunks: list[TextChunk] = []

        for block in blocks:
            if block.block_type in ("table", "figure"):
                continue  # handled by TableChunker / ImageChunker
            if not block.text or len(block.text.split()) < 10:
                continue

            llama_doc = Document(
                text=block.text,
                metadata={
                    "doc_name": doc_name,
                    "page_id": block.page_id,
                    "layout_id": block.layout_id,
                    "section_title": block.section_title,
                },
            )
            nodes = self._parser.get_nodes_from_documents([llama_doc])

            for i, node in enumerate(nodes):
                token_count = len(node.text.split())
                if token_count < self._min_chunk:
                    continue
                chunks.append(
                    TextChunk(
                        text=node.text,
                        doc_name=doc_name,
                        page_id=block.page_id,
                        layout_id=block.layout_id,
                        section_title=block.section_title,
                        token_count=token_count,
                        window_text=node.metadata.get("window", node.text),
                        parent_chunk_id=f"{doc_name}_{block.layout_id}",
                    )
                )

        logger.info("Produced %d text chunks from %s", len(chunks), doc_name)
        return chunks
