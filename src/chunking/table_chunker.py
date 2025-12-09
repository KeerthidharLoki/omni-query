"""
Atomic table chunking per PRD Layer 2.

Tables are never split — each table is a single chunk regardless of size.
Docling extracts table structure; we serialise to markdown for BM25 indexing
and cross-encoder reranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.ingestion.docling_parser import SemanticBlock

logger = logging.getLogger(__name__)


@dataclass
class TableChunk:
    """A single table chunk serialised to markdown, ready for embedding."""

    text: str            # markdown representation
    doc_name: str
    page_id: int
    layout_id: int
    section_title: str
    caption: str = ""
    chunk_type: str = "table"
    modality: str = "text"
    quote_id: str = ""
    parent_chunk_id: str = ""
    token_count: int = 0


class TableChunker:
    """
    Converts Docling table objects to atomic markdown TableChunks.

    Each table is stored as one chunk (atomic rule takes priority over token limit).
    Tables with no data rows (header-only) are discarded.
    """

    def chunk_from_blocks(self, blocks: list[SemanticBlock], doc_name: str) -> list[TableChunk]:
        """Extract table blocks and convert to TableChunks."""
        chunks: list[TableChunk] = []

        for block in blocks:
            if block.block_type != "table":
                continue
            if not block.text.strip():
                continue

            # block.text from Docling table is already markdown for table items
            markdown_text = block.text if "|" in block.text else self._plain_to_markdown(block.text)

            chunks.append(
                TableChunk(
                    text=markdown_text,
                    doc_name=doc_name,
                    page_id=block.page_id,
                    layout_id=block.layout_id,
                    section_title=block.section_title,
                    caption=block.caption,
                    token_count=len(markdown_text.split()),
                    parent_chunk_id=f"{doc_name}_{block.layout_id}_table",
                )
            )

        logger.info("Produced %d table chunks from %s", len(chunks), doc_name)
        return chunks

    def chunk_from_docling_table(self, table, page_id: int, layout_id: int,
                                  doc_name: str, section_title: str = "",
                                  caption: str = "") -> TableChunk:
        """Convert a raw Docling table object to a TableChunk."""
        markdown = self._serialise_docling_table(table)
        return TableChunk(
            text=markdown,
            doc_name=doc_name,
            page_id=page_id,
            layout_id=layout_id,
            section_title=section_title,
            caption=caption,
            token_count=len(markdown.split()),
            parent_chunk_id=f"{doc_name}_{layout_id}_table",
        )

    @staticmethod
    def _serialise_docling_table(table) -> str:
        """Convert Docling table data to a markdown table string."""
        if not table.data:
            return ""
        rows = []
        header = "| " + " | ".join(str(cell) for cell in table.data[0]) + " |"
        separator = "| " + " | ".join("---" for _ in table.data[0]) + " |"
        rows.append(header)
        rows.append(separator)
        for row in table.data[1:]:
            rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(rows)

    @staticmethod
    def _plain_to_markdown(text: str) -> str:
        """Best-effort conversion of plain table text to markdown."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return text
        header = "| " + " | ".join(lines[0].split()) + " |"
        sep = "| " + " | ".join("---" for _ in lines[0].split()) + " |"
        body = ["| " + " | ".join(ln.split()) + " |" for ln in lines[1:]]
        return "\n".join([header, sep] + body)
