"""
Docling-based layout-aware PDF parser.

Extracts semantic blocks (paragraphs, headings, tables, figures) from PDFs
while preserving reading order. Handles multi-column layouts correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class SemanticBlock:
    """A layout-aware semantic unit extracted from a PDF page."""

    block_type: str          # "paragraph" | "heading" | "table" | "figure" | "list"
    text: str
    page_id: int
    layout_id: int
    section_title: str = ""
    caption: str = ""
    bbox: dict = field(default_factory=dict)  # {"x0", "y0", "x1", "y1"} normalised 0–1


class DoclingParser:
    """
    Wraps Docling DocumentConverter for layout-aware PDF parsing.

    Configuration mirrors PRD Layer 1:
    - do_ocr=False  (corpus is digital, not scanned)
    - do_table_structure=True with cell matching
    """

    def __init__(self, do_ocr: bool = False, do_table_structure: bool = True):
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("DoclingParser initialised (do_ocr=%s)", do_ocr)

    def parse(self, pdf_path: str | Path) -> list[SemanticBlock]:
        """Parse a PDF and return ordered list of semantic blocks."""
        pdf_path = Path(pdf_path)
        logger.info("Parsing %s", pdf_path.name)

        result = self._converter.convert(str(pdf_path))
        doc = result.document

        blocks: list[SemanticBlock] = []
        current_heading = ""

        for item in doc.iterate_items():
            label = getattr(item, "label", "paragraph")
            text = getattr(item, "text", "") or ""
            page_id = self._get_page_id(item)
            layout_id = getattr(item, "self_ref", 0)

            if label in ("section_header", "title"):
                current_heading = text.strip()

            block = SemanticBlock(
                block_type=self._map_label(label),
                text=text.strip(),
                page_id=page_id,
                layout_id=int(str(layout_id).replace("#/", "").replace("/", "_")[:8], 16)
                if isinstance(layout_id, str)
                else hash(str(layout_id)) % 100000,
                section_title=current_heading,
            )
            if block.text:
                blocks.append(block)

        logger.info("Extracted %d semantic blocks from %s", len(blocks), pdf_path.name)
        return blocks

    @staticmethod
    def _map_label(label: str) -> str:
        mapping = {
            "paragraph": "paragraph",
            "section_header": "heading",
            "title": "heading",
            "table": "table",
            "figure": "figure",
            "list_item": "list",
            "caption": "paragraph",
        }
        return mapping.get(label, "paragraph")

    @staticmethod
    def _get_page_id(item) -> int:
        try:
            return item.prov[0].page_no if item.prov else 0
        except (AttributeError, IndexError):
            return 0
