"""
PyMuPDF-based image patch extractor.

Extracts raw image bytes from each PDF page along with pixel coordinates.
Used as the companion to DoclingParser — Docling handles text layout,
PyMuPDF handles image extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImagePatch:
    """A raw image patch extracted from a PDF page."""

    image_bytes: bytes
    page_id: int
    bbox: dict              # {"x0", "y0", "x1", "y1"} in points
    bbox_norm: dict         # normalised 0–1 relative to page dimensions
    xref: int               # PyMuPDF internal image reference
    ext: str = "jpeg"       # output format


class ImageExtractor:
    """
    Extracts image patches from PDF pages using PyMuPDF (fitz).

    Each image on a page is extracted as a separate ImagePatch.
    Images smaller than min_size_px on either dimension are discarded
    as likely decorative artefacts (logos, bullets, borders).
    """

    def __init__(self, min_size_px: int = 50, output_format: str = "jpeg"):
        self.min_size_px = min_size_px
        self.output_format = output_format

    def extract(self, pdf_path: str | Path) -> list[ImagePatch]:
        """Extract all images from a PDF, returning one ImagePatch per image."""
        import fitz  # PyMuPDF

        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        patches: list[ImagePatch] = []

        for page_num, page in enumerate(doc, start=1):
            page_rect = page.rect
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < self.min_size_px or height < self.min_size_px:
                    continue

                # Get bounding box on the page
                rects = page.get_image_rects(xref)
                bbox = rects[0] if rects else fitz.Rect(0, 0, width, height)

                image_bytes = base_image["image"]
                if self.output_format == "jpeg" and base_image.get("ext") != "jpeg":
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes("jpeg")

                patches.append(
                    ImagePatch(
                        image_bytes=image_bytes,
                        page_id=page_num,
                        bbox={"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
                        bbox_norm={
                            "x0": bbox.x0 / page_rect.width,
                            "y0": bbox.y0 / page_rect.height,
                            "x1": bbox.x1 / page_rect.width,
                            "y1": bbox.y1 / page_rect.height,
                        },
                        xref=xref,
                        ext=self.output_format,
                    )
                )

        doc.close()
        logger.info("Extracted %d image patches from %s", len(patches), pdf_path.name)
        return patches
