"""
Multimodal prompt builder for Gemini 2.5 Flash generation.

Assembles the final generation prompt from:
  - The user's question
  - Retrieved text/table chunks (with citation IDs)
  - Retrieved image chunks (with base64-encoded images)
  - System framing instructions

The prompt follows a strict Evidence → Question → Instruction pattern
to minimise hallucination and ensure citations anchor every claim.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional


_SYSTEM_INSTRUCTION = (
    "You are a precise document analyst. Answer the question using ONLY the evidence "
    "provided below. Cite each factual claim with the corresponding quote_id in brackets "
    "(e.g. [text_1], [img_2]). If the evidence is insufficient, say so explicitly. "
    "Do not speculate beyond what the documents show."
)


def _load_image_b64(img_path: str, data_root: str) -> Optional[str]:
    """
    Load an image from disk and return a base64-encoded data URI.

    img_path may be relative (e.g. 'images/COSTCO_2021_10K/page_51.jpg').
    data_root is the base data directory containing the images folder.
    """
    full_path = Path(data_root) / img_path
    if not full_path.exists():
        return None
    with open(full_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    suffix = full_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return f"data:{mime};base64,{encoded}"


def build_text_evidence_block(text_citations: list[dict]) -> str:
    """
    Format text/table citations into an evidence block.

    Each citation dict must have: quote_id, text_preview (or text), page_id.
    """
    if not text_citations:
        return ""
    lines = ["--- TEXT / TABLE EVIDENCE ---"]
    for c in text_citations:
        qid = c.get("quote_id", "?")
        page = c.get("page_id", "?")
        text = c.get("text_preview") or c.get("text", "")
        lines.append(f"[{qid}] (page {page}):\n{text.strip()}")
    return "\n\n".join(lines)


def build_image_evidence_block(
    image_citations: list[dict],
    data_root: str,
) -> list[dict]:
    """
    Build a list of Gemini content parts for image evidence.

    Returns a list of dicts with keys 'type' and either 'text' or 'inline_data'.
    Compatible with the google-genai Part format expected by gemini_generator.py.
    """
    parts: list[dict] = []
    for c in image_citations:
        qid = c.get("quote_id", "?")
        page = c.get("page_id", "?")
        img_path = c.get("img_path", "")
        description = c.get("description", "")

        label = f"[{qid}] (page {page})"
        if description:
            label += f": {description}"
        parts.append({"type": "text", "text": label})

        if img_path:
            b64 = _load_image_b64(img_path, data_root)
            if b64:
                # Strip the data URI header for the raw bytes part
                header, data = b64.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                parts.append(
                    {
                        "type": "inline_data",
                        "mime_type": mime,
                        "data": data,
                    }
                )
    return parts


def build_prompt(
    question: str,
    text_citations: list[dict],
    image_citations: list[dict],
    data_root: str = "",
) -> tuple[str, list[dict]]:
    """
    Assemble the full generation prompt.

    Returns:
        text_prompt   — the assembled text prompt string
        image_parts   — list of inline image part dicts (may be empty)

    The caller (gemini_generator.py) merges text_prompt with image_parts
    into the Gemini content list.
    """
    sections = [_SYSTEM_INSTRUCTION, ""]

    text_block = build_text_evidence_block(text_citations)
    if text_block:
        sections.append(text_block)
        sections.append("")

    if image_citations:
        sections.append("--- IMAGE / CHART EVIDENCE ---")
        for c in image_citations:
            qid = c.get("quote_id", "?")
            page = c.get("page_id", "?")
            desc = c.get("description", "")
            label = f"[{qid}] (page {page})"
            if desc:
                label += f" — {desc}"
            sections.append(label)
        sections.append("")

    sections.append(f"QUESTION: {question}")
    sections.append("")
    sections.append(
        "ANSWER (cite every claim with [quote_id]; be concise and factual):"
    )

    text_prompt = "\n".join(sections)

    image_parts: list[dict] = []
    if image_citations and data_root:
        image_parts = build_image_evidence_block(image_citations, data_root)

    return text_prompt, image_parts
