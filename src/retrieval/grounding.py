"""
3-level image grounding per PRD Layer 7.

Level 1 — Page grounding (always available):
    Returns page_id + img_path from the MMDocRAG dataset metadata.
    The image is rendered directly in the citation panel.

Level 2 — Region grounding (Gemini-approximated):
    Sends the image back to Gemini 2.5 Flash with a localisation prompt.
    Returns normalised bounding box coordinates (x_min, y_min, x_max, y_max).
    Marked as grounding_type="model_approximated" — not ground truth.

Level 3 — Layout anchor (structural grounding):
    Uses layout_id from MinerU's parse to construct a structural citation
    (figure number, section context, page position).

Documented limitation: Level 2 bbox coordinates are model-approximated.
They are stored with a grounding_type flag so downstream consumers can
display them with appropriate caveats.
"""

from __future__ import annotations

import json
import os
from typing import Optional


def level1_grounding(chunk_metadata: dict) -> dict:
    """
    Level 1 — Page grounding.

    Always available for any image chunk in the MMDocRAG dataset.
    Returns the page_id, img_path, and doc_name from chunk metadata.
    """
    return {
        "grounding_type": "page",
        "quote_id": chunk_metadata.get("quote_id", ""),
        "page_id": chunk_metadata.get("page_id"),
        "img_path": chunk_metadata.get("img_path", ""),
        "doc_name": chunk_metadata.get("doc_name", ""),
        "bbox": None,
    }


def level2_grounding(
    image_bytes: bytes,
    answer_sentence: str,
    model: str = "gemini-2.5-flash",
) -> Optional[dict]:
    """
    Level 2 — Region grounding via Gemini bbox localisation.

    Sends the image to Gemini 2.5 Flash with a prompt asking it to identify
    the region most relevant to `answer_sentence`. Returns normalised
    bounding box coordinates.

    Returns None if the API call fails or returns malformed JSON.

    Note: Coordinates are model-approximated, not annotated ground truth.
    Flag all Level 2 citations with grounding_type="model_approximated".
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt = (
            f"In this image, identify the region most relevant to the following statement: "
            f'"{answer_sentence}". '
            f"Return ONLY a JSON object with keys: "
            f'"x_min", "y_min", "x_max", "y_max" as normalised coordinates (0.0 to 1.0). '
            f"No other text."
        )

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        bbox = json.loads(raw.strip())

        return {
            "grounding_type": "model_approximated",
            "bbox": {
                "x_min": float(bbox["x_min"]),
                "y_min": float(bbox["y_min"]),
                "x_max": float(bbox["x_max"]),
                "y_max": float(bbox["y_max"]),
            },
        }

    except Exception as e:
        # Non-fatal: fall back to Level 1 grounding
        return None


def level3_grounding(chunk_metadata: dict) -> dict:
    """
    Level 3 — Layout anchor (structural grounding).

    Uses layout_id from MinerU's parse to construct a structured citation
    that ties the image to its position in the document's layout tree.
    """
    doc_name = chunk_metadata.get("doc_name", "unknown")
    page_id = chunk_metadata.get("page_id", "?")
    layout_id = chunk_metadata.get("layout_id", "?")
    section_title = chunk_metadata.get("section_title", "")

    citation_text = (
        f"Figure from {doc_name}, "
        f"page {page_id}, "
        f"layout region {layout_id}"
    )
    if section_title:
        citation_text += f" (section: {section_title})"

    return {
        "grounding_type": "layout_anchor",
        "citation_text": citation_text,
        "layout_id": layout_id,
        "page_id": page_id,
        "doc_name": doc_name,
    }


def ground_image_citation(
    chunk_metadata: dict,
    image_bytes: Optional[bytes] = None,
    answer_sentence: Optional[str] = None,
    use_level2: bool = False,
) -> dict:
    """
    Full 3-level grounding pipeline for a single image citation.

    Attempts Level 2 if `use_level2=True` and `image_bytes` + `answer_sentence`
    are provided. Falls back to Level 1 if Level 2 fails. Always includes
    Level 3 structural metadata.

    Returns a unified citation dict with all available grounding fields.
    """
    result = level1_grounding(chunk_metadata)
    result.update(level3_grounding(chunk_metadata))

    if use_level2 and image_bytes and answer_sentence:
        l2 = level2_grounding(image_bytes, answer_sentence)
        if l2:
            result["bbox"] = l2["bbox"]
            result["grounding_type"] = "model_approximated"

    return result
