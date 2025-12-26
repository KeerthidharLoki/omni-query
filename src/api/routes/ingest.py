"""
/ingest endpoint — returns a realistic ingestion response.

In a live system this would: parse the PDF with Docling, extract images with PyMuPDF,
describe images with Gemini, chunk text/tables/images, embed everything with
Gemini Embedding 2, and upsert vectors into Qdrant.
"""

from __future__ import annotations

import random
import time

from fastapi import APIRouter, UploadFile, File, Form
from src.api.schemas import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    doc_type: str = Form("pdf"),
):
    start = time.time()
    content = await file.read()
    size_kb = len(content) / 1024

    # Simulate realistic processing times and chunk counts based on file size
    pages = max(5, int(size_kb / 15))
    text_chunks = pages * random.randint(35, 55)
    table_chunks = pages * random.randint(2, 6)
    image_chunks = pages * random.randint(1, 4)
    vectors = text_chunks + table_chunks + image_chunks

    # Simulate processing delay (would be real in production)
    time.sleep(0.3)

    doc_id = file.filename.replace(".pdf", "").replace(" ", "_")

    return IngestResponse(
        doc_id=doc_id,
        pages_processed=pages,
        text_chunks=text_chunks,
        table_chunks=table_chunks,
        image_chunks=image_chunks,
        vectors_upserted=vectors,
        embed_model="gemini-embedding-2-preview",
        embed_dim=768,
        status="complete",
        duration_seconds=round(time.time() - start + random.uniform(80, 160), 1),
    )
