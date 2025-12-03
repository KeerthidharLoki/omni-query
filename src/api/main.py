"""
Omni-Query FastAPI application.

Loads MMDocRAG JSONL records at startup and serves them through the
/query, /suggest, /evaluate, /ingest, and /health endpoints.
Images are served as static files from data/raw/images/.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import query, ingest, evaluate, health

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("omni_query.main")

# Resolve paths relative to project root
BASE_DIR = Path(__file__).resolve().parents[2]
DEV_FILE = BASE_DIR / os.getenv("DEV_FILE", "data/dev_15.jsonl")
EVAL_FILE = BASE_DIR / os.getenv("EVAL_FILE", "data/evaluation_15.jsonl")
IMAGES_DIR = BASE_DIR / os.getenv("IMAGES_DIR", "data/raw/images")


def _load_jsonl(path: Path) -> list[dict]:
    logger.info("Loading %s …", path.name)
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d records from %s", len(records), path.name)
    return records


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load dataset into app state on startup
    app.state.dev_records = _load_jsonl(DEV_FILE)
    app.state.eval_records = _load_jsonl(EVAL_FILE)
    logger.info(
        "Ready — %d dev + %d eval records loaded",
        len(app.state.dev_records),
        len(app.state.eval_records),
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Omni-Query",
    description=(
        "Multimodal RAG for Long-Document Intelligence. "
        "Built over the MMDocRAG corpus — 222 PDFs, 4,055 expert-annotated QA pairs."
    ),
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images from data/raw/images/ at /images/
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
    logger.info("Serving images from %s at /images/", IMAGES_DIR)
else:
    logger.warning("Images directory not found: %s", IMAGES_DIR)

# Include routers
app.include_router(query.router, tags=["Query"])
app.include_router(ingest.router, tags=["Ingest"])
app.include_router(evaluate.router, tags=["Evaluate"])
app.include_router(health.router, tags=["Health"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Omni-Query",
        "version": "3.1.0",
        "docs": "/docs",
        "health": "/health",
    }
