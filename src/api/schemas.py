"""Pydantic schemas for all API request and response bodies."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── /ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: str
    pages_processed: int
    text_chunks: int
    table_chunks: int
    image_chunks: int
    vectors_upserted: int
    embed_model: str
    embed_dim: int
    status: str
    duration_seconds: float


# ── /suggest ───────────────────────────────────────────────────────────────────

class SuggestResponse(BaseModel):
    suggestions: list[str]


# ── /query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language question")
    top_k: int = Field(10, ge=1, le=50)
    doc_filter: Optional[str] = None
    domain_filter: Optional[str] = None


class TextCitation(BaseModel):
    quote_id: str
    type: str = "text"
    page_id: int
    text_preview: str
    rerank_score: float


class ImageCitation(BaseModel):
    quote_id: str
    type: str = "image"
    page_id: int
    img_filename: str          # filename only — served at /images/{filename}
    img_description: str
    relevant_region: str = ""
    rerank_score: float
    grounding_type: str = "dataset-native"


class QueryResponse(BaseModel):
    answer: str
    matched_question: str      # the dataset question this was matched to
    doc_name: str
    domain: str
    question_type: str
    evidence_modality_type: list[str]
    text_citations: list[TextCitation]
    image_citations: list[ImageCitation]
    gold_quotes: list[str]
    retrieved_quote_ids: list[str]
    recall_at_10: float
    precision_at_10: float
    llm_used: str
    embed_model: str
    retrieval_ms: int
    rerank_ms: int
    generation_ms: int
    total_ms: int
    demo_mode: bool = True


# ── /evaluate ──────────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    eval_file: str = Field("evaluation_15", description="'evaluation_15' or 'dev_15'")
    k_values: list[int] = Field([5, 10])
    max_records: Optional[int] = Field(None, description="Limit for fast iteration")
    domain_filter: Optional[str] = None


class ModalityBreakdown(BaseModel):
    recall_at_10: float
    precision_at_10: float
    answer_f1: float
    count: int


class EvaluateResponse(BaseModel):
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    answer_f1: float
    records_evaluated: int
    duration_seconds: float
    breakdown_by_modality: dict[str, ModalityBreakdown]
    breakdown_by_domain: dict[str, ModalityBreakdown]


# ── /health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    gemini_api: str
    langfuse: str
    embed_model: str
    embed_dim: int
    fallback_llm_active: bool
    fallback_embed_active: bool
    collection_vectors: int
    dev_records_loaded: int
    eval_records_loaded: int
