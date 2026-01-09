"""Tests for Pydantic schemas in src/api/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    TextCitation,
    ImageCitation,
    HealthResponse,
    EvaluateRequest,
)


# ── QueryRequest ───────────────────────────────────────────────────────────────

def test_query_request_valid():
    r = QueryRequest(query="what is the debt ratio?")
    assert r.query == "what is the debt ratio?"
    assert r.top_k == 10

def test_query_request_too_short():
    with pytest.raises(ValidationError):
        QueryRequest(query="hi")

def test_query_request_top_k_too_low():
    with pytest.raises(ValidationError):
        QueryRequest(query="valid query", top_k=0)

def test_query_request_top_k_too_high():
    with pytest.raises(ValidationError):
        QueryRequest(query="valid query", top_k=51)

def test_query_request_top_k_max():
    r = QueryRequest(query="valid query", top_k=50)
    assert r.top_k == 50


# ── QueryResponse ──────────────────────────────────────────────────────────────

def test_query_response_has_demo_mode():
    r = QueryResponse(
        answer="The ratio was 0.32.",
        matched_question="What is the debt ratio?",
        doc_name="COSTCO_2021",
        domain="Financial report",
        question_type="numerical",
        evidence_modality_type=["chart", "text"],
        text_citations=[],
        image_citations=[],
        gold_quotes=["q1"],
        retrieved_quote_ids=["q1", "q2"],
        recall_at_10=0.5,
        precision_at_10=0.2,
        llm_used="gemini-2.5-flash",
        embed_model="gemini-embedding-2-preview",
        retrieval_ms=310,
        rerank_ms=160,
        generation_ms=3200,
        total_ms=3670,
    )
    assert hasattr(r, "demo_mode")
    assert r.demo_mode is True  # default


# ── TextCitation / ImageCitation ───────────────────────────────────────────────

def test_text_citation_type_default():
    c = TextCitation(quote_id="q1", page_id=3, text_preview="some text", rerank_score=0.9)
    assert c.type == "text"

def test_image_citation_grounding_default():
    c = ImageCitation(
        quote_id="q2", page_id=4,
        img_filename="fig_001.jpg",
        img_description="A bar chart",
        rerank_score=0.85,
    )
    assert c.grounding_type == "dataset-native"


# ── HealthResponse ─────────────────────────────────────────────────────────────

def test_health_response_construction():
    h = HealthResponse(
        status="healthy",
        qdrant="healthy",
        gemini_api="healthy",
        langfuse="healthy",
        embed_model="gemini-embedding-2-preview",
        embed_dim=768,
        fallback_llm_active=False,
        fallback_embed_active=False,
        collection_vectors=102347,
        dev_records_loaded=2055,
        eval_records_loaded=2000,
    )
    assert h.status == "healthy"


# ── EvaluateRequest ────────────────────────────────────────────────────────────

def test_evaluate_request_defaults():
    r = EvaluateRequest()
    assert r.k_values == [5, 10]
    assert r.eval_file == "evaluation_15"
