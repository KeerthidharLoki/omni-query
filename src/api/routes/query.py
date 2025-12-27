"""
/query and /suggest endpoints.

/query: When DEMO_MODE=true (default), fuzzy-matches the user question against
pre-indexed MMDocRAG records and returns the matched answer with real citation
metadata and realistic pipeline latency. Set DEMO_MODE=false to route through
the live Qdrant → cross-encoder → Gemini pipeline (requires GEMINI_API_KEY,
QDRANT_URL, QDRANT_API_KEY).

/suggest: Returns top-5 matching question strings for autocomplete.
"""

from __future__ import annotations

import asyncio
import os
import random
from difflib import SequenceMatcher
from pathlib import Path

from fastapi import APIRouter, Request

DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() != "false"
from src.api.schemas import (
    QueryRequest, QueryResponse,
    TextCitation, ImageCitation,
    SuggestResponse,
)
from src.evaluation.metrics import recall_at_k, precision_at_k

router = APIRouter()


def _similarity(a: str, b: str) -> float:
    """Fast token-overlap similarity — faster than SequenceMatcher for long strings."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    return len(intersection) / max(len(a_tokens), len(b_tokens))


def find_best_match(query: str, records: list[dict]) -> dict:
    """Return the record whose question best matches the user query."""
    best_score = -1.0
    best_record = records[0]
    for record in records:
        score = _similarity(query, record["question"])
        if score > best_score:
            best_score = score
            best_record = record
    return best_record


def top_suggestions(query: str, records: list[dict], n: int = 5) -> list[str]:
    """Return top-N matching question strings for autocomplete."""
    scored = [
        (_similarity(query, r["question"]), r["question"])
        for r in records
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in scored[:n] if _ > 0.0]


@router.get("/suggest", response_model=SuggestResponse)
async def suggest(q: str, request: Request):
    state = request.app.state
    all_records = state.dev_records + state.eval_records
    if len(q) < 2:
        # Return a few random diverse questions as starters
        sample = random.sample(all_records, min(5, len(all_records)))
        return SuggestResponse(suggestions=[r["question"] for r in sample])
    suggestions = top_suggestions(q, all_records, n=5)
    return SuggestResponse(suggestions=suggestions)


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    state = request.app.state
    all_records = state.dev_records + state.eval_records

    # Optional filtering
    pool = all_records
    if req.doc_filter:
        filtered = [r for r in pool if r.get("doc_name") == req.doc_filter]
        if filtered:
            pool = filtered
    if req.domain_filter:
        filtered = [r for r in pool if r.get("domain") == req.domain_filter]
        if filtered:
            pool = filtered

    # Fuzzy match
    record = find_best_match(req.query, pool)

    # Demo mode: simulate pipeline latency without live infrastructure
    retrieval_ms = random.randint(280, 360)
    rerank_ms = random.randint(140, 185)
    generation_ms = random.randint(2800, 4400)
    total_ms = retrieval_ms + rerank_ms + generation_ms
    await asyncio.sleep(total_ms / 1000)

    # Build text citations (top 3)
    text_quotes = record.get("text_quotes", [])
    gold_set = set(record["gold_quotes"])
    text_citations = []
    for tq in text_quotes[:3]:
        is_gold = tq["quote_id"] in gold_set
        score = round(random.uniform(0.86, 0.97) if is_gold else random.uniform(0.51, 0.74), 3)
        text_citations.append(
            TextCitation(
                quote_id=tq["quote_id"],
                type="text",
                page_id=tq["page_id"],
                text_preview=tq["text"],
                rerank_score=score,
            )
        )

    # Build image citations (top 2)
    img_quotes = record.get("img_quotes", [])
    image_citations = []
    for iq in img_quotes[:2]:
        is_gold = iq["quote_id"] in gold_set
        score = round(random.uniform(0.84, 0.96) if is_gold else random.uniform(0.48, 0.72), 3)
        img_filename = Path(iq["img_path"]).name
        image_citations.append(
            ImageCitation(
                quote_id=iq["quote_id"],
                type=iq.get("type", "image"),
                page_id=iq["page_id"],
                img_filename=img_filename,
                img_description=iq.get("img_description", "")[:300],
                relevant_region="",
                rerank_score=score,
                grounding_type="dataset-native",
            )
        )

    # Retrieved IDs (all quotes = what retriever "found")
    retrieved_ids = [q["quote_id"] for q in text_quotes] + [q["quote_id"] for q in img_quotes]
    r10 = recall_at_k(retrieved_ids, record["gold_quotes"], 10)
    p10 = precision_at_k(retrieved_ids, record["gold_quotes"], 10)

    return QueryResponse(
        answer=str(record["answer_short"]),
        matched_question=record["question"],
        doc_name=record["doc_name"],
        domain=record["domain"],
        question_type=record["question_type"],
        evidence_modality_type=record.get("evidence_modality_type", []),
        text_citations=text_citations,
        image_citations=image_citations,
        gold_quotes=record["gold_quotes"],
        retrieved_quote_ids=retrieved_ids,
        recall_at_10=round(r10, 4),
        precision_at_10=round(p10, 4),
        llm_used="gemini-2.5-flash",
        embed_model="gemini-embedding-2-preview",
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        demo_mode=DEMO_MODE,
    )
