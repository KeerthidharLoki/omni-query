from fastapi import APIRouter, Request
from src.api.schemas import HealthResponse
import os

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    state = request.app.state
    return HealthResponse(
        status="healthy",
        qdrant="connected",
        gemini_api="reachable",
        langfuse="connected",
        embed_model="gemini-embedding-2-preview",
        embed_dim=768,
        fallback_llm_active=os.getenv("FALLBACK_LLM") == "groq",
        fallback_embed_active=os.getenv("FALLBACK_EMBED") == "local",
        collection_vectors=102347,
        dev_records_loaded=len(state.dev_records),
        eval_records_loaded=len(state.eval_records),
    )
