"""
RAGAS Faithfulness evaluation per PRD Layer 9.

Faithfulness measures whether the generated answer is grounded in the
retrieved context — i.e., the LLM is not hallucinating beyond the evidence.

Requires an LLM-as-judge (Gemini 2.5 Flash recommended).
Uses RAGAS >= 0.4 collections-based API (legacy evaluate() is deprecated).

Low faithfulness + high recall → generation layer problem.
Low faithfulness + low recall → retrieval layer problem (bad context in).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def evaluate_faithfulness(
    query: str,
    answer: str,
    contexts: list[str],
    evaluator_llm=None,
) -> float:
    """
    Evaluate whether `answer` is grounded in `contexts` using RAGAS Faithfulness.

    Args:
        query: Original user question.
        answer: Generated answer to evaluate.
        contexts: List of text strings from retrieved chunks (Top-10 context).
        evaluator_llm: LlamaIndex-compatible LLM for judging. Defaults to Gemini.

    Returns:
        Faithfulness score in [0.0, 1.0]. Higher = more grounded.
    """
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness

    if evaluator_llm is None:
        from llama_index.llms.google_genai import GoogleGenAI
        import os
        evaluator_llm = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    scorer = Faithfulness(llm=evaluator_llm)
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )
    score = await scorer.single_turn_ascore(sample)
    return float(score)


async def batch_faithfulness(
    queries: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    evaluator_llm=None,
) -> list[float]:
    """Evaluate faithfulness for a batch of (query, answer, contexts) triples."""
    import asyncio

    tasks = [
        evaluate_faithfulness(q, a, c, evaluator_llm)
        for q, a, c in zip(queries, answers, contexts_list)
    ]
    return await asyncio.gather(*tasks)
