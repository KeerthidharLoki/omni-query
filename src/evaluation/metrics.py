"""
Evaluation metrics per PRD Layer 9.

Four independent metrics — a low score on any one points to a specific
pipeline layer:

  Recall@K      → Low = fix retrieval layer
  Precision@K   → Low with high recall = reranker not filtering well
  Answer F1     → Low with high Recall = fix generation / prompt
  RAGAS Faith.  → Low = LLM hallucinating beyond retrieved evidence
  Citation acc. → Low = answer right but attribution wrong

All metrics are deterministic and LLM-free (except RAGAS, which uses an
LLM judge — see ragas_eval.py).
"""

from __future__ import annotations

from collections import Counter


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """
    Fraction of gold evidence quotes that appear in the Top-K retrieved results.

    recall@K = |retrieved[:K] ∩ gold| / |gold|
    """
    retrieved_top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    return len(retrieved_top_k & gold_set) / len(gold_set)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """
    Fraction of Top-K retrieved results that are in the gold set.

    precision@K = |retrieved[:K] ∩ gold| / K
    """
    if k == 0:
        return 0.0
    retrieved_top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    return len(retrieved_top_k & gold_set) / k


# ── Generation metrics ─────────────────────────────────────────────────────────

def answer_f1(prediction: str, ground_truth: str) -> float:
    """
    SQuAD-style token overlap F1 between predicted and ground truth answer.

    Fully reproducible — no LLM judge required.
    Works well for MMDocRAG answers which are typically 1–3 sentences.
    """
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    common = sum((pred_counter & truth_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Citation metrics ───────────────────────────────────────────────────────────

def citation_accuracy(
    cited_page_ids: list[int],
    gold_quotes: list[str],
    all_quotes: list[dict],
) -> float:
    """
    Whether any cited page_id matches a gold quote's page_id.

    citation_accuracy = 1.0 if any cited page matches any gold page, else 0.0.
    """
    gold_ids_set = set(gold_quotes)
    gold_pages = {
        q["page_id"]
        for q in all_quotes
        if q.get("quote_id") in gold_ids_set
    }
    if not gold_pages:
        return 0.0
    return 1.0 if any(p in gold_pages for p in cited_page_ids) else 0.0


# ── Aggregate evaluation ───────────────────────────────────────────────────────

def run_evaluation(
    eval_data: list[dict],
    retriever_fn,
    generator_fn,
    k_values: list[int] = (5, 10),
    max_records: int | None = None,
) -> dict:
    """
    Run full evaluation loop over eval_data records.

    Args:
        eval_data: List of MMDocRAG records.
        retriever_fn: Callable(query) → list of (quote_id, page_id) tuples.
        generator_fn: Callable(query, chunks) → {answer, citations}.
        k_values: K values to compute Recall@K and Precision@K for.
        max_records: Limit evaluation to first N records (for fast iteration).

    Returns:
        Dict of metric_name → mean score.
    """
    records = eval_data[:max_records] if max_records else eval_data
    results: dict[str, list[float]] = {
        **{f"recall@{k}": [] for k in k_values},
        **{f"precision@{k}": [] for k in k_values},
        "answer_f1": [],
        "citation_accuracy": [],
    }

    for record in records:
        query = record["question"]
        gold_ids = record["gold_quotes"]
        gold_short = str(record["answer_short"])

        retrieved = retriever_fn(query)
        retrieved_ids = [r[0] for r in retrieved]

        answer_obj = generator_fn(query, retrieved)

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved_ids, gold_ids, k))
            results[f"precision@{k}"].append(precision_at_k(retrieved_ids, gold_ids, k))

        results["answer_f1"].append(
            answer_f1(answer_obj.get("answer", ""), gold_short)
        )

        cited_pages = [c.get("page_id", -1) for c in answer_obj.get("citations", [])]
        all_quotes = record.get("text_quotes", []) + record.get("img_quotes", [])
        results["citation_accuracy"].append(
            citation_accuracy(cited_pages, gold_ids, all_quotes)
        )

    return {k: sum(v) / len(v) for k, v in results.items() if v}
