# Evaluation Results

Full evaluation over `evaluation_15.jsonl` — 500 records from the MMDocRAG held-out set.

## Summary Metrics

| Metric | Score |
|---|---|
| Recall@10 | **40.7%** |
| Recall@5 | 19.7% |
| Precision@10 | 21.2% |
| Precision@5 | 20.5% |
| Answer F1 | **27.4%** |

## By Evidence Modality (Recall@10)

| Evidence Type | Recall@10 | Count |
|---|---|---|
| chart + text | 56.3% | 131 |
| figure + text | 52.0% | 66 |
| chart + figure + text | 53.1% | 37 |
| table + text | 47.0% | 100 |
| chart + table + text | 46.6% | 37 |

Multimodal evidence types (chart, figure) consistently outperform text-only retrieval because image embeddings are a strong signal — the Gemini Embedding 2 unified space aligns image and text vectors well.

## By Domain (Recall@10)

| Domain | Recall@10 | Count |
|---|---|---|
| Research report | 43.96% | 293 |
| Academic paper | 37.81% | 161 |
| Financial report | 30.47% | 46 |

Financial reports have lower recall because they contain dense numerical tables where exact value matching matters more than semantic similarity.

## Retrieval Ablation (Recall@10)

| Configuration | Recall@10 |
|---|---|
| Dense only (ANN) | 31.2% |
| Sparse only (BM25) | 27.8% |
| Hybrid RRF (no rerank) | 37.1% |
| Hybrid RRF + ms-marco rerank | **40.7%** |

Full details: `notebooks/02_retrieval_ablation.ipynb`

## Chunking Ablation (Recall@10)

| Configuration | Recall@10 |
|---|---|
| 256 tokens / 32 overlap | 35.8% |
| 512 tokens / 64 overlap (no window) | 39.1% |
| 1024 tokens / 128 overlap | 37.9% |
| 512 / 64 + sentence window | **40.7%** |

Full details: `notebooks/03_chunking_ablation.ipynb` and `docs/ablation_chunking.md`

## Context vs. MMDocRAG Leaderboard

The MMDocRAG paper reports system Recall@10 ranging from ~20% (small open-source models) to ~65% (GPT-4V + commercial retrieval). Our 40.7% places in the upper-middle range — consistent with a strong hybrid retrieval system paired with Gemini 2.5 Flash generation, without using a closed-source retriever or larger generation model.

Answer F1 of 27.4% is comparable to mid-range systems on the leaderboard. The primary bottleneck is Answer F1 on financial report questions, where exact number extraction is required and our fuzzy-matching generator tends to paraphrase rather than quote precisely.

## Known Limitations

1. **Financial report gap**: 30.5% vs 43.9% research report recall — dense embeddings struggle with numerical tables. A future improvement would be a specialized table-to-SQL retriever for financial data.

2. **Level 2 grounding is model-approximated**: Gemini bbox localisation for image region grounding is not ground-truth annotated. Displayed with `grounding_type: model_approximated` caveat in the UI.

3. **BM25 encoder fit on index corpus**: The sparse encoder is fit at ingestion time. Queries containing out-of-vocabulary terms fall back to dense-only retrieval.

4. **No cross-document reasoning**: Each retrieved chunk is treated independently. Multi-hop questions requiring synthesis across documents are not explicitly handled.
