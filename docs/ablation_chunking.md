# Chunking Strategy Ablation

Evaluated over 500 records from `evaluation_15.jsonl` using the full hybrid retrieval + reranking pipeline (Conditions C+D from the retrieval ablation).

## Results

| Condition | Chunk Size | Overlap | Sentence Window | Recall@10 | Answer F1 |
|---|---|---|---|---|---|
| Small chunks | 256 tokens | 32 | No | 35.8% | 24.1% |
| **Baseline** | **512 tokens** | **64** | **No** | **39.1%** | **26.1%** |
| Large chunks | 1024 tokens | 128 | No | 37.9% | 25.5% |
| **Selected** | **512 tokens** | **64** | **Yes (+1 sent)** | **40.7%** | **27.4%** |

## Key Findings

**1. Chunk size sweet spot at 512 tokens**

Dense embeddings (Gemini Embedding 2, 768-dim) perform best with 512-token chunks. Smaller chunks lose inter-sentence context; larger chunks dilute the embedding signal across too many topics, reducing cosine similarity discrimination.

- 256→512: +3.3pp Recall@10 — context completeness matters
- 512→1024: −1.2pp Recall@10 — embedding dilution outweighs longer context

**2. Sentence-window expansion is the highest-leverage change**

Expanding retrieved chunks by ±1 sentence at generation time (not at indexing time) adds +1.6pp Recall@10 and +1.3pp Answer F1. The mechanism: the chunk boundary often cuts a sentence mid-thought; expansion restores the complete sentence, giving the generator cleaner context without changing the vector store layout.

**3. Overlap matters less than expected**

We tested 32 vs 64 vs 128 overlap (all at 512-token base size). Differences were within 0.5pp — not a primary driver. 64 was selected as a reasonable default that prevents boundary-cut information loss without excessive duplication.

## Selected Configuration

```
chunk_size=512, chunk_overlap=64
sentence_window_expansion=±1 sentence (at query time, not index time)
```

Implemented in `src/chunking/text_chunker.py` and `src/retrieval/sentence_window.py`.

## Notebook

Full analysis with visualisations: `notebooks/03_chunking_ablation.ipynb`
