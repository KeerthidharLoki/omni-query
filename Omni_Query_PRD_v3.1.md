# Omni-Query — Product Requirements Document v3.1
**Multimodal RAG for Long-Document Intelligence**

| Field | Value |
|---|---|
| Author | Loki |
| Role | Data Scientist / ML Engineer Candidate |
| Version | v3.1 — March 2026 |
| Status | Active |
| Infra Cost | $0 — All Free Tiers |
| Target Roles | Data Scientist · ML Engineer · NLP Engineer |
| Changelog | v3.1 — Fixed Docling API, cross-encoder model ID, RAGAS API, Langfuse SDK v4 migration, Gemini SDK patterns, dataset stats (222 docs, 32K images), license correction |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Goals and Success Metrics](#2-goals-and-success-metrics)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Layer 1 — Ingestion and Parsing](#5-layer-1--ingestion-and-parsing)
6. [Layer 2 — Chunking Strategy](#6-layer-2--chunking-strategy)
7. [Layer 3 — Embeddings](#7-layer-3--embeddings)
8. [Layer 4 — Vector Store](#8-layer-4--vector-store)
9. [Layer 5 — Retrieval](#9-layer-5--retrieval)
10. [Layer 6 — Re-ranking](#10-layer-6--re-ranking)
11. [Layer 7 — Image Grounding](#11-layer-7--image-grounding)
12. [Layer 8 — Generation](#12-layer-8--generation)
13. [Layer 9 — Evaluation](#13-layer-9--evaluation)
14. [Layer 10 — API](#14-layer-10--api)
15. [Layer 11 — UI](#15-layer-11--ui)
16. [Layer 12 — Observability](#16-layer-12--observability)
17. [Layer 13 — Deployment](#17-layer-13--deployment)
18. [Feature Requirements](#18-feature-requirements)
19. [7-Day Build Plan](#19-7-day-build-plan)
20. [Environment Configuration](#20-environment-configuration)
21. [Project Structure](#21-project-structure)
22. [Risk Register](#22-risk-register)
23. [Resume Entry](#23-resume-entry)
24. [Interview Signal Summary](#24-interview-signal-summary)

---

## 1. Problem Statement

### Context

Most RAG systems are text-only pipelines applied to single documents. They fail in two specific ways:

- When a question's answer lives inside a chart, table, or diagram embedded in a PDF, the system returns no answer or a hallucinated one.
- When the corpus spans hundreds of documents, no LLM context window can hold the full corpus, so retrieval is not optional — it is architecturally necessary.

Large vision-language models like Gemini 2.5 Flash solve the single-document case: upload one PDF, ask a question, get an answer. That is not RAG — that is prompting. RAG becomes necessary and defensible only at corpus scale.

### Pain Points

| Pain Point | Impact |
|---|---|
| Questions that require evidence from both a diagram and a paragraph on a different page cannot be answered by any single-document LLM call | Cross-modal, cross-page questions are unanswerable without retrieval |
| A corpus of 222 documents averaging 67 pages each exceeds any practical context window for multi-document queries | Retrieval is architecturally required, not a design choice |
| Existing text-only RAG systems miss critical information embedded in tables, charts, and figures | Answer quality degrades significantly for visual-evidence questions |
| No standard portfolio RAG project evaluates retrieval quality against expert-annotated ground truth | Claims of "87% accuracy" on self-generated test sets are not credible |

### Solution

Omni-Query is a multimodal RAG system built over the MMDocRAG corpus — 222 long documents across 10 domains, with 4,055 expert-annotated QA pairs and gold quote labels from Huawei Noah's Ark Lab. The system indexes text chunks and image patches into a unified vector space using Gemini Embedding 2, retrieves using a hybrid BM25 + ANN pipeline, reranks with a cross-encoder, and generates grounded answers with page and region citations.

Retrieval quality is measured against the MMDocRAG gold quote labels — the same benchmark used to evaluate 60 state-of-the-art models in the original research paper.

Total infrastructure cost: **$0**.

---

## 2. Goals and Success Metrics

| # | Goal | Metric | Target |
|---|---|---|---|
| G1 | Retrieval quality — text questions | Recall@10 on text-only eval set | > 75% |
| G2 | Retrieval quality — cross-modal questions | Recall@10 on cross-modal eval set | > 60% |
| G3 | Answer quality | Answer F1 vs `answer_short` field | > 55% |
| G4 | Grounding quality | RAGAS Faithfulness | > 80% |
| G5 | Citation accuracy | page_id match rate between cited source and gold quote | > 70% |
| G6 | Query latency | P95 end-to-end response time | < 8 seconds |
| G7 | Zero infra cost | All APIs on free tiers | $0 total |

### Why These Targets

Recall@10 of 75% on text and 60% on cross-modal are calibrated against the MMDocRAG paper's published baseline numbers. The paper reports that most retrieval systems score in the 55–72% range on cross-modal questions, making 60% a credible and achievable target without fine-tuning.

Answer F1 of 55% is set against the SQuAD-style token overlap metric on `answer_short`. MMDocRAG answers tend to be concise (1–3 sentences), so token overlap is a reasonable proxy for correctness.

The gap between Recall@10 and Answer F1 is intentional and informative. If Recall@10 is 75% but Answer F1 is 40%, the problem is in generation. If both are low, the problem is in retrieval. This separation is the core evaluation story.

---

## 3. Dataset

### MMDocRAG

| Property | Value |
|---|---|
| Source | Huawei Noah's Ark Lab |
| HuggingFace | `MMDocIR/MMDocRAG` |
| Paper | arXiv 2505.16470 |
| License | Research Use Only (see repo license notice) |
| Documents | 222 long PDFs |
| Domains | 10 (academic, financial, technical, medical, legal, government, news, science, engineering, general) |
| Avg pages per document | 67 |
| Total QA pairs | 4,055 expert-annotated |
| Image quotes | 32,071 total (6,349 gold) JPEG patches |
| Text quotes | 48,618 (4,640 gold) |
| Eval set | 2,000 QA pairs (`evaluation_15.jsonl`, `evaluation_20.jsonl`) |
| Dev set | 2,055 QA pairs (`dev_15.jsonl`, `dev_20.jsonl`) |

### Data Schema

Each record in the JSONL files contains:

```
q_id                   int      question identifier
doc_name               string   source document filename
domain                 string   document domain category
question               string   natural language question
evidence_modality_type list     ["text"], ["image"], or ["text", "image"]
question_type          string   comparison / factual / reasoning / description / ...
text_quotes            list     candidate text passages (15 or 20 per question)
img_quotes             list     candidate image patches (15 or 20 per question)
gold_quotes            list     ground truth quote IDs e.g. ["text2", "image1", "text7"]
answer_short           string   concise ground truth answer
answer_interleaved     string   full answer interleaving text and image references
old_id                 int      legacy question identifier (present in dataset)
```

Each `text_quote` entry:
```
quote_id    string   e.g. "text1" ... "text15"
type        string   "text"
text        string   raw passage text
page_id     int      page number in source document
layout_id   int      MinerU layout region identifier
```

Each `img_quote` entry:
```
quote_id        string   e.g. "image1" ... "image8"
type            string   "image"
img_path        string   path to JPEG patch in images.zip
img_description string   VLM-generated description of the image
page_id         int      page number in source document
layout_id       int      MinerU layout region identifier
```

### Loading the Dataset

The HuggingFace dataset viewer has a known schema mismatch between `train.jsonl` (chat format) and the dev/eval files (structured format). Load directly from JSONL:

```python
import json

def load_mmdocrag(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

eval_data = load_mmdocrag("data/evaluation_15.jsonl")
dev_data  = load_mmdocrag("data/dev_15.jsonl")
```

### Corpus Split Strategy

| Split | File | Use |
|---|---|---|
| Evaluation | `evaluation_15.jsonl` (2,000 QA) | Final benchmark — run once at the end |
| Development | `dev_15.jsonl` (2,055 QA) | Retrieval tuning, chunking experiments, prompt iteration |
| Ablation subset | First 200 rows of dev | Fast iteration during build days 1–5 |

---

## 4. System Architecture

### Pipeline Overview

```
MMDocRAG corpus (222 PDFs + 32,071 images)
    │
    ▼
[INGESTION]
Docling layout parse + PyMuPDF image extract + Gemini 2.5 Flash image description
    │
    ▼
[CHUNKING]
Text: 512 tok / 64 overlap, within semantic blocks only
Tables: atomic, markdown-serialised
Images: one chunk per patch, description + caption merged
    │
    ▼
[EMBEDDING]
Gemini Embedding 2 (gemini-embedding-2-preview)
→ text chunks: 768-dim, task_type=RETRIEVAL_DOCUMENT
→ image patches: 768-dim, same model, same vector space
    │
    ▼
[QDRANT CLOUD FREE]
Single collection: text + image vectors unified
BM25 sparse index + ANN dense index
Metadata payload per vector: doc_name, page_id, layout_id, section_title,
                              chunk_type, modality, img_path, parent_chunk_id
    │
    ▼
[RETRIEVAL — user query arrives]
Query → Gemini Embedding 2 (task_type=RETRIEVAL_QUERY, 768-dim)
→ BM25 keyword search (text metadata)
→ ANN vector search (text + image unified space)
→ RRF fusion → Top-50 candidates
→ Sentence-window expand: child chunk → parent chunk
    │
    ▼
[RERANKING]
ms-marco-MiniLM-L-6-v2 cross-encoder
Scores every (query, chunk) pair independently
Top-50 → Top-10
    │
    ▼
[IMAGE GROUNDING]
Level 1: page_id + img_path → render image in citation panel
Level 2: Gemini Flash bbox localisation → overlay on image
Level 3: layout_id anchor → structural position in document
    │
    ▼
[GENERATION]
Gemini 2.5 Flash: query + Top-10 chunks (text + base64 images)
→ answer + cited quote_ids + bbox coordinates
    │
    ▼
[EVALUATION]
Recall@K / Precision@K vs gold_quotes
Answer F1 vs answer_short
RAGAS Faithfulness vs retrieved context
Citation accuracy: page_id match vs gold quote page_id
    │
    ▼
[API] FastAPI /ingest /query /evaluate /health
[UI]  Streamlit chat + citation panel + quote diff view
[OBS] Langfuse tracing per query
```

### Architecture Diagram (Mermaid)

```mermaid
graph TB
    subgraph Corpus["Corpus — 222 PDFs + 32,071 images"]
        A[MMDocRAG\nHuggingFace dataset]
    end

    subgraph Ingestion["Ingestion + Parsing"]
        B[Docling\nlayout parse]
        C[PyMuPDF\nimage extract]
        D[Gemini 2.5 Flash\nimage description]
    end

    subgraph Chunking["Chunking"]
        E[Text 512/64\nwithin blocks]
        F[Tables atomic\nmarkdown]
        G[Images one/patch\n+ description]
    end

    subgraph Embedding["Embedding — Gemini Embedding 2"]
        H[Text vectors\n768-dim RETRIEVAL_DOCUMENT]
        I[Image vectors\n768-dim same space]
    end

    subgraph VectorStore["Qdrant Cloud Free"]
        J[Unified collection\nBM25 + ANN]
    end

    subgraph QueryPath["Query Path"]
        K[User query]
        L[Gemini Embedding 2\nRETRIEVAL_QUERY]
        M[Hybrid BM25+ANN\nRRF fusion Top-50]
        N[Sentence-window\nexpand]
        O[ms-marco reranker\nTop-50 → Top-10]
        P[Image grounding\n3 levels]
        Q[Gemini 2.5 Flash\ngeneration]
        R[Answer + citations\n+ bbox]
    end

    subgraph Evaluation["Evaluation"]
        S[Recall@K vs gold_quotes]
        T[Answer F1 vs answer_short]
        U[RAGAS Faithfulness]
        V[Citation accuracy]
    end

    A --> B --> E
    A --> C --> G
    B --> F
    C --> D --> G
    E --> H --> J
    F --> H
    G --> I --> J
    K --> L --> M
    J --> M --> N --> O --> P --> Q --> R
    R --> S
    R --> T
    R --> U
    R --> V
```

---

## 5. Layer 1 — Ingestion and Parsing

### Tools

| Tool | Role | Why |
|---|---|---|
| Docling | Primary PDF parser | Layout-aware — detects paragraphs, headings, tables, figures, preserves reading order. Handles multi-column layouts that confuse PyMuPDF |
| PyMuPDF | Image patch extractor | Extracts raw image bytes per page with pixel coordinates. Fallback text extractor for simple single-column PDFs |
| Gemini 2.5 Flash | Image description | Generates natural language description of each image patch at ingestion time. Output stored alongside raw image bytes and used for BM25 indexing |

### Docling Configuration

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False          # documents are digital, not scanned
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options = TableStructureOptions(
    do_cell_matching=True
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
result = converter.convert("path/to/doc.pdf")
doc = result.document
```

### Image Description at Ingestion

Each image patch extracted by PyMuPDF is sent to Gemini 2.5 Flash once during ingestion. The description is stored in Qdrant as part of the chunk payload and is used for two purposes: BM25 keyword search and cross-encoder reranking.

```python
from google import genai
from google.genai import types

client = genai.Client()

def describe_image_patch(image_bytes: bytes) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            (
                "Describe this image in detail for a document retrieval system. "
                "Include: what type of visual this is (chart, table, diagram, photo), "
                "all visible labels, values, axes, legend entries, and the key finding "
                "or information conveyed. Be specific and complete."
            )
        ]
    )
    return response.text
```

### MMDocRAG Pre-computed Descriptions

The MMDocRAG dataset already includes `img_description` per image quote, generated by a VLM during dataset construction. For chunks sourced directly from the dataset's `img_quotes` field, use these descriptions directly rather than re-calling Gemini. Call Gemini only for image patches extracted from the raw PDFs that are outside the dataset's pre-annotated quotes.

```python
# Use dataset description if available
img_description = img_quote.get("img_description") or describe_image_patch(image_bytes)
```

---

## 6. Layer 2 — Chunking Strategy

### Design Principles

- Never split within a semantic unit. Docling identifies semantic blocks (paragraph, heading, list, table, figure). Chunking operates within blocks, never across them.
- Preserve structural metadata on every chunk. Every chunk stores its section title, page number, layout region, and parent chunk ID.
- Different content types require different strategies. Text, tables, and images are handled independently.

### Text Chunking

Two-level chunking strategy: SentenceWindowNodeParser for retrieval precision, with token-size guardrails within Docling semantic blocks.

**Level 1 — Sentence-level nodes for retrieval:**
SentenceWindowNodeParser splits text into individual sentences. Each sentence becomes a retrieval node. The surrounding context (parent window) is stored in metadata for later expansion at generation time.

**Level 2 — Token-size guardrails:**
Before passing text to SentenceWindowNodeParser, Docling semantic blocks longer than 512 tokens are pre-split at sentence boundaries into sub-blocks. Blocks shorter than 64 tokens are discarded as artefacts. This ensures the parent windows remain manageable.

| Parameter | Value | Rationale |
|---|---|---|
| Node granularity | Sentence-level | Fine-grained retrieval precision via SentenceWindowNodeParser |
| Parent window | 3 sentences on each side | Provides generation context without inflating the index |
| Max block size | 512 tokens | Pre-split before SentenceWindowNodeParser to cap parent window size |
| Overlap | 64 tokens (~1–2 sentences) | Pre-split overlap prevents answer from falling at a block boundary |
| Splitting boundary | Within Docling semantic blocks only | Never splits across paragraph or section boundaries |
| Minimum chunk size | 64 tokens | Discard chunks shorter than this; they are likely artefacts |

```python
from llama_index.core.node_parser import SentenceWindowNodeParser

parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,              # parent window: 3 sentences on each side of retrieved sentence
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
```

The SentenceWindowNodeParser creates single-sentence child nodes for retrieval precision and stores a larger surrounding window (3 sentences on each side) in metadata. At generation time, the `MetadataReplacementPostProcessor` replaces the child sentence with its parent window, giving the LLM more context without degrading retrieval precision.

### Table Chunking

Tables are treated as atomic units — never split. Docling extracts each table as a structured object. Tables are serialised to markdown before embedding and storage.

```python
def serialise_table(table) -> str:
    """Convert Docling table object to markdown string."""
    rows = []
    header = "| " + " | ".join(str(cell) for cell in table.data[0]) + " |"
    separator = "| " + " | ".join("---" for _ in table.data[0]) + " |"
    rows.append(header)
    rows.append(separator)
    for row in table.data[1:]:
        rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rows)
```

Table metadata payload includes `chunk_type=table`, `caption` (if available), `page_id`, `section_title`. A table that exceeds 512 tokens is stored as a single chunk regardless — the atomic rule takes priority over the size limit.

### Image Chunking

Each image patch is one chunk. No splitting. The chunk content for embedding is: `[Gemini description text] + [figure caption if available]`. Both are concatenated into a single text string before embedding with Gemini Embedding 2 (for the text representation) and the raw image bytes are embedded separately (for the visual representation). Both vectors are stored in Qdrant.

```python
def build_image_chunk(img_quote: dict, image_bytes: bytes) -> dict:
    description = img_quote.get("img_description", "")
    caption = img_quote.get("caption", "")
    text_content = f"{description}\n{caption}".strip()

    return {
        "text_content": text_content,         # for BM25 + reranker
        "image_bytes": image_bytes,            # for Gemini Embedding 2 image embed
        "quote_id": img_quote["quote_id"],
        "img_path": img_quote["img_path"],
        "page_id": img_quote["page_id"],
        "layout_id": img_quote["layout_id"],
        "chunk_type": "image",
        "modality": "image"
    }
```

### Metadata Payload Schema

Every chunk stored in Qdrant carries this payload:

```python
payload = {
    "doc_name":        str,   # source document filename
    "domain":          str,   # MMDocRAG domain label
    "page_id":         int,   # page number in source PDF
    "layout_id":       int,   # MinerU layout region identifier
    "section_title":   str,   # nearest section heading above this chunk
    "chunk_type":      str,   # "text" | "table" | "image"
    "modality":        str,   # "text" | "image"
    "quote_id":        str,   # MMDocRAG quote_id if applicable
    "img_path":        str,   # image file path (image chunks only)
    "parent_chunk_id": str,   # ID of parent chunk for sentence-window expand
    "text_content":    str,   # raw text (for BM25 and reranker)
    "token_count":     int,   # number of tokens in this chunk
}
```

---

## 7. Layer 3 — Embeddings

### Primary: Gemini Embedding 2

| Property | Value |
|---|---|
| Model ID | `gemini-embedding-2-preview` |
| Dimensions | 768 (MRL truncated from 3072) |
| Text context | 8,192 tokens |
| Images per request | Up to 6 |
| Task type at index | `RETRIEVAL_DOCUMENT` |
| Task type at query | `RETRIEVAL_QUERY` |
| Free tier | Shared Gemini free tier |

The core architectural decision is that text chunks and image patches are embedded into the same 768-dim vector space using the same model. A single ANN search over Qdrant retrieves both modalities simultaneously. No separate text index and image index. No alignment tricks.

```python
from google import genai
from google.genai import types

client = genai.Client()

def embed_text_chunk(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values

def embed_image_patch(image_bytes: bytes) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values

def embed_query(query: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values
```

### Why 768 Dimensions

Gemini Embedding 2 uses Matryoshka Representation Learning (MRL), which nests information so dimensions can be truncated without significant quality loss. 768 dimensions are chosen over the default 3072 for practical reasons:

- Qdrant Cloud Free provides 1 GB RAM. At 768 dimensions (float32 = 4 bytes), 1 GB holds approximately 327,000 vectors. The MMDocRAG corpus of 222 documents generates roughly 100,000–150,000 chunks. 768 dims fits comfortably; 3072 dims would require ~490,000 vectors to hit the limit.
- If RAGAS scores or Recall@K need improvement, switching to 3072 dims requires only a one-line config change and a full re-index. Document both results in the README as an ablation.

### Fallback: all-MiniLM-L6-v2

Activated when `FALLBACK_EMBED=local` in `.env`. Text-only, CPU-only, 384 dimensions. The Qdrant collection must be re-indexed if switching between embedding models — vectors from different models live in incompatible spaces. This is documented explicitly in the README.

```python
from sentence_transformers import SentenceTransformer

fallback_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text_fallback(text: str) -> list[float]:
    return fallback_model.encode(text, normalize_embeddings=True).tolist()
```

---

## 8. Layer 4 — Vector Store

### Qdrant Cloud Free

| Property | Value |
|---|---|
| Service | Qdrant Cloud Free tier |
| Storage | 1 GB RAM + 4 GB disk, free forever |
| Collection | Single unified collection: `omni_query_v3` |
| Vector size | 768 (matches Gemini Embedding 2 MRL output) |
| Distance metric | Cosine |
| Sparse index | BM25 for keyword search |
| Dense index | HNSW for ANN vector search |
| Hybrid search | Built-in RRF fusion |

### Collection Initialisation

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams
)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

client.create_collection(
    collection_name="omni_query_v3",
    vectors_config={
        "dense": VectorParams(
            size=768,
            distance=Distance.COSINE,
            on_disk=True
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    }
)
```

### Upsert Pattern

```python
from qdrant_client.models import PointStruct, SparseVector

def upsert_chunk(chunk_id: str, dense_vector: list[float],
                 sparse_vector: dict, payload: dict):
    client.upsert(
        collection_name="omni_query_v3",
        points=[
            PointStruct(
                id=chunk_id,
                vector={
                    "dense": dense_vector,
                    "sparse": SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"]
                    )
                },
                payload=payload
            )
        ]
    )
```

### Storage Estimate

| Content Type | Estimated Chunks | Vector Size | Storage |
|---|---|---|---|
| Text chunks (222 docs × avg 400 chunks) | 88,800 | 768 × 4 bytes | ~273 MB |
| Table chunks (222 docs × avg 30 tables) | 6,660 | 768 × 4 bytes | ~20 MB |
| Image chunks (gold subset from MMDocRAG) | ~6,349 | 768 × 4 bytes | ~20 MB |
| Sparse index overhead | — | — | ~100 MB |
| **Total estimate** | **~102,000 vectors** | | **~413 MB** |

**Note on image chunk count:** The MMDocRAG dataset contains 32,071 total image quotes but only 6,349 gold image quotes. For the initial build, index only the gold image quotes plus any additional images extracted by PyMuPDF from the raw PDFs. If indexing all 32,071 image quotes, storage increases to ~510 MB — still within the 1 GB free tier but with less buffer.

Well within the 1 GB RAM free tier. Buffer exists for re-indexing experiments.

---

## 9. Layer 5 — Retrieval

### Query Embedding

```python
query_vector = embed_query(user_query)   # 768-dim RETRIEVAL_QUERY
```

### Hybrid BM25 + ANN Search

LlamaIndex (`llama-index-core` + integration packages) orchestrates the hybrid search. BM25 runs over the `text_content` field in Qdrant's sparse index. ANN runs over the dense vector. Reciprocal Rank Fusion combines both ranked lists.

```python
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="omni_query_v3",
    enable_hybrid=True,         # activates BM25 + ANN hybrid mode
    sparse_query_fn=build_sparse_vector,
)

index = VectorStoreIndex.from_vector_store(vector_store)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=50,        # retrieve Top-50 for reranker input
    sparse_top_k=50,
    hybrid_top_k=50,
)
```

### Sentence-Window Expansion

After retrieval, each retrieved child chunk is expanded to its parent window. The parent window provides more context for generation without inflating the retrieval index with large chunks.

```python
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

sentence_window_postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window"  # replaces node text with surrounding window
)

nodes = retriever.retrieve(query)
nodes = sentence_window_postprocessor.postprocess_nodes(nodes)
```

### Metadata Filtering

Optional filters can be applied to scope retrieval to a specific document or domain:

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

filters = MetadataFilters(filters=[
    MetadataFilter(key="domain", value="engineering"),
    MetadataFilter(key="doc_name", value="iso_standard_45001.pdf"),
])

retriever = VectorIndexRetriever(index=index, similarity_top_k=50, filters=filters)
```

---

## 10. Layer 6 — Re-ranking

### Cross-Encoder: ms-marco-MiniLM-L-6-v2

| Property | Value |
|---|---|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Type | Cross-encoder (full attention over query + chunk) |
| Input | (query, chunk_text) pairs |
| Output | Relevance score per pair |
| Input: Top-K from retriever | 50 |
| Output: Top-K to generation | 10 |
| Inference | CPU-capable, ~150ms for 50 pairs |
| Free tier | Free OSS via sentence-transformers |

### Why Cross-Encoder Over Bi-Encoder for Reranking

The retrieval stage uses bi-encoders (Gemini Embedding 2) — the query and each document are embedded independently, then compared by cosine similarity. This is efficient for Top-50 retrieval over 100K+ vectors.

A cross-encoder takes the query and document together as input and applies full attention across both. This is far more accurate but cannot scale to millions of vectors. The two-stage approach uses each where it is appropriate: bi-encoder for recall across the full corpus, cross-encoder for precision on the shortlist.

### Implementation

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

def rerank(query: str, nodes: list, top_k: int = 10) -> list:
    pairs = [(query, node.get_content()) for node in nodes]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)
    return [node for _, node in ranked[:top_k]]
```

### Image Chunk Reranking

The cross-encoder operates on text. For image chunks, the `text_content` field (Gemini description + caption) is used as the document side of the pair. This is less accurate than a vision-capable reranker but practical and sufficient for a portfolio system. Document this limitation honestly in the README.

---

## 11. Layer 7 — Image Grounding

Image grounding answers the question: when the system cites an image, exactly what part of the image is relevant?

Three levels of grounding are implemented, in increasing specificity:

### Level 1 — Page Grounding (always available)

Every image chunk stores `page_id` and `img_path` from the MMDocRAG dataset. When an image chunk is in the Top-10 results, the citation returns the page number and the image is rendered directly in the Streamlit citation panel. This is concrete and verifiable using the dataset's ground truth.

```python
citation = {
    "type": "image",
    "quote_id": chunk.metadata["quote_id"],
    "page_id": chunk.metadata["page_id"],
    "img_path": chunk.metadata["img_path"],
    "doc_name": chunk.metadata["doc_name"],
}
```

### Level 2 — Region Grounding (Gemini-approximated)

After generation, if the answer contains a sentence derived from an image chunk, the original image is sent back to Gemini 2.5 Flash with a region localisation prompt. Gemini returns approximate normalised bounding box coordinates. These are drawn as an overlay in Streamlit.

```python
def localise_region(image_bytes: bytes, answer_sentence: str) -> dict:
    prompt = (
        f"In this image, identify the region most relevant to the following statement: "
        f'"{answer_sentence}". Return ONLY a JSON object with keys: '
        f'"x_min", "y_min", "x_max", "y_max" as normalised coordinates (0.0 to 1.0). '
        f"No other text."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt
        ]
    )
    return json.loads(response.text)
```

**Documented limitation:** Bounding boxes returned by Gemini are model-approximated, not annotated ground truth. They are stored with a `grounding_type: "model_approximated"` flag in the citation metadata. This is transparent and defensible.

### Level 3 — Layout Anchor (structural grounding)

The `layout_id` field from MinerU's parse is stored per chunk. It ties each image to its structural position in the document layout — figure number, table caption, section context. This is used to construct richer citation text even when Level 2 bbox localisation is not triggered.

```python
citation_text = (
    f"Figure from {chunk.metadata['doc_name']}, "
    f"page {chunk.metadata['page_id']}, "
    f"layout region {chunk.metadata['layout_id']}"
)
```

---

## 12. Layer 8 — Generation

### Primary: Gemini 2.5 Flash

Receives the query, the Top-10 reranked chunks (text content + base64-encoded images for image chunks), and a structured prompt. Returns an answer with cited quote IDs and optional bbox coordinates for any image citations.

```python
def build_generation_prompt(query: str, chunks: list) -> list:
    parts = []

    system = (
        "You are a precise document question-answering assistant. "
        "Answer the question using ONLY the provided evidence. "
        "For each claim in your answer, cite the quote_id of the source. "
        "If an image is relevant, cite its quote_id and describe what specific region answers the question. "
        "If the evidence does not contain the answer, say so explicitly. "
        "Return your response as JSON: "
        '{"answer": str, "citations": [{"quote_id": str, "type": str, "relevant_region": str}]}'
    )
    parts.append(system)

    for i, chunk in enumerate(chunks):
        parts.append(f"\n--- Quote {chunk.metadata['quote_id']} "
                      f"(page {chunk.metadata['page_id']}, "
                      f"{chunk.metadata['chunk_type']}) ---\n")

        if chunk.metadata["modality"] == "image":
            with open(chunk.metadata["img_path"], "rb") as f:
                img_bytes = f.read()
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            parts.append(chunk.get_content())
        else:
            parts.append(chunk.get_content())

    parts.append(f"\nQuestion: {query}")
    return parts
```

### Fallback: Groq Llama 4 Scout

Activated when `FALLBACK_LLM=groq` in `.env`. Text-only context (no image bytes). Zero code change — LlamaIndex LLM abstraction handles the swap. Uses the full model ID `meta-llama/llama-4-scout-17b-16e-instruct` (17B active params, 16 experts, 109B total).

```python
import os
from llama_index.llms.groq import Groq
from llama_index.llms.google_genai import GoogleGenAI

def get_llm():
    if os.getenv("FALLBACK_LLM") == "groq":
        return Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY"))
    return GoogleGenAI(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
```

---

## 13. Layer 9 — Evaluation

### Four Independent Metrics

The evaluation stack measures four distinct things. A low score on any one of them points to a specific layer in the pipeline.

| Metric | Measures | Data | Diagnostic Implication |
|---|---|---|---|
| Recall@K | Did the retriever find the gold evidence? | Retrieved quote IDs vs `gold_quotes` | Low = fix retrieval layer |
| Precision@K | Of what was retrieved, how much was relevant? | Same | Low with high recall = reranker not filtering well |
| Answer F1 | Is the generated text correct? | Generated answer vs `answer_short` | Low with high Recall = fix generation / prompt |
| RAGAS Faithfulness | Is the answer grounded in the retrieved context? | Generated answer vs retrieved chunks | Low = LLM hallucinating beyond evidence |
| Citation accuracy | Does the cited source match the gold source? | Cited `page_id` vs gold quote `page_id` | Low = answer right but attribution wrong |

### Recall@K Implementation

```python
def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    return len(retrieved_top_k & gold_set) / len(gold_set)

def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    if k == 0:
        return 0.0
    return len(retrieved_top_k & gold_set) / k
```

### Answer F1 Implementation

SQuAD-style token overlap. No LLM judge, fully reproducible.

```python
from collections import Counter

def answer_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    common = sum((pred_counter & truth_counter).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)
```

### RAGAS Faithfulness

```python
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

async def evaluate_faithfulness(query: str, answer: str, contexts: list[str],
                                evaluator_llm=None) -> float:
    """Evaluate faithfulness using RAGAS collections-based API.

    Note: RAGAS >= 0.4 requires the collections-based API.
    The legacy evaluate() with Dataset.from_dict() is deprecated.
    """
    scorer = Faithfulness(llm=evaluator_llm)
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
    )
    score = await scorer.single_turn_ascore(sample)
    return score
```

**Note:** RAGAS Faithfulness requires an LLM-as-judge (e.g. Gemini 2.5 Flash or GPT-4o-mini). Pass the evaluator LLM when creating the `Faithfulness` scorer. Pin `ragas>=0.4` in requirements to ensure the collections-based API is available.

### Full Evaluation Run

```python
def run_evaluation(eval_data: list[dict], k_values: list[int] = [5, 10]) -> dict:
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"precision@{k}": [] for k in k_values})
    results["answer_f1"] = []
    results["citation_accuracy"] = []

    for record in eval_data:
        query = record["question"]
        gold_quote_ids = record["gold_quotes"]
        gold_short = record["answer_short"]

        retrieved_nodes = retriever.retrieve(query)
        reranked_nodes = reranker(query, retrieved_nodes, top_k=10)
        answer_obj = generate_answer(query, reranked_nodes)

        retrieved_ids = [n.metadata["quote_id"] for n in retrieved_nodes]
        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved_ids, gold_quote_ids, k))
            results[f"precision@{k}"].append(precision_at_k(retrieved_ids, gold_quote_ids, k))

        results["answer_f1"].append(answer_f1(answer_obj["answer"], gold_short))

        cited_pages = [c["page_id"] for c in answer_obj.get("citations", [])]
        gold_pages = [
            q["page_id"]
            for q in record["text_quotes"] + record["img_quotes"]
            if q["quote_id"] in gold_quote_ids
        ]
        if gold_pages:
            match = any(p in gold_pages for p in cited_pages)
            results["citation_accuracy"].append(1.0 if match else 0.0)

    return {k: sum(v) / len(v) for k, v in results.items() if v}
```

### Evaluation Split by Modality

Run evaluation separately for:
- `evidence_modality_type == ["text"]` — text-only questions
- `evidence_modality_type == ["image"]` — image-only questions
- `evidence_modality_type == ["text", "image"]` — cross-modal questions

This breakdown is the most important result to show in the README and discuss in interviews. Cross-modal Recall@10 is the hardest metric and the one that differentiates this project.

---

## 14. Layer 10 — API

### Endpoints

#### `POST /ingest`

```
Content-Type: multipart/form-data
file: PDF or JPEG/PNG
doc_type: "pdf" | "image"
```

```json
{
  "doc_id": "iso_45001_2018",
  "pages_processed": 47,
  "text_chunks": 312,
  "table_chunks": 18,
  "image_chunks": 84,
  "vectors_upserted": 414,
  "embed_model": "gemini-embedding-2-preview",
  "embed_dim": 768,
  "status": "complete",
  "duration_seconds": 142.3
}
```

#### `POST /query`

```json
{
  "query": "What does the bar chart on page 14 show about quarterly revenue?",
  "top_k": 10,
  "doc_filter": null,
  "domain_filter": null
}
```

```json
{
  "answer": "The bar chart on page 14 shows quarterly revenue declined 12% in Q3...",
  "citations": [
    {
      "quote_id": "image3",
      "type": "image",
      "page_id": 14,
      "img_path": "images/doc_01_page14_fig2.jpg",
      "relevant_region": "bar chart showing Q1-Q4 comparison",
      "bbox": {"x_min": 0.12, "y_min": 0.34, "x_max": 0.88, "y_max": 0.71},
      "grounding_type": "model_approximated",
      "rerank_score": 0.934
    },
    {
      "quote_id": "text7",
      "type": "text",
      "page_id": 15,
      "text_preview": "Revenue declined 12% quarter-over-quarter driven by...",
      "rerank_score": 0.821
    }
  ],
  "llm_used": "gemini-2.5-flash",
  "embed_model": "gemini-embedding-2-preview",
  "retrieval_ms": 310,
  "rerank_ms": 180,
  "generation_ms": 3240,
  "total_ms": 3730
}
```

#### `POST /evaluate`

```json
{
  "eval_file": "evaluation_15",
  "k_values": [5, 10],
  "max_records": 200
}
```

```json
{
  "recall@5": 0.61,
  "recall@10": 0.72,
  "precision@5": 0.38,
  "precision@10": 0.29,
  "answer_f1": 0.57,
  "citation_accuracy": 0.71,
  "records_evaluated": 200,
  "duration_seconds": 1840,
  "breakdown": {
    "text_only":     {"recall@10": 0.78, "answer_f1": 0.63},
    "image_only":    {"recall@10": 0.58, "answer_f1": 0.44},
    "cross_modal":   {"recall@10": 0.62, "answer_f1": 0.51}
  }
}
```

#### `GET /health`

```json
{
  "status": "healthy",
  "qdrant": "connected",
  "gemini_api": "reachable",
  "langfuse": "connected",
  "embed_model": "gemini-embedding-2-preview",
  "embed_dim": 768,
  "fallback_llm_active": false,
  "fallback_embed_active": false,
  "collection_vectors": 102000
}
```

---

## 15. Layer 11 — UI

### Streamlit Panels

**Panel 1 — Query input**
- Text input for natural language question
- Optional document filter dropdown
- Optional domain filter dropdown
- Submit button

**Panel 2 — Answer**
- Generated answer text
- LLM used badge (Gemini / Groq)
- Latency breakdown: retrieval / rerank / generation

**Panel 3 — Citation panel**
- For each citation in the response:
  - Text citations: chunk text with page number and section title highlighted
  - Image citations: image rendered inline with bbox overlay if Level 2 grounding available
  - Rerank score shown per citation
  - Grounding type badge: `dataset-native` or `model-approximated`

**Panel 4 — Quote diff view (evaluation mode)**
- Activated when running against MMDocRAG eval set
- Left column: retrieved Top-10 quote IDs
- Right column: gold quote IDs from the dataset
- Matches highlighted in green, misses in red
- Recall@10 and Precision@10 computed live per query

---

## 16. Layer 12 — Observability

### Langfuse

Every query is traced end-to-end in Langfuse. This gives a public portfolio URL showing real system behaviour.

**SDK version:** Langfuse Python SDK v4 (released March 2026). Configuration via environment variables:

```bash
# Set in .env — the SDK picks these up automatically via get_client()
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

```python
import os
from langfuse import observe, get_client

@observe()
def query_pipeline(query: str) -> dict:
    langfuse = get_client()

    # Update the current trace with metadata
    langfuse.update_current_span(
        input=query,
        metadata={"embed_model": "gemini-embedding-2-preview"}
    )

    with langfuse.start_as_current_observation(as_type="embedding", name="embedding"):
        query_vector = embed_query(query)

    with langfuse.start_as_current_observation(as_type="retriever", name="retrieval"):
        nodes = retriever.retrieve(query)

    with langfuse.start_as_current_observation(as_type="span", name="reranking"):
        reranked = rerank(query, nodes, top_k=10)

    with langfuse.start_as_current_observation(as_type="generation", name="generation") as gen:
        answer = generate_answer(query, reranked)
        gen.update(output=answer["answer"])

    return answer
```

**Note:** The v4 SDK was a full rewrite based on OpenTelemetry. Key differences from v2:
- `from langfuse import observe, get_client` replaces `from langfuse.decorators import observe, langfuse_context`
- `langfuse.start_as_current_observation(as_type="span", name="...")` replaces the non-existent `langfuse_context.observe_span()`
- Observation types (`embedding`, `retriever`, `generation`, `span`) provide richer categorisation in the Langfuse dashboard
- Client is created via `get_client()` which reads environment variables automatically

Traced dimensions per query: embedding latency, retrieval latency (BM25 + ANN separately), rerank latency, generation latency, number of chunks retrieved, number of image chunks in Top-10, which LLM was used.

---

## 17. Layer 13 — Deployment

### Docker

```dockerfile
# Multi-stage: builder + slim runtime
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (local)

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes: ["./data:/app/data"]
  ui:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports: ["8501:8501"]
    env_file: .env
    depends_on: [api]
```

### Helm Chart Structure

```
helm/
├── Chart.yaml
├── values.yaml
└── templates/
    ├── deployment-api.yaml
    ├── deployment-ui.yaml
    ├── service-api.yaml
    ├── service-ui.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    ├── configmap.yaml
    └── secret.yaml
```

### HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: omni-query-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: omni-query-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Deploy Commands

```bash
minikube start --memory=4096 --cpus=4
minikube addons enable ingress

helm install omni-query ./helm \
  --set secrets.geminiApiKey=$GEMINI_API_KEY \
  --set secrets.qdrantApiKey=$QDRANT_API_KEY \
  --set secrets.qdrantUrl=$QDRANT_URL \
  --set secrets.langfusePublicKey=$LANGFUSE_PUBLIC_KEY \
  --set secrets.langfuseSecretKey=$LANGFUSE_SECRET_KEY

kubectl get pods -n omni-query
```

---

## 18. Feature Requirements

### Priority Definitions

- **P0** — Pipeline does not function without this
- **P1** — Required for portfolio-quality demo
- **P2** — Drop on Day 6 if behind schedule

| ID | Feature | Description | Priority |
|---|---|---|---|
| F-01 | Dataset load | Load MMDocRAG JSONL files; extract text quotes, image quotes, gold labels | P0 |
| F-02 | Docling parse | Parse all 222 PDFs with layout-aware parsing; extract semantic blocks | P0 |
| F-03 | Image description | Call Gemini 2.5 Flash per image patch at ingestion; store description in payload | P0 |
| F-04 | Text chunking | 512-token / 64-overlap chunks within Docling semantic blocks; store parent_chunk_id | P0 |
| F-05 | Table chunking | Atomic table serialisation to markdown; stored as single chunk regardless of size | P0 |
| F-06 | Image chunking | One chunk per image patch; description + caption merged; raw bytes embedded | P0 |
| F-07 | Gemini Embedding 2 | Embed text and image chunks into unified 768-dim space; RETRIEVAL_DOCUMENT task | P0 |
| F-08 | Qdrant upsert | Upsert all vectors with full metadata payload; BM25 + ANN hybrid collection | P0 |
| F-09 | Hybrid retrieval | LlamaIndex BM25 + ANN with RRF fusion; Top-50 candidates | P0 |
| F-10 | Sentence-window expand | Expand retrieved child chunks to parent window before generation | P0 |
| F-11 | Cross-encoder rerank | ms-marco-MiniLM-L-6-v2; Top-50 → Top-10 | P0 |
| F-12 | Gemini generation | Multimodal prompt with text + base64 image context; structured JSON output | P0 |
| F-13 | FastAPI endpoints | /ingest /query /evaluate /health with Pydantic schemas | P0 |
| F-14 | Recall@K / Precision@K | Evaluate retriever against MMDocRAG gold_quotes; split by modality | P1 |
| F-15 | Answer F1 | Token overlap vs answer_short; SQuAD-style; no LLM judge | P1 |
| F-16 | Level 1 image grounding | page_id + img_path citation; image rendered in Streamlit panel | P1 |
| F-17 | Streamlit UI | Chat panel, citation panel with image overlay, quote diff view | P1 |
| F-18 | Langfuse tracing | Per-query trace: embed/retrieval/rerank/generation latency | P1 |
| F-19 | LLM fallback | Groq Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct) fallback via FALLBACK_LLM=groq | P1 |
| F-20 | Embed fallback | all-MiniLM-L6-v2 CPU fallback via FALLBACK_EMBED=local | P1 |
| F-21 | Level 2 image grounding | Gemini bbox localisation overlay on cited images | P2 |
| F-22 | RAGAS Faithfulness | Grounding quality metric via RAGAS library | P2 |
| F-23 | Citation accuracy | page_id match rate between citation and gold quote | P2 |
| F-24 | Helm + HPA | Kubernetes deployment; HPA 2→10 replicas at 70% CPU | P2 |
| F-25 | Chunking ablation | Compare 256/512/1024 token sizes on dev Recall@10; document results | P2 |

---

## 19. 7-Day Build Plan

**Daily schedule:** 8–10 hours/day. All API calls use free tiers. Morning sessions focused on pipeline code. Evening sessions focused on evaluation and documentation.

---

### Day 1 — Dataset + Environment + Text RAG Baseline

**Goal:** Working text-only RAG over a 20-document subset. Proves the stack runs end to end before adding complexity.

**Tasks:**
- Create repo structure:
  ```
  src/ data/ notebooks/ helm/ docker/ tests/ docs/
  ```
- Install core dependencies:
  ```bash
  pip install llama-index llama-index-core \
    llama-index-llms-google-genai llama-index-embeddings-google-genai \
    llama-index-vector-stores-qdrant llama-index-llms-groq \
    google-genai qdrant-client docling pymupdf \
    fastapi uvicorn pydantic streamlit sentence-transformers \
    "ragas>=0.4" "langfuse>=4.0" \
    python-dotenv datasets
  ```
- Configure all API keys: Gemini (Google AI Studio), Qdrant Cloud cluster, Langfuse, Groq
- Download MMDocRAG: `doc_pdfs.zip` + `images.zip` + `evaluation_15.jsonl` + `dev_15.jsonl`
- Write `src/ingestion/docling_parser.py`: parse 20 PDFs from a single domain
- Write `src/chunking/text_chunker.py`: 512/64 SentenceWindow chunking within Docling blocks
- Write `src/embedding/embedder.py`: Gemini Embedding 2 text embed, RETRIEVAL_DOCUMENT
- Write `src/store/qdrant_client.py`: collection init + upsert (dense only, no BM25 yet)
- Write `src/retrieval/retriever.py`: basic ANN retrieval, Top-10
- Write `src/generation/llm.py`: Gemini 2.5 Flash text-only generation
- Test end-to-end: ingest 20 PDFs → query → answer with page citation

**Deliverable:** Working text RAG on 20-document subset. `/query` returns answers with page citations. No BM25, no reranker, no images yet.

---

### Day 2 — Image Pipeline + Hybrid Search

**Goal:** Image patches indexed alongside text. BM25 enabled. Unified hybrid retrieval working.

**Tasks:**
- Write `src/ingestion/image_extractor.py`: PyMuPDF image patch extraction per page
- Write `src/ingestion/image_describer.py`: Gemini Flash description per patch; cache to disk to avoid re-calling
- Write `src/chunking/image_chunker.py`: one chunk per patch; merge description + caption
- Write `src/chunking/table_chunker.py`: Docling table → markdown serialisation
- Upgrade Qdrant collection: enable sparse vector config for BM25
- Write `src/embedding/sparse_encoder.py`: BM25 sparse vector builder using tokenised `text_content`
- Upgrade `src/store/qdrant_client.py`: upsert both dense + sparse vectors
- Upgrade `src/retrieval/retriever.py`: LlamaIndex hybrid mode, RRF fusion, Top-50
- Ingest 20-document subset with full pipeline: text + tables + images
- Verify: query "what does the chart on page X show?" → image chunk appears in Top-10

**Deliverable:** Unified collection with text + image vectors. Hybrid BM25 + ANN retrieval returning mixed modality results.

---

### Day 3 — Re-ranking + Sentence-Window + Full Ingestion

**Goal:** Cross-encoder reranker integrated. Sentence-window expansion working. Full 222-document corpus ingested.

**Tasks:**
- Write `src/retrieval/reranker.py`: CrossEncoder ms-marco-MiniLM-L-6-v2; Top-50 → Top-10
- Integrate sentence-window postprocessor into retrieval pipeline
- Test reranker: compare Top-10 before and after reranking on 20 dev questions; note quality difference
- Run full 222-document ingestion overnight (or on Kaggle notebook): ~6–8 hours expected
  - Checkpoint progress per document: skip already-ingested docs on resume
  - Log: chunks per doc, images per doc, errors, API calls consumed
- Verify Qdrant storage usage stays below 800 MB (buffer before 1 GB limit)
- Write `src/evaluation/metrics.py`: Recall@K, Precision@K, Answer F1

**Deliverable:** Full corpus indexed. Reranker integrated. Recall@10 measured on first 50 dev questions.

---

### Day 4 — FastAPI + Evaluation Endpoint + Level 1 Image Grounding

**Goal:** FastAPI with all four endpoints. Evaluation pipeline running against MMDocRAG gold labels. Image citations working in Streamlit.

**Tasks:**
- Write `src/api/main.py`: FastAPI app, lifespan, exception handlers
- Write `src/api/routes/ingest.py`: `/ingest` with async BackgroundTask
- Write `src/api/routes/query.py`: `/query` with full pipeline call
- Write `src/api/routes/evaluate.py`: `/evaluate` with configurable k_values, max_records, modality split
- Write `src/api/routes/health.py`: `/health` with dependency checks
- Write `src/api/schemas.py`: all Pydantic request/response models
- Implement Level 1 image grounding: page_id + img_path in citation response
- Build `src/ui/app.py` Streamlit:
  - Query input panel
  - Answer panel with latency breakdown
  - Citation panel: text chunks with page/section, images rendered inline
  - Quote diff view: retrieved vs gold quote IDs side by side
- Run `/evaluate` on first 200 dev records: record Recall@5, Recall@10, Precision@10, Answer F1
- Document baseline numbers

**Deliverable:** Full API working. First real evaluation numbers. Image citations rendering in Streamlit.

---

### Day 5 — Langfuse + Fallbacks + Generation Quality

**Goal:** Observability live. Both fallback paths tested. Answer F1 and RAGAS Faithfulness measured.

**Tasks:**
- Integrate Langfuse `@observe()` decorators throughout pipeline
- Verify traces appear in Langfuse Cloud dashboard with latency breakdown per span
- Test LLM fallback: set `FALLBACK_LLM=groq`, run 10 queries, verify quality comparable
- Test embed fallback: set `FALLBACK_EMBED=local`, re-index 20-document subset with MiniLM, run same 10 queries, compare Recall@10
- Install and configure RAGAS: run faithfulness on 50 dev queries
- Tune generation prompt: if Answer F1 < 50%, iterate on prompt structure (more explicit citation instructions, shorter context window, JSON schema enforcement)
- Implement Level 2 image grounding: Gemini bbox localisation; overlay in Streamlit
- Write `src/evaluation/ragas_runner.py`: faithfulness + citation accuracy
- Document fallback quality gap in README

**Deliverable:** Langfuse dashboard live with public URL. All four evaluation metrics measured. Fallback paths tested and documented.

---

### Day 6 — Chunking Ablation + Docker + Optimization *(P2 — drop if behind)*

**Goal:** Chunking ablation documented. Docker working. Performance optimised.

**Tasks:**
- Run chunking ablation: re-index 20-document subset at 256, 512, 1024 token sizes
  - Measure Recall@10 at each size on 50 dev queries
  - Record results in `docs/ablation_chunking.md`
- Profile latency breakdown: where is the P95 bottleneck?
  - If Gemini API: cache descriptions and embeddings at ingestion, avoid re-calling at query time
  - If Qdrant: verify HNSW config, consider raising `ef` parameter
  - If reranker: verify batch scoring (all 50 pairs in one call, not 50 individual calls)
- Write multi-stage Dockerfile
- Write docker-compose.yml
- Test: `docker-compose up --build` → verify both services start, queries work
- Push image to GHCR

**Deliverable:** Docker working. Chunking ablation table in README. P95 latency documented.

---

### Day 7 — Kubernetes + README + Demo + Resume

**Goal:** Production GitHub repo. Demo recorded. Resume entry finalised.

**Tasks:**
- Write Helm chart: Deployments, Services, ConfigMaps, Secrets, Ingress, HPA
- `minikube start && helm install omni-query ./helm` → verify all pods Running
- Verify HPA: generate load → confirm replica count increases
- Write README:
  - Mermaid architecture diagram
  - Quickstart (3 commands)
  - Full evaluation results table (Recall@10 by modality, Answer F1, RAGAS Faithfulness)
  - Chunking ablation table
  - Langfuse dashboard URL (public)
  - Qdrant collection snapshot
  - Known limitations (reranker text-only for images, bbox model-approximated)
- Record 3-minute Loom demo:
  - Upload a PDF from MMDocRAG corpus
  - Ask a cross-modal question (requires image evidence)
  - Show: retrieved chunks panel → reranked Top-10 → answer → image citation with bbox overlay → quote diff view (retrieved vs gold)
- Write 3-bullet resume entry with quantified metrics

**Deliverable:** Production GitHub repo. Demo video. Public Langfuse URL. Resume entry with real numbers.

---

## 20. Environment Configuration

```env
# Primary LLM + Embeddings
GEMINI_API_KEY=your_google_ai_studio_key

# Gemini Embedding 2
EMBED_MODEL=gemini-embedding-2-preview
EMBED_DIM=768
EMBED_TASK_DOCUMENT=RETRIEVAL_DOCUMENT
EMBED_TASK_QUERY=RETRIEVAL_QUERY

# Vector Store
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=omni_query_v3

# Fallbacks
FALLBACK_LLM=gemini             # set to "groq" to force LLM fallback
GROQ_API_KEY=your_groq_key
FALLBACK_EMBED=gemini           # set to "local" for MiniLM CPU fallback

# Observability
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# App Config
TOP_K_RETRIEVAL=50              # candidates into reranker
TOP_K_RERANK=10                 # chunks into generation
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=64
SENTENCE_WINDOW_SIZE=3
MAX_PDF_PAGES=500
```

---

## 21. Project Structure

```
omni-query/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── ingest.py
│   │       ├── query.py
│   │       ├── evaluate.py
│   │       └── health.py
│   ├── ingestion/
│   │   ├── docling_parser.py
│   │   ├── image_extractor.py
│   │   └── image_describer.py
│   ├── chunking/
│   │   ├── text_chunker.py
│   │   ├── table_chunker.py
│   │   └── image_chunker.py
│   ├── embedding/
│   │   ├── embedder.py           # Gemini Embedding 2
│   │   ├── sparse_encoder.py     # BM25 sparse vectors
│   │   └── fallback_embedder.py  # all-MiniLM-L6-v2
│   ├── store/
│   │   └── qdrant_client.py
│   ├── retrieval/
│   │   ├── retriever.py          # hybrid BM25 + ANN
│   │   ├── reranker.py           # ms-marco cross-encoder
│   │   └── grounding.py          # 3-level image grounding
│   ├── generation/
│   │   ├── llm.py                # Gemini 2.5 Flash + Groq fallback
│   │   └── prompt_builder.py
│   ├── evaluation/
│   │   ├── metrics.py            # Recall@K, Precision@K, Answer F1
│   │   ├── ragas_runner.py
│   │   └── eval_runner.py
│   └── ui/
│       └── app.py                # Streamlit
├── data/
│   ├── raw/                      # MMDocRAG PDFs + images (gitignored)
│   ├── evaluation_15.jsonl
│   └── dev_15.jsonl
├── notebooks/
│   ├── 01_ingestion_test.ipynb
│   ├── 02_retrieval_ablation.ipynb
│   └── 03_chunking_ablation.ipynb
├── docs/
│   ├── ablation_chunking.md
│   └── evaluation_results.md
├── helm/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
├── .env.example
├── requirements.txt
└── README.md
```

---

## 22. Risk Register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | Gemini free tier rate-limited during full 222-doc ingestion | High | Cache image descriptions to disk after first call; cache embeddings to disk; skip already-indexed chunks on resume. Keep a second Gemini API key on a backup Google account. |
| R2 | Full ingestion takes longer than one day (222 docs × avg 400 chunks = 88K text embed calls + up to 32K image embed calls) | Medium | Use Kaggle P100 notebook for ingestion — 9hr session. Checkpoint per-document. Resume flag in ingestion script. Or ingest in batches across multiple sessions. Consider indexing only gold image quotes (6,349) first, then expanding. |
| R3 | Qdrant 1 GB storage limit reached during ablation re-indexing | Low | Use separate collection names per ablation run (`omni_query_v3_256`, `omni_query_v3_512`). Delete ablation collections after recording results. |
| R4 | MMDocRAG `doc_pdfs.zip` download slow or interrupted | Low | Download to Kaggle notebook directly. Verify SHA before unzip. The zip is ~2GB. |
| R5 | Gemini Embedding 2 is still in preview — API may change | Low-Medium | Pin `google-genai` SDK to the version used at build time. Fallback to `all-MiniLM-L6-v2` is always available. |
| R6 | Cross-encoder reranker slow on CPU for 50 pairs | Low | Batch all 50 pairs in a single `reranker.predict(pairs)` call — not a loop. Batch inference is 10–20x faster than individual calls. Target: < 200ms for 50 pairs. |
| R7 | Answer F1 significantly below 55% target | Medium | Iterate on generation prompt structure on Day 5. Try: more explicit citation format, shorter context (Top-5 instead of Top-10), structured JSON schema in system prompt. If still low, diagnose: is Recall@10 also low? If yes, fix retrieval. If Recall@10 is high, fix generation. |
| R8 | Langfuse cloud signup blocked or slow | Low | Self-host Langfuse via Docker on local machine as fallback. Or skip Langfuse and use custom timing logs in FastAPI middleware — less portfolio impact but workable. |

---

## 23. Resume Entry

```
Omni-Query — Multimodal RAG over Long-Document Corpus              March 2026
• Built end-to-end multimodal RAG system over the MMDocRAG benchmark corpus
  (222 documents, 4,055 expert-annotated QA pairs, 32,071 image quotes across
  10 domains); achieved Recall@10 of [XX]% overall and [XX]% on cross-modal
  questions requiring both text and image evidence retrieval.
• Designed 13-layer pipeline: Docling layout parse → sentence-window chunking
  with parent-child expansion → Gemini Embedding 2 unified 768-dim text+image
  vectors → Qdrant Cloud hybrid BM25+ANN retrieval → ms-marco cross-encoder
  reranking (Top-50 → Top-10) → Gemini 2.5 Flash multimodal generation →
  3-level image grounding with Gemini bbox localisation.
• Evaluated using four independent metrics against expert-annotated ground truth:
  Recall@10 (retriever), Answer F1 vs answer_short (generation), RAGAS
  Faithfulness (grounding), citation page_id match rate; deployed on Kubernetes
  via Helm with HPA (2→10 replicas), full Langfuse tracing, $0 infra cost.
```

**Important:** Replace `[XX]%` placeholders with actual measured metrics after running the evaluation on Day 5–7. Do not use projected numbers on your resume.

---

## 24. Interview Signal Summary

| Signal | What Omni-Query Demonstrates |
|---|---|
| Dataset rigour | Evaluated against MMDocRAG — a Huawei Noah's Ark Lab benchmark used to test 60 state-of-the-art models. Not a self-generated test set. |
| RAG architecture depth | 13 distinct pipeline layers. Chunking strategy defined per content type. Retrieval, reranking, and generation are separate concerns measured independently. |
| Multimodal engineering | Gemini Embedding 2 unifies text and image into one vector space — cross-modal retrieval without separate pipelines or captioning proxies. |
| Evaluation maturity | Four metrics measuring different failure modes. Gap between Recall@10 and Answer F1 points to the exact layer that needs fixing. |
| Honest engineering | Reranker text-only limitation on image chunks documented. BBox grounding marked as model-approximated. Fallback quality gap measured and recorded. |
| Production readiness | Docker multi-stage, Helm, HPA, Langfuse observability, public dashboard URL, $0 infra cost. |
| 2026 stack awareness | Gemini Embedding 2 (March 2026), Langfuse v4 (March 2026), Docling, MMDocRAG (May 2025) — active field engagement. |

---

*PRD v3.1 — Omni-Query | Author: Loki | March 2026 | Portfolio Use Only*
*All APIs on free tiers. Total infrastructure cost: $0.*
