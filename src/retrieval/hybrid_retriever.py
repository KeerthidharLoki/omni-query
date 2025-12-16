"""
Hybrid BM25 + ANN retrieval with RRF fusion per PRD Layer 5.

Uses LlamaIndex QdrantVectorStore with enable_hybrid=True.
BM25 keyword search over text_content field in Qdrant sparse index.
ANN cosine search over 768-dim dense vectors.
Reciprocal Rank Fusion combines both ranked lists → Top-50 candidates.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "omni_query_v3"
TOP_K = 50


def build_sparse_vector(text: str) -> dict:
    """
    Build a BM25-compatible sparse vector from text.
    Uses simple TF weighting as approximation for Qdrant sparse index.
    """
    from collections import Counter
    import math

    tokens = text.lower().split()
    tf = Counter(tokens)
    total = len(tokens)

    indices = []
    values = []
    for token, count in tf.items():
        token_hash = hash(token) % (2**20)  # 20-bit vocab
        indices.append(abs(token_hash))
        values.append(count / total * math.log(1 + len(tokens)))

    return {"indices": indices, "values": values}


class HybridRetriever:
    """
    Hybrid BM25 + ANN retriever backed by Qdrant Cloud.

    Retrieves Top-50 candidates for subsequent cross-encoder reranking.
    Supports optional metadata filtering by doc_name or domain.
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection_name: str = COLLECTION_NAME,
        top_k: int = TOP_K,
        embed_dim: int = 768,
    ):
        from qdrant_client import QdrantClient
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.core import VectorStoreIndex

        self._qdrant = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL"),
            api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY"),
        )
        self._vector_store = QdrantVectorStore(
            client=self._qdrant,
            collection_name=collection_name,
            enable_hybrid=True,
            sparse_query_fn=build_sparse_vector,
        )
        self._index = VectorStoreIndex.from_vector_store(self._vector_store)
        self._top_k = top_k
        logger.info("HybridRetriever ready (collection=%s, top_k=%d)", collection_name, top_k)

    def retrieve(
        self,
        query: str,
        query_vector: list[float],
        doc_filter: str | None = None,
        domain_filter: str | None = None,
    ) -> list[Any]:
        """Retrieve Top-50 candidates using hybrid BM25+ANN with RRF fusion."""
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

        filters = []
        if doc_filter:
            filters.append(MetadataFilter(key="doc_name", value=doc_filter))
        if domain_filter:
            filters.append(MetadataFilter(key="domain", value=domain_filter))

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self._top_k,
            sparse_top_k=self._top_k,
            hybrid_top_k=self._top_k,
            filters=MetadataFilters(filters=filters) if filters else None,
        )
        return retriever.retrieve(query)

    def collection_info(self) -> dict:
        """Return current collection stats from Qdrant."""
        info = self._qdrant.get_collection(COLLECTION_NAME)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }
