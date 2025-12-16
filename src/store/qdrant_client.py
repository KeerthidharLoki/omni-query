"""
Qdrant Cloud client — collection initialisation, upsert, and search helpers.

Single unified collection: text + image vectors in the same 768-dim space.
BM25 sparse index + HNSW dense index, hybrid RRF fusion at query time.
"""

from __future__ import annotations

import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "omni_query_v3")
VECTOR_DIM = int(os.getenv("EMBED_DIM", "768"))


def get_client() -> QdrantClient:
    """Return an authenticated Qdrant Cloud client."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    return QdrantClient(url=url, api_key=api_key)


def create_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """
    Initialise the unified text+image collection with BM25 + HNSW indexes.

    Safe to call on an existing collection — skips creation if already present.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists — skipping creation.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
                on_disk=True,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"Created collection '{collection_name}' — {VECTOR_DIM}-dim dense + BM25 sparse.")


def upsert_chunk(
    client: QdrantClient,
    chunk_id: str,
    dense_vector: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    payload: dict,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Upsert a single chunk (text or image) with both dense and sparse vectors."""
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=chunk_id,
                vector={
                    "dense": dense_vector,
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
                payload=payload,
            )
        ],
    )


def upsert_batch(
    client: QdrantClient,
    points: list[dict],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 64,
) -> None:
    """
    Upsert a batch of chunks efficiently.

    Each item in `points` must have keys:
        id, dense_vector, sparse_indices, sparse_values, payload
    """
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        structs = [
            PointStruct(
                id=p["id"],
                vector={
                    "dense": p["dense_vector"],
                    "sparse": SparseVector(
                        indices=p["sparse_indices"],
                        values=p["sparse_values"],
                    ),
                },
                payload=p["payload"],
            )
            for p in batch
        ]
        client.upsert(collection_name=collection_name, points=structs)
        print(f"  Upserted {i + len(batch)}/{len(points)} chunks")


def collection_info(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> dict:
    """Return basic stats about the collection."""
    info = client.get_collection(collection_name)
    return {
        "vectors_count": info.vectors_count,
        "points_count": info.points_count,
        "status": str(info.status),
        "collection_name": collection_name,
    }


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    """Delete a collection — used for ablation experiments."""
    client.delete_collection(collection_name)
    print(f"Deleted collection '{collection_name}'.")
