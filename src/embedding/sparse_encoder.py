"""
BM25 sparse vector encoder for Qdrant hybrid search.

Converts text into sparse TF-IDF-style vectors compatible with Qdrant's
BM25 sparse index. Used alongside dense Gemini Embedding 2 vectors for
hybrid BM25 + ANN retrieval with RRF fusion.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional


# Simple English stopwords — keeps the sparse index clean
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "as", "if", "not", "no", "so", "up",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and short tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Build a token → integer index vocabulary from a corpus of texts."""
    vocab: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


class BM25SparseEncoder:
    """
    BM25 sparse vector encoder.

    Encodes text into sparse (indices, values) pairs compatible with
    Qdrant's SparseVector format. Fit on the corpus at ingestion time,
    then encode each chunk and the query at search time.

    Parameters
    ----------
    k1 : float
        BM25 term frequency saturation parameter. Default 1.5.
    b : float
        BM25 length normalisation parameter. Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self._corpus_size: int = 0

    def fit(self, texts: list[str]) -> "BM25SparseEncoder":
        """Compute IDF weights from a corpus of document texts."""
        self._corpus_size = len(texts)
        tokenized = [tokenize(t) for t in texts]
        self.avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)

        # Build vocabulary
        all_tokens: set[str] = set()
        for tokens in tokenized:
            all_tokens.update(tokens)
        self.vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}

        # Compute document frequency per token
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))

        # IDF with smoothing
        N = self._corpus_size
        self.idf = {
            tok: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }
        return self

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """
        Encode a document into BM25 sparse vector (indices, values).

        Returns empty lists if the encoder has not been fit yet — falls back
        to TF-only encoding in that case.
        """
        tokens = tokenize(text)
        if not tokens:
            return [], []

        tf = Counter(tokens)
        doc_len = len(tokens)
        indices, values = [], []

        for token, freq in tf.items():
            if token not in self.vocab:
                continue
            idx = self.vocab[token]
            idf = self.idf.get(token, 1.0)
            # BM25 TF normalisation
            tf_norm = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            )
            score = idf * tf_norm
            if score > 0:
                indices.append(idx)
                values.append(float(score))

        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """
        Encode a query into a sparse vector.

        Query encoding uses raw IDF weights (no TF normalisation) since
        queries are short and each term appears at most once.
        """
        tokens = tokenize(text)
        indices, values = [], []
        for token in set(tokens):
            if token not in self.vocab:
                continue
            indices.append(self.vocab[token])
            values.append(float(self.idf.get(token, 1.0)))
        return indices, values

    def save(self, path: str) -> None:
        """Persist the encoder to disk as JSON."""
        import json
        state = {
            "k1": self.k1,
            "b": self.b,
            "vocab": self.vocab,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "corpus_size": self._corpus_size,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "BM25SparseEncoder":
        """Load a persisted encoder from disk."""
        import json
        with open(path) as f:
            state = json.load(f)
        enc = cls(k1=state["k1"], b=state["b"])
        enc.vocab = state["vocab"]
        enc.idf = state["idf"]
        enc.avgdl = state["avgdl"]
        enc._corpus_size = state["corpus_size"]
        return enc
