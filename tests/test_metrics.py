"""Tests for src/evaluation/metrics.py — all pure Python, no external deps."""

from __future__ import annotations

import pytest
from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    answer_f1,
    citation_accuracy,
)


# ── recall_at_k ────────────────────────────────────────────────────────────────

def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=10) == 1.0

def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=10) == 0.5

def test_recall_zero():
    assert recall_at_k(["x", "y"], ["a", "b"], k=10) == 0.0

def test_recall_empty_gold():
    assert recall_at_k(["a", "b"], [], k=10) == 0.0

def test_recall_respects_k():
    # gold items appear at positions 11 and 12 — beyond k=10
    retrieved = [f"noise{i}" for i in range(10)] + ["g1", "g2"]
    assert recall_at_k(retrieved, ["g1", "g2"], k=10) == 0.0

def test_recall_k_equals_1():
    assert recall_at_k(["g1", "g2"], ["g1"], k=1) == 1.0
    assert recall_at_k(["x", "g1"], ["g1"], k=1) == 0.0


# ── precision_at_k ─────────────────────────────────────────────────────────────

def test_precision_perfect():
    assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

def test_precision_zero():
    assert precision_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

def test_precision_k_zero():
    assert precision_at_k(["a"], ["a"], k=0) == 0.0

def test_precision_partial():
    result = precision_at_k(["g1", "x", "y", "z", "a", "b", "c", "d", "e", "f"], ["g1"], k=10)
    assert abs(result - 0.1) < 1e-9


# ── answer_f1 ──────────────────────────────────────────────────────────────────

def test_f1_exact_match():
    assert answer_f1("the cat sat", "the cat sat") == 1.0

def test_f1_no_overlap():
    assert answer_f1("hello world", "foo bar baz") == 0.0

def test_f1_empty_prediction():
    assert answer_f1("", "some answer") == 0.0

def test_f1_empty_ground_truth():
    assert answer_f1("some answer", "") == 0.0

def test_f1_partial_overlap():
    # "the cat sat" vs "the dog sat" — common: "the", "sat" (2 tokens)
    # precision = 2/3, recall = 2/3, f1 = 2/3
    result = answer_f1("the cat sat", "the dog sat")
    assert abs(result - 2 / 3) < 1e-9


# ── citation_accuracy ──────────────────────────────────────────────────────────

def test_citation_hit():
    all_quotes = [{"quote_id": "q1", "page_id": 5}]
    assert citation_accuracy([5], ["q1"], all_quotes) == 1.0

def test_citation_miss():
    all_quotes = [{"quote_id": "q1", "page_id": 5}]
    assert citation_accuracy([9], ["q1"], all_quotes) == 0.0

def test_citation_empty_gold():
    all_quotes = [{"quote_id": "q1", "page_id": 5}]
    assert citation_accuracy([5], [], all_quotes) == 0.0
