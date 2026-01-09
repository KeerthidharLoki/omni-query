"""Integration tests for /query and /health endpoints via FastAPI TestClient."""

from __future__ import annotations

import pytest


def test_health_returns_200(fake_app):
    resp = fake_app.get("/health")
    assert resp.status_code == 200

def test_health_status_healthy(fake_app):
    data = fake_app.get("/health").json()
    assert data["status"] == "healthy"

def test_health_record_counts(fake_app):
    data = fake_app.get("/health").json()
    assert data["dev_records_loaded"] == 1
    assert data["eval_records_loaded"] == 1

def test_query_valid_returns_200(fake_app):
    resp = fake_app.post("/query", json={"query": "what is the long-term debt ratio?"})
    assert resp.status_code == 200

def test_query_response_has_answer(fake_app):
    data = fake_app.post("/query", json={"query": "what is the long-term debt ratio?"}).json()
    assert "answer" in data
    assert len(data["answer"]) > 0

def test_query_response_demo_mode_true(fake_app):
    data = fake_app.post("/query", json={"query": "what is the long-term debt ratio?"}).json()
    assert data["demo_mode"] is True

def test_query_too_short_returns_422(fake_app):
    resp = fake_app.post("/query", json={"query": "hi"})
    assert resp.status_code == 422

def test_suggest_returns_list(fake_app):
    data = fake_app.get("/suggest?q=debt").json()
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)

def test_query_with_doc_filter(fake_app):
    resp = fake_app.post("/query", json={
        "query": "what is the long-term debt ratio?",
        "doc_filter": "COSTCO_2021_10K",
    })
    assert resp.status_code == 200
