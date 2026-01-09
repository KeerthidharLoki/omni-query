"""Shared fixtures for the Omni-Query test suite."""

from __future__ import annotations

import json
import pytest


SAMPLE_RECORD = {
    "question": "What is the long-term debt ratio for COSTCO in FY2021?",
    "answer_short": "The long-term debt ratio was 0.32.",
    "answer_interleaved": "According to figure 3, the long-term debt ratio was 0.32.",
    "gold_quotes": ["q001", "q002"],
    "text_quotes": [
        {"quote_id": "q001", "page_id": 5, "text": "Long-term debt ratio 0.32"},
        {"quote_id": "q003", "page_id": 6, "text": "Other financial text"},
    ],
    "img_quotes": [
        {
            "quote_id": "q002",
            "page_id": 5,
            "img_path": "images/fig_001.jpg",
            "img_description": "Bar chart showing debt ratios over time",
            "layout_id": 42,
            "type": "chart",
        }
    ],
    "doc_name": "COSTCO_2021_10K",
    "domain": "Financial report",
    "question_type": "numerical",
    "evidence_modality_type": ["chart", "text"],
}


@pytest.fixture
def sample_record():
    return dict(SAMPLE_RECORD)


@pytest.fixture
def fake_app(tmp_path, monkeypatch):
    """FastAPI TestClient with a single record pre-loaded in app.state."""
    dev_file = tmp_path / "dev_15.jsonl"
    eval_file = tmp_path / "evaluation_15.jsonl"
    for f in (dev_file, eval_file):
        f.write_text(json.dumps(SAMPLE_RECORD) + "\n", encoding="utf-8")

    # Patch module-level path variables before lifespan runs
    import src.api.main as main_module
    monkeypatch.setattr(main_module, "DEV_FILE", dev_file)
    monkeypatch.setattr(main_module, "EVAL_FILE", eval_file)

    # Patch DEMO_MODE in the query router module
    import src.api.routes.query as query_module
    monkeypatch.setattr(query_module, "DEMO_MODE", True)

    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        yield client
