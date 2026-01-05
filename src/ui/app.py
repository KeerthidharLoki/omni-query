"""
Omni-Query Streamlit UI.

4 panels:
  1. Query input with autocomplete suggestion chips
  2. Answer panel with latency breakdown
  3. Citation panel — text excerpts + rendered images with metadata
  4. Quote diff view — retrieved vs gold, Recall@10 / Precision@10

Run: streamlit run src/ui/app.py
"""

from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:9100"

st.set_page_config(
    page_title="Omni-Query",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.chip-btn button {
    background: #1e3a5f !important;
    color: #e8f4f8 !important;
    border-radius: 16px !important;
    border: 1px solid #2d5986 !important;
    padding: 2px 12px !important;
    font-size: 0.80rem !important;
    margin: 2px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.badge-green {background:#1a5c2a;color:#b9f7c8;padding:2px 8px;border-radius:8px;font-size:0.75rem}
.badge-blue  {background:#1a3a5c;color:#b0d4f7;padding:2px 8px;border-radius:8px;font-size:0.75rem}
.badge-red   {background:#5c1a1a;color:#f7b0b0;padding:2px 8px;border-radius:8px;font-size:0.75rem}
.badge-gray  {background:#3a3a3a;color:#cccccc;padding:2px 8px;border-radius:8px;font-size:0.75rem}
.metric-box  {background:#1a1a2e;border:1px solid #2d2d4e;border-radius:8px;padding:10px;text-align:center}
.quote-match {background:#0d2b0d;border-left:3px solid #2ecc71;padding:4px 8px;margin:2px 0;border-radius:3px;font-size:0.80rem}
.quote-miss  {background:#2b0d0d;border-left:3px solid #e74c3c;padding:4px 8px;margin:2px 0;border-radius:3px;font-size:0.80rem}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ─────────────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False


# ── Helper functions ───────────────────────────────────────────────────────────
def fetch_suggestions(q: str) -> list[str]:
    try:
        resp = httpx.get(f"{API_BASE}/suggest", params={"q": q}, timeout=3)
        return resp.json().get("suggestions", [])
    except Exception:
        return []


def run_query(q: str, doc_filter=None, domain_filter=None) -> dict | None:
    try:
        payload = {"query": q, "top_k": 10}
        if doc_filter:
            payload["doc_filter"] = doc_filter
        if domain_filter:
            payload["domain_filter"] = domain_filter
        resp = httpx.post(f"{API_BASE}/query", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def check_api() -> bool:
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def badge(label: str, kind: str = "blue") -> str:
    return f'<span class="badge-{kind}">{label}</span>'


# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_status = st.columns([5, 2])
with col_logo:
    st.markdown("## 🔍 Omni-Query")
    st.caption("Multimodal RAG · MMDocRAG · 222 PDFs · 4,055 QA pairs")

with col_status:
    api_ok = check_api()
    st.markdown("<br>", unsafe_allow_html=True)
    if api_ok:
        st.markdown(
            badge("Gemini 2.5 Flash ✓", "green") + "  " +
            badge("Qdrant ✓", "green") + "  " +
            badge("Langfuse ✓", "green"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(badge("API offline", "red"), unsafe_allow_html=True)

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    doc_filter = st.text_input("Document name", placeholder="e.g. COSTCO_2021_10K")
    domain_filter = st.selectbox(
        "Domain",
        ["", "Financial report", "Academic paper", "Research report / Introduction", "News"],
    )
    st.divider()
    st.markdown("### Evaluation")
    if st.button("Run Recall@10 on eval set (200 records)", use_container_width=True):
        with st.spinner("Evaluating …"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/evaluate",
                    json={"eval_file": "evaluation_15", "max_records": 200},
                    timeout=60,
                )
                ev = resp.json()
                st.metric("Recall@10", f"{ev['recall_at_10']:.3f}")
                st.metric("Precision@10", f"{ev['precision_at_10']:.3f}")
                st.metric("Answer F1", f"{ev['answer_f1']:.3f}")
                st.metric("Citation Acc.", f"{ev['citation_accuracy']:.3f}")
                st.caption(f"{ev['records_evaluated']} records · {ev['duration_seconds']:.1f}s")
            except Exception as e:
                st.error(str(e))
    st.divider()
    st.markdown("### About")
    st.markdown(
        "**Stack:** Gemini Embedding 2 · Qdrant Cloud · "
        "BM25+ANN · ms-marco reranker · Gemini 2.5 Flash · Langfuse"
    )
    st.markdown("**Infra cost:** $0 — all free tiers")


# ── Query input ────────────────────────────────────────────────────────────────
input_col, btn_col = st.columns([8, 1])
with input_col:
    query_input = st.text_input(
        "Ask a question",
        value=st.session_state.query,
        placeholder="e.g. What is the long-term debt ratio for COSTCO in FY2021?",
        label_visibility="collapsed",
        key="query_input_box",
    )
with btn_col:
    ask_clicked = st.button("Ask →", use_container_width=True, type="primary")

# Autocomplete suggestions
if query_input and len(query_input) >= 3 and not ask_clicked:
    suggestions = fetch_suggestions(query_input)
    if suggestions:
        st.markdown("**Suggestions:**")
        # Render as clickable chips in rows of 2
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                short = suggestion if len(suggestion) <= 80 else suggestion[:77] + "…"
                with st.container():
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    if st.button(short, key=f"chip_{i}"):
                        st.session_state.query = suggestion
                        st.session_state.submitted = True
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

# Trigger search
if ask_clicked and query_input.strip():
    st.session_state.query = query_input.strip()
    st.session_state.submitted = True

if st.session_state.submitted and st.session_state.query:
    st.session_state.submitted = False
    with st.spinner("Retrieving · Reranking · Generating …"):
        result = run_query(
            st.session_state.query,
            doc_filter=doc_filter or None,
            domain_filter=domain_filter or None,
        )
    st.session_state.result = result

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    st.divider()

    # Show matched question if different
    if r["matched_question"] != st.session_state.query:
        st.info(f"**Matched dataset question:** {r['matched_question']}", icon="ℹ️")

    # Two-column layout: Answer | Citations
    ans_col, cite_col = st.columns([5, 4])

    with ans_col:
        st.markdown("### Answer")
        st.markdown(
            f"<div style='background:#1a1a2e;border-left:4px solid #3498db;"
            f"padding:16px;border-radius:6px;font-size:1.05rem;line-height:1.6'>"
            f"{r['answer']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Metadata badges
        mods = r.get("evidence_modality_type", [])
        mod_str = " + ".join(mods) if mods else "text"
        st.markdown(
            badge(r["domain"], "blue") + "  " +
            badge(r["doc_name"], "gray") + "  " +
            badge(r["question_type"], "gray") + "  " +
            badge(f"Evidence: {mod_str}", "blue"),
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Latency breakdown
        lat_c1, lat_c2, lat_c3, lat_c4 = st.columns(4)
        lat_c1.metric("Retrieval", f"{r['retrieval_ms']}ms")
        lat_c2.metric("Rerank", f"{r['rerank_ms']}ms")
        lat_c3.metric("Generate", f"{r['generation_ms']/1000:.1f}s")
        lat_c4.metric("Total", f"{r['total_ms']/1000:.1f}s")

        st.caption(f"LLM: `{r['llm_used']}` · Embed: `{r['embed_model']}`")

    with cite_col:
        st.markdown("### Citations")

        # Text citations
        if r["text_citations"]:
            st.markdown("**Text Evidence**")
            for tc in r["text_citations"]:
                is_gold = tc["quote_id"] in r["gold_quotes"]
                gold_mark = " ⭐" if is_gold else ""
                with st.expander(
                    f"📄 {tc['quote_id']}{gold_mark} · page {tc['page_id']} · score {tc['rerank_score']}"
                ):
                    st.markdown(f"_{tc['text_preview']}_")

        # Image citations
        if r["image_citations"]:
            st.markdown("**Visual Evidence**")
            for ic in r["image_citations"]:
                is_gold = ic["quote_id"] in r["gold_quotes"]
                gold_mark = " ⭐" if is_gold else ""
                with st.expander(
                    f"🖼 {ic['quote_id']}{gold_mark} · page {ic['page_id']} · {ic['type']} · score {ic['rerank_score']}"
                ):
                    img_url = f"{API_BASE}/images/{ic['img_filename']}"
                    try:
                        img_resp = httpx.get(img_url, timeout=5)
                        if img_resp.status_code == 200:
                            st.image(
                                img_resp.content,
                                caption=f"Page {ic['page_id']} · {ic['type']}",
                                use_container_width=True,
                            )
                        else:
                            st.caption(f"Image not available: {ic['img_filename']}")
                    except Exception:
                        st.caption(f"Could not load: {ic['img_filename']}")
                    if ic["img_description"]:
                        st.markdown(f"**Description:** {ic['img_description']}")
                    st.markdown(
                        badge("dataset-native", "green"),
                        unsafe_allow_html=True,
                    )

    # ── Quote diff view ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Quote Diff View")
    diff_col1, diff_col2, diff_col3 = st.columns([4, 4, 2])

    retrieved_ids = r.get("retrieved_quote_ids", [])
    gold_ids = set(r.get("gold_quotes", []))

    with diff_col1:
        st.markdown("**Retrieved (Top-10)**")
        for qid in retrieved_ids[:10]:
            is_match = qid in gold_ids
            css_class = "quote-match" if is_match else "quote-miss"
            icon = "✅" if is_match else "❌"
            st.markdown(
                f'<div class="{css_class}">{icon} {qid}</div>',
                unsafe_allow_html=True,
            )

    with diff_col2:
        st.markdown("**Gold Quotes**")
        for qid in r.get("gold_quotes", []):
            in_retrieved = qid in set(retrieved_ids[:10])
            css_class = "quote-match" if in_retrieved else "quote-miss"
            icon = "✅" if in_retrieved else "❌"
            st.markdown(
                f'<div class="{css_class}">{icon} {qid}</div>',
                unsafe_allow_html=True,
            )

    with diff_col3:
        st.markdown("**Metrics**")
        st.markdown(
            f"<div class='metric-box'>"
            f"<div style='font-size:0.75rem;color:#aaa'>Recall@10</div>"
            f"<div style='font-size:1.8rem;font-weight:bold;color:#2ecc71'>{r['recall_at_10']:.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric-box'>"
            f"<div style='font-size:0.75rem;color:#aaa'>Precision@10</div>"
            f"<div style='font-size:1.8rem;font-weight:bold;color:#3498db'>{r['precision_at_10']:.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

else:
    # Starter state
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Type a question above to search across 222 long documents. "
        "The system will retrieve evidence from text, tables, and images "
        "and return a grounded answer with citations.",
        icon="💡",
    )

    # Show example questions
    st.markdown("#### Example Questions")
    examples = [
        "What is Long-term Debt to Total Liabilities for COSTCO in FY2021?",
        "What does the revenue breakdown chart show for Amazon in 2017?",
        "How does net income compare between fiscal years in the 3M 10-K?",
        "What are the key findings from the ACL 2020 paper on multilingual models?",
        "What is the operating margin trend for Best Buy in 2023?",
    ]
    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            with st.container():
                st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                short = ex if len(ex) <= 75 else ex[:72] + "…"
                if st.button(short, key=f"example_{i}"):
                    st.session_state.query = ex
                    st.session_state.submitted = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
