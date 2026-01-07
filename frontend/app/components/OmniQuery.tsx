'use client'

import { useState, useRef, useCallback, useEffect } from 'react'

const API = 'http://localhost:9100'

// ── Types ─────────────────────────────────────────────────────────────────────
interface TextCitation { quote_id: string; page_id: number; text_preview: string; rerank_score: number }
interface ImageCitation { quote_id: string; type: string; page_id: number; img_filename: string; img_description: string; rerank_score: number; grounding_type: string }
interface Result {
  answer: string; matched_question: string; doc_name: string; domain: string
  question_type: string; evidence_modality_type: string[]
  text_citations: TextCitation[]; image_citations: ImageCitation[]
  gold_quotes: string[]; retrieved_quote_ids: string[]
  recall_at_10: number; precision_at_10: number
  llm_used: string; embed_model: string
  retrieval_ms: number; rerank_ms: number; generation_ms: number; total_ms: number
}
interface EvalData {
  recall_at_10: number; precision_at_10: number; answer_f1: number
  recall_at_5: number; precision_at_5: number
  records_evaluated: number; duration_seconds: number
  breakdown_by_modality: Record<string, { recall_at_10: number; answer_f1: number; count: number }>
}

function ms(n: number) { return n >= 1000 ? `${(n/1000).toFixed(1)}s` : `${n}ms` }
function pct(n: number) { return `${(n*100).toFixed(1)}%` }

// ── Eval Modal ─────────────────────────────────────────────────────────────────
function EvalModal({ onClose }: { onClose: () => void }) {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<EvalData | null>(null)

  async function run() {
    setLoading(true)
    try {
      const r = await fetch(`${API}/evaluate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ eval_file: 'evaluation_15', max_records: 200 })
      })
      setData(await r.json())
    } finally { setLoading(false) }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4"
      style={{ background: 'rgba(0,0,0,0.85)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="w-full max-w-md" style={{ background: '#0f0f0f', border: '1px solid #222' }}>
        <div className="flex items-center justify-between px-6 py-4" style={{ borderBottom: '1px solid #222' }}>
          <span className="text-sm" style={{ color: '#e8e8e8' }}>Benchmark Evaluation</span>
          <button onClick={onClose} className="text-xs" style={{ color: '#555' }}>close</button>
        </div>
        <div className="p-6">
          {!data && !loading && (
            <div className="py-4">
              <p className="text-sm mb-1" style={{ color: '#555' }}>evaluation_15.jsonl · 200 records</p>
              <button onClick={run} className="mt-4 text-sm underline underline-offset-4" style={{ color: '#e8e8e8' }}>
                Run evaluation
              </button>
            </div>
          )}
          {loading && <p className="text-sm py-4" style={{ color: '#555' }}>Computing…</p>}
          {data && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                {([
                  ['Recall@10', pct(data.recall_at_10)],
                  ['Recall@5', pct(data.recall_at_5)],
                  ['Precision@10', pct(data.precision_at_10)],
                  ['Answer F1', pct(data.answer_f1)],
                ] as [string, string][]).map(([l, v]) => (
                  <div key={l}>
                    <p className="text-2xl font-semibold" style={{ color: '#e8e8e8' }}>{v}</p>
                    <p className="text-xs mt-1" style={{ color: '#555' }}>{l}</p>
                  </div>
                ))}
              </div>
              <div style={{ borderTop: '1px solid #222', paddingTop: '16px' }}>
                <p className="text-xs mb-3" style={{ color: '#555' }}>By modality</p>
                {Object.entries(data.breakdown_by_modality).slice(0, 4).map(([k, v]) => (
                  <div key={k} className="flex justify-between py-2 text-sm" style={{ borderBottom: '1px solid #1a1a1a' }}>
                    <span style={{ color: '#555' }}>{k}</span>
                    <span style={{ color: '#e8e8e8' }}>R@10 {pct(v.recall_at_10)} · F1 {pct(v.answer_f1)}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs" style={{ color: '#333' }}>{data.records_evaluated} records · {data.duration_seconds.toFixed(1)}s</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Main ───────────────────────────────────────────────────────────────────────
export default function OmniQuery() {
  const [inputVal, setInputVal] = useState('')
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSugg, setShowSugg] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [showEval, setShowEval] = useState(false)
  const [showDiff, setShowDiff] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const fetchSuggestions = useCallback((q: string) => {
    if (timer.current) clearTimeout(timer.current)
    if (q.length < 2) { setSuggestions([]); return }
    timer.current = setTimeout(async () => {
      try {
        const r = await fetch(`${API}/suggest?q=${encodeURIComponent(q)}`)
        setSuggestions((await r.json()).suggestions ?? [])
      } catch { /* silent */ }
    }, 250)
  }, [])

  function handleChange(v: string) {
    setInputVal(v)
    fetchSuggestions(v)
    setShowSugg(true)
  }

  async function doSearch(q: string) {
    const trimmed = q.trim()
    if (!trimmed) return
    setQuery(trimmed)
    setInputVal(trimmed)
    setSuggestions([])
    setShowSugg(false)
    setLoading(true)
    setResult(null)
    setShowDiff(false)
    try {
      const r = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, top_k: 10 })
      })
      setResult(await r.json())
    } catch { /* silent */ } finally { setLoading(false) }
  }

  const EXAMPLES = [
    'What is the Long-term Debt to Total Liabilities for COSTCO in FY2021?',
    'What does the revenue chart show for Amazon in 2017?',
    'How does net income compare year over year in the 3M 10-K?',
    'What are the key findings from the ACL 2020 multilingual NLP paper?',
    'What is the operating margin trend for Best Buy in FY2023?',
    'What does the cash flow statement show for ACTIVISION in 2019?',
  ]

  // ── Search input JSX (inlined — not a component, to avoid remount on rerender)
  function searchBox(large: boolean, placeholder: string) {
    return (
      <div style={{ position: 'relative', width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', borderBottom: `1px solid ${large ? '#333' : '#222'}` }}>
          <input
            value={inputVal}
            onChange={e => handleChange(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') doSearch(inputVal)
              if (e.key === 'Escape') setShowSugg(false)
            }}
            onFocus={() => suggestions.length > 0 && setShowSugg(true)}
            placeholder={placeholder}
            autoFocus={large}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: '#e8e8e8',
              fontSize: large ? '17px' : '15px',
              padding: large ? '14px 0' : '8px 0',
              fontFamily: 'inherit',
            }}
          />
          <button
            onClick={() => doSearch(inputVal)}
            disabled={!inputVal.trim()}
            style={{
              background: 'none',
              border: 'none',
              cursor: inputVal.trim() ? 'pointer' : 'default',
              color: inputVal.trim() ? '#e8e8e8' : '#333',
              fontSize: '14px',
              padding: '4px 0',
              flexShrink: 0,
              fontFamily: 'inherit',
            }}
          >
            Ask
          </button>
        </div>

        {showSugg && suggestions.length > 0 && (
          <div style={{ position: 'absolute', left: 0, right: 0, top: '100%', background: '#141414', border: '1px solid #222', marginTop: '2px', zIndex: 50 }}>
            {suggestions.slice(0, 5).map((s, i) => (
              <button
                key={i}
                onClick={() => { setShowSugg(false); doSearch(s) }}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  background: 'none',
                  border: 'none',
                  borderBottom: i < suggestions.length - 1 ? '1px solid #1a1a1a' : 'none',
                  color: '#555',
                  cursor: 'pointer',
                  fontSize: '14px',
                  padding: '8px 12px',
                  fontFamily: 'inherit',
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#e8e8e8' }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555' }}
              >
                {s.length > 90 ? s.slice(0, 87) + '…' : s}
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }

  // ── Loading ────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p style={{ color: '#555', fontSize: '17px' }}>Thinking…</p>
      </div>
    )
  }

  // ── Landing ────────────────────────────────────────────────────────────────
  if (!result) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-6">
        <div style={{ width: '100%', maxWidth: '560px' }}>
          <div style={{ marginBottom: '40px' }}>
            <h1 style={{ fontSize: '26px', fontWeight: 600, color: '#e8e8e8', marginBottom: '8px' }}>Omni-Query</h1>
            <p style={{ fontSize: '15px', color: '#555' }}>222 documents · 4,055 QA pairs</p>
          </div>

          {searchBox(true, 'Ask anything about the document corpus…')}

          <div style={{ marginTop: '40px' }}>
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => doSearch(ex)}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: '#555',
                  fontSize: '15px',
                  padding: '9px 0',
                  borderBottom: i < EXAMPLES.length - 1 ? '1px solid #1a1a1a' : 'none',
                  lineHeight: '1.5',
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#e8e8e8' }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555' }}
              >
                {ex}
              </button>
            ))}
          </div>

          <div style={{ marginTop: '32px', borderTop: '1px solid #1a1a1a', paddingTop: '16px' }}>
            <button
              onClick={() => setShowEval(true)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#333', fontSize: '12px' }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555' }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#333' }}
            >
              View benchmark results →
            </button>
          </div>
        </div>

        {showEval && <EvalModal onClose={() => setShowEval(false)} />}
      </div>
    )
  }

  // ── Results ────────────────────────────────────────────────────────────────
  const r = result
  const goldSet = new Set(r.gold_quotes)
  const allCitations: Array<{ kind: 'text'; data: TextCitation } | { kind: 'image'; data: ImageCitation }> = [
    ...r.text_citations.map(tc => ({ kind: 'text' as const, data: tc })),
    ...r.image_citations.map(ic => ({ kind: 'image' as const, data: ic })),
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header style={{ borderBottom: '1px solid #222', padding: '12px 24px', display: 'flex', alignItems: 'center', gap: '16px' }}>
        <button
          onClick={() => { setResult(null); setInputVal('') }}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#555', fontSize: '13px', flexShrink: 0 }}
          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#e8e8e8' }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555' }}
        >
          Omni-Query
        </button>
        <div style={{ flex: 1, maxWidth: '480px' }}>
          {searchBox(false, 'Ask a new question…')}
        </div>
        <button
          onClick={() => setShowEval(true)}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#333', fontSize: '12px', flexShrink: 0, marginLeft: 'auto' }}
          onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555' }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#333' }}
        >
          Evaluate
        </button>
      </header>

      {/* Content */}
      <main style={{ maxWidth: '680px', margin: '0 auto', width: '100%', padding: '48px 24px' }}>

        {/* Question */}
        <p style={{ fontSize: '20px', color: '#666', marginBottom: '28px', lineHeight: '1.6' }}>{query}</p>

        {/* Answer */}
        <p style={{ fontSize: '20px', color: '#ffffff', lineHeight: '1.85', marginBottom: '48px' }}>{r.answer}</p>

        {/* Divider */}
        <div style={{ borderTop: '1px solid #222', marginBottom: '32px' }} />

        {/* Sources */}
        {allCitations.length > 0 && (
          <div style={{ marginBottom: '48px' }}>
            <p style={{ fontSize: '13px', color: '#333', marginBottom: '24px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Sources</p>
            {allCitations.map((c, i) => (
              <div key={i} style={{ display: 'flex', gap: '20px', paddingBottom: '24px', paddingTop: '24px', borderBottom: '1px solid #1a1a1a' }}>
                <span style={{ color: '#333', fontFamily: 'monospace', fontSize: '14px', width: '22px', flexShrink: 0 }}>{i + 1}</span>
                <div style={{ flex: 1 }}>
                  {c.kind === 'text' ? (
                    <>
                      <p style={{ fontSize: '13px', color: '#444', marginBottom: '10px', fontFamily: 'monospace' }}>
                        {c.data.quote_id} · p.{c.data.page_id}
                      </p>
                      <p style={{ fontSize: '16px', color: '#ffffff', lineHeight: '1.8' }}>"{c.data.text_preview}"</p>
                    </>
                  ) : (
                    <>
                      <p style={{ fontSize: '13px', color: '#444', marginBottom: '12px', fontFamily: 'monospace' }}>
                        {c.data.quote_id} · p.{c.data.page_id} · {c.data.type}
                      </p>
                      <img
                        src={`${API}/images/${c.data.img_filename}`}
                        alt=""
                        style={{ maxWidth: '400px', display: 'block', opacity: 0.92 }}
                        onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
                      />
                      {c.data.img_description && (
                        <p style={{ fontSize: '16px', color: '#ffffff', marginTop: '12px', lineHeight: '1.75' }}>{c.data.img_description}</p>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Divider */}
        <div style={{ borderTop: '1px solid #222', marginBottom: '20px' }} />

        {/* Footer */}
        <p style={{ fontSize: '13px', color: '#333', lineHeight: '1.8' }}>
          {r.retrieval_ms}ms retrieval · {r.rerank_ms}ms rerank · {ms(r.generation_ms)} generate · {ms(r.total_ms)} total
          {' · '}Recall@10 {pct(r.recall_at_10)} · Precision@10 {pct(r.precision_at_10)}
          {' · '}{r.doc_name} · {r.domain}
        </p>

        {/* Quote diff toggle */}
        <div style={{ marginTop: '16px' }}>
          <button
            onClick={() => setShowDiff(o => !o)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#333', fontSize: '12px' }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = '#555' }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = '#333' }}
          >
            {showDiff ? 'Hide quote diff ↑' : 'Show quote diff ↓'}
          </button>

          {showDiff && (
            <div style={{ marginTop: '16px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
              <div>
                <p style={{ fontSize: '11px', color: '#333', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Retrieved top-10</p>
                {r.retrieved_quote_ids.slice(0, 10).map(id => {
                  const hit = goldSet.has(id)
                  return (
                    <p key={id} style={{ fontSize: '12px', fontFamily: 'monospace', color: hit ? '#4a7' : '#444', padding: '3px 0' }}>
                      {hit ? '✓' : '·'} {id}
                    </p>
                  )
                })}
              </div>
              <div>
                <p style={{ fontSize: '11px', color: '#333', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Gold quotes</p>
                {r.gold_quotes.map(id => {
                  const found = r.retrieved_quote_ids.slice(0, 10).includes(id)
                  return (
                    <p key={id} style={{ fontSize: '12px', fontFamily: 'monospace', color: found ? '#4a7' : '#844', padding: '3px 0' }}>
                      {found ? '✓' : '✗'} {id}
                    </p>
                  )
                })}
              </div>
            </div>
          )}
        </div>

      </main>

      {showEval && <EvalModal onClose={() => setShowEval(false)} />}
    </div>
  )
}
