import { useState, useEffect, useRef } from 'react'
import { Toaster, toast } from 'react-hot-toast'
import {
  Brain, Send, Terminal, Globe, FileText,
  Code, Settings, Activity, Zap, ChevronRight
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_BASE  = import.meta.env.VITE_WS_URL  || 'ws://localhost:8000'

const NAV_ITEMS = [
  { id: 'agent',   label: 'Agent Console', icon: Brain },
  { id: 'tasks',   label: 'Task History',  icon: Activity },
  { id: 'memory',  label: 'Memory',        icon: FileText },
  { id: 'tools',   label: 'Tools',         icon: Zap },
  { id: 'settings',label: 'Settings',      icon: Settings },
]

const AGENT_ICONS = {
  planner:  '🧠',
  executor: '⚡',
  file:     '📁',
  web:      '🌐',
  system:   '💻',
  code:     '🧩',
  guardian: '🛡️',
  error:    '❌',
}

export default function App() {
  const [activeView, setActiveView] = useState('agent')
  const [sessionId]  = useState(() => crypto.randomUUID())
  const [messages,   setMessages]   = useState([])
  const [inputText,  setInputText]  = useState('')
  const [isRunning,  setIsRunning]  = useState(false)
  const [streamEvents, setStreamEvents] = useState([])
  const [currentStatus, setCurrentStatus] = useState(null)
  const [taskHistory, setTaskHistory] = useState([])

  const wsRef        = useRef(null)
  const messagesEndRef = useRef(null)
  const inputRef     = useRef(null)

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamEvents])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const connectWS = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return wsRef.current
    const ws = new WebSocket(`${WS_BASE}/ws/stream/${sessionId}`)
    wsRef.current = ws
    return ws
  }

  const handleSubmit = async (e) => {
    e?.preventDefault()
    if (!inputText.trim() || isRunning) return

    const goal = inputText.trim()
    setInputText('')
    setIsRunning(true)
    setStreamEvents([])
    setCurrentStatus('planning')

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: goal, timestamp: new Date() }])

    try {
      const ws = connectWS()

      ws.onopen = () => {
        ws.send(JSON.stringify({ goal, user_id: 'user', session_id: sessionId }))
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        if (data.event === 'node_update') {
          const node = data.node
          const update = data.update || {}
          setCurrentStatus(update.status || node)

          setStreamEvents(prev => [...prev, {
            id: crypto.randomUUID(),
            node,
            status: update.status,
            timestamp: new Date(),
          }])

          // Show plan steps if planner ran
          if (node === 'planner' && update.plan?.length) {
            setMessages(prev => [...prev, {
              role: 'agent',
              content: `📋 **Plan created** with ${update.plan.length} steps:\n${update.plan.map((s, i) => `${i + 1}. [${s.agent}] ${s.description}`).join('\n')}`,
              timestamp: new Date(),
              node: 'planner',
            }])
          }

          // Show executor results
          if (node === 'executor' && update.tool_results?.length) {
            const latest = update.tool_results[update.tool_results.length - 1]
            if (latest?.result?.output) {
              setMessages(prev => [...prev, {
                role: 'agent',
                content: `✅ **${latest.agent}**: ${String(latest.result.output).slice(0, 300)}`,
                timestamp: new Date(),
                node: latest.agent,
              }])
            }
          }

          if (node === 'error') {
            setMessages(prev => [...prev, {
              role: 'agent',
              content: `❌ **Error**: ${update.error || 'An error occurred'}`,
              timestamp: new Date(),
              node: 'error',
            }])
          }
        }

        if (data.event === 'complete') {
          setCurrentStatus('completed')
          setIsRunning(false)
          toast.success('Task completed!', { icon: '✅' })
          setTaskHistory(prev => [{ goal, status: 'completed', timestamp: new Date() }, ...prev])
        }

        if (data.event === 'error') {
          setCurrentStatus('failed')
          setIsRunning(false)
          toast.error(data.message || 'Task failed')
        }
      }

      ws.onerror = () => {
        // Fallback to REST API
        submitViaREST(goal)
      }

      ws.onclose = () => {
        setIsRunning(false)
      }

    } catch (err) {
      submitViaREST(goal)
    }
  }

  const submitViaREST = async (goal) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/agent/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal, user_id: 'user', session_id: sessionId }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'agent',
        content: `Task submitted (ID: \`${data.task_id}\`). Use the Tasks view to monitor progress.`,
        timestamp: new Date(),
        node: 'executor',
      }])
      toast.success('Task queued')
    } catch (err) {
      toast.error('Failed to connect to API')
    } finally {
      setIsRunning(false)
      setCurrentStatus(null)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="app-shell">
      <Toaster position="top-right" toastOptions={{ style: { background: 'var(--bg-elevated)', color: 'var(--text-primary)', border: '1px solid var(--border)' } }} />

      {/* ── Topbar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="logo-dot" />
          Agentic AI OS
        </div>
        <div className="flex items-center gap-3 text-sm text-muted">
          {isRunning && (
            <span className="badge planning">
              <span className="step-spinner" style={{ width: 10, height: 10 }} />
              {currentStatus || 'running'}
            </span>
          )}
          <span>Session: <code className="font-mono text-xs">{sessionId.slice(0, 8)}</code></span>
        </div>
      </header>

      {/* ── Sidebar ── */}
      <nav className="sidebar">
        <div className="sidebar-section">
          <div className="sidebar-label">Navigation</div>
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`sidebar-item ${activeView === id ? 'active' : ''}`}
              onClick={() => setActiveView(id)}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Agents</div>
          {['planner', 'executor', 'file', 'web', 'system', 'code'].map(agent => (
            <div key={agent} className="sidebar-item" style={{ cursor: 'default' }}>
              <span>{AGENT_ICONS[agent]}</span>
              <span style={{ textTransform: 'capitalize' }}>{agent}</span>
            </div>
          ))}
        </div>
      </nav>

      {/* ── Main Content ── */}
      <main className="main-content">

        {/* Agent Console */}
        {activeView === 'agent' && (
          <div className="chat-container">
            <div className="message-list">
              {messages.length === 0 && (
                <div style={{ textAlign: 'center', paddingTop: '20vh', color: 'var(--text-muted)' }}>
                  <Brain size={48} style={{ margin: '0 auto 16px', color: 'var(--accent-primary)' }} />
                  <h2 style={{ fontSize: 20, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
                    Agentic AI OS
                  </h2>
                  <p>Describe any goal. The agent will plan, reason, and execute autonomously.</p>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '👤' : (AGENT_ICONS[msg.node] || '🤖')}
                  </div>
                  <div className="message-body">
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                      {msg.role === 'user' ? 'You' : (msg.node ? `Agent · ${msg.node}` : 'Agent')}
                      {' · '}{msg.timestamp.toLocaleTimeString()}
                    </div>
                    <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--font-sans)', fontSize: 14, background: 'none', border: 'none', padding: 0, color: 'inherit' }}>
                      {msg.content}
                    </pre>
                  </div>
                </div>
              ))}

              {/* Live stream events */}
              {isRunning && streamEvents.length > 0 && (
                <div className="card" style={{ padding: '12px 16px' }}>
                  <div className="text-xs text-muted mb-4">Live execution stream</div>
                  <div className="step-list">
                    {streamEvents.map(ev => (
                      <div key={ev.id} className={`step-item ${ev.status === 'completed' ? 'completed' : 'active'}`}>
                        {ev.status !== 'completed' && <div className="step-spinner" />}
                        {ev.status === 'completed' && <span>✓</span>}
                        <ChevronRight size={12} />
                        <span>{AGENT_ICONS[ev.node] || '⚙️'}</span>
                        <span style={{ textTransform: 'capitalize' }}>{ev.node}</span>
                        <span className="text-muted text-xs" style={{ marginLeft: 'auto' }}>
                          {ev.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="chat-input-bar">
              <textarea
                ref={inputRef}
                className="chat-input"
                placeholder="Describe your goal… (Enter to send, Shift+Enter for newline)"
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={isRunning}
              />
              <button className="btn-send" type="submit" disabled={isRunning || !inputText.trim()} id="send-btn">
                <Send size={16} />
              </button>
            </form>
          </div>
        )}

        {/* Task History */}
        {activeView === 'tasks' && (
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Task History</h1>
            {taskHistory.length === 0
              ? <p className="text-muted">No tasks completed yet.</p>
              : taskHistory.map((t, i) => (
                  <div key={i} className="card mt-4">
                    <div className="flex items-center gap-3">
                      <span className={`badge ${t.status}`}>{t.status}</span>
                      <span className="text-xs text-muted">{t.timestamp.toLocaleString()}</span>
                    </div>
                    <p style={{ marginTop: 8, fontSize: 14 }}>{t.goal}</p>
                  </div>
                ))
            }
          </div>
        )}

        {/* Memory */}
        {activeView === 'memory' && (
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Memory Store</h1>
            <div className="card">
              <p className="text-muted">Query your semantic memory across all sessions.</p>
              <div className="chat-input-bar mt-4" style={{ maxWidth: 600 }}>
                <input className="chat-input" placeholder="Search memory…" style={{ height: 36 }} />
                <button className="btn-send"><Send size={16} /></button>
              </div>
            </div>
          </div>
        )}

        {/* Tools */}
        {activeView === 'tools' && (
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Available Tools</h1>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 16 }}>
              {[
                { name: 'Web Search', desc: 'DuckDuckGo semantic search', icon: Globe },
                { name: 'File I/O', desc: 'Sandboxed file read/write', icon: FileText },
                { name: 'Code Exec', desc: 'Docker-isolated execution', icon: Code },
                { name: 'Shell', desc: 'Sandboxed OS commands', icon: Terminal },
              ].map(({ name, desc, icon: Icon }) => (
                <div key={name} className="card">
                  <div className="flex items-center gap-3" style={{ marginBottom: 8 }}>
                    <Icon size={18} style={{ color: 'var(--accent-primary)' }} />
                    <strong>{name}</strong>
                  </div>
                  <p className="text-sm text-muted">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Settings */}
        {activeView === 'settings' && (
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Settings</h1>
            <div className="card" style={{ maxWidth: 500 }}>
              <div style={{ marginBottom: 16 }}>
                <label className="text-sm text-muted" style={{ display: 'block', marginBottom: 6 }}>API URL</label>
                <input
                  defaultValue={API_BASE}
                  style={{ width: '100%', background: 'var(--bg-overlay)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 13 }}
                />
              </div>
              <div>
                <label className="text-sm text-muted" style={{ display: 'block', marginBottom: 6 }}>Default Model</label>
                <select style={{ width: '100%', background: 'var(--bg-overlay)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', color: 'var(--text-primary)' }}>
                  <option>llama3.2:3b</option>
                  <option>mistral:7b</option>
                  <option>codellama:7b</option>
                </select>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  )
}
