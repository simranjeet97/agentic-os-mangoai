import { useState, useRef, useEffect } from 'react'
import { Send, FileUp, Mic, Terminal, Activity, Cpu, HardDrive } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export default function MainShell({ messages, setMessages, isRunning, handleSendMessage }) {
  const [input, setInput] = useState('')
  const msgsEndRef = useRef(null)

  useEffect(() => {
    msgsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const onSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || isRunning) return
    handleSendMessage(input)
    setInput('')
  }

  const AGENT_COLORS = {
    planner: 'text-accent-primary',
    code: 'text-accent-cyan',
    file: 'text-accent-orange',
    system: 'text-accent-green',
    guardian: 'text-accent-red',
    web: 'text-accent-yellow',
  }

  return (
    <div className="flex h-full w-full">
      {/* Left: Chat Pane */}
      <div className="flex-1 flex flex-col border-r border-border">
        {/* Topbar */}
        <div className="h-14 flex items-center px-6 border-b border-border glass-panel">
          <Terminal size={18} className="text-accent-primary mr-2" />
          <h2 className="font-semibold text-sm">Terminal Shell</h2>
        </div>
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-text-muted">
              <BrainIcon />
              <p className="mt-4 text-sm font-medium">Awaiting primary directive...</p>
            </div>
          )}
          
          <AnimatePresence>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-4 w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role !== 'user' && (
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center bg-bg-overlay border border-border flex-shrink-0 ${AGENT_COLORS[msg.agent] || 'text-text-primary'}`}>
                    <AgentIcon agent={msg.agent} />
                  </div>
                )}
                
                <div className={`max-w-[80%] rounded-lg p-4 text-sm whitespace-pre-wrap ${msg.role === 'user' ? 'bg-accent-primary/10 border border-accent-primary/20 text-text-primary' : 'bg-bg-elevated border border-border text-text-secondary'}`}>
                  {msg.role !== 'user' && (
                    <div className="text-[10px] font-bold uppercase tracking-wider mb-2 text-text-muted">
                      {msg.agent || 'System'}
                    </div>
                  )}
                  {msg.content}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={msgsEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-border bg-bg-surface">
          <form onSubmit={onSubmit} className="flex gap-2 items-center bg-bg-overlay border border-border hover:border-border-hover focus-within:border-accent-primary focus-within:shadow-glow rounded-xl p-2 transition-all">
            <button type="button" className="p-2 text-text-muted hover:text-accent-cyan transition-colors">
              <FileUp size={18} />
            </button>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Initialize task sequence..."
              className="flex-1 bg-transparent border-none outline-none text-sm px-2 text-text-primary placeholder:text-text-muted"
            />
            <button type="button" className="p-2 text-text-muted hover:text-accent-green transition-colors">
              <Mic size={18} />
            </button>
            <button type="submit" disabled={!input.trim() || isRunning} className="p-2 bg-accent-primary text-white rounded-lg disabled:opacity-50 hover:bg-accent-glow hover:scale-105 active:scale-95 transition-all">
              <Send size={16} />
            </button>
          </form>
        </div>
      </div>

      {/* Right: System Monitor */}
      <div className="w-72 bg-bg-surface flex flex-col">
        <div className="h-14 flex items-center px-6 border-b border-border glass-panel">
          <Activity size={18} className="text-accent-cyan mr-2" />
          <h2 className="font-semibold text-sm">Vitals</h2>
        </div>
        
        <div className="p-6 space-y-8">
          <Gauge label="CPU Utilization" icon={<Cpu size={14}/>} value={34} color="text-accent-cyan" bg="bg-accent-cyan" />
          <Gauge label="Memory (RAM)" icon={<Activity size={14}/>} value={68} color="text-accent-yellow" bg="bg-accent-yellow" />
          <Gauge label="Disk I/O" icon={<HardDrive size={14}/>} value={12} color="text-accent-green" bg="bg-accent-green" />
          
          <div className="pt-4 border-t border-border">
            <div className="text-[10px] uppercase tracking-wider text-text-muted mb-4 font-bold">Network Traffic</div>
            <div className="h-24 flex items-end gap-1.5 opacity-70">
              {[4, 7, 3, 8, 5, 9, 4, 6, 2, 7, 5, 8].map((h, i) => (
                <div key={i} className="w-full bg-accent-primary rounded-t-sm" style={{ height: `${h * 10}%` }} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Gauge({ label, icon, value, color, bg }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <div className={`flex items-center gap-2 text-xs font-semibold uppercase tracking-wider ${color}`}>
          {icon}
          {label}
        </div>
        <div className="text-xs font-mono">{value}%</div>
      </div>
      <div className="h-1.5 w-full bg-bg-elevated rounded-full overflow-hidden border border-border">
        <motion.div 
          className={`h-full ${bg} shadow-[0_0_8px_currentColor]`}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, type: "spring" }}
        />
      </div>
    </div>
  )
}

function BrainIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-accent-primary opacity-30">
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
      <path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/>
      <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/>
      <path d="M3.477 10.896a4 4 0 0 1 .585-.396"/>
      <path d="M19.938 10.5a4 4 0 0 1 .585.396"/>
      <path d="M6 18a4 4 0 0 1-1.967-.516"/>
      <path d="M19.967 17.484A4 4 0 0 1 18 18"/>
    </svg>
  )
}

function AgentIcon({ agent }) {
  // Return early letter
  return <span className="text-xs uppercase font-bold">{agent?.[0] || 'A'}</span>
}
