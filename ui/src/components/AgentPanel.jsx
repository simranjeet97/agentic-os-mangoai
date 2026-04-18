import { motion } from 'framer-motion'
import { Brain, Zap, Folder, Globe, Terminal, Code, ShieldAlert } from 'lucide-react'

const AGENTS = [
  { id: 'planner', name: 'Strategic Planner', icon: Brain, color: 'text-accent-primary', bg: 'bg-accent-primary', status: 'idle' },
  { id: 'executor', name: 'Task Executor', icon: Zap, color: 'text-accent-cyan', bg: 'bg-accent-cyan', status: 'running' },
  { id: 'file', name: 'File System', icon: Folder, color: 'text-accent-orange', bg: 'bg-accent-orange', status: 'idle' },
  { id: 'web', name: 'Web Research', icon: Globe, color: 'text-accent-yellow', bg: 'bg-accent-yellow', status: 'error' },
  { id: 'system', name: 'Native System OS', icon: Terminal, color: 'text-accent-green', bg: 'bg-accent-green', status: 'idle' },
  { id: 'code', name: 'Software Engineer', icon: Code, color: 'text-accent-primary', bg: 'bg-accent-primary', status: 'idle' },
]

export default function AgentPanel() {
  return (
    <div className="p-8 h-full flex flex-col">
      <h1 className="text-2xl font-bold mb-2">Agent Swarm</h1>
      <p className="text-text-secondary mb-8 text-sm">Monitor and control the active multi-agent system.</p>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {AGENTS.map((agent, i) => (
          <AgentCard key={agent.id} agent={agent} delay={i * 0.1} />
        ))}
      </div>
    </div>
  )
}

function AgentCard({ agent, delay }) {
  const Icon = agent.icon
  
  const statusColors = {
    idle: 'bg-bg-elevated border-border',
    running: `bg-bg-elevated border-${agent.color.split('-')[1]}-${agent.color.split('-')[2]} shadow-glow`,
    error: 'bg-bg-elevated border-accent-red shadow-[0_0_15px_theme(colors.accent.red)]',
  }

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.3 }}
      className={`relative p-6 rounded-xl border transition-all ${statusColors[agent.status]} group cursor-pointer overflow-hidden`}
    >
      {/* Background glow if running */}
      {agent.status === 'running' && (
        <div className={`absolute -inset-20 ${agent.bg} blur-[80px] opacity-10`} />
      )}

      <div className="flex justify-between items-start mb-6">
        <div className={`p-3 rounded-lg bg-bg-overlay border border-border ${agent.color}`}>
          <Icon size={24} />
        </div>
        <Badge status={agent.status} />
      </div>

      <h3 className="font-semibold text-lg mb-1">{agent.name}</h3>
      <p className="text-text-muted text-xs font-mono uppercase">ID: {agent.id}</p>

      {agent.status === 'running' && (
        <div className="mt-6 pt-4 border-t border-border">
          <div className="text-xs text-text-muted mb-2">CURRENT TASK</div>
          <p className="text-sm">Analyzing repository structure for `CodeAgent` parameters...</p>
        </div>
      )}
      {agent.status === 'error' && (
        <div className="mt-6 pt-4 border-t border-border flex gap-2 items-start text-accent-red">
          <ShieldAlert size={14} className="mt-0.5 flex-shrink-0" />
          <p className="text-xs">Connection timeout parsing duckduckgo results.</p>
        </div>
      )}
    </motion.div>
  )
}

function Badge({ status }) {
  if (status === 'running') {
    return (
      <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-accent-cyan/10 border border-accent-cyan/20 text-accent-cyan text-[10px] font-bold tracking-wider uppercase shadow-[0_0_8px_theme(colors.accent.cyan)]">
        <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
        Running
      </span>
    )
  }
  if (status === 'error') {
    return (
      <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-accent-red/10 border border-accent-red/20 text-accent-red text-[10px] font-bold tracking-wider uppercase">
        <span className="w-1.5 h-1.5 rounded-full bg-accent-red" />
        Error
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-bg-overlay border border-border text-text-muted text-[10px] font-bold tracking-wider uppercase">
      Offline
    </span>
  )
}
