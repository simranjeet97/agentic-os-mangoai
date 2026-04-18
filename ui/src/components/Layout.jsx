import { Brain, FileText, Settings, Activity, Shield, Users, Clock } from 'lucide-react'
import { motion } from 'framer-motion'

const NAV_ITEMS = [
  { id: 'shell',   label: 'Main Shell',    icon: Brain },
  { id: 'agents',  label: 'Agent Grid',    icon: Users },
  { id: 'tasks',   label: 'Tasks',         icon: Activity },
  { id: 'audit',   label: 'Audit Log',     icon: Shield },
  { id: 'memory',  label: 'Memory',        icon: Clock },
  { id: 'settings',label: 'Settings',      icon: Settings },
]

export default function Layout({ activeView, setActiveView, isRunning, children }) {
  return (
    <div className="flex h-screen w-screen bg-bg-base overflow-hidden text-text-primary">
      {/* Sidebar */}
      <nav className="w-64 bg-bg-surface border-r border-border flex flex-col z-20">
        <div className="h-14 flex items-center px-6 border-b border-border glass-panel">
          <div className="logo-dot mr-3" />
          <span className="font-bold text-lg tracking-tight">Agentic AI OS</span>
        </div>
        
        <div className="flex-1 overflow-y-auto py-6 px-4 space-y-1">
          <div className="text-[11px] font-semibold uppercase tracking-widest text-text-muted mb-3 px-3">
            Core Systems
          </div>
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
            const isActive = activeView === id;
            return (
              <button
                key={id}
                onClick={() => setActiveView(id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all relative ${isActive ? 'text-accent-glow' : 'text-text-secondary hover:text-text-primary hover:bg-accent-primary/10'}`}
              >
                {isActive && (
                  <motion.div layoutId="sidebar-active" className="absolute inset-0 bg-accent-primary/10 rounded-md -z-10" />
                )}
                <Icon size={16} />
                {label}
              </button>
            )
          })}
        </div>
        
        {isRunning && (
          <div className="p-4 m-4 rounded-lg bg-bg-elevated border border-border">
            <div className="flex items-center gap-2 text-xs font-semibold text-accent-glow mb-2">
              <span className="w-2 h-2 rounded-full border-2 border-accent-primary border-t-transparent animate-spin" />
              SYSTEM ACTIVE
            </div>
            <div className="text-xs text-text-muted">Agent execution in progress...</div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative z-10 min-w-0">
        {children}
      </main>
    </div>
  )
}
