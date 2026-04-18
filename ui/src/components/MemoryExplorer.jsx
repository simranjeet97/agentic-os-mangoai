import { Search, Brain, Clock, Network } from 'lucide-react'
import { motion } from 'framer-motion'

export default function MemoryExplorer() {
  const episodes = [
    { title: 'Implemented File System Guardrails', type: 'episodic', time: '2 hours ago', count: 14 },
    { title: 'Resolved UI routing bug in React', type: 'semantic', time: 'Yesterday', count: 3 },
    { title: 'Learned codebase architecture patterns', type: 'knowledge_graph', time: '2 days ago', count: 42 },
  ]

  return (
    <div className="p-8 h-full flex flex-col">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-2 flex items-center gap-3">
            <Brain className="text-accent-glow" />
            Memory Explorer
          </h1>
          <p className="text-text-secondary text-sm">Long-term episodic and semantic memory storage.</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <StatCard title="Total Memories" value="1,402" icon={<DatabaseIcon />} color="text-accent-primary" />
        <StatCard title="Knowledge Graph Nodes" value="488" icon={<Network />} color="text-accent-cyan" />
        <StatCard title="Semantic Vectors" value="3,912" icon={<VectorIcon />} color="text-accent-orange" />
      </div>

      <div className="card-border flex-1 p-6 flex flex-col">
        <div className="flex gap-4 mb-6">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input type="text" placeholder="Semantically search past episodes and conversations..." className="input-field pl-10" />
          </div>
          <button className="btn-primary">Search Deep Memory</button>
        </div>

        <h3 className="font-semibold text-lg mb-4 mt-2">Recent Episodes</h3>
        <div className="space-y-4">
          {episodes.map((ep, i) => (
            <motion.div 
              initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
              key={i} 
              className="p-4 rounded-lg bg-bg-elevated border border-border hover:border-accent-primary/50 cursor-pointer transition-colors flex justify-between items-center group"
            >
              <div className="flex items-center gap-4">
                <div className="p-2 rounded bg-bg-overlay text-accent-primary">
                  {ep.type === 'episodic' ? <Clock size={16} /> : ep.type === 'semantic' ? <Brain size={16} /> : <Network size={16} />}
                </div>
                <div>
                  <h4 className="font-medium text-text-primary group-hover:text-accent-primary transition-colors">{ep.title}</h4>
                  <div className="flex gap-3 text-xs text-text-muted mt-1 font-mono uppercase tracking-wider">
                    <span>TYPE: {ep.type}</span>
                    <span>•</span>
                    <span>{ep.count} ARTIFACTS</span>
                  </div>
                </div>
              </div>
              <div className="text-xs text-text-muted font-medium">{ep.time}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({ title, value, icon, color }) {
  return (
    <div className="card-border p-6 relative overflow-hidden group">
      <div className={`absolute -right-4 -top-4 opacity-10 group-hover:opacity-20 transition-opacity ${color} scale-[2]`}>
        {icon}
      </div>
      <div className={`mb-4 ${color}`}>{icon}</div>
      <div className="text-3xl font-bold font-mono tracking-tight text-text-primary mb-1">{value}</div>
      <div className="text-xs text-text-muted font-bold uppercase tracking-wider">{title}</div>
    </div>
  )
}

function VectorIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 2v7.31"/><path d="M14 9.3V1.99"/><path d="m5 13-3.23 6.13"/><path d="m11.3 17.56-4.14 7.63"/><path d="m19 13 3.23 6.13"/><path d="m12.7 17.56 4.14 7.63"/><path d="M8 3 4.2 1.3"/><path d="M16 3l3.8-1.7"/><circle cx="12" cy="12" r="3"/></svg> }
function DatabaseIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg> }
