import { Play, CheckCircle2, Circle, XCircle, Search, Clock } from 'lucide-react'
import { motion } from 'framer-motion'

export default function TaskManager() {
  const tasks = [
    { id: 'T-819', goal: 'Build Agentic UI Shell', agent: 'planner', status: 'running', time: '14s', progress: 45 },
    { id: 'T-818', goal: 'Optimize CodeAgent parameters', agent: 'code', status: 'completed', time: '2m 13s', progress: 100 },
    { id: 'T-817', goal: 'Search web for react 18 features', agent: 'web', status: 'completed', time: '41s', progress: 100 },
    { id: 'T-816', goal: 'Run cargo check on rust backend', agent: 'system', status: 'failed', time: '12s', progress: 0 },
  ]

  return (
    <div className="p-8 h-full flex flex-col">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-2">Process Manager</h1>
          <p className="text-text-secondary text-sm">Active and historical tasks orchestrated by the OS.</p>
        </div>
        
        <div className="relative w-64">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input type="text" placeholder="Search tasks..." className="input-field pl-9 text-sm" />
        </div>
      </div>

      <div className="card-border flex-1 overflow-hidden flex flex-col">
        <div className="grid grid-cols-12 gap-4 px-6 py-3 border-b border-border bg-bg-overlay/50 text-xs font-semibold text-text-muted uppercase tracking-wider">
          <div className="col-span-2">Task ID</div>
          <div className="col-span-5">Directive / Goal</div>
          <div className="col-span-2">Agent</div>
          <div className="col-span-2">Runtime</div>
          <div className="col-span-1 text-right">Action</div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {tasks.map((task, i) => (
            <TaskRow key={task.id} task={task} index={i} />
          ))}
        </div>
      </div>
    </div>
  )
}

function TaskRow({ task, index }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="grid grid-cols-12 gap-4 px-6 py-4 border-b border-border hover:bg-bg-overlay/50 transition-colors items-center group"
    >
      <div className="col-span-2 font-mono text-sm text-text-secondary">
        {task.id}
      </div>
      
      <div className="col-span-5 truncate text-sm font-medium">
        <div className="flex items-center gap-2">
          {task.status === 'running' && <Play size={14} className="text-accent-cyan animate-pulse" />}
          {task.status === 'completed' && <CheckCircle2 size={14} className="text-accent-green" />}
          {task.status === 'failed' && <XCircle size={14} className="text-accent-red" />}
          {task.status === 'pending' && <Circle size={14} className="text-text-muted" />}
          <span className="truncate">{task.goal}</span>
        </div>
        {task.status === 'running' && (
           <div className="mt-2 h-1 w-full max-w-[200px] bg-bg-base rounded-full overflow-hidden ml-5">
             <div className="h-full bg-accent-cyan shadow-[0_0_5px_theme(colors.accent.cyan)]" style={{ width: `${task.progress}%` }} />
           </div>
        )}
      </div>
      
      <div className="col-span-2">
        <span className="px-2 py-1 rounded bg-bg-elevated border border-border text-[10px] uppercase tracking-wider font-semibold">
          {task.agent}
        </span>
      </div>
      
      <div className="col-span-2 flex items-center gap-1.5 text-xs text-text-muted font-mono">
        <Clock size={12} />
        {task.time}
      </div>
      
      <div className="col-span-1 border-hidden lg:flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
        {task.status === 'running' ? (
          <button className="text-xs text-accent-red font-medium hover:underline">Cancel</button>
        ) : (
          <button className="text-xs text-text-muted hover:text-text-primary hover:underline">Logs</button>
        )}
      </div>
    </motion.div>
  )
}
