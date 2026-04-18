import { Settings2, Cpu, Shield, Bell } from 'lucide-react'
import { useState } from 'react'
import { motion } from 'framer-motion'

export default function Settings() {
  const [provider, setProvider] = useState('ollama')
  
  return (
    <div className="p-8 h-full flex flex-col">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2 flex items-center gap-3">
          <Settings2 className="text-accent-primary" />
          System Preferences
        </h1>
        <p className="text-text-secondary text-sm">Configure LLM providers, guardrails, and environment variables.</p>
      </div>

      <div className="flex-1 overflow-y-auto max-w-4xl space-y-8 pb-12">
        {/* Model Engine */}
        <section className="card-border p-6">
          <h2 className="flex items-center gap-2 font-semibold text-lg mb-4 text-accent-cyan">
            <Cpu size={18} /> Provider & Engine
          </h2>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <button 
              onClick={() => setProvider('ollama')}
              className={`p-4 rounded-xl border text-left transition-all ${provider === 'ollama' ? 'border-accent-cyan bg-accent-cyan/10 shadow-[0_0_15px_theme(colors.accent.cyan.10)]' : 'border-border bg-bg-overlay/50 hover:border-border-hover'}`}
            >
              <div className="font-bold text-text-primary mb-1">Local / Ollama</div>
              <div className="text-xs text-text-secondary">Fully private inference running locally.</div>
            </button>
            <button 
              onClick={() => setProvider('claude')}
              className={`p-4 rounded-[xl] border text-left transition-all ${provider === 'claude' ? 'border-accent-primary bg-accent-primary/10 shadow-[0_0_15px_theme(colors.accent.primary.10)]' : 'border-border bg-bg-overlay/50 hover:border-border-hover'}`}
            >
              <div className="font-bold text-text-primary mb-1">Claude API</div>
              <div className="text-xs text-text-secondary">Anthropic Cloud API for maximum reasoning.</div>
            </button>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Model Name</label>
              <select className="input-field max-w-sm">
                <option>llama3.2:3b</option>
                <option>mistral:7b</option>
                <option>llama3:8b</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Endpoint URL</label>
              <input type="text" className="input-field max-w-sm font-mono text-sm" defaultValue="http://localhost:11434" />
            </div>
          </div>
        </section>

        {/* Guardrails */}
        <section className="card-border p-6">
          <h2 className="flex items-center gap-2 font-semibold text-lg mb-4 text-accent-orange">
            <Shield size={18} /> Guardrails & Safety
          </h2>
          <div className="space-y-6">
            <ToggleOption 
              title="Strict File Isolation" 
              desc="Prevent agents from accessing files outside the immediate working directory." 
              enabled={true} 
            />
            <ToggleOption 
              title="Manual Command Approval" 
              desc="Require user confirmation for all shell commands with Risk Score > 4." 
              enabled={true} 
            />
            <ToggleOption 
              title="Audit Mode" 
              desc="Log full IO streams (stdin/stdout) for every shell execution." 
              enabled={false} 
            />
          </div>
        </section>

      </div>
    </div>
  )
}

function ToggleOption({ title, desc, enabled }) {
  const [active, setActive] = useState(enabled)
  
  return (
    <div className="flex items-start justify-between gap-6">
      <div>
        <h4 className="font-medium text-sm text-text-primary mb-1">{title}</h4>
        <p className="text-xs text-text-secondary">{desc}</p>
      </div>
      <button 
        onClick={() => setActive(!active)}
        className={`relative w-10 h-5 rounded-full transition-colors flex-shrink-0 ${active ? 'bg-accent-primary' : 'bg-bg-elevated border border-border'}`}
      >
        <motion.div 
          animate={{ x: active ? 20 : 2 }}
          className="absolute top-[2px] w-4 h-4 bg-white rounded-full shadow"
        />
      </button>
    </div>
  )
}
