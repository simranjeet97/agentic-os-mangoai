import { Shield, Search, Filter } from 'lucide-react'
import { useState } from 'react'

export default function AuditViewer() {
  const [search, setSearch] = useState('')
  
  const logs = [
    { id: 'ACT-901', agent: 'code', action: 'file_write', target: '/src/main.rs', risk: 2, status: 'allowed', time: '10:42 PM' },
    { id: 'ACT-900', agent: 'system', action: 'shell_exec', target: 'cargo build --release', risk: 4, status: 'allowed', time: '10:41 PM' },
    { id: 'ACT-899', agent: 'web', action: 'http_request', target: 'api.github.com', risk: 1, status: 'allowed', time: '10:35 PM' },
    { id: 'ACT-898', agent: 'system', action: 'shell_exec', target: 'rm -rf /', risk: 10, status: 'blocked', time: '10:12 PM' },
    { id: 'ACT-897', agent: 'file', action: 'file_read', target: '/etc/passwd', risk: 8, status: 'blocked', time: '10:10 PM' },
  ]

  return (
    <div className="p-8 h-full flex flex-col">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-2 flex items-center gap-3">
            <Shield className="text-accent-primary" />
            Security Audit Log
          </h1>
          <p className="text-text-secondary text-sm">Immutable ledger of all agent actions and guardrail decisions.</p>
        </div>
        
        <div className="flex gap-3">
          <button className="btn-outline flex items-center gap-2">
            <Filter size={14} /> Filter
          </button>
          <button className="btn-primary">
            Export CSV
          </button>
        </div>
      </div>

      <div className="mb-6 relative w-1/3">
        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
        <input 
          type="text" 
          placeholder="Search by ID, Agent, or Target..." 
          className="input-field pl-10"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      <div className="card-border flex-1 overflow-hidden flex flex-col">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm whitespace-nowrap">
            <thead className="bg-bg-overlay/50 text-xs uppercase text-text-muted border-b border-border">
              <tr>
                <th className="px-6 py-4 font-semibold">Action ID</th>
                <th className="px-6 py-4 font-semibold">Timestamp</th>
                <th className="px-6 py-4 font-semibold">Agent</th>
                <th className="px-6 py-4 font-semibold">Operation</th>
                <th className="px-6 py-4 font-semibold">Target / Context</th>
                <th className="px-6 py-4 font-semibold text-center">Risk Score</th>
                <th className="px-6 py-4 font-semibold text-right">Decision</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {logs.map((log) => (
                <tr key={log.id} className="hover:bg-bg-overlay/30 transition-colors">
                  <td className="px-6 py-4 font-mono text-text-secondary">{log.id}</td>
                  <td className="px-6 py-4 text-text-muted">{log.time}</td>
                  <td className="px-6 py-4">
                    <span className="px-2 py-1 rounded bg-bg-elevated border border-border text-[10px] uppercase font-bold tracking-wider">
                      {log.agent}
                    </span>
                  </td>
                  <td className="px-6 py-4 font-mono text-accent-cyan">{log.action}</td>
                  <td className="px-6 py-4 font-mono text-xs">{log.target}</td>
                  <td className="px-6 py-4 text-center">
                    <span className={`px-2 py-1 rounded-full text-xs font-bold ${log.risk > 7 ? 'bg-accent-red/20 text-accent-red' : log.risk > 4 ? 'bg-accent-orange/20 text-accent-orange' : 'bg-bg-elevated text-text-secondary'}`}>
                      {log.risk}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    {log.status === 'allowed' ? (
                      <span className="text-accent-green font-semibold text-xs tracking-wide uppercase">Allowed</span>
                    ) : (
                      <span className="text-accent-red font-semibold text-xs tracking-wide uppercase px-2 py-1 bg-accent-red/10 rounded border border-accent-red/20">Blocked</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
