import { useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { AlertOctagon, Terminal, ShieldAlert, Check } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export default function ApprovalDialog({ isOpen, onOpenChange, actionDetails, onApprove, onBlock }) {
  const [confirmText, setConfirmText] = useState('')
  
  if (!actionDetails) return null
  
  const isHighRisk = actionDetails.riskScore > 8
  const canApprove = !isHighRisk || confirmText === 'CONFIRM'

  return (
    <Dialog.Root open={isOpen} onOpenChange={onOpenChange}>
      <AnimatePresence>
        {isOpen && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild>
              <motion.div 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="fixed inset-0 z-50 bg-bg-base/80 backdrop-blur-sm"
              />
            </Dialog.Overlay>
            <Dialog.Content asChild>
              <motion.div 
                initial={{ opacity: 0, scale: 0.95, y: '-50%', x: '-50%' }} 
                animate={{ opacity: 1, scale: 1, y: '-50%', x: '-50%' }} 
                exit={{ opacity: 0, scale: 0.95 }}
                className="fixed top-1/2 left-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 p-6 glass-panel rounded-xl border border-border shadow-2xl"
              >
                <div className="flex items-center gap-3 mb-4 text-accent-red">
                  <div className="p-2 rounded-full bg-accent-red/10 animate-pulse">
                    <AlertOctagon size={24} />
                  </div>
                  <Dialog.Title className="text-lg font-bold text-text-primary">Action Approval Required</Dialog.Title>
                </div>
                
                <Dialog.Description className="text-sm text-text-secondary mb-6">
                  Agent <strong className="text-text-primary">{actionDetails.agent}</strong> is requesting permission to execute a system command.
                </Dialog.Description>

                <div className="bg-bg-elevated border border-border rounded-lg p-4 mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">Target Command</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${isHighRisk ? 'bg-accent-red text-white shadow-glow' : 'bg-accent-yellow text-bg-base'}`}>
                      RISK SCORE: {actionDetails.riskScore}/10
                    </span>
                  </div>
                  <div className="font-mono text-sm bg-bg-base p-3 rounded border border-border/50 text-text-primary overflow-x-auto whitespace-pre">
                    <div className="flex gap-2">
                      <Terminal size={14} className="text-accent-primary flex-shrink-0 mt-0.5" />
                      <span>{actionDetails.command}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4 mb-6 text-sm">
                  <div>
                    <strong className="text-text-primary block mb-1">Intended Impact:</strong>
                    <div className="text-text-secondary pl-3 border-l-2 border-accent-cyan/30">
                      {actionDetails.impact}
                    </div>
                  </div>
                  {actionDetails.warnings && (
                    <div>
                      <strong className="text-accent-orange flex items-center gap-1 mb-1">
                        <ShieldAlert size={14}/> Potential Risks:
                      </strong>
                      <div className="text-text-secondary pl-4 border-l-2 border-accent-orange/30 list-disc">
                        {actionDetails.warnings.map((w, i) => <li key={i}>{w}</li>)}
                      </div>
                    </div>
                  )}
                </div>

                {isHighRisk && (
                  <div className="mb-6 p-4 rounded-lg bg-accent-red/5 border border-accent-red/20">
                    <label className="block text-xs font-semibold text-accent-red mb-2 uppercase tracking-wide">
                      Type "CONFIRM" to authorize
                    </label>
                    <input 
                      type="text" 
                      className="input-field border-accent-red/30 focus:border-accent-red focus:shadow-[0_0_10px_theme(colors.accent.red)]"
                      placeholder="CONFIRM"
                      value={confirmText}
                      onChange={e => setConfirmText(e.target.value)}
                    />
                  </div>
                )}

                <div className="flex justify-end gap-3 mt-8 pt-4 border-t border-border">
                  <button onClick={onBlock} className="btn-outline border-transparent hover:border-transparent hover:bg-bg-elevated hover:text-text-primary">
                    Block Action
                  </button>
                  <button 
                    disabled={!canApprove}
                    onClick={onApprove} 
                    className={`btn-primary flex items-center gap-2 ${isHighRisk ? 'bg-accent-red hover:bg-red-500 shadow-[0_0_15px_theme(colors.accent.red)]' : ''}`}
                  >
                    <Check size={16} /> Approve Execution
                  </button>
                </div>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  )
}
