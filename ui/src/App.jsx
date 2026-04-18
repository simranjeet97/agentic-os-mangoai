import { useState, useRef, useEffect } from 'react'
import { Toaster, toast } from 'react-hot-toast'
import Layout from './components/Layout'
import MainShell from './components/MainShell'
import AgentPanel from './components/AgentPanel'
import TaskManager from './components/TaskManager'
import AuditViewer from './components/AuditViewer'
import MemoryExplorer from './components/MemoryExplorer'
import Settings from './components/Settings'
import ApprovalDialog from './components/ApprovalDialog'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_BASE  = import.meta.env.VITE_WS_URL  || 'ws://localhost:8000'

export default function App() {
  const [activeView, setActiveView] = useState('shell')
  const [sessionId] = useState(() => crypto.randomUUID())
  const [messages, setMessages] = useState([])
  const [isRunning, setIsRunning] = useState(false)
  const [systemState, setSystemState] = useState('idle')

  // Approval Overlay State
  const [approvalParams, setApprovalParams] = useState(null) // null or object

  const wsRef = useRef(null)

  const handleSendMessage = (inputText) => {
    setMessages(prev => [...prev, { role: 'user', content: inputText, timestamp: new Date() }])
    setIsRunning(true)
    setSystemState('planning')
    
    // Simulate connection for the UI if backend not wired
    setTimeout(() => {
      setMessages(prev => [...prev, {
        role: 'agent', agent: 'planner', content: `Directive received. Computing execution graph...`, timestamp: new Date()
      }])
      
      // Simulate an approval request
      if (inputText.toLowerCase().includes('sudo') || inputText.toLowerCase().includes('rm')) {
        setTimeout(() => {
          setApprovalParams({
            agent: 'system',
            command: inputText,
            riskScore: 9,
            impact: 'Removes files entirely from disk with no immediate recovery.',
            warnings: ['Data loss is highly likely if target path is incorrect.', 'Action cannot be undone.']
          })
        }, 1500)
      } else {
        setTimeout(() => {
          setIsRunning(false)
          setSystemState('idle')
          setMessages(prev => [...prev, {
            role: 'agent', agent: 'executor', content: `Task sequence completed successfully.`, timestamp: new Date()
          }])
        }, 2000)
      }
    }, 500)
  }

  const handleApprove = () => {
    setApprovalParams(null)
    toast.success('Action approved. Executing...', { icon: '🚀' })
    setTimeout(() => {
      setIsRunning(false)
      setSystemState('idle')
      setMessages(prev => [...prev, {
        role: 'agent', agent: 'executor', content: `Command executed. System is stable.`, timestamp: new Date()
      }])
    }, 1500)
  }

  const handleBlock = () => {
    setApprovalParams(null)
    toast.error('Action blocked by user.', { icon: '⛔' })
    setIsRunning(false)
    setSystemState('idle')
  }

  return (
    <>
      <Toaster position="top-center" toastOptions={{ 
        style: { background: 'var(--bg-elevated)', color: 'var(--text-primary)', border: '1px solid var(--border)' },
      }} />
      
      <Layout activeView={activeView} setActiveView={setActiveView} isRunning={isRunning}>
        {activeView === 'shell' && (
          <MainShell 
            messages={messages} 
            setMessages={setMessages} 
            isRunning={isRunning} 
            handleSendMessage={handleSendMessage} 
          />
        )}
        {activeView === 'agents' && <AgentPanel />}
        {activeView === 'tasks' && <TaskManager />}
        {activeView === 'audit' && <AuditViewer />}
        {activeView === 'memory' && <MemoryExplorer />}
        {activeView === 'settings' && <Settings />}
      </Layout>

      {/* Global Modals */}
      <ApprovalDialog 
        isOpen={!!approvalParams} 
        onOpenChange={(v) => !v && handleBlock()}
        actionDetails={approvalParams}
        onApprove={handleApprove}
        onBlock={handleBlock}
      />
    </>
  )
}
