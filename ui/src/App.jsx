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

  // WebSocket setup
  useEffect(() => {
    const wsUrl = `${WS_BASE}/ws/stream/${sessionId}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('Connected to Agentic OS Backend')
      setSystemState('idle')
    }

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data)
      console.log('WS Event:', payload)

      if (payload.event === 'node_update') {
        const { node, data } = payload
        
        // Handle transitions for the UI vite status
        if (node === 'parse_intent') setSystemState('planning')
        if (node === 'route_to_agent') setSystemState('analyzing')
        if (node === 'execute_with_guardrails') setSystemState('executing')
        
        // Handle Approval Required
        if (data.requires_approval && data.action_id) {
          setApprovalParams({
            action_id: data.action_id,
            agent: data.guardrail_result?.risk_level === 'critical' ? 'Guardian (Critical)' : 'Guardian',
            command: data.goal || 'System Action',
            riskScore: data.guardrail_result?.risk_level === 'critical' ? 10 : (data.guardrail_result?.risk_level === 'high' ? 9 : 7),
            impact: data.guardrail_result?.blocked_reason || 'Sensitive action detected.',
            warnings: data.guardrail_result?.violations || []
          })
          setSystemState('waiting')
        }

        // Add message if there's content to show
        if (data.messages && data.messages.length > 0) {
          data.messages.forEach(m => {
            setMessages(prev => [...prev, {
              role: 'agent',
              agent: node === 'respond_to_user' ? 'System' : node.split('_')[0],
              content: m.content,
              timestamp: new Date()
            }])
          })
        } else if (node === 'route_to_agent' && data.plan) {
           setMessages(prev => [...prev, {
             role: 'agent', 
             agent: 'planner', 
             content: `Plan generated: ${data.plan.length} steps identified.`,
             timestamp: new Date()
           }])
        }
      }

      if (payload.event === 'complete') {
        setIsRunning(false)
        setSystemState('idle')
        toast.success('Task execution completed')
      }

      if (payload.event === 'error') {
        setIsRunning(false)
        setSystemState('idle')
        toast.error(`Agent Error: ${payload.message}`)
      }
      
      if (payload.event === 'interrupted') {
        setIsRunning(false)
        setSystemState('idle')
        toast.error('Task interrupted')
      }
    }

    ws.onclose = () => {
      setIsRunning(false)
      setSystemState('offline')
      console.warn('Disconnected from backend')
    }

    return () => ws.close()
  }, [sessionId])

  const handleSendMessage = (inputText) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      toast.error('Not connected to backend')
      return
    }

    setMessages(prev => [...prev, { role: 'user', content: inputText, timestamp: new Date() }])
    setIsRunning(true)
    
    wsRef.current.send(JSON.stringify({
      goal: inputText,
      user_id: 'anonymous',
      session_id: sessionId,
      metadata: {}
    }))
  }

  const handleApprove = () => {
    const actionId = approvalParams?.action_id
    if (wsRef.current && actionId) {
      wsRef.current.send(JSON.stringify({
        action: 'approve',
        action_id: actionId,
        approved: true
      }))
    }
    setApprovalParams(null)
    toast.success('Action approved', { icon: '🚀' })
  }

  const handleBlock = () => {
    const actionId = approvalParams?.action_id
     if (wsRef.current && actionId) {
      wsRef.current.send(JSON.stringify({
        action: 'approve',
        action_id: actionId,
        approved: false
      }))
    }
    setApprovalParams(null)
    toast.error('Action blocked')
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
