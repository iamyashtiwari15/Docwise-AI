import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'
const WELCOME_MESSAGE = {
  id: 'welcome',
  role: 'assistant',
  content:
    'Welcome! Upload a PDF or DOCX and ask grounded questions about it, or ask for current web information when you need live answers.',
}

function getSessionId() {
  const existingSessionId = window.localStorage.getItem('session_id')
  if (existingSessionId) {
    return existingSessionId
  }

  const nextSessionId = crypto.randomUUID()
  window.localStorage.setItem('session_id', nextSessionId)
  return nextSessionId
}

function createLocalMessage(role, content, extra = {}) {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    ...extra,
  }
}

function getUploadName(content) {
  const match = /^\[Uploaded file: (.+)\]$/.exec(content || '')
  return match ? match[1] : null
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE])
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [messageInput, setMessageInput] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [selectedFilePreview, setSelectedFilePreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [indexingFile, setIndexingFile] = useState(null) // filename currently being indexed
  const messagesEndRef = useRef(null)

  const sessionId = useMemo(() => getSessionId(), [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    const abortController = new AbortController()

    async function loadHistory() {
      try {
        const response = await fetch(`${API_BASE_URL}/history`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ session_id: sessionId }),
          signal: abortController.signal,
        })

        if (!response.ok) {
          throw new Error('Failed to fetch history')
        }

        const payload = await response.json()
        setUploadedFiles(payload.uploaded_files || [])

        if (Array.isArray(payload.history) && payload.history.length > 0) {
          setMessages(
            payload.history.map((message) => ({
              id: crypto.randomUUID(),
              role: message.role,
              content: message.content,
            })),
          )
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('Failed to load chat history', error)
        }
      }
    }

    loadHistory()
    return () => abortController.abort()
  }, [sessionId])

  function resetComposer() {
    setMessageInput('')
    setSelectedFile(null)
    setSelectedFilePreview(null)
  }

  async function handleFileChange(event) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    const isSupportedDocument =
      file.type === 'application/pdf' ||
      file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
      file.type === 'text/plain' ||
      file.name.toLowerCase().endsWith('.pdf') ||
      file.name.toLowerCase().endsWith('.docx') ||
      file.name.toLowerCase().endsWith('.txt')

    if (file.type.startsWith('image/')) {
      setSelectedFile(file)
      setSelectedFilePreview({ type: 'image', value: URL.createObjectURL(file), name: file.name })
      return
    }

    if (isSupportedDocument) {
      setSelectedFile(file)
      setSelectedFilePreview({ type: 'file', value: file.name, name: file.name })

      // Eagerly send to /ingest so chunking + embedding starts immediately
      setIndexingFile(file.name)
      try {
        const formData = new FormData()
        formData.append('session_id', sessionId)
        formData.append('file', file)
        await fetch(`${API_BASE_URL}/ingest`, { method: 'POST', body: formData })
      } catch (err) {
        console.warn('Background ingest failed (will retry on send):', err)
      } finally {
        setIndexingFile(null)
      }
      return
    }

    window.alert('Please upload an image, PDF, DOCX, or TXT file.')
    event.target.value = ''
    resetComposer()
  }

  function removeSelectedFile() {
    setSelectedFile(null)
    setSelectedFilePreview(null)
  }

  async function handleSubmit(event) {
    event.preventDefault()

    const message = messageInput.trim()
    if (!message && !selectedFile) {
      return
    }

    const optimisticMessages = []
    if (message) {
      optimisticMessages.push(createLocalMessage('user', message))
    }

    if (selectedFile) {
      if (selectedFilePreview?.type === 'image') {
        optimisticMessages.push(
          createLocalMessage('user', '', {
            attachmentType: 'image',
            attachmentName: selectedFile.name,
            attachmentValue: selectedFilePreview.value,
          }),
        )
      } else {
        optimisticMessages.push(
          createLocalMessage('user', `[Uploaded file: ${selectedFile.name}]`, {
            attachmentType: 'file',
            attachmentName: selectedFile.name,
          }),
        )
      }
    }

    setMessages((currentMessages) => [...currentMessages, ...optimisticMessages])
    setLoading(true)

    const requestFile = selectedFile
    const requestMessage = message
    resetComposer()

    try {
      const formData = new FormData()
      formData.append('session_id', sessionId)

      let endpoint = `${API_BASE_URL}/chat`
      if (requestFile) {
        formData.append('file', requestFile)
        formData.append('message', requestMessage)
        endpoint = `${API_BASE_URL}/upload`

        if (!requestFile.type.startsWith('image/')) {
          setUploadedFiles((currentFiles) =>
            currentFiles.includes(requestFile.name) ? currentFiles : [...currentFiles, requestFile.name],
          )
        }
      } else {
        formData.append('message', requestMessage)
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const payload = await response.json()
      setMessages((currentMessages) => [
        ...currentMessages,
        createLocalMessage('assistant', payload.response || 'Sorry, I could not process that request.'),
      ])
    } catch (error) {
      console.error('Failed to submit message', error)
      setMessages((currentMessages) => [
        ...currentMessages,
        createLocalMessage('assistant', '⚠️ Something went wrong. Please try again.'),
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-section">
          <h1>Document & Web Assistant</h1>
          <p>
            Upload PDF or DOCX files, ask grounded questions, and fall back to the web only when the answer is not in
            your documents.
          </p>
        </div>

        <div className="sidebar-section">
          <h2>Routes</h2>
          <ul>
            <li>Document Q&A</li>
            <li>Web fallback</li>
            <li>General chat</li>
            <li>Image analysis</li>
          </ul>
        </div>

        <div className="sidebar-section">
          <h2>Uploaded knowledge</h2>
          {uploadedFiles.length > 0 ? (
            <ul className="file-list">
              {uploadedFiles.map((fileName) => (
                <li key={fileName}>{fileName}</li>
              ))}
            </ul>
          ) : (
            <p className="muted-text">No documents uploaded yet.</p>
          )}
        </div>
      </aside>

      <main className="chat-layout">
        <header className="chat-header">
          <div>
            <h2>Chat</h2>
            <p>Upload a file and ask questions against it, or ask for live web information.</p>
          </div>
        </header>

        <section className="message-list">
          {messages.map((message) => {
            const uploadName = getUploadName(message.content)

            return (
              <article key={message.id} className={`message message-${message.role}`}>
                <div className="message-card">
                  {message.attachmentType === 'image' ? (
                    <img className="message-image" src={message.attachmentValue} alt={message.attachmentName} />
                  ) : uploadName ? (
                    <div className="file-chip">{uploadName}</div>
                  ) : message.role === 'assistant' ? (
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  ) : (
                    <p>{message.content}</p>
                  )}
                </div>
              </article>
            )
          })}

          {loading ? (
            <article className="message message-assistant">
              <div className="message-card loading-card">
                <span />
                <span />
                <span />
              </div>
            </article>
          ) : null}

          <div ref={messagesEndRef} />
        </section>

        <footer className="composer">
          {selectedFilePreview ? (
            <div className="preview-card">
              {selectedFilePreview.type === 'image' ? (
                <img src={selectedFilePreview.value} alt={selectedFilePreview.name} className="preview-image" />
              ) : (
                <div className="file-chip">
                  {selectedFilePreview.name}
                  {indexingFile === selectedFilePreview.name && (
                    <span className="indexing-badge"> ⏳ indexing…</span>
                  )}
                </div>
              )}
              <button type="button" className="ghost-button" onClick={removeSelectedFile}>
                Remove
              </button>
            </div>
          ) : null}

          <form className="composer-form" onSubmit={handleSubmit}>
            <label className="icon-button" htmlFor="file-input">
              +
            </label>
            <input
              id="file-input"
              type="file"
              accept="image/*,.pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
              hidden
              onChange={handleFileChange}
            />

            <textarea
              value={messageInput}
              onChange={(event) => setMessageInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  handleSubmit(event)
                }
              }}
              placeholder="Ask a question or upload a file..."
              rows={1}
            />

            <button className="send-button" type="submit" disabled={loading}>
              Send
            </button>
          </form>
        </footer>
      </main>
    </div>
  )
}
