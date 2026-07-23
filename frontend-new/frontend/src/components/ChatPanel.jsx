import { useEffect, useRef } from 'react'
import Header from './Header'
import SuggestionChips from './SuggestionChips'
import Composer from './Composer'
import MessageBubble from './MessageBubble'
import { useChat } from '../hooks/useChat'

export default function ChatPanel() {
  const { messages, isLoading, send, loadMore, loadingMoreId } = useChat()
  const threadRef = useRef(null)

  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight
    }
  }, [messages, isLoading])

  return (
    <div className="app">
      <Header />
      <SuggestionChips onPick={send} />
      <div className="thread" ref={threadRef}>
        {messages.map((m) => (
          <MessageBubble
            key={m.id}
            message={m}
            onShowMore={loadMore}
            isLoadingMore={loadingMoreId === m.id}
          />
        ))}
        {isLoading && (
          <div className="msg bot">
            <div className="bubble">Thinking…</div>
          </div>
        )}
      </div>
      <Composer onSend={send} disabled={isLoading} />
    </div>
  )
}
