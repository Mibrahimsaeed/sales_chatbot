import { useState } from 'react'

export default function Composer({ onSend, disabled }) {
  const [value, setValue] = useState('')

  const submit = () => {
    if (!value.trim() || disabled) return
    onSend(value)
    setValue('')
  }

  return (
    <div className="composer">
      <input
        type="text"
        value={value}
        placeholder="Ask about an advisor, a team, overdue pipeline, targets..."
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') submit()
        }}
        disabled={disabled}
      />
      <button className="send" onClick={submit} disabled={disabled}>
        Ask
      </button>
    </div>
  )
}
