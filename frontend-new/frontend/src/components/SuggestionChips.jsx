const SUGGESTIONS = [
  'Top 5 advisors by revenue',
  'Top teams by achievement',
  'Worst overdue teams',
  'Who has most connects',
  'Who was late today',
  'Give me target achievement',
]

export default function SuggestionChips({ onPick }) {
  return (
    <div className="chips">
      {SUGGESTIONS.map((s) => (
        <div key={s} className="chip" onClick={() => onPick(s)}>
          {s}
        </div>
      ))}
    </div>
  )
}
