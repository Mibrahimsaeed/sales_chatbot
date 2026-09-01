import AdvisorCard from './cards/AdvisorCard'
import TeamCard from './cards/TeamCard'
import CompanyCard from './cards/CompanyCard'
import LeaderboardCard from './cards/LeaderboardCard'
import AttendanceCard from './cards/AttendanceCard'
import BreakdownCard from './cards/BreakdownCard'

export default function MessageBubble({ message, onShowMore, isLoadingMore }) {
  const { role, text, kind, data, metric, hasMore, totalCount } = message

  if (role === 'user') {
    return (
      <div className="msg user">
        <div className="bubble">{text}</div>
      </div>
    )
  }

  // `breakdown` is the one wire type that carries TWO different answer
  // shapes. The hierarchy breakdown sends a nested OBJECT (level_label,
  // teams[], mtd_cleared) which BreakdownCard renders; filtered_list and
  // population send a flat ARRAY of rows, which it cannot.
  //
  // Given an array, `if (!b) return null` does not fire — an array is
  // truthy — so every field reads `undefined`, `b.teams || []` is empty,
  // and the card draws an empty shell. Meanwhile `hasCard` suppresses
  // the reply text that carried the whole answer, so the message renders
  // blank even though the backend returned it in full.
  //
  // Deciding on the SHAPE rather than the type alone keeps the card for
  // the case it was written for and falls back to the text for the case
  // it was not.
  const breakdownIsCard = kind === 'breakdown' && data && !Array.isArray(data)

  // ROW-SHAPED LIST RESULTS RENDER FROM `data`, NOT FROM `text`.
  //
  // `filtered_list` and `population` carry the same array of rows a
  // leaderboard does, but they were rendered as the reply TEXT — and the
  // text is a snapshot of the first page. "Show More" appends the next
  // page to `data` (see useChat.loadMore) and updates the counts, but
  // nothing rendered `data` for these kinds, so the visible list never
  // changed no matter how many times it was clicked.
  //
  // LeaderboardCard already renders an accumulating row set, the
  // "Showing X of Y" tag, and a Show More button that becomes "No more
  // results" when the last page arrives — so pointing these at it fixes
  // the paging rather than reimplementing it.
  const rowList = Array.isArray(data) && data.length > 0
  const listIsCard = rowList && ['filtered_list', 'population'].includes(kind)

  const hasCard =
    breakdownIsCard ||
    listIsCard ||
    ['advisor', 'team', 'company', 'leaderboard', 'attendance'].includes(kind)

  // Part 8 (pagination): comparison/filtered_list responses have no
  // dedicated card — the "Showing X of Y" wording is already baked into
  // `text` by the backend, this just adds the button itself.
  const showMoreButton = hasMore && (
    <button
      className="chip show-more-btn"
      onClick={() => onShowMore(message.id)}
      disabled={isLoadingMore}
    >
      {isLoadingMore ? 'Loading…' : 'Show More'}
    </button>
  )

  return (
    <div className="msg bot">
      {text && !hasCard && (
        <div className="bubble">
          {text}
          {showMoreButton}
        </div>
      )}
      {kind === 'advisor' && <AdvisorCard a={data} />}
      {kind === 'team' && <TeamCard t={data} />}
      {kind === 'company' && <CompanyCard c={data} />}
      {(kind === 'leaderboard' || listIsCard) && (
        <LeaderboardCard
          rows={data}
          metric={metric}
          title={kind === 'leaderboard' ? 'Leaderboard' : 'Results'}
          totalCount={totalCount}
          hasMore={hasMore}
          onShowMore={() => onShowMore(message.id)}
          isLoadingMore={isLoadingMore}
        />
      )}
      {kind === 'attendance' && <AttendanceCard rows={data} />}
      {breakdownIsCard && <BreakdownCard b={data} />}
    </div>
  )
}