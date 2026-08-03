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

  const hasCard = ['advisor', 'team', 'company', 'leaderboard', 'attendance', 'breakdown'].includes(kind)

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
      {kind === 'leaderboard' && (
        <LeaderboardCard
          rows={data}
          metric={metric}
          title={text}
          totalCount={totalCount}
          hasMore={hasMore}
          onShowMore={() => onShowMore(message.id)}
          isLoadingMore={isLoadingMore}
        />
      )}
      {kind === 'attendance' && <AttendanceCard rows={data} />}
      {kind === 'breakdown' && <BreakdownCard b={data} />}
    </div>
  )
}