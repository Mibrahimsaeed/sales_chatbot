import { formatMetricValue } from '../../utils/format'

export default function LeaderboardCard({
  rows, metric, totalCount, hasMore, onShowMore, isLoadingMore,
}) {
  if (!rows || !rows.length) return null

  // Part 8 (pagination): "Showing X of Y" once the result was actually
  // truncated; the plain "{N} shown" tag otherwise — a <=15 result never
  // shows pagination wording at all, per spec.
  const isPaginated = totalCount != null && (hasMore || totalCount > rows.length)
  const countTag = isPaginated ? `Showing ${rows.length} of ${totalCount}` : `${rows.length} shown`

  return (
    <div className="card">
      <div className="card-title">
        Leaderboard <span className="tag">{countTag}</span>
      </div>
      {rows.map((item, i) => (
        <div className="leader" key={item.wid ?? item.name ?? i}>
          <div className="rank">{i + 1}</div>
          <div style={{ flex: 1 }}>
            <div className="nm">{item.name}</div>
            <div className="sub">{[item.company, item.team].filter(Boolean).join(' · ')}</div>
          </div>
          <div className="val">{formatMetricValue(metric, item.value)}</div>
        </div>
      ))}
      {isPaginated && (
        hasMore ? (
          <button className="chip show-more-btn" onClick={onShowMore} disabled={isLoadingMore}>
            {isLoadingMore ? 'Loading…' : 'Show More'}
          </button>
        ) : (
          <div className="no-more-results">No more results</div>
        )
      )}
    </div>
  )
}
