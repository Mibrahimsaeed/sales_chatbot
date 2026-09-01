import { formatMetricValue } from '../../utils/format'

// A row's bundled measures, when the backend attached any. Each cell
// arrives as { value, display, label }: the label and the rendered text
// are decided by the ontology on the server, so this file never decides
// what a metric is called or how it prints. That matters more than it
// looks — formatMetricValue below keeps its own hardcoded metric sets,
// which do not know `answered_calls_rate` and would print an
// already-scaled 114.7 as a plain count (or, treated as a percentage,
// multiply it again to 11470%).
const cellsOf = (item) => item?.columns || null

// Column order comes from the payload's key order, which the server
// builds primary-first.
const keysOf = (rows) => Object.keys(cellsOf(rows[0]) || {})

export default function LeaderboardCard({
  rows, metric, totalCount, hasMore, onShowMore, isLoadingMore,
  title = 'Leaderboard',
}) {
  if (!rows || !rows.length) return null

  // A POPULATION carries names with no measure — "who matches this",
  // with nothing to rank by. Rendering a value column for it would print
  // "-" beside every name, which is the exact regression the backend's
  // response planner warns about. Detected from the data rather than
  // from the response type, so any valueless row set is handled.
  const hasValues = rows.some((r) => r.value !== null && r.value !== undefined)

  // Part 8 (pagination): "Showing X of Y" once the result was actually
  // truncated; the plain "{N} shown" tag otherwise — a <=15 result never
  // shows pagination wording at all, per spec.
  const isPaginated = totalCount != null && (hasMore || totalCount > rows.length)
  const countTag = isPaginated ? `Showing ${rows.length} of ${totalCount}` : `${rows.length} shown`

  // Every row of one response carries the same measures, so the headings
  // are read once from the first row.
  const columnKeys = keysOf(rows)
  const hasColumns = columnKeys.length > 0

  const showMore = isPaginated && (
    hasMore ? (
      <button className="chip show-more-btn" onClick={onShowMore} disabled={isLoadingMore}>
        {isLoadingMore ? 'Loading…' : 'Show More'}
      </button>
    ) : (
      <div className="no-more-results">No more results</div>
    )
  )

  // Without bundled columns this is the original single-value list,
  // unchanged — every other leaderboard still renders exactly as before.
  if (!hasColumns) {
    return (
      <div className="card">
        <div className="card-title">
          {title} <span className="tag">{countTag}</span>
        </div>
        {rows.map((item, i) => (
          <div className="leader" key={item.wid ?? item.name ?? i}>
            <div className="rank">{i + 1}</div>
            <div style={{ flex: 1 }}>
              <div className="nm">{item.name}</div>
              <div className="sub">{[item.company, item.team].filter(Boolean).join(' · ')}</div>
            </div>
            {hasValues && (
              <div className="val">{formatMetricValue(metric, item.value)}</div>
            )}
          </div>
        ))}
        {showMore}
      </div>
    )
  }

  const headings = columnKeys.map((key) => cellsOf(rows[0])[key].label)

  return (
    <div className="card">
      <div className="card-title">
        {title} <span className="tag">{countTag}</span>
      </div>
      {/* Scrolls sideways rather than wrapping: three measures beside a
          long name is wider than a narrow chat column, and wrapping a
          number onto the next line detaches it from its row. */}
      <div className="metric-table-scroll">
        <table className="metric-table">
          <thead>
            <tr>
              <th className="mt-rank" />
              <th className="mt-name">Name</th>
              {headings.map((label, i) => (
                <th className="mt-num" key={columnKeys[i]}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((item, i) => {
              const cells = cellsOf(item) || {}
              return (
                <tr key={item.wid ?? `${item.name}-${i}`}>
                  <td className="mt-rank">{i + 1}</td>
                  <td className="mt-name">
                    <div className="nm">{item.name}</div>
                    <div className="sub">
                      {[item.company, item.team].filter(Boolean).join(' · ')}
                    </div>
                  </td>
                  {columnKeys.map((key) => (
                    // The server already renders a missing figure as an
                    // em dash; the fallback covers a key this row simply
                    // does not carry, so a cell is never blank and the
                    // row never shifts.
                    <td className="mt-num" key={key}>{cells[key]?.display ?? '—'}</td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {showMore}
    </div>
  )
}
