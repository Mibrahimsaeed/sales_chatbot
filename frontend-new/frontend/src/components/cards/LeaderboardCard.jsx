import { formatMetricValue } from '../../utils/format'

export default function LeaderboardCard({ rows, metric }) {
  if (!rows || !rows.length) return null

  return (
    <div className="card">
      <div className="card-title">
        Leaderboard <span className="tag">{rows.length} shown</span>
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
    </div>
  )
}
