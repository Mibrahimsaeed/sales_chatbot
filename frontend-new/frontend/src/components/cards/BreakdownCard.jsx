import { fmtNum, fmtPKR, fmtPct } from '../../utils/format'

// Renders hierarchy_service.get_level_breakdown's shape: a Unit Head /
// Zonal Head / Business Center's advisors NESTED BY TEAM (never a flat
// list) — a single head can oversee multiple teams, and this is meant to
// make that structure visible instead of collapsing it into one number.
export default function BreakdownCard({ b }) {
  if (!b) return null

  const pct = b.mtd_target ? b.mtd_cleared / b.mtd_target : 0
  const pctClass = pct >= 1 ? 'good' : pct >= 0.6 ? 'warn' : 'bad'
  const ytdPct = b.ytd_target ? b.ytd_cleared / b.ytd_target : 0
  const ytdPctClass = ytdPct >= 1 ? 'good' : ytdPct >= 0.6 ? 'warn' : 'bad'
  const teams = b.teams || []

  return (
    <div className="card">
      <div className="card-title">
        {b.level_label}: {b.value} <span className="tag">{b.advisors} advisors · {teams.length} team(s)</span>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="v">{fmtNum(b.connects)}</div>
          <div className="l">MTD Connects</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(b.overdue)}</div>
          <div className="l">Overdue</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(b.pipeline)}</div>
          <div className="l">Open Pipeline</div>
        </div>
      </div>

      <div className="row">
        <span className="l">MTD Target</span>
        <span className="r">{fmtPKR(b.mtd_target)}</span>
      </div>
      <div className="row">
        <span className="l">MTD Cleared</span>
        <span className="r">
          {fmtPKR(b.mtd_cleared)} <span className={`pill ${pctClass}`}>{fmtPct(pct)}</span>
        </span>
      </div>

      {b.ytd_target != null && (
        <div className="row">
          <span className="l">YTD Target / Cleared</span>
          <span className="r">
            {fmtPKR(b.ytd_target)} / {fmtPKR(b.ytd_cleared)}{' '}
            <span className={`pill ${ytdPctClass}`}>{fmtPct(ytdPct)}</span>
          </span>
        </div>
      )}

      {teams.map((team) => (
        <div className="breakdown-team" key={team.team}>
          <div className="breakdown-team-title">
            {team.team} <span className="tag">{team.advisor_count} advisor(s)</span>
          </div>
          {team.advisors.map((a) => (
            <div className="breakdown-advisor" key={a.wid ?? a.name}>
              <span className="nm">{a.name}</span>
              <span className="r">{fmtNum(a.connects)} connects · {fmtPKR(a.mtd_cleared)} cleared</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
