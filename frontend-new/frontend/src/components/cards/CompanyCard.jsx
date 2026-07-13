import { fmtNum, fmtPKR, fmtPct } from '../../utils/format'

export default function CompanyCard({ c }) {
  if (!c) return null

  const pct = c.mtd_target ? c.mtd_cleared / c.mtd_target : 0
  const pctClass = pct >= 1 ? 'good' : pct >= 0.6 ? 'warn' : 'bad'

  return (
    <div className="card">
      <div className="card-title">
        {c.company} <span className="tag">{c.advisors} advisors</span>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="v">{fmtNum(c.connects)}</div>
          <div className="l">MTD Connects</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(c.overdue)}</div>
          <div className="l">Overdue</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(c.pipeline)}</div>
          <div className="l">Open Pipeline</div>
        </div>
      </div>

      <div className="row">
        <span className="l">MTD Target</span>
        <span className="r">{fmtPKR(c.mtd_target)}</span>
      </div>
      <div className="row">
        <span className="l">MTD Cleared</span>
        <span className="r">
          {fmtPKR(c.mtd_cleared)} <span className={`pill ${pctClass}`}>{fmtPct(pct)}</span>
        </span>
      </div>
    </div>
  )
}
