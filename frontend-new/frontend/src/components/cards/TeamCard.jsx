import { fmtNum, fmtPKR, fmtPct } from '../../utils/format'

export default function TeamCard({ t }) {
  if (!t) return null

  const pct = t.achievement_pct != null ? t.achievement_pct : t.target ? (t.achieved || 0) / t.target : null
  const pctClass = pct >= 1 ? 'good' : pct >= 0.6 ? 'warn' : 'bad'
  const ytdPct = t.ytd_target ? t.ytd_cleared / t.ytd_target : 0
  const ytdPctClass = ytdPct >= 1 ? 'good' : ytdPct >= 0.6 ? 'warn' : 'bad'

  return (
    <div className="card">
      <div className="card-title">
        {t.team} <span className="tag">{t.advisors} advisors</span>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="v">{fmtNum(t.connects)}</div>
          <div className="l">MTD Connects</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(t.overdue)}</div>
          <div className="l">Overdue</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(t.pipeline)}</div>
          <div className="l">Open Pipeline</div>
        </div>
      </div>

      {t.target != null ? (
        <>
          <div className="row">
            <span className="l">Target</span>
            <span className="r">{fmtPKR(t.target)}</span>
          </div>
          <div className="row">
            <span className="l">Achieved</span>
            <span className="r">
              {fmtPKR(t.achieved)} <span className={`pill ${pctClass}`}>{fmtPct(pct)}</span>
            </span>
          </div>
        </>
      ) : (
        <div className="row">
          <span className="l">Target</span>
          <span className="r">No target on file</span>
        </div>
      )}

      {t.ytd_target != null && (
        <div className="row">
          <span className="l">YTD Target / Cleared</span>
          <span className="r">
            {fmtPKR(t.ytd_target)} / {fmtPKR(t.ytd_cleared)}{' '}
            <span className={`pill ${ytdPctClass}`}>{fmtPct(ytdPct)}</span>
          </span>
        </div>
      )}
    </div>
  )
}
