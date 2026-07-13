import { fmtNum, fmtPKR, fmtPct } from '../../utils/format'

export default function AdvisorCard({ a }) {
  if (!a) return null

  const connects = (a.mtd_new_connect || 0) + (a.mtd_followup_connect || 0)
  const meetings = (a.mtd_new_meeting || 0) + (a.mtd_followup_meeting || 0)
  const pctClass = a.mtd_pct >= 1 ? 'good' : a.mtd_pct >= 0.6 ? 'warn' : 'bad'
  const bioClass =
    a.biometric_status === 'On Time' ? 'good' : a.biometric_status === 'Late' ? 'warn' : 'bad'
  const loginClass = a.login_status?.includes('Before') ? 'good' : 'bad'

  return (
    <div className="card">
      <div className="card-title">
        {a.name} <span className="tag">WID {a.wid}</span>
      </div>
      <div className="row">
        <span className="l">Company / Team</span>
        <span className="r">{a.company || '-'} · {a.team || '-'}</span>
      </div>
      <div className="row">
        <span className="l">Reports to</span>
        <span className="r">{a.management_lead || a.portfolio_lead || '-'}</span>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="v">{fmtNum(connects)}</div>
          <div className="l">MTD Connects</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(meetings)}</div>
          <div className="l">MTD Meetings</div>
        </div>
        <div className="kpi">
          <div className="v">{fmtNum(a.mtd_cr)}</div>
          <div className="l">CR Booked</div>
        </div>
      </div>

      <div className="row">
        <span className="l">Pipeline / Overdue</span>
        <span className="r">
          {fmtNum(a.pipeline)} / {a.overdue ? <span className="pill bad">{fmtNum(a.overdue)}</span> : '0'}
        </span>
      </div>

      {a.mtd_target != null && (
        <div className="row">
          <span className="l">MTD Target / Cleared</span>
          <span className="r">
            {fmtPKR(a.mtd_target)} / {fmtPKR(a.mtd_cleared)}{' '}
            <span className={`pill ${pctClass}`}>{fmtPct(a.mtd_pct)}</span>
          </span>
        </div>
      )}

      {a.biometric_status && (
        <div className="row">
          <span className="l">Biometric today</span>
          <span className="r">
            {a.biometric_time || 'N/A'} <span className={`pill ${bioClass}`}>{a.biometric_status}</span>
          </span>
        </div>
      )}

      {a.login_status && (
        <div className="row">
          <span className="l">System login</span>
          <span className="r">
            {a.login_time || 'N/A'} <span className={`pill ${loginClass}`}>{a.login_status}</span>
          </span>
        </div>
      )}

      {a.portfolio_value != null && (
        <div className="row">
          <span className="l">Portfolio / Retention</span>
          <span className="r">
            {fmtPKR(a.portfolio_value)} · {fmtPct(a.portfolio_retention_pct)}
          </span>
        </div>
      )}
    </div>
  )
}
