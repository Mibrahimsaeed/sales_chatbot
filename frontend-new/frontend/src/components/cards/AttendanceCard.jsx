export default function AttendanceCard({ rows }) {
  if (!rows || !rows.length) return null

  return (
    <div className="card">
      <div className="card-title">
        Attendance Issues <span className="tag">{rows.length} advisors</span>
      </div>
      {rows.map((r, i) => (
        <div className="row" key={r.wid ?? i}>
          <span className="l">
            {r.name} <span style={{ opacity: 0.6 }}>· {r.team || '-'}</span>
          </span>
          <span className="r">
            <span className={`pill ${r.biometric_status === 'On Time' ? 'good' : 'bad'}`}>
              {r.biometric_status}
            </span>
          </span>
        </div>
      ))}
    </div>
  )
}
