import AdvisorCard from './cards/AdvisorCard'
import TeamCard from './cards/TeamCard'
import CompanyCard from './cards/CompanyCard'
import LeaderboardCard from './cards/LeaderboardCard'
import AttendanceCard from './cards/AttendanceCard'

export default function MessageBubble({ message }) {
  const { role, text, kind, data, metric } = message

  if (role === 'user') {
    return (
      <div className="msg user">
        <div className="bubble">{text}</div>
      </div>
    )
  }

  const hasCard = ['advisor', 'team', 'company', 'leaderboard', 'attendance'].includes(kind)

  return (
    <div className="msg bot">
      {text && !hasCard && <div className="bubble">{text}</div>}
      {kind === 'advisor' && <AdvisorCard a={data} />}
      {kind === 'team' && <TeamCard t={data} />}
      {kind === 'company' && <CompanyCard c={data} />}
      {kind === 'leaderboard' && <LeaderboardCard rows={data} metric={metric} title={text} />}
      {kind === 'attendance' && <AttendanceCard rows={data} />}
    </div>
  )
}