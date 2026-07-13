export const fmtNum = (n) => {
  if (n === undefined || n === null) return '-'
  return Math.round(n).toLocaleString('en-US')
}

export const fmtPKR = (n) => {
  if (n === undefined || n === null) return '-'
  if (Math.abs(n) >= 1e7) return (n / 1e7).toFixed(2) + ' Cr'
  if (Math.abs(n) >= 1e5) return (n / 1e5).toFixed(2) + ' Lac'
  return fmtNum(n)
}

export const fmtPct = (n) => {
  if (n === undefined || n === null) return '-'
  return (n * 100).toFixed(0) + '%'
}

// Metrics whose value is currency vs. a rate vs. a plain count — used by
// LeaderboardCard to pick the right formatter for whatever metric the
// backend resolved the query to.
const CURRENCY_METRICS = new Set([
  'mtd_cleared', 'ytd_cleared', 'three_month_cleared', 'mtd_target',
  'portfolio_value', 'returned_value', 'overdue_amount',
])
const PERCENT_METRICS = new Set(['achievement_pct', 'conversion'])

export function formatMetricValue(metric, value) {
  if (value === undefined || value === null) return '-'
  if (PERCENT_METRICS.has(metric)) return fmtPct(value)
  if (CURRENCY_METRICS.has(metric)) return fmtPKR(value)
  return fmtNum(value)
}
