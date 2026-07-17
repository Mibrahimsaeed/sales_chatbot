"""
Number, currency, percent, count, and date formatters.

These are the canonical formatters used by response_formatter.py and
chat_service.py. Centralized here so all metrics format consistently:
  - revenue / cleared / target -> currency
  - achievement / target_pct / attendance_pct -> percent with %
  - connects / overdue / advisors -> counts with thousands separator
  - dates -> ISO-8601 (YYYY-MM-DD) for storage, localized for display
"""

from datetime import date, datetime
from typing import Optional

# Currency symbol. Change once here, propagates everywhere.
CURRENCY_SYMBOL = "Rs"

# Date display format. Change once here.
DATE_DISPLAY_FORMAT = "%d %b %Y"   # e.g. "07 Mar 2025"
DATETIME_DISPLAY_FORMAT = "%d %b %Y, %H:%M"


def format_currency(n: Optional[float], *, symbol: str = CURRENCY_SYMBOL) -> str:
    """
    Format a number as currency.
      None  -> "—"
      < 0   -> "-Rs 5,000"
      >= 1M -> "Rs 1.2M"
      >= 1K -> "Rs 250K"
      else  -> "Rs 1,500"
    """
    if n is None:
        return "—"
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1_000_000:
        return f"{sign}{symbol} {a/1_000_000:.1f}M"
    if a >= 1_000:
        return f"{sign}{symbol} {a/1_000:.0f}K"
    return f"{sign}{symbol} {a:,.0f}"


def format_percent(n: Optional[float], *, decimals: int = 0) -> str:
    """
    Format a 0-100 number as a percent with a trailing % sign.
    Always displays the %, per spec — even for 0% and 100%.
    """
    if n is None:
        return "—"
    return f"{n:.{decimals}f}%"


def format_count(n: Optional[int]) -> str:
    """Format an integer with thousands separator. None -> '—'."""
    if n is None:
        return "—"
    return f"{n:,}"


def format_ratio(num: Optional[float], denom: Optional[float], *, decimals: int = 0) -> str:
    """
    Format a ratio as a percent, safely handling division-by-zero.
    """
    if num is None or denom is None or denom == 0:
        return "—"
    return format_percent((num / denom) * 100, decimals=decimals)


def format_date(d) -> str:
    """
    Format a date or datetime. Accepts date, datetime, or ISO string.
    """
    if d is None:
        return "—"
    if isinstance(d, str):
        try:
            d = datetime.fromisoformat(d)
        except ValueError:
            return d
    if isinstance(d, datetime):
        return d.strftime(DATETIME_DISPLAY_FORMAT)
    if isinstance(d, date):
        return d.strftime(DATE_DISPLAY_FORMAT)
    return str(d)


# ---------------------------------------------------------------------------
# Internal: kept because response_formatter.py / older code imports _pct.
# Routes through format_ratio so the formatting is consistent.
# ---------------------------------------------------------------------------

def _pct(cleared, target) -> str:
    return format_ratio(cleared, target)
