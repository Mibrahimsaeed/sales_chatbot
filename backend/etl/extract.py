from etl.google_client import get_sheets_service
from app.core.config import settings


def fetch_tab(spreadsheet_id: str, tab_name: str) -> list[dict]:
    service = get_sheets_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=tab_name
    ).execute()
    values = result.get("values", [])
    if not values:
        return []
    header, *rows = values
    return [dict(zip(header, row + [""] * (len(header) - len(row)))) for row in rows]


def extract_all() -> dict:
    ccmc = settings.ccmc_spreadsheet_id
    bio = settings.biometric_spreadsheet_id
    return {
        "master_sheet": fetch_tab(ccmc, "MasterSheet"),
        "ccmc_mtd": fetch_tab(ccmc, "CCMC DATA MTD"),
        "p1_overdue": fetch_tab(ccmc, "P1 & Overdue"),
        "connect_session": fetch_tab(ccmc, "Connect Session"),      # system-verified connects
        "biometric": fetch_tab(bio, "Biometric"),
        "login_report": fetch_tab(bio, "Login Report"),
        "mtd_perf": fetch_tab(bio, "MTD Performance"),
        "ytd_perf": fetch_tab(bio, "YTD Performance"),
        "three_m_perf": fetch_tab(bio, "3M Performance"),
        "portfolio": fetch_tab(bio, "Portfolio"),
        "answered_calls": fetch_tab(bio, "Answered Calls"),
        "npr": fetch_tab(bio, "NPR"),
        "target_achievement": fetch_tab(bio, "Target Achievement"),  # feeds team_targets directly
    }