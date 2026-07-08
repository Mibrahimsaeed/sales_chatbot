from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from app.core.config import settings

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def get_sheets_service():
    creds = Credentials.from_service_account_file(
        settings.google_service_account_path, scopes=SCOPES
    )
    return build("sheets", "v4", credentials=creds)