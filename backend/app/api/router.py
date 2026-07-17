from fastapi import APIRouter
from app.api import advisor, team, leaderboard, chat, auth_dev

api_router = APIRouter()

api_router.include_router(advisor.router)
api_router.include_router(team.router)
api_router.include_router(leaderboard.router)
api_router.include_router(chat.router)
api_router.include_router(auth_dev.router)  # DEV ONLY — remove once real dashboard auth is wired in


