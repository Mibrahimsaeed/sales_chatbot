from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router

app = FastAPI(title="CCMC Sales Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your dashboard's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}