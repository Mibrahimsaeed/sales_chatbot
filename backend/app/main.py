# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app.api.router import api_router
# from fastapi.middleware.cors import CORSMiddleware
# app = FastAPI(title="CCMC Sales Chatbot API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://172.16.8.193:5173",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(api_router, prefix="/api")


# @app.get("/health")
# def health():
#     return {"status": "ok"}
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://172.16.8.193:5173",
#         "http://localhost:5173",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(api_router, prefix="/api")


from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Best-effort: loads the chat model into Ollama's memory at boot so
    the first real chat request isn't the one that pays the ~15-30s cold
    model-load cost (see llm_client.py's _KEEP_ALIVE comment). A failure
    here (Ollama not running yet, model not pulled) must never block
    startup — the whole LLM layer is already fail-soft, this is purely an
    optimization, and call_llm_json() itself never raises regardless."""
    from app.core.logger import get_logger
    from app.llm.llm_client import call_llm_json

    log = get_logger("main")
    if call_llm_json('Return ONLY JSON: {"ok": true}') is not None:
        log.info("Ollama model warmed up")
    else:
        log.warning("Ollama warm-up failed — will load lazily on first chat request instead")
    yield


app = FastAPI(title="CCMC Sales Chatbot API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://172.16.8.193:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
