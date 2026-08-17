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
    """Best-effort: a live sanity check against the OpenAI API at boot, so
    a bad key / no quota / network issue shows up in the startup log
    instead of silently degrading every chat message to the rule-based
    fallback until someone notices. Purely a diagnostic — the whole LLM
    layer is already fail-soft (call_llm_json() itself never raises
    regardless), so a failure here must never block startup."""
    from app.core.logger import get_logger
    from app.llm.llm_client import call_llm_json
    from app.llm import entity_extractor  # noqa: F401 — import registers entity types with entity_linker
    from app.llm import entity_linker
    from app.database.session import SessionLocal

    log = get_logger("main")
    if call_llm_json('Return ONLY JSON: {"ok": true}') is not None:
        log.info("OpenAI reachable")
    else:
        log.warning("OpenAI unreachable at startup (bad key, no quota, or network) — will retry lazily on first chat request")

    # ---- embeddings: probe ONCE, then live with the answer ----
    # Graceful degradation: embeddings are an optional widening tier
    # behind exact and fuzzy matching, so an unusable provider costs a
    # little recall on paraphrased queries and nothing else. Probing once
    # here (rather than discovering it per-query, forever) is what keeps
    # a bad key from becoming a doomed API round trip on every message.
    # Never raises, never blocks startup — the chatbot must come up and
    # serve requests regardless.
    from app.llm import embeddings

    status = embeddings.probe()
    if status.ready:
        db = SessionLocal()
        try:
            entity_linker.build_index(db, force=True)
            log.info(f"Entity linking index built for: {entity_linker.registered_types()}")
        except Exception:
            log.warning("Entity linking index build failed — semantic search will build lazily on first query")
        finally:
            db.close()
    elif status.enabled:
        # embeddings.probe() already logged the single actionable WARNING
        # with the reason; don't repeat it here.
        log.info("Starting without semantic search — exact/fuzzy entity resolution is unaffected")

    yield


app = FastAPI(title="CCMC Sales Chatbot API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://172.16.8.38:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/health")
def health():
    """Liveness plus optional-subsystem status.

    `status` stays "ok" when embeddings are down ON PURPOSE: they are a
    widening tier behind exact/fuzzy matching, so their absence degrades
    recall on paraphrased queries but does not make the service
    unhealthy. Reporting "degraded" here would page someone for a
    condition the chatbot is designed to absorb — the `embeddings` block
    is what to alert on instead, and `reason` is a stable slug
    ("insufficient_quota", "invalid_api_key", ...) so a rule can match it
    without parsing prose."""
    from app.llm import embeddings

    return {"status": "ok", "embeddings": embeddings.status().to_dict()}


@app.post("/health/embeddings/rebuild")
def rebuild_embeddings():
    """Operator-triggered retry — the only way back to enabled short of a
    restart, by design (see app/llm/embeddings.py on why there is no
    timer). Re-probes the provider; indexes rebuild lazily on next use."""
    from app.llm import embeddings

    return {"embeddings": embeddings.rebuild().to_dict()}
