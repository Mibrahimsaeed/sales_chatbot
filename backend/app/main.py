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


import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router


# ---------------------------------------------------------------------
# OPTIONAL SUBSYSTEM WARM-UP
# ---------------------------------------------------------------------
#
# NOTHING BELOW MAY DELAY READINESS. All three of these operations are
# blocking NETWORK calls, and every one of them used to run before the
# lifespan's `yield`:
#
#   call_llm_json(...)          a diagnostic ping at the chat model
#   embeddings.probe()          a diagnostic ping at the embedding model
#   entity_linker.build_index() ~930 gazetteer strings, embedded in ten
#                               calls, one per registered entity type
#
# So the server printed "Waiting for application startup." and sat there:
# uvicorn does not print "Application startup complete." until the
# lifespan reaches its yield. Worse, they were synchronous calls inside an
# ASYNC function, so they blocked the event loop itself — the process
# could not have served /health even if readiness had been signalled.
#
# None of it is required to answer a request. The LLM layer is fail-soft
# (call_llm_json never raises), the probes are diagnostics, and the entity
# index ALREADY builds lazily on first use with its own TTL — see
# entity_linker.semantic_candidates, which calls build_index() itself. The
# startup build was a warm-up, never a dependency.
#
# So the work still happens, in the same order, with the same logging —
# just after readiness, on a worker thread, where being slow costs
# nothing. asyncio.to_thread is what keeps it off the event loop; an
# asyncio.create_task alone would have moved the blocking call without
# unblocking anything.


def _probe_llm() -> None:
    """Diagnostic only: surface a bad key / no quota / unreachable host in
    the startup log rather than letting every chat message silently
    degrade to the deterministic planner until somebody notices."""
    from app.core.config import settings
    from app.core.logger import get_logger
    from app.llm import llm_client
    from app.llm.llm_client import call_llm_json

    log = get_logger("main")
    if call_llm_json('Return ONLY JSON: {"ok": true}') is not None:
        log.info("LLM reachable (%s / %s)", llm_client.PROVIDER, settings.openai_model)
    else:
        log.warning(
            "LLM unreachable at startup (%s / %s — is OPENAI_API_KEY set, and does "
            "the key have access to that model?) — will retry lazily on the first "
            "chat request; until then queries answer on the deterministic planner",
            llm_client.PROVIDER, settings.openai_model,
        )


def _warm_entity_index() -> None:
    """Probe embeddings once, then pre-build the entity index.

    Probing once (rather than discovering the answer per-query, forever)
    is what keeps a bad key from becoming a doomed round trip on every
    message. The index build below is a WARM-UP: skipping it costs the
    first semantic query the build time, nothing else.
    """
    from app.core.logger import get_logger
    from app.database.session import SessionLocal
    from app.llm import embeddings, entity_linker
    from app.llm import entity_extractor  # noqa: F401 — registers entity types

    log = get_logger("main")
    status = embeddings.probe()
    if not status.ready:
        if status.enabled:
            # embeddings.probe() already logged the single actionable
            # WARNING with the reason; don't repeat it here.
            log.info("Starting without semantic search — exact/fuzzy entity "
                     "resolution is unaffected")
        return

    db = SessionLocal()
    try:
        entity_linker.build_index(db, force=True)
        log.info("Entity linking index built for: %s", entity_linker.registered_types())
    except Exception:
        log.warning("Entity linking index build failed — semantic search will "
                    "build lazily on first query")
    finally:
        db.close()


def _start_warm_up() -> list[threading.Thread]:
    """Start the optional work on DAEMON THREADS and return immediately.

    Threads rather than asyncio tasks, for two reasons:

    - These are blocking, synchronous network calls. An asyncio task would
      have moved them without unblocking anything: they would still have
      run on the event loop and still have stalled every request.
    - Daemon threads do not have to be joined. asyncio.to_thread cannot be
      interrupted, so a cancelled task still waits for its thread to
      finish — measured, that made SHUTDOWN block for the full length of
      the warm-up, so a slow probe became an equally slow Ctrl-C.
      A daemon thread is abandoned at exit, which is exactly the contract
      best-effort warm-up wants.

    Concurrent, not sequential: the probe is slow, and the index warm-up
    should not queue behind it.

    NO DEADLINE, deliberately. The probe is not only a diagnostic — it is
    also what pays the COLD START, and that cost is UNPREDICTABLE rather
    than merely large: the first call in a fresh process was measured at
    123s, 126s and 2.1s on three separate runs, against 2.7s for a second
    call in the same process. Cutting the probe short would not save the
    slow case, only move it onto the first user's question — and no fixed
    deadline can tell the two apart in advance. That variance is the
    reason this must not be on the startup path at all.
    """
    from app.core.logger import get_logger

    log = get_logger("main")

    def guarded(name, run):
        def wrapped():
            try:
                run()
            except Exception:
                log.warning("Optional startup step %r failed — continuing",
                            name, exc_info=True)
        return wrapped

    threads = [
        threading.Thread(target=guarded("LLM probe", _probe_llm),
                         name="warmup-llm-probe", daemon=True),
        threading.Thread(target=guarded("entity index", _warm_entity_index),
                         name="warmup-entity-index", daemon=True),
    ]
    for thread in threads:
        thread.start()
    return threads


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Ready immediately; warm optional subsystems in the background.

    The `yield` is reached with no network call made and nothing awaited,
    so uvicorn prints "Application startup complete." at once and serves
    requests while the warm-up is still running. Shutdown does not join
    the threads — they are daemons precisely so that stopping the server
    never waits on a probe that is mid-timeout.
    """
    app.state.warmup_threads = _start_warm_up()
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
