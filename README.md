# sales_chatbot

CREATE TABLE advisors (
    sap_id           TEXT PRIMARY KEY,
    wid              TEXT UNIQUE,
    advisor_name     TEXT,
    team             TEXT,
    company           TEXT,
    region           TEXT,
    portfolio_lead   TEXT,
    management_lead  TEXT,
    rm               TEXT,
    bm               TEXT,
    zm               TEXT,
    email            TEXT,
    last_synced_at   TIMESTAMP DEFAULT now()
);


//master recored



To make this live, here's the architecture
1. Sync layer (scheduled job, not per-request)
Pull both spreadsheets via the Google Sheets API on a schedule (every 5–15 min is plenty for MTD sales data) into a real database — Postgres or even a simple SQLite if traffic is light. Don't hit Sheets API on every chat message; it's rate-limited and slow for a chat UX.

Join key: WID (= SAP-ID/Sr.No) across every tab — this is what makes the two files relational, not just adjacent spreadsheets.
One advisors table (current snapshot) + one advisor_history table (append MTD snapshots daily) so the bot can answer trend questions like "how has X done this week" later.

2. Query layer
Your chatbot doesn't need an LLM to hit the sheet data directly — put a thin API in front of the DB with endpoints like GET /advisor?name=, GET /team/:name/summary, GET /leaderboard?metric=mtd_cleared&limit=10. The LLM (or a smaller intent classifier) maps user text → one of these calls, same pattern I used client-side in the demo above.
3. Dashboard integration
Embed the chat panel as a component in your existing dashboard (same visual language as the prototype, or restyle to match). It calls your API, not Google directly — keeps credentials off the frontend.
4. Auth/permissions — worth deciding early: should a Portfolio Lead only see their own team's advisors, or does everyone see everything? The org hierarchy (BM→ZM→RM→Portfolio Lead→Advisor) in your data supports row-level scoping if you want it.




sales-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI() instance, router mounting, CORS, startup events
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── router.py               # aggregates all routers -> included in main.py
│   │   │   ├── advisor.py
│   │   │   ├── leaderboard.py
│   │   │   ├── team.py
│   │   │   └── chat.py
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Pydantic BaseSettings, reads .env
│   │   │   ├── logger.py
│   │   │   ├── exceptions.py
│   │   │   ├── dependencies.py         # get_db(), get_current_user()
│   │   │   └── security.py             # JWT encode/decode, password/token helpers
│   │   │
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── session.py              # engine, SessionLocal, Base
│   │   │   ├── models.py               # SQLAlchemy models: Advisor, AdvisorHistory, SyncLog
│   │   │   └── schemas.py              # Pydantic response/request models
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── advisor_service.py
│   │   │   ├── leaderboard_service.py
│   │   │   ├── team_service.py
│   │   │   └── chat_service.py         # dispatches intent -> the service above
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── intent_detector.py      # rule-based regex classifier (primary path)
│   │   │   ├── prompt_builder.py       # builds the fallback prompt for Claude
│   │   │   └── llm_client.py           # wraps Anthropic API call
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── helpers.py              # number/currency formatters
│   │       └── validators.py
│   │
│   ├── etl/                            # separate deployable unit — not imported by app/
│   │   ├── __init__.py
│   │   ├── google_client.py            # service account auth + sheet fetch
│   │   ├── extract.py                  # pulls all 8 + 16 tabs as raw rows
│   │   ├── transform.py                # mapping + merge-by-WID logic
│   │   ├── load.py                     # upsert into Postgres via SQLAlchemy
│   │   ├── scheduler.py                # optional: APScheduler wrapper for local dev
│   │   └── sync.py                     # entrypoint: `python -m etl.sync`
│   │
│   ├── migrations/                     # Alembic
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   ├── alembic.ini
│   │
│   ├── requirements.txt                # or pyproject.toml if using Poetry
│   ├── .env.example
│   └── service-account.json            # gitignored — Google service account key
│
├── frontend/
│   └── (React chat widget — unchanged from earlier guide)
│
├── docs/
│   ├── data-model.md
│   └── api-spec.md
│
├── tests/
│   ├── conftest.py                     # test DB fixture, test client fixture
│   ├── api/
│   │   ├── test_advisor.py
│   │   ├── test_team.py
│   │   ├── test_leaderboard.py
│   │   └── test_chat.py
│   ├── services/
│   │   └── test_advisor_service.py
│   └── etl/
│       └── test_transform.py           # the most important tests — lock in the merge logic
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.etl
│   └── docker-compose.yml
│
├── .gitignore
└── README.md





<!-- # on time percentage(attentance- NO OF LEADS)
# worksapp ki login (PERCENTAGE ATTENDANCE AND NO OF LATES)
answered calls, meetings, CR, pipeline, target ach















 -->