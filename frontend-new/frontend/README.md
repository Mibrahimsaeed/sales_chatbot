# CCMC Sales Assistant — Frontend

React + Vite chat widget. Talks to the FastAPI backend's `/api/chat` endpoint
for everything — no data is embedded in this app.

## Setup

```bash
npm install
cp .env.example .env   # point VITE_API_URL at your running backend
npm run dev
```

Requires the backend running (`uvicorn app.main:app --reload`) and synced
(`python -m etl.sync`) — this app renders whatever the API returns and has
no fallback data of its own.

## Structure

```
src/
├── api/client.js          # fetch wrapper for POST /api/chat (+ dev token)
├── hooks/useChat.js        # message history + send()
├── components/
│   ├── ChatPanel.jsx        # top-level layout
│   ├── Header.jsx
│   ├── SuggestionChips.jsx  # example prompts only
│   ├── Composer.jsx
│   ├── MessageBubble.jsx    # picks a card component by response.type
│   └── cards/
│       ├── AdvisorCard.jsx
│       ├── TeamCard.jsx
│       ├── CompanyCard.jsx
│       ├── LeaderboardCard.jsx
│       └── AttendanceCard.jsx   # new — wasn't reachable in the old static prototype
├── utils/format.js          # fmtNum / fmtPKR / fmtPct / formatMetricValue
└── styles/
    ├── theme.css             # design tokens + page-level layout (.app, .header, .thread, .composer)
    └── components.css        # reusable pieces (.card, .chip, .bubble, .kpi, .pill, .leader)
```

## Before going to production

`src/api/client.js` currently calls the backend's dev-only `POST /api/token`
endpoint (`app/api/auth_dev.py`) to get a test JWT. Replace `getCurrentToken()`
with whatever token the real dashboard already holds once this widget is
embedded — and delete `auth_dev.py` on the backend at the same time.
