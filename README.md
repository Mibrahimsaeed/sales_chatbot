# sales_chatbot



alembic upgrade head
python -m etl.sync
uvicorn app.main:app --reload                  




CREATE TABLE advisors (
    sap_id           TEXT PRIMARY KEY,
    wid              TEXT UNIQUE,
    
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



 percentage sign
 hr bndy ka alag sy info jb kch specific manga jaye
 aur rel/teams nh bny huy
 air jo tha bana hua woh kr de 

 

End-to-End Parity Audit
Executive summary
I ran 212 realistic queries across 9 categories through the full pipeline against a fixture with hand-computable values, then audited all 15 checkpoints per KPI.

The computation layer is sound. All 27 aggregation values matched hand calculations exactly — every funnel ratio, every roll-up, every YTD sibling. Ranking polarity is correct (top-of-overdue is the lowest, bottom-of-overdue the highest). All 7 threshold rules match the spec.

The layers around it were not. Six defects, three of them crashes on ordinary questions:

Defect	Layer	Severity
D1	Leaderboard reply raises TypeError when any row has a NULL value	Response generation	Critical
D2	advisor_service ignores join_models → cartesian → MultipleResultsFound	SQL generation	Critical
D3	Same function self-joins Advisor → "ambiguous column name"	SQL generation	Critical
D4	achievement_pct: 99% or 84.7% for the same person	Metric ontology	Critical
D5	overdue/overdue_amount — one column, two labels, different units	Metric ontology	Critical
D6	Every percentage narrated as target attainment	Response generation	High
D7	Trend questions answered with a current-state leaderboard	NLU	High — not fixed
D8	Leaderboard omitted units (90 for 90%)	Response generation	Medium — fixed with D1
D9	Displayed values unrounded vs dashboard's round()	Response generation	Medium — not fixed
D10	No tie-break in ORDER BY	SQL generation	Medium — not fixed
Root causes
D1 — aggregation.value_expression documents that a zero denominator "compiles to NULL, which the callers already render as no data". No caller did. One advisor with no recorded attendance days failed the entire reply. 26 of the first sweep's exceptions were a fixture artefact; these 3 were real.

D2/D3 — There are three query builders reading ColumnBinding: the compiler, the aggregation engine, and advisor_service.get_advisor_metric. I taught the first two about join_models and the Advisor root in earlier phases and missed the third. Both new bindings (connect_to_cr_rate, one_unit_ratio) crashed on "what is X's …".

D4 — The advisor binding read the sheet's precomputed pct column while the group binding computed cleared/target. Wherever the sheet disagreed with its own components, the same advisor had two answers. The spec computes, so computing is also the parity-correct side.

D5 — Both metrics bound Pipeline.overdue ← the single "Total Overdue" column, labelled "MTD Overdue Pipeline" and "MTD Overdue Amount". A count and an amount are different quantities; one label was always wrong.

D6 — The narrative applied "has achieved X% of the assigned target … remaining Y% short of the monthly goal" to every percentage. A 1-Unit ratio's denominator is team size; an attendance rate's is recorded days. Neither has a target.

Files changed
File	Change
response_formatter.py	format_metric_value() — NULL-safe + unit-aware; leaderboard uses it
advisor_service.py	honours join_models; skips the Advisor self-join
metric_ontology.py	achievement_pct computes; overdue_amount removed; measures_target_attainment
metric_aliases.py	overdue phrasings folded into one metric
narrative.py	target language gated on the new flag
test_parity_audit_defects.py	new, 38 tests
3 existing test files	fixtures/assertions updated with provenance
Tests added
38, weighted toward properties over instances:

test_no_metric_crashes_a_leaderboard_at_any_level — every metric × 2 levels through the real formatter.
test_every_metric_is_answerable_for_one_advisor — catches a fourth query builder appearing.
test_the_advisor_path_agrees_with_the_aggregation_engine — all three builders, one answer.
test_no_two_metrics_share_a_binding_expression — generalises D5 (period is part of the identity, so mtd_cleared/ytd_cleared are correctly distinct).
test_only_achievement_claims_a_target — every other metric asserted not to.
Two of my own test errors surfaced and were corrected: a wrong arithmetic expectation (25% not 50%) and a signature that ignored period.

Dashboard parity score
Weighted by KPI, 61%.

✅ Full parity (10 of 17): Conversion, Portfolio, Pipeline, Overdue, Connect→CR, CR→Meeting, Meeting→Conversion, Performance, 1 Unit, Meetings Conducted
⚠️ Partial (3): Biometric & WorksApp Login (denominator is recorded-days, spec says team-size × working-days); displayed values unrounded
❌ Blocked (4): Connect %, CR %, Meetings %, KPI Overview — all on workingDays, verified absent from all 24 sheet tabs
By period: MTD 100%, YTD ~50% (21 metrics), Daily 0%, custom ranges 0%.

Remaining limitations
D7 not fixed. Trend queries return a snapshot leaderboard. compute_trends exists but only augments a reply and needs AdvisorHistory snapshots. Fixing it properly is a design fork — refuse, or answer with a caveat — and I didn't want to remove the partial capability on my own judgement.
D9/D10 not fixed. Rounding displayed values and adding a tie-break are both broad-blast-radius behaviour changes; status colours already round correctly.
Spec's target=0 & cleared>0 → 100 rule still unimplemented (returns "no data").
Which unit Total Overdue holds is undeterminable from the repo; the surviving label follows the spec ("count").
Production readiness
Before this audit: not ready. Three crash paths were reachable from "what is X's connect to cr", "top advisors by attendance rate" and "what is X's 1 unit ratio" — all plain phrasings, all HTTP 500s.

Now: ready for a monitored rollout, with two caveats. The remaining failure modes are honest refusals (working-day KPIs, daily, custom ranges) rather than wrong answers — with the one exception of D7, which is the highest-value next fix. And the 3 long-standing suite failures have been red across every phase; I'd want those understood before calling the baseline green.

hehehehhehe







1–2. Audit: intent proposers and their suppression
Scorer	Proposed	Suppressed on
_score_advisor_metric	advisor_metric 0.80	rival evidence: ranking word, comparison phrase, reverse, relation ← P1
_score_comparison	comparison 1.51	required a comparison phrase ← P2
_score_roster	roster 0.95	ignored whether a measure was named ← P3
(none)	group_metric	did not exist ← P4
Scorers did two jobs — propose and suppress. "Top revenue for Omar Farooq" produced exactly one candidate, so the ranking was never a contest: a candidate that is never proposed cannot lose, and cannot be explained.

3–5. Implemented
New intent_precedence.py — Evidence (structured facts, read from the components that already own them), an explicit PRECEDENCE table, and rank().


comparison     >  two or more grounded subjects with a measure
advisor_metric >  one named person with a measure
group_metric   >  one named group with a measure
leaderboard    >  a measure with no subject, or a ranking inside one
roster         >  "who is in X", no measure, no ranking
profile        >  a named person with no measure
The one asymmetry — a ranking word suppresses group_metric but not advisor_metric — is principled, not a special case: an advisor is a leaf (nothing inside to rank); a group contains members. That's the same reading subject_level.decide() already applies, asked rather than restated.

group_metric is a first-class action, not an inferred one. Scorers now only propose; rank() decides; scores break ties within a tier. plan.action = "leaderboard" in the context-inheritance path is gone — it re-plans with the inherited measure so the planner stays the owner.

6–7. Before vs after
Before	After
Phase 6 corpus	28/36 (78%)	36/36 (100%)
P1 "Top revenue for Omar Farooq"	Yasir Ali's leaderboard	Omar Farooq's figure
P2 "Blue Area and Downtown revenue"	Blue Area only	both compared
P3 "advisors in Blue Area by revenue"	alphabetical roster	ranked leaderboard
P4 "Blue Area revenue"	leaderboard (compensated downstream)	group_metric
Comparison corpus	22/22	22/22
Behavioral corpus	0 exceptions	0 exceptions
Tests	2700	2745 (+45)
8. Report
Three of my own errors, caught by the suite. My first table promoted advisor_profile over reverse_hierarchy for every manager lookup — 155 failures. The table now governs only the subject/measure family; specialised intents keep their scored behaviour. My roster rule then demoted a 0.98 leaderboard because the metric was defaulted rather than named. And I conflated ambiguity with multiplicity — "top 20 advisors" matched the seeded person "Advisor 20", and counting that as a named subject beat a legitimate leaderboard; ambiguity is now its own evidence field, since treating it as absent wrongly triggers the no-subject rule.

A closed capability gap. KNOWN_DIVERGENCES for "Attendance of Agency21" read "the shape that would actually answer it — ONE metric for ONE entity — doesn't exist yet in either planner." Phase 7 created it; the rule planner is now the more correct of the two, reversing the note. Recorded in place rather than deleted.

Remaining debt: two post-selection intent writers survive and are legitimate — the pending-slot promotion (clarify → executable once the slot is filled) and conversation_context inheriting a prior comparison's intent. The metric-inheritance path now runs build_query_plan a second time, which trades Phase 4's one-run invariant for single ownership of intent; it fires only on subject-only follow-ups. QueryIR.intent still has no group_metric member — the classification is first-class at the planner, the IR still compiles it as a scoped leaderboard, and changing that means a schema and prompt change.









ab ye krna hai k jb kisi ka puchy tou options nh dey wo, agr tou herarichy mn jo evel hoga usi k mutabiq woi ans dega hirarechy smjhani hai usko phly mtlb ik bnda jo zonal hai tou wo advispr bhi hoga tou uska puchna nh hai okay 


secondly, if asked about the connects if team of oerson A it fails


thirst, to give answered calls, connects and ans called% all 3 things collectively






3 things
connects as whole w ans call and %
complex querys
bcm who have 0 team size
etc
