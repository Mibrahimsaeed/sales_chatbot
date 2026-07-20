"""NL benchmark runner (WI-10). Runs every YAML case in cases/ through
nlu_pipeline.resolve() against the deterministic fixture dataset and
scores four dimensions:

- intent:   resolution label == expect.intent
- entities: set-F1 over extracted teams+companies (+ advisor when expected)
- ir:       fragment-subset match on expect.ir (metric, level, limit,
            sort_direction, filters-as-subset)
- sql:      compile_and_run rows vs expect.sql (row_count / first_row)

Each dimension is scored only when the case declares it. Requires a real
OPENAI_API_KEY (this measures the actual LLM parser) — run explicitly:

    pytest tests/benchmark -m benchmark -s

Prints per-tag and per-dimension aggregates and writes
benchmark_report.json next to this file. The test only hard-fails on
harness errors, not on model misses — it's a measurement, not a gate.
"""

import json
import pathlib
import uuid

import pytest

from app.core.config import settings
from app.llm import conversation_memory, entity_extractor, nlu_pipeline
from app.llm.query_compiler import compile_and_run

from tests.benchmark import fixture_data

CASES_DIR = pathlib.Path(__file__).parent / "cases"
REPORT_PATH = pathlib.Path(__file__).parent / "benchmark_report.json"

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not settings.openai_api_key, reason="benchmark needs a real OPENAI_API_KEY"
    ),
]


def load_cases() -> list[dict]:
    import yaml

    cases = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        with open(path) as f:
            cases.extend(yaml.safe_load(f) or [])
    return cases


def _resolution_label(resolution) -> str:
    if resolution.kind == "shortcut":
        return resolution.shortcut_intent
    if resolution.kind == "plan":
        return resolution.plan.action
    if resolution.kind == "ir":
        return resolution.ir.intent
    return "clarify"


def _set_f1(expected: set, actual: set) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    tp = len(expected & actual)
    precision = tp / len(actual)
    recall = tp / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_entities(expect: dict, entities: dict) -> float:
    scores = []
    if "teams" in expect:
        scores.append(_set_f1(set(expect["teams"]), set(entities.get("teams", []))))
    if "companies" in expect:
        scores.append(_set_f1(set(expect["companies"]), set(entities.get("companies", []))))
    if "advisor" in expect:
        scores.append(1.0 if entities.get("advisor_name") == expect["advisor"] else 0.0)
    return sum(scores) / len(scores) if scores else 1.0


def _score_ir(expect: dict, ir) -> float:
    if ir is None:
        return 0.0
    checks = []
    if "metric" in expect:
        actual = ir.sort.metric or (ir.metric.key if ir.metric else None)
        checks.append(actual == expect["metric"])
    if "subject_level" in expect:
        checks.append(ir.subject_level == expect["subject_level"])
    if "limit" in expect:
        checks.append(ir.limit == expect["limit"])
    if "sort_direction" in expect:
        checks.append(ir.sort.direction == expect["sort_direction"])
    for expected_filter in expect.get("filters", []):
        checks.append(any(
            f.field == expected_filter.get("field", f.field)
            and (str(f.value).lower() == str(expected_filter["value"]).lower()
                 if "value" in expected_filter else True)
            and (f.operator == expected_filter["operator"]
                 if "operator" in expected_filter else True)
            for f in ir.filters
        ))
    return sum(checks) / len(checks) if checks else 1.0


def _score_sql(expect: dict, rows) -> float:
    if rows is None:
        return 0.0
    checks = []
    if "row_count" in expect:
        checks.append(len(rows) == expect["row_count"])
    if "first_row" in expect:
        if rows:
            first, expected_first = rows[0], expect["first_row"]
            ok = first["name"] == expected_first["name"]
            if "value" in expected_first and first["value"] is not None:
                ok = ok and abs(float(first["value"]) - float(expected_first["value"])) < 0.05
            checks.append(ok)
        else:
            checks.append(False)
    return sum(checks) / len(checks) if checks else 1.0


def test_benchmark(db_session):
    fixture_data.seed(db_session)
    entity_extractor._cache["loaded_at"] = 0
    cases = load_cases()
    assert cases, "no benchmark cases found"

    results = []
    for case in cases:
        conversation_memory._store.clear()
        session_id = f"bench-{case['id']}-{uuid.uuid4().hex[:6]}"
        turns = case.get("turns") or [case["query"]]

        resolution = None
        for turn in turns:
            resolution = nlu_pipeline.resolve(turn, db_session, session_id=session_id)

        expect = case["expect"]
        scores: dict = {}
        if "intent" in expect:
            scores["intent"] = 1.0 if _resolution_label(resolution) == expect["intent"] else 0.0
        if "entities" in expect:
            scores["entities"] = _score_entities(expect["entities"], resolution.entities or {})
        if "ir" in expect:
            scores["ir"] = _score_ir(expect["ir"], resolution.ir)
        if "sql" in expect:
            rows = compile_and_run(db_session, resolution.ir) if resolution.ir is not None else None
            scores["sql"] = _score_sql(expect["sql"], rows)

        results.append({"id": case["id"], "tags": case.get("tags", []), "scores": scores})

    # ---- aggregate + report ----
    dims = ("intent", "entities", "ir", "sql")
    per_dim = {
        d: [r["scores"][d] for r in results if d in r["scores"]] for d in dims
    }
    all_tags = sorted({t for r in results for t in r["tags"]})
    per_tag = {}
    for tag in all_tags:
        tagged = [s for r in results if tag in r["tags"] for s in r["scores"].values()]
        per_tag[tag] = round(sum(tagged) / len(tagged), 3) if tagged else None

    summary = {
        "cases": len(results),
        "per_dimension": {
            d: {"n": len(v), "score": round(sum(v) / len(v), 3) if v else None}
            for d, v in per_dim.items()
        },
        "per_tag": per_tag,
        "worst_cases": sorted(
            (
                {"id": r["id"], "avg": round(sum(r["scores"].values()) / len(r["scores"]), 3)}
                for r in results if r["scores"]
            ),
            key=lambda x: x["avg"],
        )[:10],
    }
    REPORT_PATH.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n===== NL BENCHMARK =====")
    print(f"cases: {summary['cases']}")
    for dim, agg in summary["per_dimension"].items():
        print(f"  {dim:<9} n={agg['n']:<4} score={agg['score']}")
    print("per tag:")
    for tag, score in per_tag.items():
        print(f"  {tag:<12} {score}")
    print(f"worst: {summary['worst_cases'][:5]}")
    print(f"full report: {REPORT_PATH}")
