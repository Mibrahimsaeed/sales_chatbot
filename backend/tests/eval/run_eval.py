"""Run the complex-query evaluation and attribute every failure to a stage.

    python -m tests.eval.run_eval

Read-only against the configured database. Modifies nothing.

FIVE STAGES PER CASE, scored separately, because a single pass/fail on
the final answer cannot say WHERE meaning was lost:

    1 meaning     did the query reach a plan/IR that means what was asked
    2 queryir     can the IR even REPRESENT it (oracle buildable?)
    3 validation  does validate_ir preserve the oracle's meaning
    4 sql         does the compiler build and execute it
    5 answer      does the result match independently computed truth

Stage 1 is judged on the LIVE run; 2-5 on the ORACLE run. That split is
the point: it separates "the parser never produced this" from "nothing
downstream could have executed it".
"""

from __future__ import annotations

import sys
import traceback

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.llm import nlu_pipeline, operations
from app.llm.ir_validator import validate_ir
from app.llm.query_compiler import compile_and_run
from app.services import chat_service
from tests.eval.complex_cases import CASES, CHAINS

OK, FAIL, NA = "PASS", "FAIL", "n/a"


def _truth(engine, case):
    if not case.truth_sql:
        return None
    with engine.connect() as c:
        return c.execute(text(case.truth_sql), case.truth_params).scalar()


def _live(db, query):
    """Stage 1: what the running system does with this query today."""
    try:
        resolution = nlu_pipeline.resolve(query, db, session_id=None)
        response = chat_service._dispatch(db, resolution)
        return resolution, response, None
    except Exception as exc:  # noqa: BLE001 - the harness reports, never raises
        return None, None, f"{type(exc).__name__}: {exc}"


def _oracle(db, case):
    """Stages 2-5 on a hand-built, correct IR."""
    result = {"queryir": NA, "validation": NA, "sql": NA, "answer": NA,
              "detail": "", "rows": None}

    if case.oracle is None:
        result["queryir"] = FAIL
        result["detail"] = "no IR shape expresses this question"
        return result
    result["queryir"] = OK

    ir = case.oracle.model_copy(deep=True)
    before = ([f.field for f in ir.filter_leaves()], ir.metric_keys(),
              len(ir.subjects), ir.filter_tree is not None)
    try:
        validate_ir(ir, db)
    except Exception as exc:  # noqa: BLE001
        result["validation"] = FAIL
        result["detail"] = f"validator raised {type(exc).__name__}: {exc}"
        return result

    after = ([f.field for f in ir.filter_leaves()], ir.metric_keys(),
             len(ir.subjects), ir.filter_tree is not None)
    if before != after:
        result["validation"] = FAIL
        result["detail"] = f"validation changed the IR: {before} -> {after}"
        return result
    result["validation"] = OK

    try:
        rows = compile_and_run(db, ir)
    except Exception as exc:  # noqa: BLE001
        result["sql"] = FAIL
        result["detail"] = f"compiler raised {type(exc).__name__}: {exc}"
        return result
    if rows is None:
        result["sql"] = FAIL
        result["detail"] = "compiler returned None (unanswerable)"
        return result
    result["sql"] = OK
    result["rows"] = rows
    return result


def _score_answer(case, oracle_result, truth):
    if oracle_result["sql"] != OK:
        return NA, ""
    rows = oracle_result["rows"]
    if case.rows_to_answer is None and truth is None:
        return NA, "no answer assertion declared"
    got = case.rows_to_answer(rows) if case.rows_to_answer else len(rows)
    if truth is None:
        return OK, f"produced {got}"
    if isinstance(got, list):
        return (OK if len(got) == truth else FAIL), f"got {len(got)} want {truth}"
    return (OK if got == truth else FAIL), f"got {got} want {truth}"


def _attribute(stages, live_ok):
    """Which layer lost the meaning."""
    if stages["queryir"] == FAIL:
        return "QueryIR — no representation"
    if stages["validation"] == FAIL:
        return "Validation"
    if stages["sql"] == FAIL:
        return "Compiler"
    if stages["answer"] == FAIL:
        return "Business logic"
    if not live_ok:
        return "LLM / routing"
    return ""


def _silent_wrong(rows_out):
    return [r for r in rows_out
            if r["case"].live_expectation == "refuse" and r["live"] == FAIL]


def main() -> int:
    engine = create_engine(settings.database_url)
    db = sessionmaker(bind=engine)()
    rows_out = []

    for case in CASES:
        truth = _truth(engine, case)
        resolution, response, live_err = _live(db, case.query)
        oracle_result = _oracle(db, case)
        answer, answer_detail = _score_answer(case, oracle_result, truth)
        oracle_result["answer"] = answer

        # Stage 1: did the LIVE run reach something that MEANS the query?
        # A clarification is an honest refusal, not an understanding.
        refused = live_err is None and resolution.kind == "clarify"
        if live_err:
            live, live_note = FAIL, live_err
        elif case.live_expectation == "refuse":
            # Answering an inexpressible question is a SILENT WRONG
            # ANSWER, which is worse than refusing it.
            live = OK if refused else FAIL
            live_note = ("refused, correctly" if refused
                         else "ANSWERED a question it cannot express — silent wrong answer")
        elif refused:
            live, live_note = FAIL, "refused / clarified"
        else:
            live, live_note = OK, f"{resolution.kind}"

        rows_out.append({
            "case": case,
            "live": live, "live_note": live_note,
            **{k: oracle_result[k] for k in ("queryir", "validation", "sql", "answer")},
            "detail": oracle_result["detail"] or answer_detail,
            "blame": _attribute(oracle_result, live == OK),
        })

    _report(rows_out, db)
    db.rollback()
    db.close()
    return 0


def _chain_results(db):
    out = []
    for chain in CHAINS:
        session = f"eval-{abs(hash(tuple(chain.turns)))}"
        resolution = response = None
        error = None
        try:
            for turn in chain.turns:
                resolution = nlu_pipeline.resolve(turn, db, session_id=session)
                response = chat_service._dispatch(db, resolution)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        if error:
            out.append((chain, FAIL, error))
            continue
        try:
            held = bool(chain.expect_final(resolution, response))
        except Exception as exc:  # noqa: BLE001
            held, error = False, str(exc)
        out.append((chain, OK if held else FAIL,
                    error or f"kind={resolution.kind}"))
    return out


def _report(rows_out, db):
    line = "-" * 112
    print("\n" + "=" * 112)
    print("COMPLEX QUERY UNDERSTANDING — STAGE EVALUATION")
    print("=" * 112)
    print(f"{'CATEGORY':<30}{'LIVE':<7}{'IR':<7}{'VALID':<7}{'SQL':<7}{'ANSWER':<8}BLAME")
    print(line)
    for r in rows_out:
        c = r["case"]
        print(f"{c.category:<30}{r['live']:<7}{r['queryir']:<7}{r['validation']:<7}"
              f"{r['sql']:<7}{r['answer']:<8}{r['blame']}")
        print(f"  q: {c.query}")
        if r["detail"]:
            print(f"     {r['detail']}")
        if r["live"] == FAIL:
            print(f"     live: {r['live_note']}")
    print(line)

    print("\nMULTI-TURN")
    print(line)
    for chain, verdict, note in _chain_results(db):
        print(f"{chain.category:<30}{verdict:<7}{' -> '.join(chain.turns)}")
        if verdict == FAIL:
            print(f"     {note}")

    print("\n" + "=" * 112)
    print("TOTALS")
    print("=" * 112)
    for stage in ("live", "queryir", "validation", "sql", "answer"):
        passed = sum(1 for r in rows_out if r[stage] == OK)
        failed = sum(1 for r in rows_out if r[stage] == FAIL)
        na = sum(1 for r in rows_out if r[stage] == NA)
        print(f"  {stage:<12} pass={passed:<3} fail={failed:<3} n/a={na}")

    blames: dict[str, int] = {}
    for r in rows_out:
        if r["blame"]:
            blames[r["blame"]] = blames.get(r["blame"], 0) + 1
    print("\n  failures by layer:")
    for layer, n in sorted(blames.items(), key=lambda kv: -kv[1]):
        print(f"    {layer:<28} {n}")
    silent = _silent_wrong(rows_out)
    if silent:
        print("\n  SILENT WRONG ANSWERS (answered a question no path expresses):")
        for r in silent:
            print(f"    - {r['case'].query}")
    print(f"\n  operations declared plan-only: {len(operations.PLAN_ONLY)} of "
          f"{len(operations.OPERATIONS)}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
