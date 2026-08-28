"""
Ollama runtime benchmark — Task 4, Phases 10 and 11.

WHAT THIS IS FOR. Telemetry (llm_client._log_llm_call) makes ONE call
measurable. This makes a CONFIGURATION measurable, which is a different
question: a single request cannot separate model load from prompt prefill
from generation, because the first request of a session pays all three and
reports them as one number. So every figure here is a median over repeats,
with a discarded warm-up, and cold is measured separately and deliberately.

WHAT IT DELIBERATELY DOES NOT DO. It does not run in the unit suite, it
never starts Ollama, and it makes no claim when Ollama is absent. There is
no offline "estimate" mode: an invented latency figure is worse than no
figure, because it will be quoted later as if it were measured.

RUN IT:
    python -m app.llm.benchmark_ollama              # baseline config only
    python -m app.llm.benchmark_ollama --experiments  # Phase 11 A/B/C matrix

Reads the same settings and the same provider seam the application uses,
so what it measures is what production does.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import httpx

from app.core.config import settings

# The representative queries from the task, spanning the shapes that
# produce measurably different prompts: a resolved single metric, a
# scoped lookup, a comparison, a compound numeric filter, an unresolved
# business phrase (which retires nothing and sends the FULL catalog), a
# multi-metric query, and a novel phrase with no alias at all.
BENCH_QUERIES: list[tuple[str, str]] = [
    ("single_metric", "top 5 advisors by connects"),
    ("scoped_lookup", "show revenue for Blue Area"),
    ("comparison", "compare Blue Area and Downtown by connects"),
    ("compound_filter", "show advisors with connects above 100 and revenue below 5 million"),
    ("unresolved_phrase", "which advisors are struggling?"),
    ("multi_metric", "show me connects, revenue and attendance for Blue Area advisors"),
    ("novel_phrase", "which advisors have the best pipeline velocity"),
]


def ollama_is_up(timeout: float = 2.0) -> bool:
    """One cheap probe against the configured host. Any failure means
    'not available' — the caller's job is to skip, never to fabricate."""
    try:
        return httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def model_is_present(model: str | None = None) -> bool:
    """Whether the CONFIGURED model is actually pulled. Benchmarking a
    model Ollama has to download first measures the network."""
    model = model or settings.ollama_model
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        names = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:
        return False
    return model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)


def unload_model(model: str | None = None) -> bool:
    """Evict the model so the NEXT call is genuinely cold.

    keep_alive=0 is Ollama's documented 'unload now'. Without this, 'cold'
    can only be measured once per machine boot, which is not a
    measurement — it is an anecdote.
    """
    model = model or settings.ollama_model
    try:
        httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30.0,
        )
        return True
    except Exception:
        return False


@dataclass
class CallSample:
    """One call's telemetry, taken from the provider's own metadata."""

    wall_ms: float
    total_ms: float | None = None
    load_ms: float | None = None
    prompt_tokens: int | None = None
    prompt_eval_ms: float | None = None
    output_tokens: int | None = None
    eval_ms: float | None = None
    ok: bool = True
    error: str | None = None


@dataclass
class Series:
    label: str
    samples: list[CallSample] = field(default_factory=list)

    def _vals(self, attr):
        return [v for v in (getattr(s, attr) for s in self.samples if s.ok) if v is not None]

    def median(self, attr):
        vals = self._vals(attr)
        return round(statistics.median(vals), 1) if vals else None

    def p95(self, attr):
        """Reported only with enough samples to mean anything — the p95 of
        three requests is just the maximum wearing a percentile's name."""
        vals = sorted(self._vals(attr))
        if len(vals) < 5:
            return None
        return round(vals[min(int(0.95 * len(vals)), len(vals) - 1)], 1)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "n": len(self.samples),
            "failures": sum(1 for s in self.samples if not s.ok),
            "median_total_ms": self.median("total_ms"),
            "p95_total_ms": self.p95("total_ms"),
            "median_load_ms": self.median("load_ms"),
            "median_prompt_eval_ms": self.median("prompt_eval_ms"),
            "median_eval_ms": self.median("eval_ms"),
            "median_prompt_tokens": self.median("prompt_tokens"),
            "median_output_tokens": self.median("output_tokens"),
        }


@contextmanager
def override(**kw):
    """Temporarily rebind settings the SEAM reads.

    The experiment matrix has to vary the same fields production reads, or
    it measures something other than production. Restored unconditionally
    so a failed experiment cannot leak configuration into later ones.
    """
    old = {k: getattr(settings, k) for k in kw}
    for k, v in kw.items():
        object.__setattr__(settings, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            object.__setattr__(settings, k, v)


def _sample_from(response, wall_ms: float) -> CallSample:
    from app.llm.llm_client import _ns_to_ms

    return CallSample(
        wall_ms=wall_ms,
        total_ms=_ns_to_ms(getattr(response, "total_duration", None)),
        load_ms=_ns_to_ms(getattr(response, "load_duration", None)),
        prompt_tokens=getattr(response, "prompt_eval_count", None),
        prompt_eval_ms=_ns_to_ms(getattr(response, "prompt_eval_duration", None)),
        output_tokens=getattr(response, "eval_count", None),
        eval_ms=_ns_to_ms(getattr(response, "eval_duration", None)),
    )


def run_one(prompt: str, schema) -> CallSample:
    """One structured call through the real provider seam."""
    from app.llm.llm_client import _chat

    started = time.perf_counter()
    try:
        response = _chat(
            messages=[{"role": "user", "content": prompt}],
            fmt=schema,
            purpose="benchmark:structured",
        )
    except Exception as exc:
        return CallSample(
            wall_ms=round((time.perf_counter() - started) * 1000, 1),
            ok=False,
            error=type(exc).__name__,
        )
    return _sample_from(response, round((time.perf_counter() - started) * 1000, 1))


def _prompts() -> list[tuple[str, str]]:
    """Real prompts from the real builder — a synthetic string would
    measure a prompt size the application never sends."""
    from app.database.session import SessionLocal
    from app.llm.entity_extractor import extract_entities
    from app.llm.prompt_builder import build_ir_prompt
    from app.llm.semantic_parser import (
        get_known_bcms,
        get_known_companies,
        get_known_teams,
        get_known_unit_heads,
        get_known_zonal_heads,
    )

    db = SessionLocal()
    try:
        teams, companies = get_known_teams(db), get_known_companies(db)
        kw = dict(
            known_unit_heads=get_known_unit_heads(db),
            known_zonal_heads=get_known_zonal_heads(db),
            known_bcms=get_known_bcms(db),
        )
        return [
            (name, build_ir_prompt(q, teams, companies,
                                   grounded_entities=extract_entities(q, db), **kw))
            for name, q in BENCH_QUERIES
        ]
    finally:
        db.close()


def measure(label: str, repeats: int = 3, include_cold: bool = True) -> dict:
    """Cold and warm, measured apart.

    Cold is a SINGLE call after an explicit unload — it is the model-load
    figure, and averaging it with warm calls is what makes 'the model is
    slow' unfalsifiable. Warm discards one warm-up call, then repeats.
    """
    from app.llm.llm_client import QUERY_IR_JSON_SCHEMA

    prompts = _prompts()
    result: dict = {"label": label, "model": settings.ollama_model,
                    "config": {"think": settings.ollama_think,
                               "keep_alive": settings.ollama_keep_alive,
                               "num_ctx": settings.ollama_num_ctx,
                               "num_predict": settings.ollama_num_predict,
                               "temperature": settings.ollama_temperature}}

    if include_cold:
        unload_model()
        cold = Series("cold")
        cold.samples.append(run_one(prompts[0][1], QUERY_IR_JSON_SCHEMA))
        result["cold"] = cold.summary()

    run_one(prompts[0][1], QUERY_IR_JSON_SCHEMA)  # warm-up, discarded

    per_query = {}
    overall = Series("warm")
    for name, prompt in prompts:
        s = Series(name)
        for _ in range(repeats):
            sample = run_one(prompt, QUERY_IR_JSON_SCHEMA)
            s.samples.append(sample)
            overall.samples.append(sample)
        per_query[name] = s.summary()
    result["warm"] = overall.summary()
    result["per_query"] = per_query
    return result


EXPERIMENTS = {
    "baseline": {},
    "A_think_off": {"ollama_think": False},
    "B_think_off_keepalive": {"ollama_think": False, "ollama_keep_alive": "30m"},
    "C_full": {"ollama_think": False, "ollama_keep_alive": "30m",
               "ollama_num_ctx": 16384, "ollama_num_predict": 768},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--experiments", action="store_true",
                    help="run the Phase 11 configuration matrix")
    ap.add_argument("--json", help="write raw results here")
    args = ap.parse_args()

    if not ollama_is_up():
        print(f"Ollama is not reachable at {settings.ollama_base_url} — nothing measured.")
        print("No latency figures are produced when the provider is absent.")
        return 2
    if not model_is_present():
        print(f"Model {settings.ollama_model} is not pulled — nothing measured.")
        return 2

    results = []
    if args.experiments:
        for label, cfg in EXPERIMENTS.items():
            with override(**cfg):
                results.append(measure(label, repeats=args.repeats))
    else:
        results.append(measure("baseline", repeats=args.repeats))

    for r in results:
        print(f"\n=== {r['label']}  model={r['model']}  {r['config']}")
        if "cold" in r:
            print("  cold:", r["cold"])
        print("  warm:", r["warm"])
        for name, s in r["per_query"].items():
            print(f"    {name:<18} total={s['median_total_ms']} "
                  f"load={s['median_load_ms']} prefill={s['median_prompt_eval_ms']} "
                  f"gen={s['median_eval_ms']} ptok={s['median_prompt_tokens']} "
                  f"otok={s['median_output_tokens']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
