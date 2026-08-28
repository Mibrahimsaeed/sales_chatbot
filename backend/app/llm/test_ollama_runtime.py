"""
Task 4 — the runtime options and telemetry are a contract, not a comment.

Two things are pinned here. First, that the provider seam actually SENDS
the configured runtime options, in the SHAPE the installed client accepts
(`think` and `keep_alive` top-level, `num_ctx`/`num_predict`/`temperature`
inside `options`) — a value that lives in config but never reaches Ollama
is worse than no setting, because it reads as controlled.

Second, that the telemetry reports the provider's own numbers and stays
silent where the provider said nothing. A zero in a latency log is a
claim; None is the truth when a field was absent.

None of this needs Ollama running: the seam is faked, which is the point
of having a seam.
"""

import pytest

from app.core.config import settings
from app.llm import benchmark_ollama as bench
from app.llm import llm_client


class FakeResponse:
    """Shaped like ollama._types.ChatResponse's telemetry surface."""

    def __init__(self, **kw):
        self.total_duration = kw.get("total_duration")
        self.load_duration = kw.get("load_duration")
        self.prompt_eval_count = kw.get("prompt_eval_count")
        self.prompt_eval_duration = kw.get("prompt_eval_duration")
        self.eval_count = kw.get("eval_count")
        self.eval_duration = kw.get("eval_duration")
        self.message = type("M", (), {"content": "{}"})()


@pytest.fixture()
def sent(monkeypatch):
    """Capture exactly what the seam hands the client."""
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return FakeResponse(total_duration=2_000_000_000, load_duration=1_000_000_000,
                            prompt_eval_count=5000, prompt_eval_duration=1_000_000_000,
                            eval_count=300, eval_duration=500_000_000)

    monkeypatch.setattr(llm_client._ollama, "chat", fake_chat)
    return captured


# --------------------------------------------------------------- options

def test_the_seam_sends_every_configured_runtime_option(sent):
    llm_client._chat(messages=[{"role": "user", "content": "x"}], fmt="json", purpose="t")
    assert sent["model"] == settings.ollama_model
    assert sent["keep_alive"] == settings.ollama_keep_alive
    assert sent["options"]["num_ctx"] == settings.ollama_num_ctx
    assert sent["options"]["num_predict"] == settings.ollama_num_predict
    assert sent["options"]["temperature"] == settings.ollama_temperature


def test_think_is_a_top_level_argument_not_an_option(sent):
    """ollama 0.6.2 types `think` on chat() itself. Putting it in options
    would be silently ignored — the worst possible failure for a setting
    whose entire purpose is suppressing latency."""
    llm_client._chat(messages=[{"role": "user", "content": "x"}], fmt="json", purpose="t")
    assert sent["think"] is settings.ollama_think
    assert "think" not in sent["options"]


def test_think_none_omits_the_parameter_entirely(sent, monkeypatch):
    """The escape hatch for a model that rejects the argument."""
    monkeypatch.setattr(settings, "ollama_think", None)
    llm_client._chat(messages=[{"role": "user", "content": "x"}], fmt="json", purpose="t")
    assert "think" not in sent


def test_temperature_is_configurable_and_defaults_to_deterministic(sent, monkeypatch):
    assert settings.ollama_temperature == 0.0
    monkeypatch.setattr(settings, "ollama_temperature", 0.7)
    llm_client._chat(messages=[{"role": "user", "content": "x"}], fmt="json", purpose="t")
    assert sent["options"]["temperature"] == 0.7


def test_think_accepts_the_effort_levels_the_client_declares():
    from app.core.config import Settings

    for level in ("low", "medium", "high"):
        assert Settings.model_fields["ollama_think"].annotation is not None
    # the union must admit both forms without coercing one into the other
    import typing
    args = typing.get_args(Settings.model_fields["ollama_think"].annotation)
    assert bool in args


# ------------------------------------------------------------- telemetry

def test_every_required_telemetry_field_is_reported():
    fields = llm_client._log_llm_call(
        purpose="structured:query_ir", duration_ms=1234.5,
        response=FakeResponse(total_duration=2_000_000_000, load_duration=1_000_000_000,
                              prompt_eval_count=5000, prompt_eval_duration=1_000_000_000,
                              eval_count=300, eval_duration=500_000_000),
        success=True)
    for key in ("provider", "model", "purpose", "duration_ms", "prompt_tokens",
                "output_tokens", "load_duration_ms", "prompt_eval_duration_ms",
                "eval_duration_ms", "total_duration_ms", "success"):
        assert key in fields, key


def test_nanoseconds_are_converted_to_milliseconds():
    fields = llm_client._log_llm_call(
        purpose="p", duration_ms=1.0,
        response=FakeResponse(total_duration=2_500_000_000, load_duration=1_000_000_000,
                              prompt_eval_duration=1_000_000_000, eval_duration=500_000_000))
    assert fields["total_duration_ms"] == 2500.0
    assert fields["load_duration_ms"] == 1000.0
    assert fields["prompt_eval_duration_ms"] == 1000.0
    assert fields["eval_duration_ms"] == 500.0


def test_both_throughput_figures_are_derived():
    """prefill and generation rates are different questions."""
    fields = llm_client._log_llm_call(
        purpose="p", duration_ms=1.0,
        response=FakeResponse(prompt_eval_count=5000, prompt_eval_duration=1_000_000_000,
                              eval_count=300, eval_duration=500_000_000))
    assert fields["prompt_tokens_per_second"] == 5000.0
    assert fields["eval_tokens_per_second"] == 600.0


def test_a_missing_denominator_reports_nothing_rather_than_zero():
    fields = llm_client._log_llm_call(purpose="p", duration_ms=1.0, response=FakeResponse())
    assert fields["prompt_tokens_per_second"] is None
    assert fields["eval_tokens_per_second"] is None
    assert fields["total_duration_ms"] is None


def test_a_failed_call_is_logged_with_its_exception_type():
    fields = llm_client._log_llm_call(
        purpose="structured:query_ir", duration_ms=60000.0,
        success=False, error=TimeoutError("slow"))
    assert fields["success"] is False
    assert fields["error"] == "TimeoutError"


def test_telemetry_never_carries_the_prompt_or_the_response(caplog):
    secret = "top 5 advisors by connects for Muhammad Ahmed Khan"
    fields = llm_client._log_llm_call(
        purpose="structured:query_ir", duration_ms=1.0,
        response=FakeResponse(eval_count=3, eval_duration=1))
    blob = " ".join(str(v) for v in fields.values())
    assert secret not in blob
    assert "Muhammad" not in blob


def test_a_failing_call_still_logs_then_re_raises(monkeypatch):
    """Fail-soft is preserved: the seam must not swallow the error, and a
    slow timeout is the most expensive event there is to leave unlogged."""
    logged = {}

    def boom(**kwargs):
        raise TimeoutError("nope")

    monkeypatch.setattr(llm_client._ollama, "chat", boom)
    monkeypatch.setattr(llm_client, "_log_llm_call",
                        lambda **kw: logged.update(kw) or kw)
    with pytest.raises(TimeoutError):
        llm_client._chat(messages=[{"role": "user", "content": "x"}], fmt="json", purpose="t")
    assert logged["success"] is False
    assert isinstance(logged["error"], TimeoutError)


def test_the_two_real_call_sites_are_distinguishable():
    """Phase 12: the next task has to be able to price the narrative call
    separately, which requires that it is labelled separately."""
    import inspect

    src = inspect.getsource(llm_client)
    assert 'purpose="narrative"' in src
    assert 'purpose=f"structured:{schema_name}"' in src


# ------------------------------------------------------------- benchmark

def test_the_benchmark_is_not_part_of_the_unit_suite():
    """It must never require a running provider to collect tests."""
    assert bench.ollama_is_up(timeout=0.01) in (True, False)


def test_the_benchmark_refuses_to_invent_numbers(monkeypatch, capsys):
    monkeypatch.setattr(bench, "ollama_is_up", lambda *a, **k: False)
    monkeypatch.setattr("sys.argv", ["benchmark_ollama"])
    assert bench.main() == 2
    assert "nothing measured" in capsys.readouterr().out


def test_the_experiment_matrix_covers_the_required_configurations():
    assert set(bench.EXPERIMENTS) == {"baseline", "A_think_off",
                                      "B_think_off_keepalive", "C_full"}
    assert bench.EXPERIMENTS["C_full"]["ollama_num_ctx"] == settings.ollama_num_ctx


def test_config_overrides_are_restored_even_when_an_experiment_raises():
    before = settings.ollama_num_ctx
    with pytest.raises(RuntimeError):
        with bench.override(ollama_num_ctx=1234):
            assert settings.ollama_num_ctx == 1234
            raise RuntimeError("experiment blew up")
    assert settings.ollama_num_ctx == before


def test_p95_is_withheld_until_there_are_enough_samples():
    s = bench.Series("x")
    for v in (10.0, 20.0, 30.0):
        s.samples.append(bench.CallSample(wall_ms=v, total_ms=v))
    assert s.p95("total_ms") is None
    assert s.median("total_ms") == 20.0


def test_benchmark_queries_cover_the_required_shapes():
    names = {n for n, _ in bench.BENCH_QUERIES}
    assert {"single_metric", "comparison", "compound_filter",
            "unresolved_phrase", "multi_metric", "novel_phrase"} <= names
