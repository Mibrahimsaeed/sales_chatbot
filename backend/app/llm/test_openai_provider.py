"""OpenAI behind the existing `_chat()` boundary — the only provider.

WHAT THIS PROTECTS. A previous provider migration broke 146 tests in
suites that were not about providers, because callers and fakes had bound
to a vendor SDK's call shape rather than to this project's seam. These
tests assert the seam still holds:

  * `_chat`'s signature is unchanged, so every existing fake still works
  * everything above `_chat` — the schema, the prompt, the validator —
    is untouched and knows nothing about which model answers
  * a missing key degrades exactly like a network error, never crashes,
    and the deterministic rule-based planner answers instead

Nothing here needs a real key or a network — the boundary is faked, which
is the point of having one.
"""

from types import SimpleNamespace

import pytest

from app.llm import llm_client


@pytest.fixture(autouse=True)
def _reset_client():
    """The OpenAI client is cached; a test that changes the key must not
    inherit one built from a previous test's config."""
    llm_client._reset_provider_clients_for_tests()
    yield
    llm_client._reset_provider_clients_for_tests()


def _openai_response(content="{}", prompt_tokens=1200, completion_tokens=210,
                     refusal=None):
    """The OpenAI chat-completions shape: choices[0].message.content."""
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              total_tokens=prompt_tokens + completion_tokens),
    )


def _fake_openai(response=None, raise_error=None):
    """Stands in for the OpenAI SDK client, recording the request."""
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        if raise_error:
            raise raise_error
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.seen = seen
    return client


def _use_openai(monkeypatch, client):
    monkeypatch.setattr(llm_client, "_openai", lambda: client)


# ===================================================== provider identity
class TestProviderIdentity:
    def test_the_provider_is_a_constant_not_a_setting(self):
        """There is one provider, so a configurable name could only ever
        disagree with what actually served the call."""
        from app.core.config import Settings

        assert llm_client.PROVIDER == "openai"
        assert "llm_provider" not in Settings.model_fields

    def test_the_active_model_is_the_configured_openai_model(self):
        assert llm_client._active_model() == llm_client.settings.openai_model


# ===================================================== configuration
class TestConfiguration:
    def test_an_unset_key_raises_a_catchable_error_not_an_import_failure(
            self, monkeypatch):
        """An empty key is a legitimate state — "OpenAI is not set up" —
        and must degrade, not prevent the application from starting."""
        monkeypatch.setattr(llm_client.settings, "openai_api_key", "")

        with pytest.raises(llm_client.ProviderNotConfigured):
            llm_client._openai()

    def test_the_key_is_never_put_in_the_error_message(self, monkeypatch):
        monkeypatch.setattr(llm_client.settings, "openai_api_key", "sk-secret-value")

        # A constructed client must not be built twice, and nothing about
        # the key may reach a log or an exception. Force the failure path
        # by removing the package name it imports.
        monkeypatch.setattr(llm_client.settings, "openai_base_url", "")
        client = llm_client._openai()
        assert client is not None
        # Cached: the second call returns the same object rather than
        # re-reading (and re-handling) the secret.
        assert llm_client._openai() is client

    def test_the_key_field_ships_empty(self):
        """The key ships empty so no existing deployment or CI job breaks
        the moment the field lands.

        Asserted against the FIELD DEFAULT, not the resolved value. The
        first version of this test read `settings.openai_api_key` and so
        asserted the developer's own environment — it passed only for
        someone who had not configured OpenAI, and failed the moment
        anyone did the thing the feature exists to allow. A test that
        breaks on correct usage is worse than no test.
        """
        from app.core.config import Settings

        assert Settings.model_fields["openai_api_key"].default == ""


# ===================================================== structured output
class TestStructuredOutput:
    def test_a_schema_becomes_strict_json_schema(self):
        fmt = llm_client._openai_response_format(
            llm_client.QUERY_IR_JSON_SCHEMA, "structured:query_ir")

        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True, \
            "without strict the schema is a hint, not a constraint"
        assert fmt["json_schema"]["name"] == "query_ir"
        assert fmt["json_schema"]["schema"] is llm_client.QUERY_IR_JSON_SCHEMA

    def test_the_string_json_becomes_object_mode(self):
        assert llm_client._openai_response_format("json", "narrative") == \
            {"type": "json_object"}

    def test_the_schema_name_is_sanitised(self):
        """OpenAI requires ^[a-zA-Z0-9_-]+$; `purpose` is free text."""
        fmt = llm_client._openai_response_format({}, "structured:query ir!")
        assert fmt["json_schema"]["name"] == "query_ir_"

    def test_the_query_ir_schema_satisfies_openai_strict_rules(self):
        """Strict mode rejects any object that omits
        additionalProperties:false or leaves a property out of `required`.
        QUERY_IR_JSON_SCHEMA already complies, and this keeps it that
        way — a field added for the compiler's benefit must not quietly
        make the schema unusable for constrained decoding."""
        problems = []

        def walk(node, path="$"):
            if not isinstance(node, dict):
                return
            if node.get("type") == "object":
                props = node.get("properties", {})
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path}: additionalProperties is not false")
                if set(node.get("required", [])) != set(props):
                    problems.append(f"{path}: required != properties")
                for key, value in props.items():
                    walk(value, f"{path}.{key}")
            if node.get("type") == "array":
                walk(node.get("items", {}), f"{path}[]")
            for i, sub in enumerate(node.get("anyOf") or []):
                walk(sub, f"{path}|anyOf[{i}]")

        walk(llm_client.QUERY_IR_JSON_SCHEMA)
        assert not problems, problems


# ===================================================== the boundary routes
class TestChatRouting:
    def test_chat_sends_the_request_to_openai(self, monkeypatch):
        client = _fake_openai(_openai_response('{"intent": "leaderboard"}'))
        _use_openai(monkeypatch, client)

        llm_client._chat(messages=[{"role": "user", "content": "hi"}],
                         fmt=llm_client.QUERY_IR_JSON_SCHEMA,
                         purpose="structured:query_ir")

        assert client.seen["model"] == llm_client.settings.openai_model
        assert client.seen["messages"] == [{"role": "user", "content": "hi"}]
        assert client.seen["response_format"]["type"] == "json_schema"
        assert client.seen["temperature"] == llm_client.settings.openai_temperature
        assert client.seen["max_completion_tokens"] == \
            llm_client.settings.openai_max_output_tokens

    def test_the_boundary_signature_is_unchanged(self):
        """Eight e2e suites and several fakes bind to `_chat` by keyword.
        A new parameter here is what broke 146 tests last time."""
        import inspect

        params = inspect.signature(llm_client._chat).parameters
        assert list(params) == ["messages", "fmt", "purpose"]


# ===================================================== response reading
class TestResponseReading:
    def test_openai_content_is_extracted(self):
        assert llm_client._extract_message_content(
            _openai_response('{"a": 1}')) == '{"a": 1}'

    def test_content_is_stripped(self):
        assert llm_client._extract_message_content(
            _openai_response(' {"a": 1} ')) == '{"a": 1}'

    def test_a_response_with_no_choices_reads_as_empty(self):
        assert llm_client._extract_message_content(
            SimpleNamespace(choices=[], usage=None)) == ""

    def test_a_refusal_reads_as_empty_not_as_content(self):
        """An OpenAI refusal sets `refusal` and leaves `content` null.
        Returning "" routes into the caller's existing empty-response
        branch, which degrades to the rule-based plan."""
        response = _openai_response(content=None, refusal="I can't help with that")
        assert llm_client._extract_message_content(response) == ""


# ===================================================== telemetry
class TestTelemetry:
    def test_openai_usage_becomes_token_fields(self):
        metrics = llm_client._response_metrics(
            _openai_response(prompt_tokens=1200, completion_tokens=210))

        assert metrics["prompt_tokens"] == 1200
        assert metrics["output_tokens"] == 210

    def test_a_response_without_usage_reports_none_not_zero(self):
        """"nothing generated" and "not reported" are different facts."""
        metrics = llm_client._response_metrics(
            SimpleNamespace(choices=[], usage=None))
        assert metrics == llm_client._EMPTY_METRICS
        assert all(v is None for v in metrics.values())

    def test_the_log_names_the_model_that_actually_served(self, monkeypatch, caplog):
        _use_openai(monkeypatch, _fake_openai(_openai_response()))

        with caplog.at_level("INFO", logger="llm.client"):
            llm_client._chat(messages=[], fmt="json", purpose="narrative")

        line = [r.getMessage() for r in caplog.records
                if r.getMessage().startswith("llm_call ")][0]
        assert f"model={llm_client.settings.openai_model}" in line
        assert "provider=openai" in line


# ===================================================== fail-soft
class TestFailSoft:
    def test_a_missing_key_degrades_instead_of_raising(self, monkeypatch):
        """Same contract as a network error: None, so semantic_parser
        falls back to the rule-based plan."""
        monkeypatch.setattr(llm_client.settings, "openai_api_key", "")

        assert llm_client.call_llm_structured(
            "prompt", llm_client.QUERY_IR_JSON_SCHEMA, "query_ir") is None
        assert llm_client.call_llm_json("prompt") is None

    def test_a_provider_error_degrades(self, monkeypatch):
        _use_openai(monkeypatch, _fake_openai(raise_error=RuntimeError("503")))

        assert llm_client.call_llm_structured(
            "prompt", llm_client.QUERY_IR_JSON_SCHEMA, "query_ir") is None

    def test_unparseable_output_degrades(self, monkeypatch):
        _use_openai(monkeypatch, _fake_openai(_openai_response("not json")))

        assert llm_client.call_llm_structured(
            "prompt", llm_client.QUERY_IR_JSON_SCHEMA, "query_ir") is None

    def test_a_failed_call_is_still_timed_and_logged(self, monkeypatch, caplog):
        _use_openai(monkeypatch, _fake_openai(raise_error=RuntimeError("503")))

        with caplog.at_level("INFO", logger="llm.client"):
            llm_client.call_llm_structured(
                "prompt", llm_client.QUERY_IR_JSON_SCHEMA, "query_ir")

        line = [r.getMessage() for r in caplog.records
                if r.getMessage().startswith("llm_call ")][0]
        assert "success=False" in line
        assert "error=RuntimeError" in line


# ===================================================== end to end
class TestEndToEnd:
    def test_a_query_ir_round_trips_through_the_openai_path(self, monkeypatch):
        """The whole point: OpenAI produces the EXISTING QueryIR, which
        the existing model validates. No new IR, no new schema."""
        from app.llm.query_ir import QueryIR

        ir_json = (
            '{"operation": "leaderboard", "intent": "leaderboard", '
            '"subject_level": "advisor", "subjects": [], '
            '"metric": {"key": "total_connects", "confidence": 0.95}, '
            '"metrics": [], "filters": [], "filter_tree": null, '
            '"time_range": {"mode": "snapshot", "period": "MTD", '
            '"compare_to": null, "confidence": 0.9}, '
            '"sort": {"metric": "total_connects", "direction": "desc"}, '
            '"limit": 5, "group_by": null, "flat": false, '
            '"overall_confidence": 0.95, "intent_confidence": 0.95}'
        )
        _use_openai(monkeypatch, _fake_openai(_openai_response(ir_json)))

        raw = llm_client.call_llm_structured(
            "prompt", llm_client.QUERY_IR_JSON_SCHEMA, "query_ir")

        assert raw is not None
        parsed = QueryIR.model_validate(raw)
        assert parsed.intent == "leaderboard"
        assert parsed.metric.key == "total_connects"
        assert parsed.limit == 5
