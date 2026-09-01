"""The test suite must not talk to a paid provider, and must not be able
to start doing so by accident.

WHY THIS FILE EXISTS. `backend/.env` is read by pydantic-settings at
import, so on any machine with OPENAI_API_KEY set — which is every
machine that can run the app — every NLU_MODE="llm_first" test silently
became a live, billable, non-deterministic API call. It was invisible:
the suite still passed, just slowly (340s against the 20s it takes
offline) and with a handful of tests that changed their answer between
runs. Two files' worth of "regressions" turned out to be that noise.

These tests pin the two mechanisms that close it, so neither can be
removed without a failure that says why.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from app.llm import llm_client

_BACKEND = Path(__file__).resolve().parents[1]


def test_the_provider_credential_is_absent_by_default():
    """The guarantee itself, asserted from inside an ordinary test.

    `_hermetic_llm_provider` (conftest.py) is autouse, so by the time any
    test body runs there is no credential to spend.
    """
    assert llm_client.settings.openai_api_key == "", (
        "a provider credential is visible to an unmarked test — "
        "conftest._hermetic_llm_provider is not doing its job"
    )


def test_the_network_client_cannot_be_built():
    """One level deeper than the setting: the client itself is
    unconstructable, so no call site can reach the network however it is
    written."""
    with pytest.raises(llm_client.ProviderNotConfigured):
        llm_client._openai()


def test_a_structured_call_degrades_instead_of_raising():
    """The offline state is one the pipeline already models. This is what
    makes blanking the key honest rather than a fake: nothing is told what
    the model 'would have said' — the parser simply gets None and falls
    back, exactly as it does when the provider is down in production."""
    assert llm_client.call_llm_structured(
        "irrelevant", llm_client.QUERY_IR_JSON_SCHEMA, schema_name="query_ir"
    ) is None


def test_live_and_benchmark_are_deselected_by_default():
    """The other half of the guard: `-m live` tests are not collected
    unless asked for by name. Asserted by running collection in a
    subprocess, because the marker expression is applied at collection —
    an in-process check could not see it."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "tests/test_hermeticity.py"],
        cwd=_BACKEND, capture_output=True, text=True,
    )
    assert "deselected" in result.stdout, (
        "pytest.ini's addopts no longer deselects the live/benchmark markers:\n"
        + result.stdout[-2000:]
    )


@pytest.mark.live
def test_a_live_marked_test_keeps_its_credential():
    """The escape hatch works, so a genuine model test is still possible.

    Never collected by a default run — `pytest -m live` is the only way
    here. It asserts the FIXTURE's behaviour (that it did not blank the
    key), not that a key happens to be configured, so it reports the
    mechanism rather than the operator's environment.
    """
    from app.core.config import Settings

    assert llm_client.settings.openai_api_key == Settings().openai_api_key, (
        "_hermetic_llm_provider blanked the key for a test marked `live`"
    )
