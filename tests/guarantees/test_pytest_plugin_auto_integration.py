from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)


@pytest.mark.skipif(
    os.getenv("AGENTTEST_RUN_LANGCHAIN_RUNTIME") != "1",
    reason="Set AGENTTEST_RUN_LANGCHAIN_RUNTIME=1 to run LangChain runtime interception integration",
)
def test_plugin_auto_wraps_marked_tests_without_manual_replayer(pytester, monkeypatch):
    pytester.syspathinsert(str(Path.cwd()))

    pytester.makepyfile(
        test_agent="""
import os
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from agentgit.langgraph_callback import langgraph_callback

BASELINE_RESPONSE = "baseline-response"

@pytest.mark.agenttest
@pytest.mark.baseline("auto-flow")
def test_agent(agenttest_session):
    callback = langgraph_callback(agenttest_session.ag.eventbus)
    live_response = os.environ.get("TEST_RESPONSE", BASELINE_RESPONSE)
    llm = FakeListChatModel(responses=[live_response])

    out = llm.invoke(
        [HumanMessage(content="hello")],
        config={
            "callbacks": [callback],
            "configurable": {"user_id": "pytest", "session_id": "test-session"},
        },
    )

    # In replay, locked mode must return baseline even if live model differs.
    assert out.content == BASELINE_RESPONSE
"""
    )

    monkeypatch.setenv("TEST_RESPONSE", "baseline-response")
    record = pytester.runpytest(
        "-p",
        "agenttest.pytest_plugin.plugin",
        "--agenttest",
        "--agenttest-record",
        "-q",
    )
    record.assert_outcomes(passed=1)

    monkeypatch.setenv("TEST_RESPONSE", "live-should-not-be-used")
    replay = pytester.runpytest(
        "-p",
        "agenttest.pytest_plugin.plugin",
        "--agenttest",
        "--agenttest-mode=locked",
        "-q",
    )
    replay.assert_outcomes(passed=1)
