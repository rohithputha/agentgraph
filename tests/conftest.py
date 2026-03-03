from __future__ import annotations

import os
from typing import Dict, Any

import pytest

from agentgit.langgraph_callback import langgraph_callback
from agenttest.session import AgentTestSession


@pytest.fixture()
def agenttest_session(tmp_path) -> AgentTestSession:
    session = AgentTestSession.standalone(
        project_dir=str(tmp_path),
        user_id="test-user",
        session_id="test-session",
    )
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def callback(agenttest_session: AgentTestSession):
    return langgraph_callback(agenttest_session.ag.eventbus)


@pytest.fixture()
def invoke_config(agenttest_session: AgentTestSession, callback) -> Dict[str, Any]:
    return {
        "callbacks": [callback],
        "configurable": {
            "user_id": agenttest_session.user_id,
            "session_id": agenttest_session.session_id,
        },
    }


@pytest.fixture(autouse=True)
def disable_runtime_langchain_patch(monkeypatch, request):
    """
    Keep tests deterministic in environments where importing BaseChatModel can
    pull in heavy optional stacks. Wrapper-based replay coverage remains active.
    """
    if (
        request.node.get_closest_marker("agenttest_runtime") is not None
        or os.getenv("AGENTTEST_RUN_LANGCHAIN_RUNTIME") == "1"
        or os.getenv("AGENTTEST_RUN_LIVE") == "1"
    ):
        return

    monkeypatch.setattr(
        "agenttest.replayer.install_global_runtime_interception",
        lambda: False,
    )
    monkeypatch.setattr(
        "agenttest.recorder.install_global_runtime_interception",
        lambda: False,
    )
