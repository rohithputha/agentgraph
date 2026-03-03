from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from agentgit.langgraph_callback import langgraph_callback
from agenttest.recorder import Recorder
from agenttest.replayer import Replayer
from agenttest.session import AgentTestSession

pytestmark = pytest.mark.integration


def _live_enabled() -> bool:
    return os.getenv("AGENTTEST_RUN_LIVE") == "1" and bool(os.getenv("GOOGLE_API_KEY"))


@pytest.mark.skipif(not _live_enabled(), reason="Set AGENTTEST_RUN_LIVE=1 and GOOGLE_API_KEY to run")
def test_live_gemini_record_then_locked_replay(tmp_path):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"langchain_google_genai not installed: {exc}")

    session = AgentTestSession.standalone(
        project_dir=str(tmp_path),
        user_id="live-user",
        session_id="live-session",
    )

    try:
        callback = langgraph_callback(session.ag.eventbus)
        cfg = {
            "callbacks": [callback],
            "configurable": {
                "user_id": session.user_id,
                "session_id": session.session_id,
            },
        }

        prompt = [HumanMessage(content="Say exactly: LIVE_BASELINE_OK")]

        with Recorder(session=session, name="gemini-live"):
            baseline_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            baseline_out = baseline_model.invoke(prompt, config=cfg)
            assert baseline_out.content

        with Replayer(session=session, baseline_name="gemini-live", mode="locked") as replay:
            replay_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            replay_out = replay_model.invoke(prompt, config=cfg)

        assert replay.passed is True
        assert replay.cache_stats["live_calls"] == 0
        assert replay_out.content == baseline_out.content
    finally:
        session.close()
