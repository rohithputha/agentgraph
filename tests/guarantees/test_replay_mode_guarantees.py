from __future__ import annotations

import httpx
import pytest

from agenttest.interceptors.runtime import (
    LockedModeNetworkError,
    install_global_runtime_interception,
    reset_active_replay_context,
    set_active_replay_context,
)
from agenttest.recorder import Recorder
from agenttest.replayer import Replayer
from tests.helpers import ScriptedLLM, invoke_scripted_prompts


def _new_scripted_llm(agenttest_session, responses):
    return ScriptedLLM(
        eventbus=agenttest_session.ag.eventbus,
        user_id=agenttest_session.user_id,
        session_id=agenttest_session.session_id,
        responses=responses,
    )


def test_c1_locked_mode_makes_zero_live_llm_calls(agenttest_session):
    with Recorder(agenttest_session, name="locked-base"):
        baseline_model = _new_scripted_llm(agenttest_session, ["base-1", "base-2"])
        baseline_outputs = invoke_scripted_prompts(baseline_model, ["Q1", "Q2"])

    live_model = _new_scripted_llm(agenttest_session, ["live-1", "live-2"])
    with Replayer(agenttest_session, baseline_name="locked-base", mode="locked") as replay:
        wrapped = replay.wrap_model(live_model)
        replay_outputs = invoke_scripted_prompts(wrapped, ["Q1", "Q2"])

    assert [m.content for m in replay_outputs] == [m.content for m in baseline_outputs]
    assert live_model.call_count == 0
    assert replay.passed is True
    assert replay.cache_stats["live_calls"] == 0
    assert replay.cache_stats["cache_hits"] == 2

    replay_recording = agenttest_session.get_recording_by_name("locked-base-replay")
    replay_details = agenttest_session.get_recording_details(replay_recording.recording_id)
    assert [d.was_cache_hit for d in replay_details] == [True, True]


def test_c2_selective_mode_cost_scales_with_changed_nodes(agenttest_session):
    with Recorder(agenttest_session, name="selective-base"):
        baseline_model = _new_scripted_llm(agenttest_session, ["base-1", "base-2"])
        invoke_scripted_prompts(baseline_model, ["Q1", "Q2"])

    live_model = _new_scripted_llm(agenttest_session, ["delta-live"])
    with Replayer(agenttest_session, baseline_name="selective-base", mode="selective") as replay:
        wrapped = replay.wrap_model(live_model)
        replay_outputs = invoke_scripted_prompts(wrapped, ["Q1", "Q2", "Q3"])

    assert [m.content for m in replay_outputs] == ["base-1", "base-2", "delta-live"]
    assert replay.cache_stats["cache_hits"] == 2
    assert replay.cache_stats["cache_misses"] == 1
    assert replay.cache_stats["live_calls"] == 1
    assert live_model.call_count == 1

    replay_recording = agenttest_session.get_recording_by_name("selective-base-replay")
    replay_details = agenttest_session.get_recording_details(replay_recording.recording_id)
    assert [d.was_cache_hit for d in replay_details] == [True, True, False]


def test_locked_cache_miss_converts_to_failed_comparison(agenttest_session):
    with Recorder(agenttest_session, name="miss-base"):
        baseline_model = _new_scripted_llm(agenttest_session, ["base-only"])
        invoke_scripted_prompts(baseline_model, ["Q1"])

    with Replayer(agenttest_session, baseline_name="miss-base", mode="locked") as replay:
        # Different structure (2-message request) forces a cache miss.
        live_model = _new_scripted_llm(agenttest_session, ["should-never-run"])
        wrapped = replay.wrap_model(live_model)
        wrapped.invoke(
            [
                {"role": "system", "content": "sys"},
                {"role": "human", "content": "Q1"},
            ]
        )

    assert replay.passed is False
    assert replay.comparison_result is not None
    assert replay.comparison_result.has_regression is True
    assert replay.cache_stats["cache_misses"] == 1
    assert replay.cache_stats["live_calls"] == 0
    assert live_model.call_count == 0


def test_g34_locked_mode_independent_of_provider_outage(agenttest_session):
    with Recorder(agenttest_session, name="outage-base"):
        baseline_model = _new_scripted_llm(agenttest_session, ["safe-cache"])
        invoke_scripted_prompts(baseline_model, ["Q1"])

    class _FailingScriptedLLM(ScriptedLLM):
        def invoke(self, messages, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    live_model = _FailingScriptedLLM(
        eventbus=agenttest_session.ag.eventbus,
        user_id=agenttest_session.user_id,
        session_id=agenttest_session.session_id,
        responses=["unused"],
    )

    with Replayer(agenttest_session, baseline_name="outage-base", mode="locked") as replay:
        wrapped = replay.wrap_model(live_model)
        out = invoke_scripted_prompts(wrapped, ["Q1"])[0]

    assert out.content == "safe-cache"
    assert replay.passed is True
    assert replay.cache_stats["live_calls"] == 0


def test_locked_mode_network_guard_blocks_provider_requests(monkeypatch):
    # Avoid importing LangChain BaseChatModel in this environment; we only need
    # network guard installation for this test.
    monkeypatch.setattr(
        "agenttest.interceptors.runtime._install_langchain_model_patches",
        lambda: False,
    )
    install_global_runtime_interception()
    token = set_active_replay_context(
        gatekeeper=object(),
        mode="locked",
        baseline_name="dummy",
    )

    try:
        with pytest.raises(LockedModeNetworkError):
            httpx.Client().request("GET", "https://api.openai.com/v1/models")
    finally:
        reset_active_replay_context(token)


def test_g44_locked_replay_is_repeatable(agenttest_session):
    with Recorder(agenttest_session, name="determinism-base"):
        baseline_model = _new_scripted_llm(agenttest_session, ["stable"])
        invoke_scripted_prompts(baseline_model, ["Q1"])

    for idx, live_response in enumerate(["variant-a", "variant-b"], start=1):
        live_model = _new_scripted_llm(agenttest_session, [live_response])
        with Replayer(
            agenttest_session,
            baseline_name="determinism-base",
            replay_name=f"determinism-base-replay-{idx}",
            mode="locked",
        ) as replay:
            wrapped = replay.wrap_model(live_model)
            out = invoke_scripted_prompts(wrapped, ["Q1"])[0]

        assert out.content == "stable"
        assert replay.passed is True
        assert replay.cache_stats["live_calls"] == 0
        assert live_model.call_count == 0
