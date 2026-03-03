from __future__ import annotations

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


def test_x1_x2_consistent_but_wrong_baseline_still_passes(agenttest_session):
    # Intentionally incorrect baseline behavior.
    with Recorder(agenttest_session, name="wrong-base"):
        wrong_model = _new_scripted_llm(agenttest_session, ["2 + 2 = 5"])
        invoke_scripted_prompts(wrong_model, ["what is 2+2?"])

    # Live model has corrected behavior, but locked replay must stay consistent
    # with the approved baseline.
    corrected_live_model = _new_scripted_llm(agenttest_session, ["2 + 2 = 4"])
    with Replayer(agenttest_session, baseline_name="wrong-base", mode="locked") as replay:
        wrapped = replay.wrap_model(corrected_live_model)
        out = invoke_scripted_prompts(wrapped, ["what is 2+2?"])[0]

    assert out.content == "2 + 2 = 5"
    assert replay.passed is True


def test_x3_test_consistency_does_not_guarantee_live_output_match(agenttest_session):
    with Recorder(agenttest_session, name="prod-gap"):
        baseline_model = _new_scripted_llm(agenttest_session, ["Paris is in Germany"])
        invoke_scripted_prompts(baseline_model, ["Where is Paris?"])

    live_model = _new_scripted_llm(agenttest_session, ["Paris is in France"])
    with Replayer(agenttest_session, baseline_name="prod-gap", mode="locked") as replay:
        wrapped = replay.wrap_model(live_model)
        out = invoke_scripted_prompts(wrapped, ["Where is Paris?"])[0]

    # Replay proves consistency with baseline, not production truth.
    assert out.content == "Paris is in Germany"
    assert replay.passed is True
