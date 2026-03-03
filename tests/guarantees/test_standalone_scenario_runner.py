from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import pytest

from agenttest.cli.main import cli
from agenttest.session import AgentTestSession


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


@pytest.mark.agenttest_runtime
def test_standalone_scenario_record_and_locked_replay_without_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_TORCH", "0")
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("USE_FLAX", "0")
    monkeypatch.syspath_prepend(str(tmp_path))

    _write(
        tmp_path / "demo_scenarios_runtime.py",
        """
import os


def run_no_callback(payload):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import HumanMessage

    response = os.getenv("SCENARIO_RESPONSE", payload["baseline_response"])
    llm = FakeListChatModel(responses=[response])
    llm.invoke([HumanMessage(content=payload["prompt"])])
""",
    )

    _write(
        tmp_path / "agenttest.toml",
        """
[agenttest]
default_replay_mode = "locked"

[[scenarios]]
name = "refund_flow"
entrypoint = "demo_scenarios_runtime:run_no_callback"
expects_llm = true
input = { prompt = "hello", baseline_response = "BASELINE" }
""",
    )

    runner = CliRunner()

    monkeypatch.setenv("SCENARIO_RESPONSE", "BASELINE")
    record_result = runner.invoke(
        cli,
        [
            "--project-dir",
            str(tmp_path),
            "record",
            "--backend",
            "scenario",
            "--scenario",
            "refund_flow",
        ],
    )
    assert record_result.exit_code == 0, record_result.output

    monkeypatch.setenv("SCENARIO_RESPONSE", "LIVE_SHOULD_NOT_BE_USED")
    replay_result = runner.invoke(
        cli,
        [
            "--project-dir",
            str(tmp_path),
            "replay",
            "--backend",
            "scenario",
            "--scenario",
            "refund_flow",
            "--mode",
            "locked",
        ],
    )
    assert replay_result.exit_code == 0, replay_result.output

    session = AgentTestSession.standalone(project_dir=str(tmp_path))
    try:
        baseline = session.get_recording_by_name("refund_flow")
        assert baseline is not None
        baseline_details = session.get_recording_details(baseline.recording_id)
        assert len(baseline_details) > 0

        replay = session.get_recording_by_name("refund_flow-replay")
        assert replay is not None
        replay_details = session.get_recording_details(replay.recording_id)
        assert len(replay_details) > 0
        assert all(detail.was_cache_hit for detail in replay_details)
    finally:
        session.close()


def test_scenario_expects_llm_guard_fails_on_zero_capture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))

    _write(
        tmp_path / "demo_scenarios_nollm.py",
        """
def run_no_llm(payload):
    return {"ok": True, "payload": payload}
""",
    )
    _write(
        tmp_path / "agenttest.toml",
        """
[agenttest]
default_replay_mode = "locked"

[[scenarios]]
name = "no_llm"
entrypoint = "demo_scenarios_nollm:run_no_llm"
expects_llm = true
input = { hello = "world" }
""",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--project-dir",
            str(tmp_path),
            "record",
            "--backend",
            "scenario",
            "--scenario",
            "no_llm",
        ],
    )

    assert result.exit_code == 3
    assert "captured 0 steps" in result.output


def test_record_auto_prefers_scenario_backend_when_scenarios_exist(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    called = {"count": 0}

    _write(
        tmp_path / "demo_scenarios_auto.py",
        """
def run_no_llm(payload):
    return payload
""",
    )
    _write(
        tmp_path / "agenttest.toml",
        """
[agenttest]
default_replay_mode = "locked"

[[scenarios]]
name = "auto_path"
entrypoint = "demo_scenarios_auto:run_no_llm"
expects_llm = false
input = { demo = "ok" }
""",
    )

    def _fake_pytest_record(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("pytest backend should not execute in scenario auto mode")

    monkeypatch.setattr("agenttest.cli.main._run_pytest_record", _fake_pytest_record)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--project-dir",
            str(tmp_path),
            "record",
            "--backend",
            "auto",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["count"] == 0
