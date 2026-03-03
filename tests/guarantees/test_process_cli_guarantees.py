from __future__ import annotations

from click.testing import CliRunner

from agenttest.cli.main import cli
from agenttest.recorder import Recorder
from agenttest.session import AgentTestSession
from tests.helpers import ScriptedLLM, invoke_scripted_prompts


def _record_once(session: AgentTestSession, name: str, response: str) -> str:
    model = ScriptedLLM(
        eventbus=session.ag.eventbus,
        user_id=session.user_id,
        session_id=session.session_id,
        responses=[response],
    )

    with Recorder(session=session, name=name) as recorder:
        invoke_scripted_prompts(model, ["Q"])

    return recorder.recording_id


def test_g41_accept_promotes_latest_replay_to_baseline(tmp_path):
    project_dir = str(tmp_path)

    session = AgentTestSession.standalone(project_dir=project_dir)
    try:
        _record_once(session, "checkout", "baseline")
        replay_id = _record_once(session, "checkout-replay", "candidate")

        # Deterministic ordering for `accept`: newest created_at wins.
        conn = session.test_store.conn
        conn.execute(
            "UPDATE at_recordings SET created_at = created_at + 100 WHERE recording_id = ?",
            (replay_id,),
        )
        conn.commit()
    finally:
        session.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", project_dir, "accept", "checkout"])

    assert result.exit_code == 0
    assert "Accepted recording 'checkout-replay'" in result.output

    verify = AgentTestSession.standalone(project_dir=project_dir)
    try:
        baseline = verify.get_baseline("checkout")
        assert baseline is not None
        assert baseline.metadata["recording_id"] == replay_id
    finally:
        verify.close()


def test_g42_baseline_tag_is_auditable_metadata(tmp_path):
    project_dir = str(tmp_path)
    session = AgentTestSession.standalone(project_dir=project_dir)

    try:
        recording_id = _record_once(session, "support-flow", "baseline")
        tag = session.set_baseline("support-flow", recording_id)

        assert tag.tag_name == "baseline/support-flow"
        assert tag.metadata["recording_id"] == recording_id

        tags = session.list_baselines()
        assert any(t.tag_name == "baseline/support-flow" for t in tags)
    finally:
        session.close()


def test_cli_record_includes_agenttest_flag_in_pytest_command(monkeypatch, tmp_path):
    calls = []

    class _Result:
        returncode = 0

    def _fake_run(args, cwd):
        calls.append((args, cwd))
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--project-dir", str(tmp_path), "record", "--name", "my-test"],
    )

    assert result.exit_code == 0
    assert calls, "expected subprocess.run to be called"

    args, cwd = calls[0]
    assert cwd == str(tmp_path)
    assert "--agenttest" in args
    assert "--agenttest-record" in args
    assert "--agenttest-mode=full" in args
