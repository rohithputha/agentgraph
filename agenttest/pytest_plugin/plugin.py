import pytest
from typing import Optional

from agenttest.session import AgentTestSession
from agenttest.recorder import Recorder
from agenttest.replayer import Replayer
from agenttest.config_loader import load_config
from agenttest.models.config import AgentTestConfig


def pytest_addoption(parser):
    group = parser.getgroup("agenttest")

    group.addoption(
        "--agenttest",
        action="store_true",
        default=False,
        help="Enable AgentTest recording and replay"
    )

    group.addoption(
        "--agenttest-mode",
        type=str,
        default=None,
        choices=["full", "selective", "locked"],
        help="Replay mode: full, selective, or locked"
    )

    group.addoption(
        "--agenttest-record",
        action="store_true",
        default=False,
        help="Force recording mode (create new baselines)"
    )

    group.addoption(
        "--agenttest-config",
        type=str,
        default=None,
        help="Path to AgentTest config file"
    )

    group.addoption(
        "--agenttest-project-dir",
        type=str,
        default=".",
        help="Project directory for AgentTest"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "agenttest: mark test as an AgentTest test"
    )
    config.addinivalue_line(
        "markers",
        "baseline(name): mark test to use specific baseline"
    )


@pytest.fixture(scope="session")
def agenttest_config(request) -> AgentTestConfig:
    config_path = request.config.getoption("--agenttest-config")

    if config_path:
        return load_config(config_path)

    return load_config()


@pytest.fixture(scope="session")
def agenttest_session(request, agenttest_config):
    project_dir = request.config.getoption("--agenttest-project-dir")

    session = AgentTestSession.standalone(
        project_dir=project_dir,
        user_id="pytest",
        session_id="test-session"
    )

    yield session

    session.close()


@pytest.fixture
def agenttest_record(request, agenttest_session, agenttest_config):
    def _record(name: Optional[str] = None, set_as_baseline: bool = False):
        test_name = name or request.node.name

        recorder = Recorder(
            session=agenttest_session,
            name=test_name,
            config=agenttest_config
        )

        if set_as_baseline:
            recorder.set_as_baseline_on_exit = True

        return recorder

    return _record


@pytest.fixture
def agenttest_replay(request, agenttest_session, agenttest_config):
    def _replay(
        baseline_name: Optional[str] = None,
        mode: Optional[str] = None,
        replay_name: Optional[str] = None
    ):
        if baseline_name is None:
            baseline_marker = request.node.get_closest_marker("baseline")
            if baseline_marker:
                baseline_name = baseline_marker.args[0]
            else:
                baseline_name = request.node.name

        if mode is None:
            mode = request.config.getoption("--agenttest-mode")
            if mode is None:
                mode = agenttest_config.default_replay_mode

        if replay_name is None:
            replay_name = f"{baseline_name}-replay"

        replayer = Replayer(
            session=agenttest_session,
            baseline_name=baseline_name,
            replay_name=replay_name,
            mode=mode,
            config=agenttest_config
        )

        return replayer

    return _replay


@pytest.fixture
def agenttest_auto(request, agenttest_record, agenttest_replay, agenttest_config):
    is_recording = request.config.getoption("--agenttest-record")
    is_enabled = request.config.getoption("--agenttest")

    if not is_enabled:
        yield None
        return

    test_name = request.node.name

    if is_recording:
        with agenttest_record(name=test_name, set_as_baseline=True) as rec:
            yield rec
    else:
        baseline_marker = request.node.get_closest_marker("baseline")
        if baseline_marker:
            baseline_name = baseline_marker.args[0]
        else:
            baseline_name = test_name

        with agenttest_replay(baseline_name=baseline_name) as rep:
            yield rep

            if not rep.passed:
                summary = rep.root_cause_summary or "No root cause identified"
                pytest.fail(f"Regression detected: {summary}")
