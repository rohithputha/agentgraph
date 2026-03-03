from dataclasses import dataclass
from typing import Any, Optional

import pytest

from agenttest.config_loader import load_config
from agenttest.interceptors.runtime import (
    install_global_runtime_interception,
    uninstall_global_runtime_interception,
)
from agenttest.models.config import AgentTestConfig, TestConfig
from agenttest.recorder import Recorder
from agenttest.replayer import Replayer
from agenttest.session import AgentTestSession


_PLUGIN_STATE_ATTR = "_agenttest_state"
_MANUAL_FIXTURES = {"agenttest_auto", "agenttest_record", "agenttest_replay"}


@dataclass
class _AgentTestPluginState:
    enabled: bool
    config: AgentTestConfig
    session: AgentTestSession
    runtime_installed: bool


def pytest_addoption(parser):
    group = parser.getgroup("agenttest")

    group.addoption(
        "--agenttest",
        action="store_true",
        default=False,
        help="Enable AgentTest recording and replay",
    )

    group.addoption(
        "--agenttest-mode",
        type=str,
        default=None,
        choices=["full", "selective", "locked"],
        help="Replay mode: full, selective, or locked",
    )

    group.addoption(
        "--agenttest-record",
        action="store_true",
        default=False,
        help="Force recording mode (create new baselines)",
    )

    group.addoption(
        "--agenttest-config",
        type=str,
        default=None,
        help="Path to AgentTest config file",
    )

    group.addoption(
        "--agenttest-project-dir",
        type=str,
        default=".",
        help="Project directory for AgentTest",
    )

    group.addoption(
        "--agenttest-tier",
        type=str,
        default="all",
        choices=["always", "local", "ci-only", "all"],
        help="Run only tests that match this configured tier",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "agenttest(manual=False): mark test as an AgentTest test",
    )
    config.addinivalue_line(
        "markers",
        "baseline(name): mark test to use specific baseline",
    )

    if config.getoption("--agenttest"):
        _get_or_create_state(config)


def pytest_unconfigure(config):
    state = getattr(config, _PLUGIN_STATE_ATTR, None)
    if not state:
        return

    try:
        state.session.close()
    finally:
        if state.runtime_installed:
            uninstall_global_runtime_interception()
        setattr(config, _PLUGIN_STATE_ATTR, None)


def pytest_collection_modifyitems(config, items):
    state = _get_or_create_state(config, create=False)
    if not state or not state.enabled:
        return

    selected_tier = config.getoption("--agenttest-tier")
    if selected_tier == "all":
        return

    kept = []
    deselected = []

    for item in items:
        marker = item.get_closest_marker("agenttest")
        if marker is None:
            kept.append(item)
            continue

        test_cfg = _lookup_test_config(state.config, item)
        test_tier = test_cfg.tier if test_cfg else "always"

        if _tier_matches(selected_tier, test_tier):
            kept.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    state = _get_or_create_state(item.config, create=False)
    marker = item.get_closest_marker("agenttest")

    if (
        not state
        or not state.enabled
        or marker is None
        or marker.kwargs.get("manual", False)
        or _uses_manual_fixtures(item)
    ):
        yield
        return

    baseline_name = _resolve_baseline_name(item)
    is_recording = item.config.getoption("--agenttest-record")

    manager: Any
    replay: Optional[Replayer] = None

    if is_recording:
        recorder = Recorder(
            session=state.session,
            name=baseline_name,
            config=state.config,
        )
        recorder.set_as_baseline_on_exit = True
        manager = recorder
    else:
        mode = _resolve_mode(item, state.config)
        replay_name = f"{baseline_name}-replay"
        replay = Replayer(
            session=state.session,
            baseline_name=baseline_name,
            replay_name=replay_name,
            mode=mode,
            config=state.config,
        )
        manager = replay

    manager.__enter__()
    outcome = yield

    if outcome.excinfo is None:
        suppress = manager.__exit__(None, None, None)
    else:
        exc_type, exc_val, exc_tb = outcome.excinfo
        suppress = manager.__exit__(exc_type, exc_val, exc_tb)

    if suppress and outcome.excinfo is not None:
        outcome.force_result(None)

    if outcome.excinfo is None and replay is not None and not replay.passed:
        summary = replay.root_cause_summary or "No root cause identified"
        pytest.fail(f"Regression detected: {summary}")


@pytest.fixture(scope="session")
def agenttest_config(request) -> AgentTestConfig:
    state = _get_or_create_state(request.config)
    return state.config


@pytest.fixture(scope="session")
def agenttest_session(request, agenttest_config):
    state = _get_or_create_state(request.config)
    return state.session


@pytest.fixture
def agenttest_record(request, agenttest_session, agenttest_config):
    def _record(name: Optional[str] = None, set_as_baseline: bool = False):
        test_name = name or request.node.name

        recorder = Recorder(
            session=agenttest_session,
            name=test_name,
            config=agenttest_config,
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
        replay_name: Optional[str] = None,
    ):
        if baseline_name is None:
            baseline_name = _resolve_baseline_name(request.node)

        if mode is None:
            mode = _resolve_mode(request.node, agenttest_config)

        if replay_name is None:
            replay_name = f"{baseline_name}-replay"

        replayer = Replayer(
            session=agenttest_session,
            baseline_name=baseline_name,
            replay_name=replay_name,
            mode=mode,
            config=agenttest_config,
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

    baseline_name = _resolve_baseline_name(request.node)

    if is_recording:
        with agenttest_record(name=baseline_name, set_as_baseline=True) as rec:
            yield rec
    else:
        with agenttest_replay(baseline_name=baseline_name) as rep:
            yield rep

            if not rep.passed:
                summary = rep.root_cause_summary or "No root cause identified"
                pytest.fail(f"Regression detected: {summary}")


def _get_or_create_state(config, create: bool = True) -> Optional[_AgentTestPluginState]:
    state = getattr(config, _PLUGIN_STATE_ATTR, None)
    if state is not None or not create:
        return state

    is_enabled = bool(config.getoption("--agenttest"))
    config_path = config.getoption("--agenttest-config")
    project_dir = config.getoption("--agenttest-project-dir")

    loaded_config = load_config(config_path) if config_path else load_config()

    session = AgentTestSession.standalone(
        project_dir=project_dir,
        user_id="pytest",
        session_id="test-session",
        config=loaded_config,
    )

    runtime_installed = False
    if is_enabled:
        runtime_installed = install_global_runtime_interception()

    state = _AgentTestPluginState(
        enabled=is_enabled,
        config=loaded_config,
        session=session,
        runtime_installed=runtime_installed,
    )
    setattr(config, _PLUGIN_STATE_ATTR, state)
    return state


def _lookup_test_config(config: AgentTestConfig, item) -> Optional[TestConfig]:
    nodeid = item.nodeid
    name = item.name

    for test_cfg in config.tests:
        if test_cfg.name == name or test_cfg.name == nodeid:
            return test_cfg
        if nodeid.endswith(f"::{test_cfg.name}"):
            return test_cfg

    return None


def _resolve_baseline_name(item) -> str:
    marker = item.get_closest_marker("baseline")
    if marker and marker.args:
        return str(marker.args[0])
    return item.name


def _resolve_mode(item, config: AgentTestConfig) -> str:
    option_mode = item.config.getoption("--agenttest-mode")
    if option_mode:
        return option_mode

    test_cfg = _lookup_test_config(config, item)
    if test_cfg and test_cfg.mode:
        return test_cfg.mode

    return config.default_replay_mode


def _tier_matches(selected_tier: str, test_tier: str) -> bool:
    if selected_tier == "all":
        return True

    if test_tier == "always":
        return True

    return selected_tier == test_tier


def _uses_manual_fixtures(item) -> bool:
    fixtures = set(getattr(item, "fixturenames", []) or [])
    return bool(fixtures.intersection(_MANUAL_FIXTURES))
