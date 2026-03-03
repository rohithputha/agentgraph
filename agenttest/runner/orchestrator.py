from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agenttest.config_loader import load_config
from agenttest.models.config import AgentTestConfig, ScenarioConfig
from agenttest.recorder import Recorder
from agenttest.replayer import Replayer
from agenttest.runner.scenario_backend import (
    build_scenario_context,
    load_scenario_payload,
    run_scenario_callable,
)
from agenttest.session import AgentTestSession


@dataclass
class RunResult:
    exit_code: int
    status: str
    message: str
    comparison_id: Optional[str] = None


def _find_scenario(config: AgentTestConfig, name: str) -> ScenarioConfig:
    for scenario in config.scenarios:
        if scenario.name == name:
            return scenario
    known = [s.name for s in config.scenarios]
    raise ValueError(f"Scenario '{name}' not found. Configured scenarios: {known}")


def _make_session(project_dir: str, config: AgentTestConfig, scenario: ScenarioConfig) -> AgentTestSession:
    return AgentTestSession.standalone(
        project_dir=project_dir,
        user_id=scenario.user_id,
        session_id=scenario.session_id,
        config=config,
    )


def run_scenario_record(
    *,
    project_dir: str,
    scenario_name: str,
    config_path: Optional[str] = None,
    set_as_baseline: bool = True,
    recording_name: Optional[str] = None,
) -> RunResult:
    resolved_config_path = config_path
    if resolved_config_path is None:
        candidate = Path(project_dir) / "agenttest.toml"
        if candidate.exists():
            resolved_config_path = str(candidate)
    config = load_config(resolved_config_path)
    scenario = _find_scenario(config, scenario_name)
    payload = load_scenario_payload(project_dir, scenario)
    run_name = recording_name or scenario.baseline_name or scenario.name

    session = _make_session(project_dir, config, scenario)
    try:
        recorder = Recorder(session=session, name=run_name, config=config)
        recorder.set_as_baseline_on_exit = False
        with recorder:
            ctx = build_scenario_context(session, payload)
            run_scenario_callable(scenario.entrypoint, ctx)

        detail_count = len(session.get_recording_details(recorder.recording_id))
        if scenario.expects_llm and detail_count == 0:
            return RunResult(
                exit_code=3,
                status="CONFIG_ERROR",
                message=(
                    f"Scenario '{scenario.name}' expected LLM activity but captured 0 steps. "
                    "Ensure your scenario invokes an LLM through a supported path."
                ),
            )

        if set_as_baseline and detail_count > 0:
            session.set_baseline(run_name, recorder.recording_id)

        return RunResult(
            exit_code=0,
            status="RECORDED",
            message=f"Recorded scenario '{scenario.name}' as '{run_name}' ({detail_count} steps).",
        )
    finally:
        session.close()


def run_scenario_replay(
    *,
    project_dir: str,
    scenario_name: str,
    mode: str = "locked",
    config_path: Optional[str] = None,
    replay_name: Optional[str] = None,
) -> RunResult:
    resolved_config_path = config_path
    if resolved_config_path is None:
        candidate = Path(project_dir) / "agenttest.toml"
        if candidate.exists():
            resolved_config_path = str(candidate)
    config = load_config(resolved_config_path)
    scenario = _find_scenario(config, scenario_name)
    payload = load_scenario_payload(project_dir, scenario)
    baseline_name = scenario.baseline_name or scenario.name
    resolved_replay_name = replay_name or f"{baseline_name}-replay"

    session = _make_session(project_dir, config, scenario)
    try:
        with Replayer(
            session=session,
            baseline_name=baseline_name,
            replay_name=resolved_replay_name,
            mode=mode,
            config=config,
        ) as replay:
            ctx = build_scenario_context(session, payload)
            run_scenario_callable(scenario.entrypoint, ctx)

        if scenario.expects_llm and len(replay.replay_details) == 0:
            return RunResult(
                exit_code=3,
                status="CONFIG_ERROR",
                message=(
                    f"Scenario '{scenario.name}' expected LLM activity but replay captured 0 steps. "
                    "Ensure your scenario invokes an LLM through a supported path."
                ),
            )

        comparison_id = replay.comparison_result.comparison_id if replay.comparison_result else None
        if replay.comparison_result is None:
            return RunResult(
                exit_code=3,
                status="CONFIG_ERROR",
                message="Replay completed but no comparison result was produced.",
            )

        result = replay.comparison_result
        if result.overall_pass:
            return RunResult(
                exit_code=0,
                status="PASS",
                message=f"Scenario '{scenario.name}' replay passed.",
                comparison_id=comparison_id,
            )

        if result.has_regression:
            return RunResult(
                exit_code=1,
                status="REGRESSION",
                message=f"Scenario '{scenario.name}' replay detected regression.",
                comparison_id=comparison_id,
            )

        return RunResult(
            exit_code=2,
            status="DELTA",
            message=f"Scenario '{scenario.name}' replay detected delta requiring acceptance.",
            comparison_id=comparison_id,
        )
    finally:
        session.close()
