"""
AgentTest - Record-replay regression testing framework for AI agents
"""

__version__ = "0.1.0"

from agenttest.session import AgentTestSession
from agenttest.recorder import Recorder
from agenttest.replayer import Replayer
from agenttest.comparator import Comparison
from agenttest.config_loader import load_config
from agenttest.models.comparison import ComparisonResult, StepComparison, StepStatus, MatchType
from agenttest.models.recording import Recording, RecordingStatus
from agenttest.models.tag import Tag
from agenttest.models.config import AgentTestConfig, ScenarioConfig
from agenttest.pytest_plugin.assertions import assert_no_regression
from agenttest.runner.orchestrator import run_scenario_record, run_scenario_replay

__all__ = [
    "AgentTestSession", "Recorder", "Replayer", "Comparison",
    "ComparisonResult", "StepComparison", "StepStatus", "MatchType",
    "Recording", "RecordingStatus", "Tag", "AgentTestConfig", "ScenarioConfig",
    "load_config", "assert_no_regression",
    "run_scenario_record", "run_scenario_replay",
]
