"""
Configuration model for AgentTest.
"""

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


@dataclass
class TestConfig:
    """Per-test configuration including tier assignment."""
    name: str
    tier: str = "always"        # always | local | ci-only
    mode: Optional[str] = None  # per-test override for default_replay_mode

    def to_dict(self) -> dict:
        d = {"name": self.name, "tier": self.tier}
        if self.mode is not None:
            d["mode"] = self.mode
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TestConfig':
        return cls(
            name=d["name"],
            tier=d.get("tier", "always"),
            mode=d.get("mode")
        )


@dataclass
class ScenarioConfig:
    """
    Standalone CLI scenario configuration.

    Scenarios are executed by the AgentTest runner without pytest.
    """

    name: str
    entrypoint: str
    baseline_name: Optional[str] = None
    input_file: Optional[str] = None
    input_data: Any = None
    expects_llm: bool = True
    user_id: str = "agenttest"
    session_id: str = "default"

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "expects_llm": self.expects_llm,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        if self.baseline_name:
            data["baseline_name"] = self.baseline_name
        if self.input_file:
            data["input_file"] = self.input_file
        if self.input_data is not None:
            data["input"] = self.input_data
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        return cls(
            name=d["name"],
            entrypoint=d["entrypoint"],
            baseline_name=d.get("baseline_name"),
            input_file=d.get("input_file"),
            input_data=d.get("input"),
            expects_llm=bool(d.get("expects_llm", True)),
            user_id=str(d.get("user_id", "agenttest")),
            session_id=str(d.get("session_id", "default")),
        )


@dataclass
class AgentTestConfig:
    """Configuration for AgentTest recording/replay"""

    # Comparison settings
    similarity_threshold: float = 0.85
    default_replay_mode: str = "selective"  # locked, selective, full
    ignore_fields: List[str] = field(default_factory=list)

    # Interceptor configuration
    interceptors: Dict[str, Dict] = field(default_factory=dict)

    # Paths (database is managed by agentgit)
    agentgit_dir: str = ".agentgit"
    project_dir: str = "."

    # Test definitions with tier assignments
    tests: List[TestConfig] = field(default_factory=list)
    scenarios: List[ScenarioConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "similarity_threshold": self.similarity_threshold,
            "default_replay_mode": self.default_replay_mode,
            "ignore_fields": self.ignore_fields,
            "interceptors": self.interceptors,
            "agentgit_dir": self.agentgit_dir,
            "project_dir": self.project_dir,
            "tests": [t.to_dict() for t in self.tests],
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AgentTestConfig':
        tests = [TestConfig.from_dict(t) for t in data.get("tests", [])]
        scenarios = [ScenarioConfig.from_dict(s) for s in data.get("scenarios", [])]
        return cls(
            similarity_threshold=data.get("similarity_threshold", 0.85),
            default_replay_mode=data.get("default_replay_mode", "selective"),
            ignore_fields=data.get("ignore_fields", []),
            interceptors=data.get("interceptors", {}),
            agentgit_dir=data.get("agentgit_dir", ".agentgit"),
            project_dir=data.get("project_dir", "."),
            tests=tests,
            scenarios=scenarios,
        )
