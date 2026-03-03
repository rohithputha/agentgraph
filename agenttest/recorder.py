from typing import Optional
from agenttest.session import AgentTestSession
from agenttest.models.config import AgentTestConfig
from agenttest.interceptors.runtime import (
    install_global_runtime_interception,
    reset_active_recording_context,
    set_active_recording_context,
)


class Recorder:
    def __init__(
        self,
        session: AgentTestSession,
        name: str,
        config: Optional[AgentTestConfig] = None
    ):
        self.session = session
        self.name = name
        self.config = config or session.config
        self.recording = None
        self.set_as_baseline_on_exit = False
        self._runtime_recording_token = None

    def __enter__(self):
        install_global_runtime_interception()
        self.recording = self.session.create_recording(
            name=self.name,
            config=self.config
        )
        self._runtime_recording_token = set_active_recording_context(
            session=self.session,
            mode="record",
            run_name=self.name,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            error = str(exc_val) if exc_type else None
            self.session.complete_recording(self.recording.recording_id, error=error)
            if exc_type is None and self.set_as_baseline_on_exit:
                self.session.set_baseline(self.name, self.recording.recording_id)
            return False
        finally:
            reset_active_recording_context(self._runtime_recording_token)
            self._runtime_recording_token = None

    def set_as_baseline(self, name: Optional[str] = None):
        if not self.recording:
            raise RuntimeError("No active recording")
        baseline_name = name or self.name
        self.session.set_baseline(baseline_name, self.recording.recording_id)

    @property
    def recording_id(self):
        return self.recording.recording_id if self.recording else None

    @property
    def step_count(self):
        return self.recording.step_count if self.recording else 0
