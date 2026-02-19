from typing import Optional
from agenttest.session import AgentTestSession
from agenttest.models.config import AgentTestConfig


class Recorder:
    def __init__(
        self,
        session: AgentTestSession,
        name: str,
        config: Optional[AgentTestConfig] = None
    ):
        self.session = session
        self.name = name
        self.config = config or getattr(session, 'config', AgentTestConfig())
        self.recording = None
        self.set_as_baseline_on_exit = False

    def __enter__(self):
        self.recording = self.session.create_recording(
            name=self.name,
            config=self.config
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.session.complete_recording(
                self.recording.recording_id
            )
        else:
            self.session.complete_recording(
                self.recording.recording_id,
                error=str(exc_val)
            )

        return False

    @property
    def recording_id(self):
        return self.recording.recording_id if self.recording else None

    @property
    def step_count(self):
        return self.recording.step_count if self.recording else 0
