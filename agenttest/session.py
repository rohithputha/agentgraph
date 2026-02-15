"""
AgentTestSession - Central orchestrator for AgentTest.

Wraps an AgentGit instance, shares its database connection,
and subscribes to the eventbus for unified LLM capture.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from agentgit import AgentGit
from agentgit.models.dag import ExecutionNode, ActionType, CallerType
from agentgit.event import Event, EventType

from agenttest.storage.test_store import TestStore
from agenttest.models.recording import Recording, RecordingStatus
from agenttest.models.tag import Tag
from agenttest.models.llm_call_detail import LLMCallDetail
from agenttest.models.comparison import ComparisonResult
from agenttest.models.config import AgentTestConfig


class AgentTestSession:
    """
    Central orchestrator for agenttest.

    Wraps an AgentGit instance, shares its database connection,
    and subscribes to the eventbus for unified LLM capture.

    Node creation is handled by the existing Tracer.
    This session only creates sidecar LLMCallDetail records
    when recording mode is active.
    """

    def __init__(
        self,
        agentgit: AgentGit,
        user_id: str = "agenttest",
        session_id: str = "default"
    ):
        """
        Initialize AgentTestSession.

        Args:
            agentgit: AgentGit instance to wrap
            user_id: User identifier for multi-user support
            session_id: Session identifier for isolation
        """
        self.ag = agentgit
        self.user_id = user_id
        self.session_id = session_id

        # Connection sharing - the key V3 mechanism
        self.test_store = TestStore(self.ag.dag_store.conn)

        # Recording state
        self._active_recording: Optional[Recording] = None

        # Subscribe to eventbus - unified LLM capture
        # The Tracer is subscribed FIRST (in AgentGit.__init__ via subscribe_all),
        # so by the time our handler runs, the node already exists.
        self.ag.eventbus.subscribe(EventType.LLM_CALL_END, self._on_llm_call_end)

    @classmethod
    def standalone(
        cls,
        project_dir: str = ".",
        user_id: str = "agenttest",
        session_id: str = "default"
    ) -> 'AgentTestSession':
        """
        Create a standalone session with its own AgentGit instance.

        Args:
            project_dir: Project directory path
            user_id: User identifier
            session_id: Session identifier

        Returns:
            AgentTestSession instance
        """
        ag = AgentGit(project_dir=project_dir)
        return cls(ag, user_id, session_id)

    # ==================== Recording Lifecycle ====================

    def create_recording(
        self,
        name: str,
        config: Optional[AgentTestConfig] = None
    ) -> Recording:
        """
        Create a recording and activate its branch.

        Args:
            name: Recording name
            config: Optional configuration snapshot

        Returns:
            Created Recording object
        """
        recording_id = f"rec_{uuid.uuid4().hex[:12]}"
        branch_name = f"recording/{name}"

        # Create branch in AgentGit
        branch_id = self.ag.create_branch(
            user_id=self.user_id,
            session_id=self.session_id,
            name=branch_name,
            base_node_id=None
        )

        # Create Recording object
        recording = Recording(
            recording_id=recording_id,
            name=name,
            user_id=self.user_id,
            session_id=self.session_id,
            branch_id=branch_id,
            status=RecordingStatus.IN_PROGRESS,
            created_at=datetime.now(),
            step_count=0,
            error=None,
            config=config.to_dict() if config else None,
            metadata={}
        )

        # Store recording
        self.test_store.insert_recording(recording)

        # Note: Branch is automatically active after creation (status='active')
        # No need to explicitly activate it

        # Activate recording so _on_llm_call_end creates sidecar records
        self._active_recording = recording

        return recording

    def complete_recording(
        self,
        recording_id: str,
        error: Optional[str] = None
    ) -> Recording:
        """
        Mark recording as completed or failed and deactivate.

        Args:
            recording_id: Recording ID to complete
            error: Optional error message if failed

        Returns:
            Updated Recording object
        """
        status = RecordingStatus.FAILED if error else RecordingStatus.COMPLETED
        self.test_store.update_recording_status(
            recording_id=recording_id,
            status=status.value,
            error=error
        )

        # Deactivate recording
        self._active_recording = None

        return self.test_store.get_recording(recording_id)

    # ==================== Eventbus Subscriber ====================

    def _on_llm_call_end(self, event: Event) -> None:
        """
        Eventbus subscriber: creates sidecar LLMCallDetail when recording is active.

        Ordering guarantee: The Tracer (subscribed first via subscribe_all in
        AgentGit.__init__) has already:
          1. Created the ExecutionNode in the nodes table
          2. Updated the branch head to point to the new node

        So we can safely read the branch head to get the node_id.

        Testing-specific data (provider, method, request_params, response_data,
        fingerprint) is carried in Event.metadata - the existing dict field.
        No AgentGit Event model changes needed.

        Args:
            event: LLM_CALL_END event from eventbus
        """
        if not self._active_recording:
            return  # Not recording, skip

        # Read the node_id from branch head (just created by Tracer)
        user_id = event.user_id or self.user_id
        session_id = event.session_id or self.session_id
        branch = self.ag.dag_store.get_active_branch(user_id, session_id)

        if not branch or not branch.head_node_id:
            return

        node_id = int(branch.head_node_id)

        # Extract testing-specific data from event.metadata
        meta = event.metadata or {}

        # Create LLMCallDetail sidecar record
        detail = LLMCallDetail(
            id=None,
            node_id=node_id,
            recording_id=self._active_recording.recording_id,
            step_index=self._active_recording.step_count,
            provider=meta.get("provider", "unknown"),
            method=meta.get("method", "unknown"),
            model=event.model or "unknown",
            fingerprint=meta.get("fingerprint", ""),
            request_params=meta.get("request_params", {}),
            response_data=meta.get("response_data", {}),
            is_streaming=meta.get("is_streaming", False),
            stream_id=meta.get("stream_id"),
            duration_ms=event.duration_ms or 0,
            token_usage=event.usage,
            error=None,
            metadata={}
        )

        # Insert into database
        detail_id = self.test_store.insert_llm_call_detail(detail)
        detail.id = detail_id

        # Update recording step count
        self._active_recording.step_count += 1
        self.test_store.update_recording_status(
            self._active_recording.recording_id,
            status=RecordingStatus.IN_PROGRESS.value,
            step_count=self._active_recording.step_count
        )

    # ==================== Tag/Baseline Management ====================

    def set_baseline(self, name: str, recording_id: str) -> Tag:
        """
        Tag the recording's branch head node as a baseline.

        Args:
            name: Baseline name
            recording_id: Recording to promote to baseline

        Returns:
            Created Tag object
        """
        # Get recording and its branch
        recording = self.test_store.get_recording(recording_id)
        if not recording:
            raise ValueError(f"Recording {recording_id} not found")

        branch = self.ag.dag_store.get_branch_by_id(recording.branch_id)
        if not branch or not branch.head_node_id:
            raise ValueError(f"Recording {recording_id} has no nodes")

        # Create baseline tag
        tag = Tag(
            tag_id=None,
            tag_name=f"baseline/{name}",
            user_id=self.user_id,
            session_id=self.session_id,
            node_id=int(branch.head_node_id),
            tag_type="baseline",
            description=f"Baseline for {name}",
            metadata={"recording_id": recording_id},
            created_at=datetime.now()
        )

        tag_id = self.test_store.insert_tag(tag)
        tag.tag_id = tag_id

        return tag

    def get_baseline(self, name: str) -> Optional[Tag]:
        """
        Get baseline tag by name.

        Args:
            name: Baseline name (without "baseline/" prefix)

        Returns:
            Tag object if found, None otherwise
        """
        tag_name = f"baseline/{name}"
        return self.test_store.get_tag(self.user_id, self.session_id, tag_name)

    def list_baselines(self) -> List[Tag]:
        """
        List all baseline tags.

        Returns:
            List of baseline Tag objects
        """
        return self.test_store.list_tags(
            self.user_id,
            self.session_id,
            tag_type="baseline"
        )

    def delete_baseline(self, name: str) -> bool:
        """
        Delete a baseline tag.

        Args:
            name: Baseline name (without "baseline/" prefix)

        Returns:
            True if deleted, False if not found
        """
        tag_name = f"baseline/{name}"
        return self.test_store.delete_tag(self.user_id, self.session_id, tag_name)

    # ==================== Comparison Storage ====================

    def store_comparison(self, result: ComparisonResult) -> str:
        """
        Store comparison result in database.

        Args:
            result: ComparisonResult to store

        Returns:
            Comparison ID
        """
        return self.test_store.insert_comparison(
            result,
            user_id=self.user_id,
            session_id=self.session_id
        )

    def get_comparison(self, comparison_id: str) -> Optional[ComparisonResult]:
        """
        Get comparison result by ID.

        Args:
            comparison_id: Comparison ID

        Returns:
            ComparisonResult if found, None otherwise
        """
        return self.test_store.get_comparison(comparison_id)

    def list_comparisons(
        self,
        baseline_recording_id: Optional[str] = None,
        failed_only: bool = False
    ) -> List[ComparisonResult]:
        """
        List comparison results with optional filters.

        Args:
            baseline_recording_id: Filter by baseline recording
            failed_only: Only return failed comparisons

        Returns:
            List of ComparisonResult objects
        """
        return self.test_store.list_comparisons(
            self.user_id,
            self.session_id,
            baseline_recording_id=baseline_recording_id,
            failed_only=failed_only
        )

    # ==================== Query Helpers ====================

    def get_recording(self, recording_id: str) -> Optional[Recording]:
        """
        Get recording by ID.

        Args:
            recording_id: Recording ID

        Returns:
            Recording if found, None otherwise
        """
        return self.test_store.get_recording(recording_id)

    def get_recording_details(self, recording_id: str) -> List[LLMCallDetail]:
        """
        Get all LLM call details for a recording.

        Args:
            recording_id: Recording ID

        Returns:
            List of LLMCallDetail objects ordered by step_index
        """
        return self.test_store.get_recording_details(recording_id)

    def get_recording_by_name(self, name: str) -> Optional[Recording]:
        """
        Get recording by name.

        Args:
            name: Recording name

        Returns:
            Recording if found, None otherwise
        """
        return self.test_store.get_recording_by_name(
            self.user_id,
            self.session_id,
            name
        )

    def list_recordings(
        self,
        status: Optional[RecordingStatus] = None
    ) -> List[Recording]:
        """
        List recordings with optional status filter.

        Args:
            status: Filter by status (IN_PROGRESS, COMPLETED, FAILED)

        Returns:
            List of Recording objects ordered by created_at DESC
        """
        status_str = status.value if status else None
        return self.test_store.list_recordings(
            self.user_id,
            self.session_id,
            status=status_str
        )

    # ==================== Lifecycle ====================

    def close(self):
        """Close the session (delegates to AgentGit)"""
        self.ag.close()
