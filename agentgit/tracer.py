"""
Tracer - Subscribes to events from LangGraph callbacks and records nodes in the DAG.
"""

import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from .eventbus import Eventbus
from .event import Event, EventType
from .models.dag import ExecutionNode, ActionType, CallerType

if TYPE_CHECKING:
    from core import AgentGit


def _generate_id() -> str:
    """Generate a unique ID for nodes."""
    return str(uuid.uuid4())[:8]


class Tracer:
    """Listens to events and creates DAG nodes for each action."""

    def __init__(self, store: 'DagStore'):
        self.store = store
        self.eventbus = None  # Will be set by AgentGit
        self.current_turn = 0
        
    def handle_event(self, event: Event):
        handlers = {
            EventType.USER_INPUT: self._on_user_input,
            EventType.LLM_CALL_START: self._on_llm_call_start,
            EventType.LLM_CALL_END: self._on_llm_call_end,
            EventType.LLM_STREAM_CHUNK: self._on_stream_chunk,
            EventType.LLM_STREAM_END: self._on_stream_end,
            EventType.LLM_ERROR: self._on_llm_error,
            EventType.TOOL_CALL_START: self._on_tool_call_start,
            EventType.TOOL_CALL_END: self._on_tool_call_end,
            EventType.TOOL_ERROR: self._on_tool_error,
            EventType.AGENT_TURN_START: self._on_turn_start,
            EventType.AGENT_TURN_END: self._on_turn_end,
            EventType.AGENT_THINKING: self._on_thinking,
        }
        handler = handlers.get(event.type)
        if handler: 
            handler(event)
    


    def _on_user_input(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.USER_INPUT,
            triggered_by=CallerType.HUMAN_UI,
            content={"message": event.content},
        )

    def _on_llm_call_start(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.LLM_CALL,
            triggered_by=CallerType.SYSTEM,
            content={
                "model": event.model,
                "message_count": len(event.messages) if event.messages else 0,
            },
        )

    def _on_llm_call_end(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.LLM_RESPONSE,
            triggered_by=CallerType.SYSTEM,
            content={
                "model": event.model,
                "response": event.content,
                "usage": event.usage,
                "duration_ms": event.duration_ms,
            },
        )

    def _on_stream_chunk(self, event: Event):
        """Handle streaming chunk - typically don't create nodes for each chunk."""
        pass  # Streaming chunks are ephemeral, we record the final response

    def _on_stream_end(self, event: Event):
        """Handle stream end - create node with full response."""
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.LLM_RESPONSE,
            triggered_by=CallerType.SYSTEM,
            content={"response": event.content, "streamed": True},
        )

    def _on_llm_error(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.LLM_ERROR,
            triggered_by=CallerType.SYSTEM,
            content={"error": event.error, "model": event.model},
        )

    def _on_thinking(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.LLM_RESPONSE,
            triggered_by=CallerType.SYSTEM,
            content={"stage": "thinking", "reasoning": event.content},
        )

    def _on_tool_call_start(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.TOOL_CALL,
            triggered_by=CallerType.AGENT_TOOL,
            content={
                "tool": event.tool_name,
                "args": event.tool_args,
                "tool_call_id": event.metadata.get("tool_call_id") if event.metadata else None,
            },
        )

    def _on_tool_call_end(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.TOOL_RESULT,
            triggered_by=CallerType.AGENT_TOOL,
            content={
                "tool": event.tool_name,
                "result": event.content,
                "tool_call_id": event.metadata.get("tool_call_id") if event.metadata else None,
                "duration_ms": event.duration_ms,
            },
        )

    def _on_tool_error(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.TOOL_ERROR,
            triggered_by=CallerType.AGENT_TOOL,
            content={"tool": event.tool_name, "error": event.error},
        )

    def _on_turn_start(self, event: Event):
        self.current_turn += 1

    def _on_turn_end(self, event: Event):
        user_id = event.user_id or "default"
        session_id = event.session_id or "default"
        self._create_node(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.AGENT_TURN_END,
            triggered_by=CallerType.SYSTEM,
            content={},
        )

    # ─── Node Creation ─────────────────────────────────────────────

    def _create_node(self, user_id: str, session_id: str, action_type: ActionType, triggered_by: CallerType, content: dict) -> ExecutionNode:
        """Create node using session context from event (stateless!)."""
        # Query DB for active branch for this session
        branch = self.store.get_active_branch(user_id, session_id)
        if not branch:
            return None  # No active branch for this session

        parent_id = branch.head_node_id

        node = ExecutionNode(
            user_id=user_id,
            session_id=session_id,
            id=_generate_id(),
            parent_id=parent_id,
            action_type=action_type,
            content=content,
            triggered_by=triggered_by,
            caller_context={"turn": self.current_turn},
            state_hash=None,
            timestamp=datetime.now(),
            duration_ms=content.get("duration_ms", 0),
            token_count=None,
        )
        new_node_id = self.store.insert_node(user_id, session_id, node, branch.branch_id)
        self.store.update_branch_head(user_id, session_id, branch.branch_id, new_node_id)
        return node
    