

import time
import uuid
from datetime import datetime
from eventbus  import Eventbus
from models.dag import *
from models.agit import Agit

class Tracer:
    def __init__(self,agit: Agit):
        self.eventbus = agit.eventbus
        self.store = agit.store
        self.agit = agit
        self.eventbus.subscribe_all(self.handle_event)
        
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
        self._create_node(
            action_type=ActionType.USER_INPUT,
            triggered_by=CallerType.HUMAN_UI,
            content={"message": event.content},
        )

    def _on_llm_call_start(self, event: Event):
        self._create_node(
            action_type=ActionType.LLM_CALL,
            triggered_by=CallerType.SYSTEM,
            content={
                "model": event.model,
                "message_count": len(event.messages) if event.messages else 0,
            },
        )

    def _on_llm_call_end(self, event: Event):
        self._create_node(
            action_type=ActionType.LLM_RESPONSE,
            triggered_by=CallerType.SYSTEM,
            content={
                "model": event.model,
                "response": event.content,
                "usage": event.usage,
                "duration_ms": event.duration_ms,
            },
        )

    def _on_llm_error(self, event: Event):
        self._create_node(
            action_type=ActionType.LLM_ERROR,
            triggered_by=CallerType.SYSTEM,
            content={"error": event.error, "model": event.model},
        )

    def _on_thinking(self, event: Event):
        self._create_node(
            action_type=ActionType.LLM_RESPONSE,
            triggered_by=CallerType.SYSTEM,
            content={"stage": "thinking", "reasoning": event.content},
        )

    def _on_tool_call_start(self, event: Event):
        self._create_node(
            action_type=ActionType.TOOL_CALL,
            triggered_by=CallerType.AGENT_TOOL,
            content={
                "tool": event.tool_name,
                "args": event.tool_args,
                "tool_call_id": event.metadata.get("tool_call_id"),
            },
        )

    def _on_tool_call_end(self, event: Event):
        dedup_key = f"{event.tool_name}:{event.content}"
        if self._is_duplicate(dedup_key):
            return
        self._mark_seen(dedup_key)

        self._create_node(
            action_type=ActionType.TOOL_RESULT,
            triggered_by=CallerType.AGENT_TOOL,
            content={
                "tool": event.tool_name,
                "result": event.content,
                "tool_call_id": event.metadata.get("tool_call_id"),
                "duration_ms": event.duration_ms,
            },
        )

    def _on_tool_error(self, event: Event):
        self._create_node(
            action_type=ActionType.TOOL_ERROR,
            triggered_by=CallerType.AGENT_TOOL,
            content={"tool": event.tool_name, "error": event.error},
        )

    def _on_turn_start(self, event: Event):
        self.current_turn += 1

    def _on_turn_end(self, event: Event):
        self._create_node(
            action_type=ActionType.AGENT_TURN_END,
            triggered_by=CallerType.SYSTEM,
            content={},
        )
    

    def _create_node(self, action_type: ActionType, triggered_by: CallerType, content: dict) -> ExecutionNode:
        node = ExecutionNode(
            id=_generate_id(),
            parent_id=self.agit.current_node_id,
            thread_id=self.agit.current_thread_id,
            action_type=action_type,
            content=content,
            triggered_by=triggered_by,
            caller_context={"turn": self.current_turn},
            state_hash=None,
            timestamp=datetime.now(),
            duration_ms=content.get("duration_ms", 0),
            token_count=None,
        )
        new_node_id = self.store.insert_node(node)
        self.agit.current_node_id = new_node_id
        self.store.update_branch_head(self.agit.current_branch_id, new_node_id)
        return node
    