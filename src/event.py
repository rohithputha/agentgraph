from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


class EventType(Enum):
    USER_INPUT = "user_input"
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    LLM_STREAM_CHUNK = "llm_stream_chunk"
    LLM_STREAM_END = "llm_stream_end"
    LLM_ERROR = "llm_error"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_ERROR = "tool_error"
    AGENT_TURN_START = "agent_turn_start"
    AGENT_TURN_END = "agent_turn_end"
    AGENT_THINKING = "agent_thinking"


@dataclass
class Event:
    """Payload for every event in the system."""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    content: Optional[str] = None
    text: Optional[str] = None 
    messages: Optional[list] = None
    outputs: Any = None  
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Any = None
    error: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[dict] = None
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)