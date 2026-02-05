from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ActionType(Enum):
    USER_INPUT = "user_input"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    CHECKPOINT = "checkpoint"
    BRANCH_CREATE = "branch_create"
    BRANCH_SWITCH = "branch_switch"
    BACKTRACK = "backtrack"
    AGENT_TURN_END = "agent_turn_end"


class CallerType(Enum):
    HUMAN_CLI = "human_cli"
    HUMAN_UI = "human_ui"
    AGENT_TOOL = "agent_tool"
    SYSTEM = "system"


class BranchStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    MERGED = "merged"


@dataclass
class ExecutionNode:
    """Lightweight node (~1KB) created for every agent action. Forms the DAG."""
    id: str
    parent_id: Optional[str]
    thread_id: str

    action_type: ActionType
    content: dict

    triggered_by: CallerType
    caller_context: dict

    state_hash: Optional[str]
    timestamp: datetime
    duration_ms: int
    token_count: Optional[int]


@dataclass
class Branch:
    """Named pointer to a position in the DAG."""
    name: str
    thread_id: str
    head_node_id: str
    base_node_id: str

    status: BranchStatus
    intent: str

    created_by: CallerType
    created_at: datetime

    tokens_used: int = 0
    time_elapsed_seconds: float = 0.0


@dataclass
class Checkpoint:
    """Heavy state snapshot. Only created explicitly."""
    hash: str
    agent_memory: dict
    conversation_history: list
    filesystem_ref: Optional[str]
    files_changed: list
    created_at: datetime
    compressed: bool
    size_bytes: int
    label: str = ""
