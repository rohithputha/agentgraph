"""
AgentTest interceptors for LLM API calls.

The gatekeeper path depends on langchain_core. Runtime interception helpers are
always importable so the rest of the package can load in minimal environments.
"""

from agenttest.interceptors.runtime import (
    LockedModeNetworkError,
    get_active_recording_context,
    get_active_replay_context,
    install_global_runtime_interception,
    is_runtime_interception_installed,
    reset_active_recording_context,
    reset_active_replay_context,
    set_active_recording_context,
    set_active_replay_context,
    uninstall_global_runtime_interception,
)

LLMGatekeeper = None
LLMCacheMissError = None
ReplayMode = None
ResponseBuilder = None

try:
    from agenttest.interceptors.gatekeeper import (  # type: ignore
        LLMGatekeeper,
        LLMCacheMissError,
        ReplayMode,
    )
    from agenttest.interceptors.response_builder import ResponseBuilder  # type: ignore
except ImportError:
    pass

__all__ = [
    "LLMGatekeeper",
    "LLMCacheMissError",
    "ReplayMode",
    "ResponseBuilder",
    "LockedModeNetworkError",
    "set_active_recording_context",
    "reset_active_recording_context",
    "get_active_recording_context",
    "set_active_replay_context",
    "reset_active_replay_context",
    "get_active_replay_context",
    "install_global_runtime_interception",
    "uninstall_global_runtime_interception",
    "is_runtime_interception_installed",
]
