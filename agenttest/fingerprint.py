
import hashlib
import json
from typing import Dict, List, Any, Optional
from agenttest.models.llm_call_detail import LLMCallDetail

def compute_fingerprint(
    provider: str,
    method: str,
    request_params: Dict[str, Any]
) -> str:

    model = request_params.get("model", "")
    message_roles = _extract_message_roles(request_params)
    tool_names = _extract_tool_names(request_params)

    # Build structural signature
    # Order matters for ALL components: provider, method, model, roles, tools
    structural_parts = [
        provider,
        method,
        model,
        json.dumps(message_roles),      # Preserve message order
        json.dumps(tool_names)           # PRESERVE TOOL ORDER (not sorted!)
    ]

    # Combine with separator and hash
    combined = "|".join(structural_parts)
    hash_digest = hashlib.sha256(combined.encode('utf-8')).hexdigest()

    # Return first 16 characters (64 bits)
    # Collision probability: ~1 in 18 quintillion
    return hash_digest[:16]


def _extract_message_roles(params: Dict[str, Any]) -> List[str]:
    """
    Extract message role sequence from request parameters.

    Order is preserved: ["system", "user", "assistant"] ≠ ["user", "system", "assistant"]

    Supports both OpenAI and Anthropic message formats.

    Args:
        params: Request parameters dict

    Returns:
        List of role strings in order (e.g., ["user", "assistant", "user"])

    Example:
        >>> params = {
        ...     "messages": [
        ...         {"role": "system", "content": "You are helpful"},
        ...         {"role": "user", "content": "Hello"},
        ...         {"role": "assistant", "content": "Hi!"}
        ...     ]
        ... }
        >>> _extract_message_roles(params)
        ['system', 'user', 'assistant']
    """
    messages = params.get("messages", [])

    # Handle different message formats
    roles = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            if role:
                roles.append(role)

    return roles


def _extract_tool_names(params: Dict[str, Any]) -> List[str]:
    tool_names = []

    # OpenAI tools format: [{"type": "function", "function": {"name": "..."}}]
    tools = params.get("tools", [])
    for tool in tools:
        if isinstance(tool, dict):
            # OpenAI format
            if "function" in tool:
                func = tool["function"]
                if isinstance(func, dict) and "name" in func:
                    tool_names.append(func["name"])
            # Anthropic format: {"name": "...", "description": "..."}
            elif "name" in tool:
                tool_names.append(tool["name"])

    # OpenAI functions (deprecated): [{"name": "...", "description": "..."}]
    functions = params.get("functions", [])
    for func in functions:
        if isinstance(func, dict) and "name" in func:
            tool_names.append(func["name"])

    return tool_names


def compute_fingerprint_from_detail(detail: LLMCallDetail) -> str:
    return compute_fingerprint(detail.provider, detail.method, detail.request_params)


def verify_fingerprint(fingerprint: str, provider: str, method: str, request_params: Dict[str, Any]) -> bool:
    computed = compute_fingerprint(provider, method, request_params)
    return computed == fingerprint
