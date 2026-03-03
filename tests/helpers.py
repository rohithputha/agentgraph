from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from langchain_core.messages import HumanMessage
from agentgit.event import Event, EventType
from agenttest.fingerprint import compute_fingerprint
from agenttest.models.llm_call_detail import LLMCallDetail


def make_detail(
    *,
    recording_id: str,
    step_index: int,
    node_id: int,
    roles: Sequence[str],
    tool_names: Sequence[str] = (),
    model: str = "gpt-4",
    provider: str = "openai",
    method: str = "chat.completions.create",
    response_text: str = "ok",
    was_cache_hit: bool = True,
) -> LLMCallDetail:
    tools = [
        {
            "type": "function",
            "function": {"name": name},
        }
        for name in tool_names
    ]
    request_params = {
        "model": model,
        "messages": [{"role": role, "content": f"{role}-content"} for role in roles],
        "tools": tools,
    }
    fingerprint = compute_fingerprint(provider, method, request_params)
    return LLMCallDetail(
        id=step_index + 1,
        node_id=node_id,
        recording_id=recording_id,
        step_index=step_index,
        provider=provider,
        method=method,
        model=model,
        fingerprint=fingerprint,
        request_params=request_params,
        response_data={"content": response_text},
        was_cache_hit=was_cache_hit,
    )


def invoke_prompts(llm, prompts: Iterable[str], config: dict) -> List:
    outputs = []
    for prompt in prompts:
        outputs.append(
            llm.invoke(
                [HumanMessage(content=prompt)],
                config=config,
            )
        )
    return outputs


@dataclass
class SimpleMessage:
    content: str


class ScriptedLLM:
    """
    Lightweight test LLM that emits real AgentGit/AgentTest events without any network calls.
    """

    def __init__(
        self,
        *,
        eventbus,
        user_id: str,
        session_id: str,
        responses: Sequence[str],
        model_name: str = "scripted-llm",
        provider: str = "openai",
        method: str = "chat.completions.create",
    ):
        self._eventbus = eventbus
        self._user_id = user_id
        self._session_id = session_id
        self._responses = list(responses)
        self._idx = 0
        self.model_name = model_name
        self.provider = provider
        self.method = method

    @property
    def call_count(self) -> int:
        return self._idx

    def invoke(self, messages, *args, **kwargs):
        if self._idx >= len(self._responses):
            raise RuntimeError("No scripted response left")

        request_messages = []
        for message in messages:
            if isinstance(message, dict):
                request_messages.append(
                    {
                        "role": str(message.get("role", "unknown")),
                        "content": str(message.get("content", "")),
                    }
                )
            else:
                role = getattr(message, "role", None) or getattr(message, "type", None) or "unknown"
                content = getattr(message, "content", "")
                request_messages.append({"role": str(role), "content": str(content)})

        request_params = {
            "model": self.model_name,
            "messages": request_messages,
            "tools": [],
        }
        fingerprint = compute_fingerprint(self.provider, self.method, request_params)

        run_id = f"scripted-{uuid.uuid4().hex[:8]}"
        self._eventbus.publish(
            EventType.LLM_CALL_START,
            Event(
                type=EventType.LLM_CALL_START,
                user_id=self._user_id,
                session_id=self._session_id,
                run_id=run_id,
                model=self.model_name,
                messages=request_messages,
                metadata={
                    "provider": self.provider,
                    "method": self.method,
                    "request_params": request_params,
                    "fingerprint": fingerprint,
                },
            ),
        )

        content = self._responses[self._idx]
        self._idx += 1

        response_data = {
            "content": content,
            "tool_calls": [],
            "usage": {},
        }
        self._eventbus.publish(
            EventType.LLM_CALL_END,
            Event(
                type=EventType.LLM_CALL_END,
                user_id=self._user_id,
                session_id=self._session_id,
                run_id=run_id,
                model=self.model_name,
                content=content,
                usage={},
                duration_ms=1,
                metadata={
                    "provider": self.provider,
                    "method": self.method,
                    "fingerprint": fingerprint,
                    "request_params": request_params,
                    "response_data": response_data,
                    "is_streaming": False,
                    "was_cache_hit": False,
                },
            ),
        )

        return SimpleMessage(content=content)


def invoke_scripted_prompts(llm: ScriptedLLM, prompts: Iterable[str]) -> List[SimpleMessage]:
    outputs: List[SimpleMessage] = []
    for prompt in prompts:
        outputs.append(
            llm.invoke(
                [{"role": "human", "content": prompt}],
            )
        )
    return outputs
