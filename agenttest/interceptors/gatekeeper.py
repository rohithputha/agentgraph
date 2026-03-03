from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

from langchain_core.messages import AIMessage

from agenttest.session import AgentTestSession
from agenttest.interceptors.response_builder import ResponseBuilder
from agentgit.event import Event, EventType


class ReplayMode(Enum):
    FULL = "full"
    SELECTIVE = "selective"
    LOCKED = "locked"


class LLMCacheMissError(Exception):
    pass


class LLMGatekeeper:
    def __init__(self, session: AgentTestSession):
        self.session = session
        self._cache: List[Any] = []
        self._mode = ReplayMode.FULL
        self._response_builder = ResponseBuilder()

        self.cache_hits = 0
        self.cache_misses = 0
        self.live_calls = 0

    def load_baseline_cache(self, baseline_recording_id: str) -> None:
        details = self.session.get_recording_details(baseline_recording_id)
        self._cache = list(details)
        print(f"✅ Loaded {len(self._cache)} cached responses from baseline")

    def _find_cached_detail(self, message_dicts: List[Dict[str, str]]):
        for detail in self._cache:
            cached_messages = detail.request_params.get("messages", [])
            if cached_messages == message_dicts:
                return detail
        return None

    def create_middleware(self, mode: ReplayMode = ReplayMode.SELECTIVE) -> List:
        self._mode = mode

        if mode == ReplayMode.FULL:
            return []

        def check_llm_cache(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])
            if not messages:
                return state

            message_dicts = [
                {"role": getattr(msg, "type", "unknown"), "content": str(msg.content)}
                for msg in messages
            ]

            cached_detail = self._find_cached_detail(message_dicts)

            if cached_detail is not None:
                self.cache_hits += 1
                print(f"✅ LLM cache hit [{self._mode.value}]: {cached_detail.fingerprint[:8]}...")

                # Emit LLM_CALL_END event so replay recording captures this step
                self.session.ag.eventbus.publish(
                    EventType.LLM_CALL_END,
                    Event(
                        type=EventType.LLM_CALL_END,
                        user_id=self.session.user_id,
                        session_id=self.session.session_id,
                        model=cached_detail.model,
                        content=cached_detail.response_data.get("content", ""),
                        usage=cached_detail.token_usage,
                        duration_ms=cached_detail.duration_ms,
                        metadata={
                            "provider": cached_detail.provider,
                            "method": cached_detail.method,
                            "fingerprint": cached_detail.fingerprint,
                            "request_params": cached_detail.request_params,
                            "response_data": cached_detail.response_data,
                            "is_streaming": False,
                            "cached": True,
                            "was_cache_hit": True,
                        }
                    )
                )

                ai_message = self._response_builder.build_message(cached_detail.response_data)
                state["messages"].append(ai_message)
                return {"messages": state["messages"], "_skip_model": True}

            self.cache_misses += 1

            if self._mode == ReplayMode.LOCKED:
                raise LLMCacheMissError(
                    "LOCKED mode: no cached response for this LLM call. "
                    "Run with --mode=selective to allow live calls for new steps."
                )

            print(f"⚠️ LLM cache miss [{self._mode.value}]: (calling live)")
            self.live_calls += 1
            return state

        def log_llm_response(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                print(f"📝 LLM response captured (live call)")
            return state

        return [check_llm_cache, log_llm_response]

    def get_stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "live_calls": self.live_calls,
            "total_calls": self.cache_hits + self.live_calls,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.live_calls)
                if (self.cache_hits + self.live_calls) > 0
                else 0.0
            )
        }

    def reset_stats(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.live_calls = 0
