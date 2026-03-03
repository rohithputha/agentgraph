import copy
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Deque, Tuple, Set
from enum import Enum

from langchain_core.messages import AIMessage

from agenttest.session import AgentTestSession
from agenttest.interceptors.response_builder import ResponseBuilder
from agenttest.fingerprint import compute_fingerprint
from agenttest.models.llm_call_detail import LLMCallDetail
from agentgit.event import Event, EventType

logger = logging.getLogger(__name__)


class ReplayMode(Enum):
    FULL = "full"
    SELECTIVE = "selective"
    LOCKED = "locked"


class LLMCacheMissError(Exception):
    pass


@dataclass
class CacheDecision:
    hit: bool
    ai_message: Optional[AIMessage]
    cached_detail: Optional[LLMCallDetail]
    fingerprint: str


class LLMGatekeeper:
    def __init__(self, session: AgentTestSession):
        self.session = session
        self._cache: List[LLMCallDetail] = []
        self._cache_by_fingerprint: Dict[str, Deque[LLMCallDetail]] = defaultdict(deque)
        self._cache_by_role_signature: Dict[Tuple[str, ...], Deque[LLMCallDetail]] = defaultdict(deque)
        self._consumed_detail_keys: Set[Tuple[str, int, int]] = set()
        self._sequence_cursor = 0

        self._mode = ReplayMode.FULL
        self._response_builder = ResponseBuilder()

        self.interception_attempts = 0
        self.wrapper_interceptions = 0
        self.middleware_interceptions = 0
        self.runtime_interceptions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.live_calls = 0

    def _rebuild_indices_from_cache(self) -> None:
        self._cache_by_fingerprint = defaultdict(deque)
        self._cache_by_role_signature = defaultdict(deque)
        self._consumed_detail_keys = set()
        self._sequence_cursor = 0

        for detail in self._cache:
            provider = getattr(detail, "provider", "unknown")
            method = getattr(detail, "method", "chat")
            request_params = getattr(detail, "request_params", {}) or {}
            fingerprint = getattr(detail, "fingerprint", "") or compute_fingerprint(
                provider,
                method,
                request_params
            )
            try:
                detail.fingerprint = fingerprint
            except Exception:
                pass

            self._cache_by_fingerprint[fingerprint].append(detail)
            role_sig = self._extract_role_signature(
                request_params.get("messages", [])
            )
            self._cache_by_role_signature[role_sig].append(detail)

    def load_baseline_cache(self, baseline_recording_id: str) -> None:
        details = self.session.get_recording_details(baseline_recording_id)
        self.reset_stats()

        self._cache = []
        for detail in details:
            snap = copy.deepcopy(detail)
            if not snap.fingerprint:
                snap.fingerprint = compute_fingerprint(
                    snap.provider,
                    snap.method,
                    snap.request_params
                )

            self._cache.append(snap)
        self._rebuild_indices_from_cache()

        logger.info(
            "Loaded %d cached responses from baseline %s",
            len(self._cache),
            baseline_recording_id,
        )

    def _detail_key(self, detail: LLMCallDetail) -> Tuple[str, int, int]:
        recording_id = getattr(detail, "recording_id", "")
        node_id = getattr(detail, "node_id", -1)
        step_index = getattr(detail, "step_index", -1)

        if not isinstance(recording_id, str):
            recording_id = ""
        if not isinstance(node_id, int):
            node_id = -1
        if not isinstance(step_index, int):
            step_index = -1

        if not recording_id and node_id < 0 and step_index < 0:
            return ("__obj__", id(detail), 0)

        if node_id < 0:
            node_id = 0
        if step_index < 0:
            step_index = 0
        return (recording_id, node_id, step_index)

    def _extract_role_signature(self, messages: List[Dict[str, Any]]) -> Tuple[str, ...]:
        return tuple(str(msg.get("role", "")) for msg in messages if isinstance(msg, dict))

    def _peek_unconsumed(self, queue: Optional[Deque[LLMCallDetail]]) -> Optional[LLMCallDetail]:
        if not queue:
            return None
        while queue and self._detail_key(queue[0]) in self._consumed_detail_keys:
            queue.popleft()
        return queue[0] if queue else None

    def _next_expected_locked_detail(self) -> Optional[LLMCallDetail]:
        while self._sequence_cursor < len(self._cache):
            detail = self._cache[self._sequence_cursor]
            if self._detail_key(detail) in self._consumed_detail_keys:
                self._sequence_cursor += 1
                continue
            return detail
        return None

    def _is_allowed_candidate(self, detail: LLMCallDetail) -> bool:
        if self._mode != ReplayMode.LOCKED:
            return True
        expected = self._next_expected_locked_detail()
        if expected is None:
            return False
        return self._detail_key(expected) == self._detail_key(detail)

    def _consume_head(self, queue: Deque[LLMCallDetail]) -> Optional[LLMCallDetail]:
        detail = self._peek_unconsumed(queue)
        if detail is None:
            return None
        popped = queue.popleft()
        key = self._detail_key(popped)
        self._consumed_detail_keys.add(key)
        if self._mode == ReplayMode.LOCKED:
            expected = self._next_expected_locked_detail()
            if expected and self._detail_key(expected) == key:
                self._sequence_cursor += 1
        return popped

    def _find_cached_detail_internal(
        self,
        message_dicts: Optional[List[Dict[str, str]]] = None,
        provider: str = "unknown",
        method: str = "chat",
        request_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[LLMCallDetail], str]:
        if request_params is None:
            request_params = {
                "model": "",
                "messages": message_dicts or [],
                "tools": [],
            }
        if not self._cache_by_fingerprint and self._cache:
            self._rebuild_indices_from_cache()

        fingerprint = compute_fingerprint(provider, method, request_params)
        fp_queue = self._cache_by_fingerprint.get(fingerprint)
        fp_candidate = self._peek_unconsumed(fp_queue)

        if fp_candidate and self._is_allowed_candidate(fp_candidate):
            return self._consume_head(fp_queue), fingerprint

        role_sig = self._extract_role_signature(request_params.get("messages", []))
        role_queue = self._cache_by_role_signature.get(role_sig)
        role_candidate = self._peek_unconsumed(role_queue)

        if role_candidate and self._is_allowed_candidate(role_candidate):
            return self._consume_head(role_queue), fingerprint

        return None, fingerprint

    # Backward-compatible helper used by older integration tests.
    def _find_cached_detail(self, message_dicts: List[Dict[str, str]]):
        for detail in self._cache:
            cached_messages = getattr(detail, "request_params", {}).get("messages", [])
            if cached_messages == message_dicts:
                return detail
        return None

    def _to_message_dict(self, msg: Any) -> Dict[str, str]:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type") or "unknown"
            content = msg.get("content", "")
            return {"role": str(role), "content": str(content)}

        role = getattr(msg, "type", None) or getattr(msg, "role", None) or "unknown"
        content = getattr(msg, "content", "")
        return {"role": str(role), "content": str(content)}

    def _build_request_params(
        self,
        messages: List[Any],
        model: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        invocation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        invocation = invocation or {}
        safe_tools: List[Dict[str, Any]] = []
        if isinstance(tools, list):
            safe_tools = [t for t in tools if isinstance(t, dict)]
        return {
            "model": model,
            "messages": [self._to_message_dict(msg) for msg in messages],
            "tools": safe_tools,
            "temperature": invocation.get("temperature"),
            "max_tokens": invocation.get("max_tokens"),
            "top_p": invocation.get("top_p"),
            "stream": bool(invocation.get("stream", False)),
        }

    def _emit_cached_event(self, detail: LLMCallDetail):
        self.session.ag.eventbus.publish(
            EventType.LLM_CALL_END,
            Event(
                type=EventType.LLM_CALL_END,
                user_id=self.session.user_id,
                session_id=self.session.session_id,
                model=detail.model,
                content=detail.response_data.get("content", ""),
                usage=copy.deepcopy(detail.token_usage),
                duration_ms=detail.duration_ms,
                metadata={
                    "provider": detail.provider,
                    "method": detail.method,
                    "fingerprint": detail.fingerprint,
                    "request_params": copy.deepcopy(detail.request_params),
                    "response_data": copy.deepcopy(detail.response_data),
                    "is_streaming": False,
                    "cached": True,
                    "was_cache_hit": True,
                }
            )
        )

    def check_request(
        self,
        provider: str,
        method: str,
        request_params: Dict[str, Any],
        source: str = "unknown"
    ) -> CacheDecision:
        self.interception_attempts += 1
        if source == "wrapper":
            self.wrapper_interceptions += 1
        elif source == "middleware":
            self.middleware_interceptions += 1
        elif source == "runtime":
            self.runtime_interceptions += 1

        cached_detail, fingerprint = self._find_cached_detail_internal(
            provider=provider,
            method=method,
            request_params=request_params
        )

        if cached_detail is not None:
            self.cache_hits += 1
            logger.info(
                "LLM cache hit [%s]: %s...",
                self._mode.value,
                cached_detail.fingerprint[:8],
            )
            self._emit_cached_event(cached_detail)
            ai_message = self._response_builder.build_message(
                copy.deepcopy(cached_detail.response_data)
            )
            return CacheDecision(
                hit=True,
                ai_message=ai_message,
                cached_detail=cached_detail,
                fingerprint=fingerprint,
            )

        self.cache_misses += 1
        if self._mode == ReplayMode.LOCKED:
            raise LLMCacheMissError(
                "LOCKED mode: no cached response for this LLM call "
                f"(fingerprint={fingerprint}). "
                "Run with --mode=selective to allow live calls for new steps."
            )

        self.live_calls += 1
        logger.warning("LLM cache miss [%s]: calling live model", self._mode.value)
        return CacheDecision(
            hit=False,
            ai_message=None,
            cached_detail=None,
            fingerprint=fingerprint,
        )

    def check_messages(self, messages: List[Any], state: Optional[Dict[str, Any]] = None) -> CacheDecision:
        state = state or {}
        provider = str(state.get("provider", "unknown"))
        method = str(state.get("method", "chat"))
        invocation = {
            "temperature": state.get("temperature"),
            "max_tokens": state.get("max_tokens"),
            "top_p": state.get("top_p"),
            "stream": state.get("stream", False),
        }
        request_params = self._build_request_params(
            messages=messages,
            model=str(state.get("model", "")),
            tools=state.get("tools"),
            invocation=invocation,
        )
        return self.check_request(
            provider=provider,
            method=method,
            request_params=request_params,
            source="middleware"
        )

    def _infer_provider(self, llm: Any) -> str:
        name = llm.__class__.__name__.lower()
        if "openai" in name:
            return "azure_openai" if "azure" in name else "openai"
        if "anthropic" in name:
            return "anthropic"
        if "google" in name:
            return "google"
        if "cohere" in name:
            return "cohere"
        if "mistral" in name:
            return "mistral"
        return "unknown"

    def _infer_method(self, provider: str) -> str:
        method_map = {
            "openai": "chat.completions.create",
            "azure_openai": "chat.completions.create",
            "anthropic": "messages.create",
            "google": "generateContent",
            "cohere": "chat",
            "mistral": "chat",
        }
        return method_map.get(provider, "chat")

    def wrap_model(self, llm: Any, provider: Optional[str] = None, method: Optional[str] = None) -> Any:
        gatekeeper = self
        resolved_provider = provider or self._infer_provider(llm)
        resolved_method = method or self._infer_method(resolved_provider)

        class ReplayAwareModel:
            def __init__(self):
                self._llm = llm
                self._provider = resolved_provider
                self._method = resolved_method

            def _resolve_model_name(self, kwargs: Dict[str, Any]) -> str:
                inv = kwargs.get("invocation_params", {})
                return (
                    inv.get("model_name")
                    or inv.get("model")
                    or getattr(self._llm, "model_name", "")
                    or getattr(self._llm, "model", "")
                    or ""
                )

            def _resolve_tools(self, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
                inv = kwargs.get("invocation_params", {})
                tools = inv.get("tools", [])
                if not tools:
                    tools = inv.get("functions", [])
                return tools if isinstance(tools, list) else []

            def invoke(self, messages: List[Any], *args: Any, **kwargs: Any) -> Any:
                request_params = gatekeeper._build_request_params(
                    messages=messages,
                    model=self._resolve_model_name(kwargs),
                    tools=self._resolve_tools(kwargs),
                    invocation=kwargs.get("invocation_params", {}),
                )
                decision = gatekeeper.check_request(
                    provider=self._provider,
                    method=self._method,
                    request_params=request_params,
                    source="wrapper",
                )
                if decision.hit and decision.ai_message is not None:
                    return decision.ai_message
                return self._llm.invoke(messages, *args, **kwargs)

            async def ainvoke(self, messages: List[Any], *args: Any, **kwargs: Any) -> Any:
                request_params = gatekeeper._build_request_params(
                    messages=messages,
                    model=self._resolve_model_name(kwargs),
                    tools=self._resolve_tools(kwargs),
                    invocation=kwargs.get("invocation_params", {}),
                )
                decision = gatekeeper.check_request(
                    provider=self._provider,
                    method=self._method,
                    request_params=request_params,
                    source="wrapper",
                )
                if decision.hit and decision.ai_message is not None:
                    return decision.ai_message
                return await self._llm.ainvoke(messages, *args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._llm, name)

        return ReplayAwareModel()

    def create_middleware(self, mode: ReplayMode = ReplayMode.SELECTIVE) -> List:
        self._mode = mode

        if mode == ReplayMode.FULL:
            return []

        def check_llm_cache(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])
            if not messages:
                return state

            decision = self.check_messages(messages, state=state)
            updated_state = dict(state)
            updated_messages = list(messages)
            updated_state["agenttest_cache_hit"] = decision.hit

            if decision.hit and decision.ai_message is not None:
                updated_messages.append(decision.ai_message)
                updated_state["messages"] = updated_messages
                return updated_state

            updated_state["messages"] = updated_messages
            return updated_state

        def log_llm_response(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                logger.info("LLM response captured (live call)")
            return state

        return [check_llm_cache, log_llm_response]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "interception_attempts": self.interception_attempts,
            "wrapper_interceptions": self.wrapper_interceptions,
            "middleware_interceptions": self.middleware_interceptions,
            "runtime_interceptions": self.runtime_interceptions,
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
        self.interception_attempts = 0
        self.wrapper_interceptions = 0
        self.middleware_interceptions = 0
        self.runtime_interceptions = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.live_calls = 0
